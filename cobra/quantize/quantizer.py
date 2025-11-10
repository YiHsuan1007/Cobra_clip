from re import U
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import tqdm
import numpy as np
import pdb
import math
from .observers.hist_observers import PercentileObserver, KLObserver, MSEObserver
from .observers.minmax_observers import MinMaxObserver


CLIPMIN = 1e-5


class _ObserverDTypeAdapter:
    """
    Minimal dtype adapter supplying qmin/qmax/bitwidth expected by observers.
    """

    __slots__ = ("torch_dtype", "bitwidth", "qmin", "qmax", "signed")

    def __init__(self, torch_dtype: Optional[torch.dtype], bitwidth: int, signed: bool):
        if bitwidth <= 0:
            raise ValueError("bitwidth must be positive")
        self.torch_dtype = torch_dtype
        self.bitwidth = bitwidth
        self.signed = signed
        if signed:
            limit = 1 << (bitwidth - 1)
            self.qmin = -limit
            self.qmax = limit - 1
        else:
            self.qmin = 0
            self.qmax = (1 << bitwidth) - 1

    def __repr__(self) -> str:
        prefix = "int" if self.signed else "uint"
        return f"{prefix}{self.bitwidth}"

class ClampSte(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,min_,max_):
        return x.clamp(min_,max_)
    
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output.clone(),None,None

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
        rescale=False,
        rescale_limit=False,
        has_batch_dim = False,
        is_weight=False,
        observe="minmax",
        percent = 0.999999,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point or self.symmetric:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.rescale = rescale # for channel-rescale
        self.rescale_limit = rescale_limit

        self.mode = str(observe or "minmax").lower()
        self.percent = float(percent)

        init_value = 4.0  # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric  # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
        
        if rescale:
            if rescale_limit:
                self.rescale_param = nn.Parameter(torch.zeros(dim1,1) )
            else:
                self.rescale_param = nn.Parameter(torch.ones(dim1,1) )

        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size
        
        self.has_batch_dim = has_batch_dim
        self.is_observing = False
        self.is_dynamic_quant = True
        granularity = 'dim{}'.format(per_channel_axes[0]) if len(per_channel_axes) > 0 else 'tensor'
        
        if observe == "percentile":
            self.observer = PercentileObserver(percent=self.percent,granularity=granularity)
        else:
            self.observer = MinMaxObserver(granularity=granularity)
            self.observer.owner = self
        
            self.observered = False
            
            self.is_weight = is_weight
            self._reference_shape = torch.Size(shape) if shape is not None else None
            self._scale_storage_shape: Optional[torch.Size] = None
            self._zero_storage_shape: Optional[torch.Size] = None
            self._pending_percentile = False

    @property
    def pending(self) -> bool:
        return bool(getattr(self, "_pending_percentile", False))

    @pending.setter
    def pending(self, value: bool) -> None:
        self._pending_percentile = bool(value)

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def _update_storage_shapes(self) -> None:
        if isinstance(getattr(self, "scale", None), torch.Tensor):
            self._scale_storage_shape = self.scale.shape
        else:
            self._scale_storage_shape = None

        zero_attr = getattr(self, "round_zero_point", None)
        if isinstance(zero_attr, torch.Tensor):
            self._zero_storage_shape = zero_attr.shape
        else:
            self._zero_storage_shape = None

    def _reshape_canonical(self, value: Optional[torch.Tensor], shape: Optional[torch.Size]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if shape is None:
            return value
        if len(shape) == 0:
            return value
        expected = math.prod(shape)
        if value.numel() != expected:
            return value
        return value.reshape(shape)

    def _infer_scale_shape(self, numel: int) -> torch.Size:
        if self._scale_storage_shape is not None:
            return self._scale_storage_shape
        if self._reference_shape is not None and self.per_channel_axes:
            shape = [1] * len(self._reference_shape)
            remaining = numel
            for axis in self.per_channel_axes:
                dim_val = self._reference_shape[axis]
                shape[axis] = dim_val
                remaining //= max(dim_val, 1)
            return torch.Size(shape)
        return torch.Size([numel])

    def init_from_weight(self, w: torch.Tensor) -> None:
        """
        Initialise quantizer parameters from a weight tensor when no observer
        statistics are available.

        This is a lightweight fallback used prior to percentile stats being
        replayed. Existing parameters or previously applied percentile stats are
        left untouched.
        """
        if isinstance(getattr(self, "scale", None), torch.Tensor) and self.scale.numel() > 0:
            return
        if getattr(self, "observered", False):
            return
        if not isinstance(w, torch.Tensor):
            raise TypeError("init_from_weight expects a torch.Tensor.")
        if w.numel() == 0:
            return

        with torch.no_grad():
            weight = w.detach()
            if weight.is_sparse:
                weight = weight.to_dense()

            finite_mask = torch.isfinite(weight)
            if not bool(finite_mask.all().item()):
                weight = weight[finite_mask]
            if weight.numel() == 0:
                return

            weight = weight.to(dtype=torch.float32)
            xmin = weight.min()
            xmax = weight.max()
            if not (torch.isfinite(xmin) and torch.isfinite(xmax)):
                return

            xmin = xmin.view(1)
            xmax = xmax.view(1)
            if self.symmetric or self.disable_zero_point:
                abs_max = torch.max(xmax.abs(), xmin.abs())
                scale = abs_max / (2 ** (self.n_bits - 1) - 1)
                scale = scale.clamp(min=CLIPMIN, max=1e4)
                zero_point_tensor: Optional[torch.Tensor] = None
            else:
                dynamic_range = (xmax - xmin).clamp(min=0)
                scale = dynamic_range / (2 ** self.n_bits - 1)
                scale = scale.clamp(min=CLIPMIN, max=1e4)
                zero_point_tensor = -(xmin / scale)
                zero_point_tensor = zero_point_tensor.clamp(min=-1e4, max=1e4).round()

            if not torch.isfinite(scale).all():
                return

            scale = scale.detach()
            zero_arg = None if zero_point_tensor is None else zero_point_tensor.detach()
            self.register_qparams(scale, zero_arg)

    def _resolve_runtime_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(getattr(self, "scale", None), torch.Tensor):
            raise RuntimeError("UniformAffineQuantizer.scale is not initialised.")

        scale = self.scale
        zero_point = getattr(self, "round_zero_point", None)

        scale_shape = self._scale_storage_shape
        scale = self._reshape_canonical(scale, scale_shape)
        if scale.dim() == 0:
            scale = scale.reshape([1])
        if scale_shape is not None and len(scale_shape) == x.dim():
            scale = scale.reshape(scale_shape)
        elif self.per_channel_axes:
            scale = self.expand_scale_shape_2_x(x, scale)
        else:
            while scale.dim() < x.dim():
                scale = scale.unsqueeze(0)

        zero_point = self._reshape_canonical(zero_point, self._zero_storage_shape)
        if zero_point is not None:
            if zero_point.dim() == 0:
                zero_point = zero_point.reshape([1])
            if self._zero_storage_shape is not None and len(self._zero_storage_shape) == x.dim():
                zero_point = zero_point.reshape(self._zero_storage_shape)
            elif self.per_channel_axes:
                zero_point = self.expand_scale_shape_2_x(x, zero_point)
            else:
                while zero_point.dim() < x.dim():
                    zero_point = zero_point.unsqueeze(0)

        return scale.to(x.device), None if zero_point is None else zero_point.to(x.device)

    def export_percentile_stats(self) -> Optional[Dict[str, Any]]:
        """Return percentile statistics suitable for per-quantizer export."""
        if str(getattr(self, "mode", "")).lower() != "percentile":
            return None
        if getattr(self, "_pending_percentile", False):
            return None

        scale_tensor = getattr(self, "scale", None)
        if not isinstance(scale_tensor, torch.Tensor) or scale_tensor.numel() == 0:
            return None
        scale_tensor = scale_tensor.detach().float().cpu()

        zero_tensor = getattr(self, "round_zero_point", None)
        if isinstance(zero_tensor, torch.Tensor) and zero_tensor.numel() > 0:
            zero_tensor = zero_tensor.detach().float().cpu()
        else:
            zero_tensor = None

        def _tensor_to_python(tensor: Optional[torch.Tensor]) -> Any:
            if tensor is None:
                return None
            tensor = tensor.detach().cpu()
            if tensor.numel() == 1:
                return float(tensor.item())
            return tensor.reshape(-1).tolist()

        def _tensor_to_int_payload(tensor: Optional[torch.Tensor]) -> Any:
            value = _tensor_to_python(tensor)
            if value is None:
                return None
            if isinstance(value, list):
                return [int(round(v)) for v in value]
            return int(round(value))

        def _observer_meta() -> Optional[Dict[str, Any]]:
            observer = getattr(self, "observer", None)
            if observer is None:
                return None
            meta: Dict[str, Any] = {
                "type": observer.__class__.__name__,
                "granularity": getattr(observer, "granularity", None),
            }
            for attr in ("left_percent", "right_percent", "percentile_mode"):
                if hasattr(observer, attr):
                    raw_value = getattr(observer, attr)
                    if isinstance(raw_value, torch.Tensor):
                        if raw_value.numel() == 1:
                            raw_value = float(raw_value.item())
                        else:
                            raw_value = raw_value.detach().cpu().reshape(-1).tolist()
                    meta[attr] = raw_value
            return meta

        bits = int(getattr(self, "n_bits", getattr(self, "bits", 0)) or 0)
        if bits <= 0:
            return None
        qmax = float(getattr(self, "qmax", 0))
        qmin = float(getattr(self, "qmin", 0))
        if self.symmetric or self.disable_zero_point or zero_tensor is None:
            clip_tensor = scale_tensor * (2 ** (bits - 1) - 1)
            clip_max = clip_tensor
            clip_min = -clip_tensor
        else:
            clip_max = (qmax - zero_tensor) * scale_tensor
            clip_min = (qmin - zero_tensor) * scale_tensor

        percentile = float(getattr(self, "percent", getattr(self, "percentile", 0.0)))
        payload: Dict[str, Any] = {
            "mode": "percentile",
            "percentile": percentile,
            "clip_min": _tensor_to_python(clip_min),
            "clip_max": _tensor_to_python(clip_max),
            "scale": _tensor_to_python(scale_tensor),
            "zero_point": _tensor_to_int_payload(zero_tensor) if zero_tensor is not None else 0,
            "bits": bits,
            "observer_meta": _observer_meta(),
        }
        return payload

    def apply_percentile_stats(self, stats: Mapping[str, Any]) -> None:
        """Ingest percentile statistics produced by :meth:`export_percentile_stats`."""
        if str(getattr(self, "mode", "")).lower() != "percentile":
            return
        if not isinstance(stats, Mapping):
            raise TypeError("Percentile stats must be provided as a mapping.")
        if stats.get("pending"):
            return

        def _to_tensor(value: Any) -> torch.Tensor:
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim == 0:
                tensor = tensor.reshape(1)
            return tensor

        clip_value = stats.get("clip")
        clip_min_value = stats.get("clip_min")
        clip_max_value = stats.get("clip_max")
        min_value = stats.get("min")
        max_value = stats.get("max")

        if clip_min_value is not None and clip_max_value is not None:
            min_value = clip_min_value
            max_value = clip_max_value

        for key in ("p99_999", "p99_99", "p99_9", "p99"):
            if key in stats:
                clip_value = stats[key]
                break

        if (min_value is None or max_value is None) and stats.get("scale") is not None:
            scale_tensor = _to_tensor(stats["scale"])
            zero_value = stats.get("zero_point")
            zero_tensor = _to_tensor(zero_value) if zero_value is not None else None
            bits_value = int(stats.get("bits", getattr(self, "n_bits", 0)))
            if bits_value > 0:
                if self.symmetric or self.disable_zero_point or zero_tensor is None:
                    clip_tensor = scale_tensor * (2 ** (bits_value - 1) - 1)
                    min_value = -clip_tensor
                    max_value = clip_tensor
                else:
                    qmax = float(getattr(self, "qmax", 0))
                    qmin = float(getattr(self, "qmin", 0))
                    max_value = (qmax - zero_tensor) * scale_tensor
                    min_value = (qmin - zero_tensor) * scale_tensor

        clip_tensor = _to_tensor(clip_value) if clip_value is not None else None
        min_tensor = _to_tensor(min_value) if min_value is not None else None
        max_tensor = _to_tensor(max_value) if max_value is not None else None

        if clip_tensor is not None:
            min_tensor = -clip_tensor
            max_tensor = clip_tensor

        if min_tensor is None or max_tensor is None:
            return

        device = self.scale.device if isinstance(getattr(self, "scale", None), torch.Tensor) else torch.device("cpu")
        min_tensor = min_tensor.to(device=device, dtype=torch.float32)
        max_tensor = max_tensor.to(device=device, dtype=torch.float32)

        if self.symmetric or self.disable_zero_point:
            abs_max = torch.max(max_tensor.abs(), min_tensor.abs())
            self.symmetric_cal_scale(-abs_max, abs_max)
        else:
            self.assymmetric_cal_scale(min_tensor, max_tensor)

        if hasattr(self, "observer"):
            self.observer = None
        if hasattr(self, "observered"):
            self.observered = True
        self.is_observing = False
        self._update_storage_shapes()
        self._pending_percentile = False

    ## dtype 需正規化成帶 qmin/qmax/bitwidth 的物件，方便 observer 使用硬體語義做縮放。
    def _normalize_observer_dtype(self, candidate: Any, signed_fallback: bool) -> Any:
        if candidate is None:
            bitwidth = getattr(self, "n_bits", 8)
            base = torch.qint8 if signed_fallback else torch.quint8
            return _ObserverDTypeAdapter(base, bitwidth, signed_fallback)
        if isinstance(candidate, _ObserverDTypeAdapter):
            return candidate
        if hasattr(candidate, "qmin") and hasattr(candidate, "qmax") and hasattr(candidate, "bitwidth"):
            return candidate
        if isinstance(candidate, str):
            key = candidate.lower()
            if key.startswith("int"):
                bitwidth = int(key[3:])
                return _ObserverDTypeAdapter(torch.qint8, bitwidth, True)
            if key.startswith("uint"):
                bitwidth = int(key[4:])
                return _ObserverDTypeAdapter(torch.quint8, bitwidth, False)
        if isinstance(candidate, torch.dtype):
            bitwidth = getattr(self, "n_bits", 8)
            signed = candidate == torch.qint8
            return _ObserverDTypeAdapter(candidate, bitwidth, signed)
        if isinstance(candidate, int):
            bitwidth = int(candidate)
            base = torch.qint8 if signed_fallback else torch.quint8
            return _ObserverDTypeAdapter(base, bitwidth, signed_fallback)
        if hasattr(candidate, "dtype"):
            return self._normalize_observer_dtype(candidate.dtype, signed_fallback)
        return candidate

    def _align_params_with_input(self, x: torch.Tensor) -> None:
        if not isinstance(getattr(self, "scale", None), torch.Tensor):
            return

        if self.per_channel_axes:
            self.scale = self.expand_scale_shape_2_x(x, self.scale)
            if isinstance(self.round_zero_point, torch.Tensor):
                self.round_zero_point = self.expand_scale_shape_2_x(x, self.round_zero_point)
        else:
            while self.scale.dim() < x.dim():
                self.scale = self.scale.unsqueeze(0)
            if isinstance(self.round_zero_point, torch.Tensor):
                while self.round_zero_point.dim() < x.dim():
                    self.round_zero_point = self.round_zero_point.unsqueeze(0)

        self._update_storage_shapes()

    ## finalize 階段需確保 dtype 來源一致，權重預設 qint8、啟用預設 quint8。
    def calculate_qparams(self, dtype: Any = None, symmetric: Optional[bool] = None):
        if symmetric is None:
            if hasattr(self, "signed"):
                symmetric = bool(self.signed)
            else:
                symmetric = bool(getattr(self, "symmetric", True))
        dtype_candidate = dtype if dtype is not None else getattr(self, "dtype", None)
        if dtype_candidate is None:
            base_dtype = torch.qint8 if getattr(self, "is_weight", False) or getattr(self, "signed", False) else torch.quint8
            dtype_candidate = base_dtype
        signed_fallback = bool(getattr(self, "signed", False)) or bool(getattr(self, "is_weight", False))
        dtype_normalized = self._normalize_observer_dtype(dtype_candidate, signed_fallback)
        impl = getattr(super(UniformAffineQuantizer, self), "calculate_qparams", None)
        if callable(impl):
            return impl(dtype=dtype_normalized, symmetric=symmetric)
        observer = getattr(self, "observer", None)
        if observer is None or not hasattr(observer, "calculate_qparams"):
            return None
        quant_param = observer.calculate_qparams(dtype_normalized, symmetric)
        existing_scale = getattr(self, "scale", None)
        existing_zero = getattr(self, "round_zero_point", None)

        def _is_valid_tensor(candidate: Optional[torch.Tensor]) -> bool:
            return (
                isinstance(candidate, torch.Tensor)
                and candidate.numel() > 0
                and torch.isfinite(candidate.detach().float()).all()
            )

        def _resolve_label() -> str:
            for attr in ("_quant_label", "_module_path", "_logical_path"):
                value = getattr(self, attr, None)
                if isinstance(value, str) and value:
                    return value
            return self.__class__.__name__

        scale_tensor = getattr(quant_param, "scale", None)
        zero_point = getattr(quant_param, "zero_point", None)

        scale_valid = _is_valid_tensor(scale_tensor)
        if not scale_valid:
            if _is_valid_tensor(existing_scale):
                logging.warning(
                    "[Quant] Observer produced empty/invalid scale; preserving existing parameters for %s",
                    _resolve_label(),
                )
                return quant_param
            scale_tensor = torch.full((1,), CLIPMIN, dtype=torch.float32)
            zero_point = None
        else:
            scale_tensor = scale_tensor.detach().clone()

        if _is_valid_tensor(existing_scale):
            scale_tensor = scale_tensor.to(device=existing_scale.device, dtype=existing_scale.dtype)
        self.scale = scale_tensor

        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            zero_valid = _is_valid_tensor(zero_point)
            if zero_valid:
                zero_tensor = zero_point.detach().clone()
                target_device = self.scale.device if isinstance(self.scale, torch.Tensor) else None
                if target_device is not None:
                    zero_tensor = zero_tensor.to(device=target_device)
                self.round_zero_point = zero_tensor
            elif _is_valid_tensor(existing_zero):
                logging.warning(
                    "[Quant] Observer produced empty/invalid zero-point; preserving existing parameters for %s",
                    _resolve_label(),
                )
                self.round_zero_point = existing_zero
            else:
                self.round_zero_point = None
        self._update_storage_shapes()
        return quant_param

    def fake_quant(self, x):
        scale, round_zero_point = self._resolve_runtime_params(x)
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = 0.5 + F.sigmoid(rescale_param)
            if len(rescale_param.shape) == 2 and len(x_dequant.shape)==3:
                rescale_param = rescale_param.unsqueeze(-1)
            x_dequant = x_dequant*rescale_param.to(x_dequant.device)
        return x_dequant

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        
        if self.is_weight:#权重量化，没有observe过程
            if True:#not self.is_dynamic_quant:
                if  self.is_observing:
                    return x
                if self.observer is not None:
                    self.observer.update(x)
                    xmin,xmax = self.observer.cal_min_max()
                    self.assymmetric_cal_scale(xmin,xmax)
                    self._align_params_with_input(x)
                    self.observer = None
                x_dequant = self.fake_quant(x)
                return x_dequant.type_as(x)
            # else:
            #     if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            #         self.per_token_dynamic_calibration(x)
            #     else:
            #         self.dynamic_per_tensor_calibration(x)
            #     x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
            #     return x_dequant
        else:#激活量化
            if not self.is_dynamic_quant:
                if self.is_observing:
                    self.observer.update(x)
                    return x.type_as(x)
                else:
                    if not self.observered:
                        xmin,xmax = self.observer.cal_min_max()
                        self.assymmetric_cal_scale(xmin,xmax)
                        self._align_params_with_input(x)
                        self.observered = True
                        self.observer = None
                    x_dequant = self.fake_quant(x)
                    return x_dequant.type_as(x)
                    
            else:
                if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
                    self.per_token_dynamic_calibration(x)
                else:
                    self.dynamic_per_tensor_calibration(x)

                self._align_params_with_input(x)
                x_dequant = self.fake_quant(x)
                return x_dequant.type_as(x)

    def expand_scale_shape_2_x(self, x, scale):
        if self.per_channel_axes:
            dim=self.per_channel_axes[0]
            for i in range(len(x.shape)):
                if i != dim:
                    scale = scale.unsqueeze(i)
        return scale

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros(
                    (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
                )
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)
        if self.dynamic_method == "per_channel":
            if len(self.per_channel_axes):
                assert len(self.per_channel_axes) == 1,"must be one"
                reduce_shape = list(range(x.dim()))
                reduce_shape.remove(self.per_channel_axes[0])
            else:
                reduce_shape = list(range(x.dim()-1))
        else:
            reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        self.xmin_tmp = xmin.detach()
        self.xmax_tmp = xmax.detach()
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            dynamic_range = xmax - xmin
            scale = dynamic_range / (2**self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        self._update_storage_shapes()
    
    def MaxMin_except_first_dim(self,tensor,func):
        # 获取张量的维度数
        dims = list(range(1, tensor.dim()))
        # 逐步在每个维度上取最大值
        for dim in dims:
            tensor, _ = func(tensor, dim=dim, keepdim=True)
        return tensor
    
    def dynamic_per_tensor_calibration(self,x):
        if not self.has_batch_dim:
            xmin = x.min()
            xmax = x.max()
        else:
            shape = [1] * len(x.shape)
            shape[0] = -1
            xmin = self.MaxMin_except_first_dim(x,torch.min).view(shape)
            xmax = self.MaxMin_except_first_dim(x,torch.max).view(shape)
        if self.symmetric or self.disable_zero_point:
            self.symmetric_cal_scale(xmin,xmax)
        else:
            self.assymmetric_cal_scale(xmin,xmax)

    def symmetric_cal_scale(self,xmin,xmax):
        abs_max = torch.max(xmax.abs(), xmin.abs())
        scale = abs_max / (2 ** (self.n_bits - 1) - 1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        self.round_zero_point = None
        self._update_storage_shapes()
        
    def assymmetric_cal_scale(self,xmin,xmax):
        dynamic_range = xmax - xmin
        scale = dynamic_range / (2**self.n_bits - 1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        self._update_storage_shapes()
    
    def normal_quantize(self, x, scales: torch.Tensor, mig_cof: torch.Tensor):
        s = (scales / mig_cof).max()
        s = s / (2**self.n_bits - 1)
        self.scale = s
        # only support symmetric quantization
        self.round_zero_point = None
        self._update_storage_shapes()
        
    def scale_frexp(self):
        k = 16
        m = (self.scale * (2**k)).round()
        self.scale = m * (2**(-k))
        self._update_storage_shapes()

        return self.scale

    def export_params(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(getattr(self, "scale", None), torch.Tensor):
            raise RuntimeError("Cannot export parameters before scale is initialised.")

        self._update_storage_shapes()
        scale_tensor = self.scale.detach().clone()
        if scale_tensor.numel() == 0:
            raise RuntimeError("Scale tensor is empty.")
        if scale_tensor.numel() == 1:
            scale_tensor = scale_tensor.reshape(1)
        else:
            scale_tensor = scale_tensor.reshape(-1)

        zero_tensor: Optional[torch.Tensor] = None
        zero_buffer = getattr(self, "_buffers", {}).get("zero_point") if hasattr(self, "_buffers") else None
        if isinstance(zero_buffer, torch.Tensor) and zero_buffer.numel() > 0:
            zero_tensor = zero_buffer.detach().clone()
        elif isinstance(self.round_zero_point, torch.Tensor):
            zero_tensor = self.round_zero_point.detach().clone()
        if isinstance(zero_tensor, torch.Tensor):
            zero_tensor = zero_tensor.reshape(1) if zero_tensor.numel() == 1 else zero_tensor.reshape(-1)
        else:
            zero_tensor = None

        return scale_tensor, zero_tensor

    def load_exported_params(
        self,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
    ) -> None:
        if scale.dim() > 1 or scale.numel() == 0:
            raise ValueError("`scale` must be a non-empty 1D tensor.")

        target_scale_shape = self._infer_scale_shape(scale.numel())
        restored_scale = scale.reshape(target_scale_shape).to(scale.device)

        restored_zero: Optional[torch.Tensor] = None
        if zero_point is not None and zero_point.numel() > 0:
            if zero_point.dim() > 1:
                raise ValueError("`zero_point` must be 1D when provided.")
            target_zero_shape = self._zero_storage_shape or target_scale_shape
            restored_zero = zero_point.reshape(target_zero_shape).to(scale.device)

        if isinstance(getattr(self, "_buffers", {}).get("scale", None), torch.Tensor):
            self._buffers["scale"] = restored_scale
        self.scale = restored_scale
        self.round_zero_point = restored_zero
        if restored_zero is not None and isinstance(getattr(self, "_buffers", {}).get("zero_point", None), torch.Tensor):
            self._buffers["zero_point"] = restored_zero
        self._update_storage_shapes()

    ## 統一走 register_qparams，避免重複維護緩衝區型態。
    def register_scales_and_zeros(self):
        scale_canonical, zero_canonical = self.export_params()
        self.register_qparams(scale_canonical, zero_canonical)

    ## 將量化參數註冊為 buffer，保留原本名稱/型別供匯出或重載。
    def register_qparams(self, scale: torch.Tensor, zero_point: Optional[torch.Tensor]) -> None:
        if not isinstance(scale, torch.Tensor):
            raise TypeError("scale must be a torch.Tensor")
        if scale.numel() == 0:
            raise ValueError("scale must be non-empty")
        if zero_point is not None and not isinstance(zero_point, torch.Tensor):
            raise TypeError("zero_point must be a torch.Tensor or None")

        if "scale" in self.__dict__.get("_buffers", {}):
            del self._buffers["scale"]
        if hasattr(self, "scale"):
            delattr(self, "scale")
        scale_buffer = scale.detach().clone()
        self.register_buffer("scale", scale_buffer)

        if "zero_point" in self.__dict__.get("_buffers", {}):
            del self._buffers["zero_point"]
        if hasattr(self, "zero_point"):
            delattr(self, "zero_point")

        scale_device = scale_buffer.device

        if zero_point is None or (isinstance(zero_point, torch.Tensor) and zero_point.numel() == 0):
            zero_buffer = torch.zeros(0, dtype=torch.int32, device=scale_device)
            runtime_zero: Optional[torch.Tensor] = None
        else:
            zero_source = zero_point.detach().clone()
            if isinstance(zero_source, torch.Tensor) and zero_source.is_floating_point():
                zero_source = zero_source.round()
            zero_buffer = zero_source.to(device=scale_device)
            if zero_buffer.dtype.is_floating_point:
                zero_buffer = zero_buffer.to(dtype=torch.int32)
            runtime_zero = zero_source.to(device=scale_device, dtype=scale_buffer.dtype)

        if hasattr(self, "round_zero_point"):
            delattr(self, "round_zero_point")
        self.register_buffer("zero_point", zero_buffer)

        self.round_zero_point = runtime_zero
        self._update_storage_shapes()

    def quant2int(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.n_bits >= 16 or not self.enable:
            unity = torch.ones(1, dtype=x.dtype, device=x.device)
            return x, unity, None
        if self.metric == "fix0to1":
            scale = torch.full((1,), 1.0 / (2**self.n_bits - 1), dtype=x.dtype, device=x.device)
            x_int = (x * (2**self.n_bits - 1)).round().clamp(self.qmin, self.qmax)
            return x_int, scale, None
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        scale, zero_point = self._resolve_runtime_params(x)
        x_int = round_ste(x / scale)
        if zero_point is not None:
            x_int = x_int.add(zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        
        if self.group_size:
            x_int = x_int.reshape(dim1, dim2)
            scale = scale.reshape(dim1, dim2)
            if zero_point is not None:
                zero_point = zero_point.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_int = x_int[:, : -self.deficiency]
            scale = scale[:, : -self.deficiency]
            if zero_point is not None:
                zero_point = zero_point[:, : -self.deficiency]
        return x_int, scale.detach(), None if zero_point is None else zero_point.detach()
    
    def dequant(
        self,
        x_int: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
    ):
        target_shape = x_int.shape

        scale = scale.to(x_int.device)
        if scale.shape != target_shape:
            scale = torch.broadcast_to(scale, target_shape)

        x_dequant = x_int.to(scale.dtype)

        if zero_point is not None:
            zero_tensor = zero_point.to(x_int.device, dtype=scale.dtype)
            if zero_tensor.shape != target_shape:
                zero_tensor = torch.broadcast_to(zero_tensor, target_shape)
            x_dequant = x_dequant.sub(zero_tensor)

        x_dequant = x_dequant.mul(scale)

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = 0.5 + F.sigmoid(rescale_param)
            if len(rescale_param.shape) == 2 and len(x_dequant.shape) == 3:
                rescale_param = rescale_param.unsqueeze(-1)
            x_dequant = x_dequant * rescale_param.to(x_dequant.device)
        return x_dequant



class ActQuantizer(nn.Module):
    def __init__(self):
        self.register_parameter("scale",torch.ones(1))
        self.register_buffer("calibed_enabled",torch.tensor([0],dtype=torch.uint8))
    
    # @property
    # def calib
    
    def forward(self,x):
        pass


if __name__ == "__main__":
    cfg = {"dynamic_method":"per_tensor","n_bits":8,"symmetric":True}
    weight = torch.randn(100,100)
    quantizer = UniformAffineQuantizer(**cfg)
    weight_quant = quantizer.forward(weight)
    diff = weight-weight_quant
    print(diff.sum())

