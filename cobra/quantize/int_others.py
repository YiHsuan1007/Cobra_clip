import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Sequence

from .quantizer import UniformAffineQuantizer


def _shift_to_int32(t: torch.Tensor, zero: Optional[torch.Tensor]) -> torch.Tensor:
    shifted = t.to(torch.int32)
    if zero is not None:
        shifted = shifted - zero.to(device=t.device, dtype=torch.int32)
    return shifted


def _prepare_scale(scale: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=ref.device, dtype=torch.float32)
    if scale.ndim == 0:
        return scale
    if scale.ndim == 1:
        scale = scale.unsqueeze(0)
    while scale.ndim < ref.ndim:
        scale = scale.unsqueeze(-1)
    return scale


def _prepare_bias(bias: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    bias = bias.to(device=ref.device, dtype=ref.dtype)
    if bias.ndim == 1:
        view_shape = [1] * ref.ndim
        view_shape[1] = -1
        bias = bias.view(*view_shape)
    return bias


def int_conv2d(
    x_int: torch.Tensor,
    w_int: torch.Tensor,
    a_scale: torch.Tensor,
    w_scale: torch.Tensor,
    a_zero: Optional[torch.Tensor] = None,
    w_zero: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """
    Integer convolution helper that mirrors the behaviour of an INT8/INT32 kernel with float dequantisation.
    """
    x_shift = _shift_to_int32(x_int, a_zero)
    w_shift = _shift_to_int32(w_int, w_zero)

    y_int = F.conv2d(
        x_shift,
        w_shift,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    act_scale = _prepare_scale(a_scale, y_int)
    weight_scale = _prepare_scale(w_scale, y_int)
    effective_scale = act_scale * weight_scale

    y_fp = y_int.to(torch.float32) * effective_scale
    if bias is not None:
        y_fp = y_fp + _prepare_bias(bias, y_fp)
    return y_fp


def _dequant_tensor(
    tensor_int: torch.Tensor,
    scale: torch.Tensor,
    zero: Optional[torch.Tensor],
) -> torch.Tensor:
    shifted = _shift_to_int32(tensor_int, zero)
    return shifted.to(torch.float32) * scale.to(torch.float32)


def int_add(
    lhs_int: torch.Tensor,
    rhs_int: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    lhs_zero: Optional[torch.Tensor] = None,
    rhs_zero: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    lhs = _dequant_tensor(lhs_int, lhs_scale, lhs_zero)
    rhs = _dequant_tensor(rhs_int, rhs_scale, rhs_zero)
    return lhs + rhs


def int_mul(
    lhs_int: torch.Tensor,
    rhs_int: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    lhs_zero: Optional[torch.Tensor] = None,
    rhs_zero: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    lhs = _dequant_tensor(lhs_int, lhs_scale, lhs_zero)
    rhs = _dequant_tensor(rhs_int, rhs_scale, rhs_zero)
    return lhs * rhs


def int_cat(
    tensors_int: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    zeros: Optional[Sequence[Optional[torch.Tensor]]] = None,
    dim: int = 0,
) -> torch.Tensor:
    if zeros is None:
        zeros = [None] * len(tensors_int)
    dequantised = [
        _dequant_tensor(t_int, scale, zero)
        for t_int, scale, zero in zip(tensors_int, scales, zeros)
    ]
    return torch.cat(dequantised, dim=dim)



# C8C8Add
class QuantAdd(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict | None = None,
        x2_quant_params: dict | None = None,
        *,
        base_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.x1_quantizer = UniformAffineQuantizer(**(x1_quant_params or {}))
        self.x2_quantizer = UniformAffineQuantizer(**(x2_quant_params or {}))
        self.use_act_quant = False
        object.__setattr__(self, "base_module", base_module)
        if "base_module" in self._modules:
            self._modules.pop("base_module", None)
        if base_module is not None:
            setattr(base_module, "_quant_pct_wrapped", True)
        self._is_cobra_internal = True
        self._quant_pct_wrapped = True

    def forward(self, *inputs, **kwargs):
        if not inputs:
            raise ValueError("QuantAdd expects at least one tensor input.")
        args = list(inputs)
        if self.use_act_quant:
            args[0] = self.x1_quantizer(args[0])
            if len(args) > 1:
                args[1] = self.x2_quantizer(args[1])

        if self.base_module is not None:
            return self.base_module(*args, **kwargs)

        if len(args) == 1:
            return args[0]

        result = args[0] + args[1]
        for extra in args[2:]:
            result = result + extra
        return result

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.use_act_quant = act_quant
        if self.base_module is not None and hasattr(self.base_module, "set_quant_state"):
            self.base_module.set_quant_state(weight_quant, act_quant)


class QuantSoftmax(nn.Module):
    def __init__(
        self,
        act_quant_params: dict | None = None,
        dim: int = -1,
        *,
        base_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.act_quantizer = UniformAffineQuantizer(**(act_quant_params or {}))
        object.__setattr__(self, "base_module", base_module)
        if "base_module" in self._modules:
            self._modules.pop("base_module", None)
        if base_module is not None:
            setattr(base_module, "_quant_pct_wrapped", True)
            dim = getattr(base_module, "dim", dim)
        self.dim = dim
        self.use_act_quant = False
        self._is_cobra_internal = True
        self._quant_pct_wrapped = True

    def forward(self, attn_weights, attention_mask=None, *extra_args, **kwargs):
        if self.use_act_quant:
            attn_weights = self.act_quantizer(attn_weights)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            min_val = torch.finfo(attn_weights.dtype).min
            attn_weights = torch.clamp(attn_weights, min=min_val)

        if self.base_module is not None:
            try:
                return self.base_module(attn_weights, *extra_args, **kwargs)
            except TypeError:
                pass

        return F.softmax(attn_weights, dim=self.dim, dtype=torch.float32).to(attn_weights.dtype)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.use_act_quant = act_quant
        if self.base_module is not None and hasattr(self.base_module, "set_quant_state"):
            self.base_module.set_quant_state(weight_quant, act_quant)

class QuantSwiglu(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict | None = None,
        x2_quant_params: dict | None = None,
        *,
        base_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.x1_quantizer = UniformAffineQuantizer(**(x1_quant_params or {}))
        self.x2_quantizer = UniformAffineQuantizer(**(x2_quant_params or {}))
        self.use_act_quant = False
        object.__setattr__(self, "base_module", base_module)
        if "base_module" in self._modules:
            self._modules.pop("base_module", None)
        if base_module is not None:
            setattr(base_module, "_quant_pct_wrapped", True)
        self.smooth = getattr(base_module, "smooth", None)
        self.extra_scale = getattr(base_module, "s", None)
        self._is_cobra_internal = True
        self._quant_pct_wrapped = True

    def forward(self, *inputs, **kwargs):
        if not inputs:
            raise ValueError("QuantSwiglu expects at least one tensor input.")

        args = list(inputs)
        if self.use_act_quant:
            args[0] = self.x1_quantizer(args[0])
            if len(args) > 1:
                args[1] = self.x2_quantizer(args[1])

        if self.base_module is not None:
            return self.base_module(*args, **kwargs)

        if len(args) == 1:
            x1 = args[0]
            gate = self._apply_gate(x1)
            return x1 * gate

        x1, x2 = args[0], args[1]
        gate = self._apply_gate(x1)
        return x1 * gate * x2

    def _apply_gate(self, ref: torch.Tensor) -> torch.Tensor:
        if self.smooth is not None:
            return torch.sigmoid(ref / self.smooth.to(ref.device))
        if self.extra_scale is not None:
            scale = torch.as_tensor(self.extra_scale, device=ref.device, dtype=ref.dtype)
            return torch.sigmoid(ref * scale)
        return torch.sigmoid(ref)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.use_act_quant = act_quant
        if self.base_module is not None and hasattr(self.base_module, "set_quant_state"):
            self.base_module.set_quant_state(weight_quant, act_quant)


class QuantSwilu(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict | None = None,
        x2_quant_params: dict | None = None,
        *,
        base_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.x1_quantizer = UniformAffineQuantizer(**(x1_quant_params or {}))
        self.x2_quantizer = UniformAffineQuantizer(**(x2_quant_params or {}))
        self.use_act_quant = False
        object.__setattr__(self, "base_module", base_module)
        if "base_module" in self._modules:
            self._modules.pop("base_module", None)
        if base_module is not None:
            setattr(base_module, "_quant_pct_wrapped", True)
        self.smooth = getattr(base_module, "smooth", None)
        self._is_cobra_internal = True
        self._quant_pct_wrapped = True

    def forward(self, *inputs, **kwargs):
        if not inputs:
            raise ValueError("QuantSwilu expects at least one tensor input.")

        args = list(inputs)
        if self.use_act_quant:
            args[0] = self.x1_quantizer(args[0])

        if self.base_module is not None:
            return self.base_module(*args, **kwargs)

        x1 = args[0]
        if self.smooth is not None:
            return x1 * torch.sigmoid(x1 * self.smooth.to(x1.device))
        return x1 * torch.sigmoid(x1)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.use_act_quant = act_quant
        if self.base_module is not None and hasattr(self.base_module, "set_quant_state"):
            self.base_module.set_quant_state(weight_quant, act_quant)

