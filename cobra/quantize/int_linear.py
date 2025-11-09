import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .quantizer import UniformAffineQuantizer


class QuantLinear(nn.Linear):
    """
    Quantized ``nn.Linear`` wrapper that supports weight and activation fake quantisation.

    The module mirrors the legacy interface used throughout the codebase while exposing
    explicit ``weight_bits`` / ``act_bits`` attributes so downstream utilities can introspect
    the configured bit-widths.
    """

    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict | None = None,
        act_quant_params: dict | None = None,
        disable_input_quant: bool = False,
        observe: str = "minmax",
        weight_bits: int = 8,
        act_bits: int = 8,
    ):
        super().__init__(org_module.in_features, org_module.out_features, bias=org_module.bias is not None)
        self.fwd_kwargs: dict = {}
        self.fwd_func = F.linear

        self.weight = org_module.weight
        self.bias = org_module.bias if org_module.bias is not None else None

        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.weight_quantized = False
        self.real_quant_enabled = False
        self._guard_weight_scale_logged = False

        self.weight_bits = int(weight_bits)
        self.act_bits = int(act_bits)

        weight_params = dict(weight_quant_params or {"dynamic_method": "per_tensor"})
        weight_params.setdefault("n_bits", self.weight_bits)
        weight_params.setdefault("shape", org_module.weight.shape)
        weight_params.setdefault("is_weight", True)
        weight_params.setdefault("observe", observe)
        self.weight_quantizer = UniformAffineQuantizer(**weight_params)

        if not disable_input_quant:
            act_params = dict(act_quant_params or {"dynamic_method": "per_tensor"})
            act_params.setdefault("n_bits", self.act_bits)
            act_params.setdefault("has_batch_dim", True)
            act_params.setdefault("observe", observe)
            self.act_quantizer: UniformAffineQuantizer | None = UniformAffineQuantizer(**act_params)
        else:
            self.act_quantizer = None

        self.weight_int: Optional[torch.Tensor] = None
        self.w_scale: Optional[torch.Tensor] = None
        self.w_zero: Optional[torch.Tensor] = None
        self._warned_invalid_scale = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        use_real_quant = (
            self.real_quant_enabled
            and self.weight_int is not None
            and self.w_scale is not None
            and self.use_act_quant
            and not self.disable_input_quant
            and self.act_quantizer is not None
        )
        if use_real_quant:
            x_int, a_scale, a_zero = self.act_quantizer.quant2int(input.detach())

            if self._should_fallback_real_quant(a_scale, self.w_scale):
                use_real_quant = False
            else:
                act_scale = self._collapse_activation_scale(a_scale.to(input.device), batch_size=input.shape[0])
                weight_scale_matrix = self._collapse_weight_scale(self.w_scale.to(input.device)).transpose(0, 1)

                if getattr(self, "use_int_kernel", False):
                    y_fp = int_gemm(
                        x_int,
                        self.weight_int,
                        act_scale,
                        weight_scale_matrix,
                        a_zero=a_zero,
                        b_zero=self.w_zero,
                    ).to(input.dtype)
                else:
                    x_shift = x_int.to(torch.int32)
                    if a_zero is not None:
                        a_zero_broadcast = torch.broadcast_to(
                            a_zero.to(device=x_shift.device, dtype=torch.int32),
                            x_shift.shape,
                        )
                        x_shift = x_shift - a_zero_broadcast

                    weight_shift = self.weight_int.to(torch.int32)
                    if self.w_zero is not None:
                        w_zero_broadcast = torch.broadcast_to(
                            self.w_zero.to(device=weight_shift.device, dtype=torch.int32),
                            weight_shift.shape,
                        )
                        weight_shift = weight_shift - w_zero_broadcast

                    y_int = torch.matmul(x_shift, weight_shift.transpose(0, 1))
                    effective_scale = act_scale * weight_scale_matrix
                    y_fp = y_int.to(input.dtype) * effective_scale.to(input.dtype)

                if self.bias is not None:
                    y_fp = y_fp + self.bias.to(y_fp.dtype)
                return y_fp

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            if self.weight_quantizer.is_observing:
                weight = self.weight
            elif not self.weight_quantized:
                self.weight = torch.nn.Parameter(self.weight_quantizer(self.weight))
                weight = self.weight
                self.weight_quantized = True
            else:
                weight = self.weight
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant and self.act_quantizer is not None:
            input = self.act_quantizer(input)

        if bias is not None:
            bias = bias.to(weight)
        return self.fwd_func(input.to(weight), weight, bias, **self.fwd_kwargs)

    def _resolve_guard_label(self) -> str:
        cached = getattr(self, "_quant_guard_label", None)
        if isinstance(cached, str) and cached:
            return cached
        for attr in ("_quant_label", "_quant_path", "_module_path", "_logical_path"):
            candidate = getattr(self, attr, None)
            if isinstance(candidate, str) and candidate:
                self._quant_guard_label = candidate
                return candidate
        label = self.__class__.__name__
        self._quant_guard_label = label
        return label

    def _guard_weight_scale_initialized(self) -> None:
        quantizer = getattr(self, "weight_quantizer", None)
        if quantizer is None:
            return
        scale = getattr(quantizer, "scale", None)
        if isinstance(scale, torch.Tensor) and scale.numel() > 0:
            return
        weight = getattr(self, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.numel() == 0:
            return

        initialized = False
        if hasattr(quantizer, "init_from_weight"):
            try:
                quantizer.init_from_weight(weight.detach())
                scale = getattr(quantizer, "scale", None)
                initialized = isinstance(scale, torch.Tensor) and scale.numel() > 0
            except Exception as exc:
                logging.debug(
                    "[Guard] init_from_weight failed for %s: %s",
                    self._resolve_guard_label(),
                    exc,
                )
        if not initialized and hasattr(quantizer, "calculate_qparams"):
            try:
                quantizer.calculate_qparams()
                scale = getattr(quantizer, "scale", None)
                initialized = isinstance(scale, torch.Tensor) and scale.numel() > 0
            except Exception as exc:
                logging.debug(
                    "[Guard] calculate_qparams failed for %s: %s",
                    self._resolve_guard_label(),
                    exc,
                )
        if initialized and not getattr(self, "_guard_weight_scale_logged", False):
            logging.warning(
                "[Guard] weight quant scale initialized on-demand for %s",
                self._resolve_guard_label(),
            )
            self._guard_weight_scale_logged = True

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

        if self.use_weight_quant:
            self._guard_weight_scale_initialized()
            with torch.no_grad():
                q_weight, scale, zero = self.weight_quantizer.quant2int(self.weight.detach())
                self.weight_int = q_weight.to(torch.int32).detach()
                self.w_scale = scale.detach()
                self.w_zero = None if zero is None else zero.detach()
        else:
            self.weight_int = None
            self.w_scale = None
            self.w_zero = None

    @staticmethod
    def _collapse_activation_scale(scale: torch.Tensor, batch_size: int) -> torch.Tensor:
        if scale.numel() == 1:
            return scale.reshape(1, 1)
        return scale.reshape(batch_size, -1).mean(dim=1, keepdim=True)

    def _collapse_weight_scale(self, scale: torch.Tensor) -> torch.Tensor:
        if scale.numel() == 1:
            return scale.reshape(1, 1)
        return scale.reshape(self.out_features, -1).mean(dim=1, keepdim=True)

    def _should_fallback_real_quant(
        self,
        act_scale: Optional[torch.Tensor],
        weight_scale: Optional[torch.Tensor],
    ) -> bool:
        def _invalid(scale: Optional[torch.Tensor]) -> bool:
            if scale is None:
                return True
            if scale.numel() == 0:
                return True
            if torch.any(scale == 0):
                return True
            if not torch.isfinite(scale).all():
                return True
            return False

        if _invalid(weight_scale) or _invalid(act_scale):
            if not self._warned_invalid_scale:
                logging.warning(
                    "QuantLinear detected invalid quantization scale; falling back to fake-quant path."
                )
                self._warned_invalid_scale = True
            return True
        return False


if __name__ == "__main__":
    torch.manual_seed(0)

    base_linear = nn.Linear(1024, 1024, bias=True)
    quant_linear = QuantLinear(base_linear)
    quant_linear.eval()

    x = torch.randn(8, 1024)

    with torch.no_grad():
        quant_linear.weight_quantizer.dynamic_per_tensor_calibration(quant_linear.weight)
        quant_linear.weight_quantizer._align_params_with_input(quant_linear.weight)
        if quant_linear.act_quantizer is not None:
            quant_linear.act_quantizer.dynamic_per_tensor_calibration(x)
            quant_linear.act_quantizer._align_params_with_input(x)

        quant_linear.set_quant_state(weight_quant=True, act_quant=True)

        quant_linear.real_quant_enabled = False
        y_fake = quant_linear(x)

        quant_linear.real_quant_enabled = True
        y_int = quant_linear(x)

    y_fake_flat = y_fake.flatten()
    y_int_flat = y_int.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(y_fake_flat, y_int_flat, dim=0).item()
    mae = torch.mean(torch.abs(y_fake - y_int)).item()

    print(f"Cosine similarity (fake vs real): {cos_sim:.6f}")
    print(f"Mean absolute error (fake vs real): {mae:.6f}")

