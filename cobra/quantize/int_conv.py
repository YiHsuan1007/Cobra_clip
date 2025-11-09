import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .quantizer import UniformAffineQuantizer
from .int_others import int_conv2d


def _scale_to_scalar(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    scalar = scale.to(device=device, dtype=torch.float32)
    if scalar.numel() > 1:
        scalar = scalar.mean()
    return scalar.reshape(1)


def _invalid_scale(scale: Optional[torch.Tensor]) -> bool:
    if scale is None:
        return True
    if scale.numel() == 0:
        return True
    if torch.any(scale == 0):
        return True
    if not torch.isfinite(scale).all():
        return True
    return False


def _should_use_int_path(module, act_scale: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor]) -> bool:
    if _invalid_scale(weight_scale) or _invalid_scale(act_scale):
        if not getattr(module, "_warned_invalid_scale", False):
            logging.warning(
                "%s detected invalid quantization scale; falling back to fake-quant path.",
                module.__class__.__name__,
            )
            module._warned_invalid_scale = True
        return False
    return True


class QuantConvBase(nn.Module):
    """Common base for quantized convolution wrappers."""

    def __init__(self) -> None:
        super().__init__()
        self._quant_pct_wrapped = True
        self._observer_enabled = False
        self._guard_weight_scale_logged = False
        modules_dict = getattr(self, "_modules", None)
        if isinstance(modules_dict, dict):
            modules_dict.pop("_origin_conv", None)

    def _resolve_guard_label(self) -> str:
        cached = getattr(self, "_quant_guard_label", None)
        if isinstance(cached, str) and cached:
            return cached
        for attr in ("_quant_label", "_quant_path", "_module_path", "_logical_path"):
            candidate = getattr(self, attr, None)
            if isinstance(candidate, str) and candidate:
                self._quant_guard_label = candidate
                return candidate
        origin = getattr(self, "_origin_conv", None)
        if origin is not None:
            origin_label = getattr(origin, "_quant_guard_label", None)
            if isinstance(origin_label, str) and origin_label:
                self._quant_guard_label = origin_label
                return origin_label
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


class QuantConv1d(QuantConvBase):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Conv1d,
        weight_quant_params: dict = {"dynamic_method":"per_tensor"},
        act_quant_params: dict = {"dynamic_method":"per_tensor"},
        observe = "minmax",
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.conv1d
        weight_param = nn.Parameter(
            org_module.weight.detach().clone(),
            requires_grad=org_module.weight.requires_grad,
        )
        self.weight = weight_param
        bias_param = None
        if org_module.bias is not None:
            bias_param = nn.Parameter(
                org_module.bias.detach().clone(),
                requires_grad=org_module.bias.requires_grad,
            )
            self.bias = bias_param
        else:
            self.register_parameter("bias", None)
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        weight_params = dict(weight_quant_params or {"dynamic_method": "per_tensor"})
        weight_params.setdefault("shape", org_module.weight.shape)
        weight_params.setdefault("is_weight", True)
        weight_params.setdefault("observe", observe)
        self.weight_quantizer = UniformAffineQuantizer(**weight_params)
        if not disable_input_quant:
            act_params = dict(act_quant_params or {"dynamic_method": "per_tensor"})
            act_params.setdefault("has_batch_dim", True)
            act_params.setdefault("observe", observe)
            self.act_quantizer = UniformAffineQuantizer(**act_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.temp_weight: Optional[torch.Tensor] = None
        self.temp_bias: Optional[torch.Tensor] = None

        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups
        self.kernel_size = org_module.kernel_size
        self.in_channels = org_module.in_channels
        self.out_channels = org_module.out_channels
        self.padding_mode = getattr(org_module, "padding_mode", "zeros")
        
        self.weight_quantized = False
        self.weight_int: Optional[torch.Tensor] = None
        self.w_scale: Optional[torch.Tensor] = None
        self.w_zero: Optional[torch.Tensor] = None
        self.real_quant_enabled = False
        self._warned_invalid_scale = False

     
    def forward(self, input: torch.Tensor):
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
            if not _should_use_int_path(self, a_scale, self.w_scale):
                use_real_quant = False
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

                y_int = self.fwd_func(
                    x_shift,
                    weight_shift,
                    None,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    **self.fwd_kwargs,
                )

                act_scale_scalar = _scale_to_scalar(a_scale, input.device)
                weight_scale_scalar = _scale_to_scalar(self.w_scale, input.device)
                effective_scale = (act_scale_scalar * weight_scale_scalar).to(input.dtype)

                y_fp = y_int.to(input.dtype) * effective_scale
                if self.bias is not None:
                    y_fp = y_fp + self.bias.to(y_fp.dtype)
                return y_fp

        if self._observer_enabled:
            with torch.no_grad():
                self.weight_quantizer(self.weight)
            if not self.disable_input_quant and self.act_quantizer is not None:
                with torch.no_grad():
                    self.act_quantizer(input)

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            if not self.weight_quantized:
                self.weight = torch.nn.Parameter(self.weight_quantizer(self.weight))
                weight = self.weight
                self.weight_quantized = True
            else:
                weight = self.weight
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if weight is None:
            raise RuntimeError("Temporary weight is not initialised.")
        bias_tensor = bias.to(weight.dtype) if bias is not None else None

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)

        out = self.fwd_func(
                input.to(weight.dtype), weight, bias_tensor,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                **self.fwd_kwargs)
        return out

    def set_quant_state(
        self,
        weight_quant: bool = False,
        act_quant: bool = False,
        observer: bool = False,
    ) -> None:
        logging.debug(
            "%s.set_quant_state(w=%s, a=%s, observer=%s)",
            self.__class__.__name__,
            weight_quant,
            act_quant,
            observer,
        )

        observer_enabled = bool(observer)
        if observer_enabled:
            self._observer_enabled = True
            self.use_weight_quant = False
            self.use_act_quant = False
            self.real_quant_enabled = False
            self.weight_int = None
            self.w_scale = None
            self.w_zero = None
            if hasattr(self.weight_quantizer, "is_observing"):
                self.weight_quantizer.is_observing = True
            if self.act_quantizer is not None and hasattr(self.act_quantizer, "is_observing"):
                self.act_quantizer.is_observing = True
            return

        if self._observer_enabled:
            if hasattr(self.weight_quantizer, "is_observing"):
                self.weight_quantizer.is_observing = False
            if self.act_quantizer is not None and hasattr(self.act_quantizer, "is_observing"):
                self.act_quantizer.is_observing = False
        self._observer_enabled = False

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


class QuantConv2d(QuantConvBase):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module,
        weight_quant_params: dict = {"dynamic_method":"per_tensor"},
        act_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_input_quant=False,
        observe = "minmax",
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.conv2d
        weight_param = nn.Parameter(
            org_module.weight.detach().clone(),
            requires_grad=org_module.weight.requires_grad,
        )
        self.weight = weight_param
        bias_param = None
        if org_module.bias is not None:
            bias_param = nn.Parameter(
                org_module.bias.detach().clone(),
                requires_grad=org_module.bias.requires_grad,
            )
            self.bias = bias_param
        else:
            self.register_parameter("bias", None)
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        weight_params = dict(weight_quant_params or {"dynamic_method": "per_tensor"})
        weight_params.setdefault("shape", org_module.weight.shape)
        weight_params.setdefault("is_weight", True)
        weight_params.setdefault("observe", observe)
        self.weight_quantizer = UniformAffineQuantizer(**weight_params)
        if not disable_input_quant:
            act_params = dict(act_quant_params or {"dynamic_method": "per_tensor"})
            act_params.setdefault("has_batch_dim", True)
            act_params.setdefault("observe", observe)
            self.act_quantizer = UniformAffineQuantizer(**act_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.temp_weight: Optional[torch.Tensor] = None
        self.temp_bias: Optional[torch.Tensor] = None

        self.in_channels = org_module.in_channels
        self.out_channels = org_module.out_channels
        self.kernel_size = org_module.kernel_size
        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups
        self.padding_mode = getattr(org_module, "padding_mode", "zeros")
        self.weight_int: Optional[torch.Tensor] = None
        self.w_scale: Optional[torch.Tensor] = None
        self.w_zero: Optional[torch.Tensor] = None
        self.real_quant_enabled = False
        self._warned_invalid_scale = False

     
    def forward(self, input: torch.Tensor):
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
            if not _should_use_int_path(self, a_scale, self.w_scale):
                use_real_quant = False
            else:
                if getattr(self, "use_int_kernel", False):
                    y_fp = int_conv2d(
                        x_int,
                        self.weight_int,
                        a_scale,
                        self.w_scale,
                        a_zero=a_zero,
                        w_zero=self.w_zero,
                        bias=self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    ).to(input.dtype)
                    return y_fp

                x_shift = x_int.to(torch.int32)
                if a_zero is not None:
                    x_shift = x_shift - a_zero.to(device=x_shift.device, dtype=torch.int32)

                weight_shift = self.weight_int.to(torch.int32)
                if self.w_zero is not None:
                    weight_shift = weight_shift - self.w_zero.to(device=weight_shift.device, dtype=torch.int32)

                y_int = self.fwd_func(
                    x_shift,
                    weight_shift,
                    None,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    **self.fwd_kwargs,
                )

                act_scale_scalar = _scale_to_scalar(a_scale, input.device)
                weight_scale_scalar = _scale_to_scalar(self.w_scale, input.device)
                effective_scale = (act_scale_scalar * weight_scale_scalar).to(input.dtype)

                y_fp = y_int.to(input.dtype) * effective_scale
                if self.bias is not None:
                    y_fp = y_fp + self.bias.to(y_fp.dtype)
                return y_fp

        if self._observer_enabled:
            with torch.no_grad():
                self.weight_quantizer(self.weight)
            if not self.disable_input_quant and self.act_quantizer is not None:
                with torch.no_grad():
                    self.act_quantizer(input)

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(
                input, weight, bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                **self.fwd_kwargs)
        return out

    def set_quant_state(
        self,
        weight_quant: bool = False,
        act_quant: bool = False,
        observer: bool = False,
    ) -> None:
        logging.debug(
            "%s.set_quant_state(w=%s, a=%s, observer=%s)",
            self.__class__.__name__,
            weight_quant,
            act_quant,
            observer,
        )

        observer_enabled = bool(observer)
        if observer_enabled:
            self._observer_enabled = True
            self.use_weight_quant = False
            self.use_act_quant = False
            self.real_quant_enabled = False
            self.weight_int = None
            self.w_scale = None
            self.w_zero = None
            if hasattr(self.weight_quantizer, "is_observing"):
                self.weight_quantizer.is_observing = True
            if self.act_quantizer is not None and hasattr(self.act_quantizer, "is_observing"):
                self.act_quantizer.is_observing = True
            return

        if self._observer_enabled:
            if hasattr(self.weight_quantizer, "is_observing"):
                self.weight_quantizer.is_observing = False
            if self.act_quantizer is not None and hasattr(self.act_quantizer, "is_observing"):
                self.act_quantizer.is_observing = False
        self._observer_enabled = False

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

class QuantConv3d(QuantConvBase):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module,
        weight_quant_params: dict = {"dynamic_method":"per_tensor"},
        act_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_input_quant=False,
        observe = "minmax",
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.conv3d
        weight_param = nn.Parameter(
            org_module.weight.detach().clone(),
            requires_grad=org_module.weight.requires_grad,
        )
        self.weight = weight_param
        bias_param = None
        if org_module.bias is not None:
            bias_param = nn.Parameter(
                org_module.bias.detach().clone(),
                requires_grad=org_module.bias.requires_grad,
            )
            self.bias = bias_param
        else:
            self.register_parameter("bias", None)
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        weight_params = dict(weight_quant_params or {"dynamic_method": "per_tensor"})
        weight_params.setdefault("shape", org_module.weight.shape)
        weight_params.setdefault("is_weight", True)
        weight_params.setdefault("observe", observe)
        self.weight_quantizer = UniformAffineQuantizer(**weight_params)
        if not disable_input_quant:
            act_params = dict(act_quant_params or {"dynamic_method": "per_tensor"})
            act_params.setdefault("has_batch_dim", True)
            act_params.setdefault("observe", observe)
            self.act_quantizer = UniformAffineQuantizer(**act_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.temp_weight: Optional[torch.Tensor] = None
        self.temp_bias: Optional[torch.Tensor] = None

        self.in_channels = org_module.in_channels
        self.out_channels = org_module.out_channels
        self.kernel_size = org_module.kernel_size
        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups
        self.kernel_size = org_module.kernel_size
        self.padding_mode = getattr(org_module, "padding_mode", "zeros")
        self.weight_int: Optional[torch.Tensor] = None
        self.w_scale: Optional[torch.Tensor] = None
        self.w_zero: Optional[torch.Tensor] = None
        self.real_quant_enabled = False
        self._warned_invalid_scale = False

     
    def forward(self, input: torch.Tensor):
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
            if not _should_use_int_path(self, a_scale, self.w_scale):
                use_real_quant = False
            else:
                x_int = x_int.to(torch.int32)
                weight_int = self.weight_int.to(torch.int32)

                if a_zero is not None:
                    x_int = x_int - a_zero.to(device=x_int.device, dtype=torch.int32)

                if self.w_zero is not None:
                    weight_centered = weight_int - self.w_zero.to(device=weight_int.device, dtype=torch.int32)
                else:
                    weight_centered = weight_int

                y_int = self.fwd_func(
                    x_int,
                    weight_centered,
                    None,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    **self.fwd_kwargs,
                )

                act_scale_scalar = _scale_to_scalar(a_scale, input.device)
                weight_scale_scalar = _scale_to_scalar(self.w_scale, input.device)
                effective_scale = (act_scale_scalar * weight_scale_scalar).to(input.dtype)

                y_fp = y_int.to(input.dtype) * effective_scale
                if self.bias is not None:
                    y_fp = y_fp + self.bias.to(y_fp.dtype)
            return y_fp

        if self._observer_enabled:
            with torch.no_grad():
                self.weight_quantizer(self.weight)
            if not self.disable_input_quant and self.act_quantizer is not None:
                with torch.no_grad():
                    self.act_quantizer(input)

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(
                input, weight, bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                **self.fwd_kwargs)
        return out

    def set_quant_state(
        self,
        weight_quant: bool = False,
        act_quant: bool = False,
        observer: bool = False,
    ) -> None:
        logging.debug(
            "%s.set_quant_state(w=%s, a=%s, observer=%s)",
            self.__class__.__name__,
            weight_quant,
            act_quant,
            observer,
        )

        observer_enabled = bool(observer)
        if observer_enabled:
            self._observer_enabled = True
            self.use_weight_quant = False
            self.use_act_quant = False
            self.real_quant_enabled = False
            self.weight_int = None
            self.w_scale = None
            self.w_zero = None
            if hasattr(self.weight_quantizer, "is_observing"):
                self.weight_quantizer.is_observing = True
            if self.act_quantizer is not None and hasattr(self.act_quantizer, "is_observing"):
                self.act_quantizer.is_observing = True
            return

        if self._observer_enabled:
            if hasattr(self.weight_quantizer, "is_observing"):
                self.weight_quantizer.is_observing = False
            if self.act_quantizer is not None and hasattr(self.act_quantizer, "is_observing"):
                self.act_quantizer.is_observing = False
        self._observer_enabled = False

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

