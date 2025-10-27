import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

