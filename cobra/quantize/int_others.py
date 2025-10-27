import torch,torch.nn as nn,torch.nn.functional as F
from typing import Optional

from .quantizer import UniformAffineQuantizer



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
        self.base_module = base_module
        if base_module is not None:
            setattr(base_module, "_quant_pct_wrapped", True)
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
        self.base_module = base_module
        if base_module is not None:
            setattr(base_module, "_quant_pct_wrapped", True)
            dim = getattr(base_module, "dim", dim)
        self.dim = dim
        self.use_act_quant = False
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
        self.base_module = base_module
        if base_module is not None:
            setattr(base_module, "_quant_pct_wrapped", True)
        self.smooth = getattr(base_module, "smooth", None)
        self.extra_scale = getattr(base_module, "s", None)
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
        self.base_module = base_module
        if base_module is not None:
            setattr(base_module, "_quant_pct_wrapped", True)
        self.smooth = getattr(base_module, "smooth", None)
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

