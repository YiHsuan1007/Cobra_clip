import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import UniformAffineQuantizer


def int_gemm(
    a_int: torch.Tensor,
    b_int: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    a_zero: torch.Tensor | None = None,
    b_zero: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Perform an integer GEMM followed by dequantisation using the provided scales/zero-points.
    """
    a_shift = a_int.to(torch.int32)
    if a_zero is not None:
        a_shift = a_shift - a_zero.to(device=a_int.device, dtype=torch.int32)

    b_shift = b_int.to(torch.int32)
    if b_zero is not None:
        b_shift = b_shift - b_zero.to(device=b_int.device, dtype=torch.int32)

    y_int = torch.matmul(a_shift, b_shift.transpose(-1, -2))

    a_scale = a_scale.to(device=y_int.device, dtype=torch.float32)
    b_scale = b_scale.to(device=y_int.device, dtype=torch.float32)
    y_fp = y_int.to(torch.float32) * (a_scale * b_scale)
    return y_fp


class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict | None = None,
        x2_quant_params: dict | None = None,
        disable_act_quant=False,
        observe = "minmax",
        matmul_func=torch.matmul,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        x1_params = dict({"dynamic_method": "per_tensor"} if x1_quant_params is None else x1_quant_params)
        x1_has_batch = x1_params.pop("has_batch_dim", True)
        x1_observe = observe if observe is not None else x1_params.pop("observe", "minmax")
        self.x1_quantizer = UniformAffineQuantizer(
            has_batch_dim=x1_has_batch,
            observe=x1_observe,
            **x1_params,
        )

        x2_params = dict({"dynamic_method": "per_tensor"} if x2_quant_params is None else x2_quant_params)
        x2_has_batch = x2_params.pop("has_batch_dim", True)
        x2_observe = observe if observe is not None else x2_params.pop("observe", "minmax")
        self.x2_quantizer = UniformAffineQuantizer(
            has_batch_dim=x2_has_batch,
            observe=x2_observe,
            **x2_params,
        )
        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1, x2):
        if hasattr(self,"pertoken"):
            B,L,ED,N = x1.shape
            x1 = x1.reshape(B,L*ED,N)
            x1 = self.quant_x1(x1)
            x1 = x1.reshape(B,L,ED,N)
            x2 = self.quant_x2(x2)
            out = self.matmul_func(x1, x2)
            pass
        else:
            x1 = self.quant_x1(x1)
            x2 = self.quant_x2(x2)
            out = self.matmul_func(x1, x2)
        return out


class QuantMatmulWrapper(QuantMatMul):
    def __init__(self, base_module: nn.Module, **kw):
        super().__init__(**kw)
        object.__setattr__(self, "_origin_module", base_module)
        object.__setattr__(self, "_quant_wrapped", True)

    def named_children(self):
        """Hide internal quantizers from recursive walkers to avoid infinite wrapping."""
        return iter(())

    def children(self):
        return iter(())

