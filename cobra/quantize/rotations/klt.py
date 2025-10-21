"""Offline KLT rotations used during quantization calibration."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from ..hadamard_utils import is_pow2, matmul_hadU
from ..utils.dtype import force_calib_dtype, scoped_no_autocast


def compute_klt_from_stats(stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the KLT rotation matrix derived from covariance statistics."""
    cov = stats.get("cov")
    if cov is None:
        raise ValueError("stats must contain 'cov'")
    original_dtype = cov.dtype
    work_dtype = torch.float32 if original_dtype in {torch.float16, torch.bfloat16} else original_dtype
    cov_work = force_calib_dtype(cov, target=work_dtype)
    with scoped_no_autocast():
        cov_sym = 0.5 * (cov_work + cov_work.transpose(0, 1))
        eigvals, eigvecs = torch.linalg.eigh(cov_sym)
        order = torch.argsort(eigvals, descending=True)
        eigvecs = eigvecs[:, order]
        rotation = eigvecs.transpose(0, 1)
    return rotation.to(dtype=original_dtype)


def apply_wht_then_klt(stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply a Walsh-Hadamard transform before performing KLT if feasible."""
    cov = stats.get("cov")
    if cov is None:
        raise ValueError("stats must contain 'cov'")
    dim = cov.shape[0]
    if not is_pow2(dim):
        return compute_klt_from_stats(stats)
    work_dtype = torch.float32 if cov.dtype in {torch.float16, torch.bfloat16} else cov.dtype
    identity = torch.eye(dim, device=cov.device, dtype=work_dtype)
    with scoped_no_autocast():
        hadamard = matmul_hadU(identity)
        rotated_cov = hadamard @ force_calib_dtype(cov, target=work_dtype) @ hadamard.transpose(0, 1)
    rotated_stats = dict(stats)
    rotated_stats["cov"] = rotated_cov
    rotation_klt = compute_klt_from_stats(rotated_stats)
    with scoped_no_autocast():
        rotation = hadamard.transpose(0, 1) @ rotation_klt.to(dtype=work_dtype)
    return rotation.to(dtype=cov.dtype)


def fold_rotation_into_linear(
    linear_module: nn.Linear,
    R_in: Optional[torch.Tensor] = None,
    R_out: Optional[torch.Tensor] = None,
) -> None:
    """Fold input/output rotations into a linear weight tensor."""
    if R_in is None and R_out is None:
        return

    weight = linear_module.weight.data
    device = weight.device
    dtype = weight.dtype

    calc_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    with scoped_no_autocast():
        new_weight = weight.to(dtype=calc_dtype)
        if R_in is not None:
            if R_in.shape != (linear_module.in_features, linear_module.in_features):
                raise ValueError("R_in has incompatible shape")
            new_weight = new_weight @ R_in.to(device=device, dtype=calc_dtype)

        if R_out is not None:
            if R_out.shape != (linear_module.out_features, linear_module.out_features):
                raise ValueError("R_out has incompatible shape")
            R_out_cast = R_out.to(device=device, dtype=calc_dtype)
            new_weight = R_out_cast @ new_weight
            if linear_module.bias is not None:
                updated_bias = (R_out_cast @ linear_module.bias.data.to(device=device, dtype=calc_dtype).unsqueeze(-1)).squeeze(-1)
                linear_module.bias.data.copy_(updated_bias.to(dtype=dtype))

    linear_module.weight.data.copy_(new_weight.to(dtype=dtype))
