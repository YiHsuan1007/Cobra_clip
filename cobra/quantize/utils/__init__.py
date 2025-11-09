import random
import logging
from datetime import datetime
import os
import sys
import numpy as np
import torch
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple, Optional, Set

from ..int_linear import QuantLinear
from ..int_conv import QuantConv1d, QuantConv2d, QuantConv3d
from ..int_matmul import QuantMatMul
from ..quantizer import UniformAffineQuantizer
from ..observers.observer_abc import ObserverABC


def iter_named_modules(module: torch.nn.Module) -> Iterator[Tuple[str, torch.nn.Module]]:
    """Yield ``(name, module)`` pairs even if the module exposes triplet-style named iteration."""
    original = getattr(module, "_quant_pct_named_modules_original", None)
    if original is not None:
        yield from original()
    else:
        yield from module.named_modules()

class NoHookContext:
    def __init__(self, module):
        self.module = module
        self.hooks = []

    def __enter__(self):
        # 保存hooks
        for hook_id in list(self.module._forward_hooks.keys()):
            self.hooks.append((hook_id, self.module._forward_hooks.pop(hook_id)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复hooks
        for hook_id, hook in self.hooks:
            self.module._forward_hooks[hook_id] = hook

class Logger(object):
    def __init__(self, folder="logs"):
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 创建日志文件夹（如果不存在）
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 定义日志文件名
        filename = os.path.join(folder, f"log_{current_time}.txt")

        # 打开日志文件
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


def set_seed(seed):
    torch.manual_seed(seed)  # 设置 CPU 上的随机数种子
    torch.cuda.manual_seed(seed)  # 设置当前 GPU 上的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 上的随机数种子（如果有多个 GPU）
    np.random.seed(seed)  # 设置 NumPy 的随机数种子
    random.seed(seed)  # 设置 Python 自带的随机数种子

    # 如果使用了 CuDNN 后端
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 确保卷积算法的选择是确定的
def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )



def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for name, m in iter_named_modules(self):
        if isinstance(m, (QuantLinear, QuantMatMul,QuantConv1d,QuantConv2d,QuantConv3d)):
            m.set_quant_state(weight_quant, act_quant)

def set_static_quant(self, static_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    total = 0
    changed = 0
    desired_dynamic = not static_quant
    for m in self.modules():
        if isinstance(m, UniformAffineQuantizer):
            total += 1
            previous = getattr(m, "is_dynamic_quant", None)
            if previous != desired_dynamic:
                changed += 1
            m.is_dynamic_quant = desired_dynamic
    message = (
        "[Quant] set_static_quant -> "
        f"static={static_quant} total={total} updated={changed}"
    )
    logging.info(message)
    print(message)

def set_static_quant_weight(self, static_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    for name, m in iter_named_modules(self):
        if "weight" in name:
            if isinstance(m, UniformAffineQuantizer):
                m.is_dynamic_quant = not static_quant

def set_observing(self, observing: bool = True):
    self.use_observing = observing
    for name, m in iter_named_modules(self):
        if isinstance(m, (UniformAffineQuantizer)):
           m.is_observing = observing


## dtype推斷確保 finalize 時可提供 observer 期望的 qmin/qmax 範圍。
def _infer_dtype(q: Any) -> torch.dtype:
    dtype = getattr(q, "dtype", None)
    if dtype is not None:
        return dtype
    ## 權重量化預設走 qint8 方便雙向符號範圍，啟用量化預設用 quint8 對應常見非負輸入。
    if getattr(q, "is_weight", False) or getattr(q, "signed", False):
        return torch.qint8
    return torch.quint8


def _flatten_quantizer_objects(value: Any) -> Iterator[Any]:
    """Yield all quantizer-like objects from nested containers."""
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten_quantizer_objects(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten_quantizer_objects(item)
        return
    if isinstance(value, torch.nn.ModuleDict):
        for item in value.values():
            yield from _flatten_quantizer_objects(item)
        return
    if isinstance(value, (torch.nn.ModuleList, torch.nn.Sequential)):
        for item in value:
            yield from _flatten_quantizer_objects(item)
        return
    yield value


def _iter_all_quantizers(model: torch.nn.Module) -> Iterator[Any]:
    """Iterate unique quantizer-like objects attached to the model."""
    def register(candidate: Any) -> Iterator[Any]:
        if candidate is None:
            return
        for obj in _flatten_quantizer_objects(candidate):
            if isinstance(obj, ObserverABC):
                owner = getattr(obj, "owner", None)
                if owner is None:
                    continue
                owner_id = id(owner)
                if owner_id in seen:
                    continue
                if not any(hasattr(owner, attr) for attr in ("finalize", "calculate_qparams", "scale", "zero", "observe")):
                    continue
                seen.add(owner_id)
                yield owner
                continue
            obj_id = id(obj)
            if obj_id in seen:
                continue
            if any(hasattr(obj, attr) for attr in ("finalize", "calculate_qparams", "scale", "zero", "observe")):
                seen.add(obj_id)
                yield obj

    seen: set[int] = set()

    for module in model.modules():
        yield from register(module)
        for attr in ("weight_quantizer", "act_quantizer"):
            if hasattr(module, attr):
                yield from register(getattr(module, attr))


def _iter_quantizers_with_roles(model: torch.nn.Module) -> Iterator[Tuple[Any, str, str]]:
    """Yield (quantizer, role, label) pairs with deduplication."""
    seen: set[int] = set()
    role_by_attr = {
        "weight_quantizer": "weight",
        "act_quantizer": "activation",
        "input_quantizer": "activation",
        "output_quantizer": "activation",
    }

    for module_name, module in iter_named_modules(model):
        qual_name = module_name or module.__class__.__name__
        for attr, role in role_by_attr.items():
            if not hasattr(module, attr):
                continue
            quant_attr = getattr(module, attr)
            for quantizer in _flatten_quantizer_objects(quant_attr):
                if quantizer is None:
                    continue
                qid = id(quantizer)
                if qid in seen:
                    continue
                if not any(hasattr(quantizer, attr_name) for attr_name in ("finalize", "calculate_qparams", "scale", "zero", "observe")):
                    continue
                seen.add(qid)
                label = f"{qual_name}.{attr}".strip(".")
                yield quantizer, role, label

    for module in model.modules():
        if isinstance(module, UniformAffineQuantizer):
            qid = id(module)
            if qid in seen:
                continue
            seen.add(qid)
            label = module.__class__.__name__
            yield module, "other", label


@dataclass(frozen=True)
class QuantizerHandle:
    quantizer: Any
    kind: str
    label: str

    def is_initialized(self) -> bool:
        return _is_quantizer_initialized(self.quantizer)


def iter_quantizers(model: torch.nn.Module) -> Iterator[QuantizerHandle]:
    """Yield lightweight wrappers for quantizers attached to ``model``."""
    for quantizer, role, label in _iter_quantizers_with_roles(model):
        yield QuantizerHandle(quantizer=quantizer, kind=role, label=label)


def _is_quantizer_initialized(quantizer: torch.nn.Module) -> bool:
    """Return True when quantizer exposes scale and, if required, zero parameters."""
    scale = getattr(quantizer, "scale", None)
    scale_ready = isinstance(scale, torch.Tensor) and scale.numel() > 0
    if not scale_ready:
        return False

    zero_required = not (
        getattr(quantizer, "disable_zero_point", False) or getattr(quantizer, "symmetric", False)
    )
    if not zero_required:
        return True

    for name in ("zero_point", "round_zero_point", "zero"):
        value = getattr(quantizer, name, None)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            return True
    return False


def _collect_uninitialized_quantizers(model: torch.nn.Module) -> Dict[str, List[str]]:
    """Return mapping of role -> labels for quantizers missing parameters."""
    summary: Dict[str, List[str]] = {"weight": [], "activation": [], "other": []}
    for quantizer, role, label in _iter_quantizers_with_roles(model):
        if not _is_quantizer_initialized(quantizer):
            summary.setdefault(role, []).append(label)
    return summary


def count_observers(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count observer modules attached to quantizers.

    Returns a dict with ``total`` observers discovered, ``observing`` currently
    receiving updates, and ``initialized`` quantizers whose observers have populated
    scale/zero statistics.
    """
    total = 0
    observing = 0
    initialized = 0
    seen: set[int] = set()

    def _is_observing(candidate: Any) -> bool:
        for attr in ("observe", "enabled", "is_enabled", "active"):
            value = getattr(candidate, attr, None)
            if isinstance(value, bool):
                if value:
                    return True
            elif value is not None:
                try:
                    if bool(value):
                        return True
                except Exception:
                    continue
        return False

    for quantizer in _iter_all_quantizers(model):
        observer = getattr(quantizer, "observer", None)
        if observer is None:
            continue
        obs_id = id(observer)
        if obs_id in seen:
            continue
        seen.add(obs_id)
        total += 1
        if _is_observing(observer) or _is_observing(quantizer):
            observing += 1
        if _is_quantizer_initialized(quantizer):
            initialized += 1

    return {
        "total": total,
        "observing": observing,
        "initialized": initialized,
    }


def enable_observation(model: torch.nn.Module) -> None:
    """Turn on observation flags for modules and their quantizers."""
    for module in model.modules():
        if hasattr(module, "observe"):
            setattr(module, "observe", True)
        for attr in ("weight_quantizer", "act_quantizer"):
            if hasattr(module, attr):
                for quantizer in _flatten_quantizer_objects(getattr(module, attr)):
                    if hasattr(quantizer, "observe"):
                        setattr(quantizer, "observe", True)
    stats = count_observers(model)
    message = (
        "[Quant] enable_observation -> "
        f"observers total={stats['total']} observing={stats['observing']} initialized={stats['initialized']}"
    )
    logging.info(message)
    print(message)


def finalize_all_quantizers(model: torch.nn.Module, *, kind: str | None = None) -> None:
    """Finalize quantizers so they expose scale and zero parameters."""
    allowed_ids: Optional[Set[int]] = None
    if kind is not None:
        normalized = kind.strip().lower()
        if normalized in {"weight", "weights"}:
            roles = {"weight"}
        elif normalized in {"act", "activation", "activations"}:
            roles = {"activation"}
        elif normalized in {"all", ""}:
            roles = None
        else:
            raise ValueError("`kind` must be one of {'weight', 'activation', 'all'} or None.")
        if roles is not None:
            allowed_ids = {
                id(quantizer)
                for quantizer, role, _ in _iter_quantizers_with_roles(model)
                if role in roles
            }
            if not allowed_ids:
                return

    for quantizer in _iter_all_quantizers(model):
        if allowed_ids is not None and id(quantizer) not in allowed_ids:
            continue
        ## observer 僅承載統計資訊，略過以避免誤觸缺少 dtype 的 finalize 路徑。
        if isinstance(quantizer, ObserverABC):
            continue
        if hasattr(quantizer, "finalize"):
            quantizer.finalize()
        elif hasattr(quantizer, "calculate_qparams"):
            dtype = _infer_dtype(quantizer)
            quantizer.calculate_qparams(dtype)


def count_uninitialized_quantizers(model: torch.nn.Module) -> int:
    """Count quantizers missing scale or zero parameters."""
    summary = _collect_uninitialized_quantizers(model)
    return sum(len(entries) for entries in summary.values())


def summarize_quantizer_init(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Summarize UniformAffineQuantizer initialisation state.

    Returns a dict with keys:
        total: total quantizers observed
        initialized: quantizers ready for export
        missing: quantizers lacking scale/zero
        missing_examples: up to 10 module names missing params
    """
    total = 0
    initialized = 0
    missing = 0
    missing_examples: List[str] = []
    seen: set[int] = set()

    for name, module in iter_named_modules(model):
        if not isinstance(module, UniformAffineQuantizer):
            continue
        qid = id(module)
        if qid in seen:
            continue
        seen.add(qid)
        total += 1
        label = name or module.__class__.__name__
        if _is_quantizer_initialized(module):
            initialized += 1
        else:
            missing += 1
            if len(missing_examples) < 10:
                missing_examples.append(label)

    return {
        "total": total,
        "initialized": initialized,
        "missing": missing,
        "missing_examples": missing_examples,
    }


def assert_all_initialized(model: torch.nn.Module, msg: str | None = None) -> None:
    """Assert every quantizer has initialized scale and zero parameters."""
    missing = count_uninitialized_quantizers(model)
    if missing == 0:
        return
    base = f"{missing} quantizer(s) missing scale/zero parameters; 先跑校正 forward 再切換 real-quant."
    detail = f"{msg} {base}" if msg else base
    raise RuntimeError(detail)


def freeze_weight_qparams(model: torch.nn.Module) -> int:
    """Disable dynamic initialisation for weight quantizers after calibration."""
    frozen = 0
    for quantizer, role, _ in _iter_quantizers_with_roles(model):
        if role != "weight":
            continue
        observer = getattr(quantizer, "observer", None)
        if observer is not None:
            quantizer.observer = None
        if hasattr(quantizer, "observered"):
            quantizer.observered = True
        if hasattr(quantizer, "is_dynamic_quant"):
            quantizer.is_dynamic_quant = False
        if hasattr(quantizer, "is_observing"):
            quantizer.is_observing = False
        if hasattr(quantizer, "observe"):
            quantizer.observe = False
        frozen += 1
    logging.info("[Quant] freeze_weight_qparams -> frozen=%d", frozen)
    return frozen


def register_scales_and_zeros(module: torch.nn.Module) -> None:
    """
    Persist quantizer parameters as buffers for every UniformAffineQuantizer under ``module``.

    The helper should be invoked after calibration (and typically after enabling static quant)
    so that each quantizer has collected its scale and zero-point statistics.
    """
    quant_modules = [q for q in module.modules() if isinstance(q, UniformAffineQuantizer)]
    qz_cnt = len(quant_modules)
    if qz_cnt == 0:
        raise RuntimeError("No quantizers found; did you run replacement?")

    summary = _collect_uninitialized_quantizers(module)
    pending = sum(len(entries) for entries in summary.values())
    if pending:
        finalize_all_quantizers(module)
        summary = _collect_uninitialized_quantizers(module)
        pending = sum(len(entries) for entries in summary.values())
        if pending:
            counts = ", ".join(
                f"{role}={len(entries)}" for role, entries in summary.items() if entries
            )
            if counts:
                logging.error("[Quant] Pending quantizers before export: %s", counts)
            message = "Cannot export parameters before scale is initialised."
            if counts:
                message += f" Pending counts: {counts}."
            message += " 請先跑校正 forward 或移除 --real-quant."
            raise RuntimeError(message)

    registered_scales = 0
    registered_zero_points = 0

    def _zero_tensor(q: UniformAffineQuantizer) -> Optional[torch.Tensor]:
        for name in ("zero_point", "round_zero_point", "zero"):
            value = getattr(q, name, None)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return value
        return None

    for quantizer in quant_modules:
        quantizer.register_scales_and_zeros()
        scale = getattr(quantizer, "scale", None)
        if isinstance(scale, torch.Tensor) and scale.numel() > 0:
            registered_scales += 1
        zero_value = _zero_tensor(quantizer)
        zero_required = not (
            getattr(quantizer, "disable_zero_point", False) or getattr(quantizer, "symmetric", False)
        )
        if zero_value is not None or not zero_required:
            registered_zero_points += 1

    print(
        "[QuantExport] register_scales_and_zeros -> "
        f"scales={registered_scales}/{qz_cnt}, zero_points={registered_zero_points}/{qz_cnt}"
    )


def convert_to_int(
    model: torch.nn.Module,
    propagate_int: bool = False,
    use_int_kernel: bool = False,
) -> None:
    """
    Prepare quantized wrappers for real-int execution by materialising integer weights.

    When ``propagate_int`` is True the function also annotates downstream modules with
    ``in_scale`` / ``in_zero`` so that alternate execution paths can ingest integer
    activations without re-quantising them.
    """

    prev_scale: torch.Tensor | None = None
    prev_zero: torch.Tensor | None = None

    quant_wrappers = (QuantLinear, QuantConv1d, QuantConv2d, QuantConv3d)

    for module in model.modules():
        if not isinstance(module, quant_wrappers):
            continue

        module.set_quant_state(weight_quant=True, act_quant=True)
        if hasattr(module, "real_quant_enabled"):
            module.real_quant_enabled = True
        supports_kernel = isinstance(module, (QuantLinear, QuantConv2d))
        if supports_kernel:
            setattr(module, "use_int_kernel", bool(use_int_kernel))
        elif hasattr(module, "use_int_kernel"):
            setattr(module, "use_int_kernel", False)

        if propagate_int:
            module.expect_int_input = True
            module.in_scale = prev_scale.clone() if prev_scale is not None else None
            module.in_zero = prev_zero.clone() if prev_zero is not None else None
        else:
            if hasattr(module, "expect_int_input"):
                module.expect_int_input = False
            if hasattr(module, "in_scale"):
                module.in_scale = None
            if hasattr(module, "in_zero"):
                module.in_zero = None

        if hasattr(module, "act_quantizer") and module.act_quantizer is not None:
            try:
                next_scale, next_zero = module.act_quantizer.export_params()
            except RuntimeError:
                prev_scale = None
                prev_zero = None
            else:
                prev_scale = next_scale.detach().clone()
                prev_zero = None if next_zero is None else next_zero.detach().clone()
        else:
            prev_scale = None
            prev_zero = None


from .dtype import force_calib_dtype, scoped_no_autocast

__all__ = [
    "NoHookContext",
    "Logger",
    "set_seed",
    "cleanup_memory",
    "set_quant_state",
    "set_static_quant",
    "set_static_quant_weight",
    "set_observing",
    "enable_observation",
    "finalize_all_quantizers",
    "freeze_weight_qparams",
    "iter_quantizers",
    "summarize_quantizer_init",
    "count_uninitialized_quantizers",
    "assert_all_initialized",
    "register_scales_and_zeros",
    "convert_to_int",
    "force_calib_dtype",
    "scoped_no_autocast",
]

