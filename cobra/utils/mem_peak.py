from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List

try:
    import psutil
except Exception:
    psutil = None

import torch


def _to_gib(x_bytes: float) -> float:
    return float(x_bytes) / (1024.0 ** 3)


def init_peak_track() -> float:
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
    except Exception:
        pass
    return time.perf_counter()


def gather_peaks() -> Dict[str, Any]:
    summary: Dict[str, Any] = {"cpu_gb": None, "gpus": [], "ddp": None}
    # CPU
    try:
        if psutil is not None:
            rss = psutil.Process(os.getpid()).memory_info().rss
        else:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss 單位在 Linux 通常是 KiB
            if rss < 1e12:  # 以 KiB 估計
                rss = rss * 1024
        summary["cpu_gb"] = round(_to_gib(rss), 2)
    except Exception:
        summary["cpu_gb"] = None

    # GPU
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.max_memory_allocated(i)
                reserv = torch.cuda.max_memory_reserved(i)
                summary["gpus"].append(
                    {
                        "device": i,
                        "alloc_gb": round(_to_gib(alloc), 2),
                        "reserved_gb": round(_to_gib(reserv), 2),
                    }
                )
    except Exception:
        pass

    # DDP 聚合（僅彙整數值，格式化交給 caller）
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            import json

            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
            payload = {"cpu": summary["cpu_gb"], "gpus": summary["gpus"], "rank": rank}
            tensor = torch.tensor(
                [len(json.dumps(payload))],
                dtype=torch.long,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            sizes = [tensor.clone() for _ in range(world)]
            torch.distributed.all_gather(sizes, tensor)
            max_size = int(max(int(t.item()) for t in sizes))
            buf = bytearray(max_size)
            enc = json.dumps(payload).encode("utf-8")
            buf[: len(enc)] = enc
            tbuf = torch.tensor(
                list(buf), dtype=torch.uint8, device=tensor.device
            )
            gathered = [torch.zeros_like(tbuf) for _ in range(world)]
            torch.distributed.all_gather(gathered, tbuf)
            all_payloads = []
            for gb in gathered:
                s = bytes(bytearray(gb.tolist())).rstrip(b"\x00")
                all_payloads.append(json.loads(s.decode("utf-8")))
            # 聚合統計
            cpu_vals = [p["cpu"] for p in all_payloads if p["cpu"] is not None]
            alloc_vals, reserv_vals = [], []
            for p in all_payloads:
                for g in p["gpus"]:
                    alloc_vals.append(g["alloc_gb"])
                    reserv_vals.append(g["reserved_gb"])
            agg = {
                "world": world,
                "cpu_max": round(max(cpu_vals), 2) if cpu_vals else None,
                "cpu_mean": round(sum(cpu_vals) / len(cpu_vals), 2)
                if cpu_vals
                else None,
                "alloc_max": round(max(alloc_vals), 2) if alloc_vals else None,
                "alloc_mean": round(sum(alloc_vals) / len(alloc_vals), 2)
                if alloc_vals
                else None,
                "reserv_max": round(max(reserv_vals), 2) if reserv_vals else None,
                "reserv_mean": round(sum(reserv_vals) / len(reserv_vals), 2)
                if reserv_vals
                else None,
            }
            summary["ddp"] = {"rank": rank, "world": world, "agg": agg}
    except Exception:
        pass
    return summary


def format_block(
    summary: Dict[str, Any], quant_meta: Dict[str, Any], elapsed_s: float
) -> str:
    lines = []
    lines.append("=== Peak Memory & Runtime ===")
    lines.append(f"Mode: {quant_meta.get('mode', 'N/A')}")
    lines.append(f"Bits: W{quant_meta.get('weight_bits', '?')}A{quant_meta.get('act_bits', '?')}")
    lines.append(f"Percentile: {quant_meta.get('percentile', 'N/A')}")
    lines.append(
        f"Rotation: hadamard={quant_meta.get('hadamard', False)}, klt={quant_meta.get('klt', False)}"
    )
    lines.append("")
    cpu = summary.get("cpu_gb")
    lines.append(
        f"CPU Peak RSS: {cpu:.2f} GB" if cpu is not None else "CPU Peak RSS: N/A"
    )
    # GPUs
    if summary.get("gpus"):
        lines.append("GPU Peaks:")
        for g in summary["gpus"]:
            lines.append(
                f"  - device:{g['device']}  alloc:{g['alloc_gb']:.2f} GB  reserved:{g['reserved_gb']:.2f} GB"
            )
    # DDP
    ddp = summary.get("ddp")
    if ddp and ddp.get("rank") == 0:
        agg = ddp["agg"]
        lines.append("(DDP) Aggregates:")
        lines.append(f"  - ranks: {agg['world']}")
        amx, amn = agg.get("alloc_max"), agg.get("alloc_mean")
        rmx, rmn = agg.get("reserv_max"), agg.get("reserv_mean")
        cmx, cmn = agg.get("cpu_max"), agg.get("cpu_mean")
        lines.append(
            f"  - GPU alloc max/mean: {amx if amx is not None else 'N/A'}/{amn if amn is not None else 'N/A'} GB"
        )
        lines.append(
            f"  - GPU reserved max/mean: {rmx if rmx is not None else 'N/A'}/{rmn if rmn is not None else 'N/A'} GB"
        )
        lines.append(
            f"  - CPU RSS max/mean: {cmx if cmx is not None else 'N/A'}/{cmn if cmn is not None else 'N/A'} GB"
        )
    lines.append("")
    lines.append(f"Elapsed: {elapsed_s:.2f} s")
    lines.append("=============================")
    return "\n".join(lines)

