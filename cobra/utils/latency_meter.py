"""Utility to benchmark model latency."""
from __future__ import annotations

import time

import torch


class LatencyMeter:
    def __init__(self, repeat: int = 10, sync: bool = True):
        self.repeat = repeat
        self.sync = sync
        self.records = []

    def measure(self, fn, *args, **kwargs):
        """Run ``fn(*args, **kwargs)`` multiple times and record average latency in seconds."""
        # Warmup run
        _ = fn(*args, **kwargs)
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(self.repeat):
            _ = fn(*args, **kwargs)
            if self.sync and torch.cuda.is_available():
                torch.cuda.synchronize()
        end = time.perf_counter()
        avg_s = (end - start) / self.repeat
        self.records.append(avg_s)
        return avg_s

    def summary(self):
        if not self.records:
            return "No latency data recorded."
        avg_ms = sum(self.records) / len(self.records) * 1000
        return f"Average Latency: {avg_ms:.2f} ms/batch (repeat={self.repeat})"
