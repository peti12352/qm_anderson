"""
Runtime monitoring utilities for TMM runs.

CompileMonitor  — background thread that prints GPU stats every N seconds
                  while block_until_ready() is blocking. Distinguishes
                  XLA compilation (GPU idle) from GPU execution (GPU busy).

print_system_info — prints JAX/CUDA/GPU details at startup.
"""
from __future__ import annotations

import subprocess
import threading
import time

import jax


def _nvidia_smi(*fields: str) -> list[str] | None:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return [x.strip() for x in r.stdout.strip().split(",")]
    except Exception:
        return None


def print_system_info() -> None:
    """Print GPU + JAX details. Call once at the top of each run script."""
    print("\n" + "=" * 65)
    print("ENVIRONMENT")
    print(f"  JAX version  : {jax.__version__}")
    print(f"  JAX backend  : {jax.default_backend()}")
    print(f"  JAX devices  : {jax.devices()}")
    print(f"  x64 enabled  : {jax.config.x64_enabled}")

    vals = _nvidia_smi("name", "driver_version", "cuda_version",
                       "memory.total", "memory.free", "temperature.gpu",
                       "power.draw", "power.limit")
    if vals and len(vals) == 8:
        name, driver, cuda, mem_tot, mem_free, temp, pwr, pwr_lim = vals
        print(f"  GPU          : {name}")
        print(f"  Driver/CUDA  : {driver} / {cuda}")
        print(f"  VRAM         : {mem_free} MiB free / {mem_tot} MiB total")
        print(f"  Temp / Power : {temp}°C / {pwr}W (limit {pwr_lim}W)")
    else:
        print("  GPU info     : (nvidia-smi unavailable)")
    print("=" * 65 + "\n")


def _gpu_line() -> str:
    vals = _nvidia_smi("utilization.gpu", "memory.used", "memory.free",
                       "temperature.gpu", "power.draw")
    if not vals or len(vals) < 5:
        return "(no GPU stats)"
    util, used, free, temp, pwr = vals
    phase = "XLA compiling — GPU idle" if int(util) < 5 else f"GPU executing  {util}% util"
    return f"{phase} | VRAM {used}/{int(used)+int(free)} MiB | {temp}°C | {pwr}W"


class CompileMonitor:
    """
    Context manager: prints GPU stats every `interval` seconds while waiting
    for block_until_ready(). Use this to distinguish silent XLA compilation
    from actual GPU execution.

    Usage::

        with CompileMonitor(label="M=4 warmup", interval=10):
            result = tmm_batched(...).block_until_ready()
    """

    def __init__(self, label: str = "", interval: int = 10):
        self._label    = label
        self._interval = interval
        self._stop     = threading.Event()
        self._t0: float | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        elapsed = time.time() - self._t0
        stats = _gpu_line()
        print(f"\n    ✓ done in {elapsed:.0f}s | {stats}", flush=True)

    def _run(self):
        self._stop.wait(self._interval)          # skip first interval
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            stats   = _gpu_line()
            tag     = f" [{self._label}]" if self._label else ""
            print(f"\n    ↳{tag} {elapsed:5.0f}s | {stats}", flush=True)
            self._stop.wait(self._interval)
