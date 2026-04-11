"""
Global JAX configuration.

Must be imported before any jax.numpy call that expects float64.
"""
import os
from pathlib import Path

# XLA GPU autotuning: default level 4 benchmarks hundreds of CUDA kernel
# configurations per op — this is the most common cause of 10-60 min
# silent compile hangs on first run. Level 0 uses heuristic selection
# (negligible runtime cost for sequential QR-dominated workloads like TMM).
# Set XLA_FLAGS before importing jax so it takes effect.
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_autotune_level=0 --xla_gpu_enable_fast_min_max=true",
)

import jax

# Enable 64-bit precision globally. Without this, JAX defaults to float32
# everywhere, which is insufficient for accumulating log-norms over 10^5 steps.
jax.config.update("jax_enable_x64", True)

# Silence the "No GPU/TPU found" warning when running tests on CPU.
# Remove or comment out if debugging CUDA issues.
jax.config.update("jax_platform_name", "gpu")

# Persistent XLA compilation cache.
# First run of each (M, L, lattice) combination triggers a 5–20 min XLA compile.
# With this cache, subsequent runs load the compiled kernel from disk in seconds.
_cache_dir = Path.home() / ".cache" / "jax_qm_anderson"
_cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_cache_dir))

# Physical constant: on-site energy assigned to absent (diluted) sites.
# Must be >> max disorder + bandwidth. For SC bandwidth ≈ 12t (W=25t max),
# 1e6 t is safely irrelevant. Wavefunction amplitude at absent sites ~ t/ABSENT_ENERGY.
ABSENT_ENERGY: float = 1e6
