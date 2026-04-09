"""
Global JAX configuration.

Must be imported before any jax.numpy call that expects float64.
"""
import jax

# Enable 64-bit precision globally. Without this, JAX defaults to float32
# everywhere, which is insufficient for accumulating log-norms over 10^5 steps.
jax.config.update("jax_enable_x64", True)

# Silence the "No GPU/TPU found" warning when running tests on CPU.
# Remove or comment out if debugging CUDA issues.
jax.config.update("jax_platform_name", "gpu")

# Physical constant: on-site energy assigned to absent (diluted) sites.
# Must be >> max disorder + bandwidth. For SC bandwidth ≈ 12t (W=25t max),
# 1e6 t is safely irrelevant. Wavefunction amplitude at absent sites ~ t/ABSENT_ENERGY.
ABSENT_ENERGY: float = 1e6
