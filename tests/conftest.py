"""
Pytest session configuration: prints device info and timing context
so the user can see whether JAX found the GPU and roughly how long
each compilation/run phase is taking.
"""
import sys
from pathlib import Path

# Add project root to path so 'src' is importable regardless of where
# pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def pytest_sessionstart(session):
    import jax
    import src.config  # noqa — enables x64 + GPU before any jax call

    print("\n" + "=" * 60)
    print("JAX devices:", jax.devices())
    print("JAX default backend:", jax.default_backend())
    print("x64 enabled:", jax.config.x64_enabled)
    print("=" * 60)


def pytest_runtest_logreport(report):
    """Print elapsed time for each test phase (setup/call/teardown)."""
    if report.when == "call":
        duration = report.duration
        outcome  = "PASSED" if report.passed else ("FAILED" if report.failed else "ERROR")
        print(f"\n  [{outcome}] {report.nodeid.split('::')[-1]}  ({duration:.1f}s)")
