"""
Week 1 gate: reproduce the 3D SC Anderson metal-insulator transition.

Expected result (from Slevin & Ohtsuki 2014, arXiv:1307.4483):
    W_c ≈ 16.53 ± 0.05   (in units of t)
    ν   ≈ 1.571 ± 0.007

If your numbers disagree by more than 5%, stop and debug before continuing.
Common failure modes:
    - Wrong transfer matrix (see TRANSFER MATRIX BUG CHECKLIST in lattice.py)
    - Missing sign fix in QR (see tmm.py step function)
    - Too-short bars (increase L)
    - Not enough realizations near W_c (increase N_REAL)

Usage:
    pixi run benchmark
    # or
    python runs/sc_benchmark.py [--fast]   # --fast uses small N_REAL for quick check
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config   # noqa: F401  — must import first to enable x64 + XLA flags
from src.tmm     import make_tmm_fns, lambda_M
from src.fss     import fit_fss
from src.io_utils import save_run
from src.monitor  import print_system_info, CompileMonitor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M_LIST  = [4, 6, 8, 10, 12]
W_LIST  = list(np.linspace(13.0, 20.0, 40))   # dense coverage of transition
L       = 100_000   # bar length; increase to 300k near W_c if variance is high
N_REAL  = 2_000     # disorder realizations per (M, W) point
E       = 0.0       # band centre
T_HOP   = 1.0       # hopping amplitude (energy unit)
P       = 1.0       # no site dilution
OUTFILE = Path("data/sc_benchmark.h5")
SEED    = 42

FAST_N_REAL = 200   # for --fast mode

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> None:
    n_real = FAST_N_REAL if fast else N_REAL
    print_system_info()
    print(f"SC benchmark  |  M={M_LIST}  |  {len(W_LIST)} W values  |  "
          f"L={L}  |  N_real={n_real}")

    rng = jax.random.PRNGKey(SEED)
    all_gammas: dict[int, dict[float, np.ndarray]] = {}

    t0 = time.time()
    for M in M_LIST:
        all_gammas[M] = {}
        N = M * M
        _, tmm_batched = make_tmm_fns(M, "sc", E=E, t=T_HOP)

        # Warm up JIT (first call compiles — can take several minutes on first ever run)
        # Warm-up MUST use the production batch size (n_real).
        # A different size (e.g. 4) produces a different XLA kernel and causes
        # a silent recompile on the first real point — appearing as a 30-min hang.
        keys_warmup = jax.random.split(rng, n_real)
        print(f"\n  M={M} | N={N} | JIT compile+warmup (batch={n_real}, L={L})")
        print(f"  GPU idle = XLA compiling (CPU-bound); GPU busy = executing")
        with CompileMonitor(label=f"M={M} warmup", interval=10):
            _ = tmm_batched(keys_warmup, L, W_LIST[len(W_LIST)//2], P).block_until_ready()

        lam_by_W = {}
        t_M = time.time()
        with tqdm(W_LIST, desc=f"  M={M:2d}", unit="W", ncols=100) as pbar:
            for W in pbar:
                rng, subkey = jax.random.split(rng)
                keys = jax.random.split(subkey, n_real)
                t_pt = time.time()
                gammas = np.asarray(tmm_batched(keys, L, W, P).block_until_ready())
                dt = time.time() - t_pt
                all_gammas[M][W] = gammas
                lam, err = lambda_M(jnp.array(gammas), M)
                lam_by_W[W] = lam
                pbar.set_postfix(
                    W=f"{W:.2f}",
                    lam=f"{lam:.3f}±{err:.3f}",
                    s_pt=f"{dt:.1f}s",
                    real_s=f"{n_real/dt:.0f}",
                )

        elapsed_M = time.time() - t_M
        lam_vals = list(lam_by_W.values())
        print(f"  M={M} finished | {elapsed_M/60:.1f} min | "
              f"λ_M ∈ [{min(lam_vals):.3f}, {max(lam_vals):.3f}]")

    elapsed = time.time() - t0
    print(f"\nRaw data collected in {elapsed/60:.1f} min")

    # FSS fit
    M_flat, W_flat, lam_flat, err_flat = [], [], [], []
    for M in M_LIST:
        N = M * M
        for W in W_LIST:
            mean, stderr = lambda_M(jnp.array(all_gammas[M][W]), M)
            M_flat.append(M)
            W_flat.append(W)
            lam_flat.append(mean)
            err_flat.append(max(stderr, 1e-6))   # guard against zero stderr

    result = fit_fss(M_flat, W_flat, lam_flat, err_flat,
                     x_c_guess=16.5, nu_guess=1.57, poly_order=3)

    print("\n" + "="*50)
    print("FSS RESULT (SC Anderson disorder benchmark)")
    print(f"  W_c = {result.x_c:.4f} ± {result.x_c_err:.4f}  (expected ≈ 16.53)")
    print(f"  ν   = {result.nu:.4f} ± {result.nu_err:.4f}  (expected ≈ 1.571)")
    print(f"  χ²/dof = {result.chi2_dof:.3f}  (ideal ≈ 1.0)")
    print(f"  converged: {result.converged}")
    print("="*50)

    if abs(result.nu - 1.571) > 0.10:
        print("\n  WARNING: ν deviates from expected by >6%. Do NOT proceed.")
        print("  Debug checklist: transfer matrix construction, QR sign fix, L vs ξ_M.")
    else:
        print("\n  PASS: benchmark within tolerance. Proceed to sc_dilution.py")

    metadata = dict(lattice="sc", E=E, t=T_HOP, L=L, n_real=n_real,
                    run_type="benchmark", W_c=result.x_c, nu=result.nu)
    save_run(OUTFILE, M_LIST, W_LIST, all_gammas, metadata)
    print(f"\nData saved to {OUTFILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use small N_real for quick sanity check")
    args = parser.parse_args()
    main(fast=args.fast)
