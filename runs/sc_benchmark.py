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
M_LIST  = [4, 6, 8, 10]      # 4-point FSS sufficient; M=12 (dim=288) not worth it
W_LIST  = list(np.linspace(13.0, 20.0, 40))
# L=10_000 is >> ξ_M near W_c (ξ_M ~ M ~ 10); L=100_000 was 10× over-sampled.
# N_REAL=500 gives err~0.003 vs err~0.000 at 2000 — negligible for FSS.
# MGS (M≤6) and cuSOLVER (M≥8) in tmm.py give ~10–40s/point across all M.
L       = 10_000
N_REAL  = 500
E       = 0.0
T_HOP   = 1.0
P       = 1.0
OUTFILE  = Path("data/sc_benchmark.h5")
CKPT_DIR = Path("data/checkpoints")
SEED    = 42

FAST_N_REAL = 50

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> None:
    n_real = FAST_N_REAL if fast else N_REAL
    print_system_info()
    print(f"SC benchmark  |  M={M_LIST}  |  {len(W_LIST)} W values  |  "
          f"L={L}  |  N_real={n_real}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(SEED)
    all_gammas: dict[int, dict[float, np.ndarray]] = {}

    t0 = time.time()
    for M in M_LIST:
        all_gammas[M] = {}
        N = M * M

        # Resume: if a completed-M checkpoint exists, load it and skip.
        ckpt = CKPT_DIR / f"sc_benchmark_M{M}.npz"
        if ckpt.exists():
            data = np.load(ckpt)
            all_gammas[M] = {W: data[f"W_{i}"] for i, W in enumerate(W_LIST)}
            print(f"\n  M={M} | loaded from checkpoint {ckpt}  (skipping)")
            continue

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

        # Resume mid-M: load any per-W checkpoints saved from a previous interrupted run.
        ckpt_w_dir = CKPT_DIR / f"sc_benchmark_M{M}_partial"
        ckpt_w_dir.mkdir(exist_ok=True)
        done_W = set()
        for i, W in enumerate(W_LIST):
            wf = ckpt_w_dir / f"W_{i}.npy"
            if wf.exists():
                all_gammas[M][W] = np.load(wf)
                done_W.add(W)
        if done_W:
            print(f"  M={M} | resumed {len(done_W)}/{len(W_LIST)} W values from partial checkpoint")

        remaining_W = [W for W in W_LIST if W not in done_W]
        lam_by_W = {W: float(lambda_M(jnp.array(all_gammas[M][W]), M)[0])
                    for W in done_W}

        lam_by_W = {}
        t_M = time.time()
        with tqdm(remaining_W, desc=f"  M={M:2d}", unit="W", ncols=100,
                  initial=len(done_W), total=len(W_LIST)) as pbar:
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
                # Per-W checkpoint: survives a mid-M kill.
                np.save(ckpt_w_dir / f"W_{W_LIST.index(W)}.npy", gammas)

        elapsed_M = time.time() - t_M
        lam_vals = list(lam_by_W.values())
        print(f"  M={M} finished | {elapsed_M/60:.1f} min | "
              f"λ_M ∈ [{min(lam_vals):.3f}, {max(lam_vals):.3f}]")

        # Completed-M checkpoint.
        np.savez(ckpt, **{f"W_{i}": all_gammas[M][W] for i, W in enumerate(W_LIST)},
                 W_list=np.array(W_LIST))
        print(f"  ✓ checkpoint saved → {ckpt}")
        # Clean up per-W partials now that the full M checkpoint exists.
        for f in ckpt_w_dir.iterdir():
            f.unlink()
        ckpt_w_dir.rmdir()

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

    # FSS fit on a narrow window around W_c where the polynomial approximation is valid.
    # Full W range [13,20] gives |u|_max ≈ 16 for M=10 — far outside scaling regime.
    # Window [15.0, 18.5] gives |u|_max ≈ 1.5 × 4.6 ≈ 7 — much more controlled.
    W_FIT_RANGE = (15.0, 18.5)
    result = fit_fss(M_flat, W_flat, lam_flat, err_flat,
                     x_c_guess=16.5, nu_guess=1.57, poly_order=4,
                     x_range=W_FIT_RANGE, y_irrel=1.12,
                     fit_y_irrel=True, n_corrections=2)

    print("\n" + "="*55)
    print("FSS RESULT (SC Anderson disorder benchmark)")
    print(f"  W fit range: {W_FIT_RANGE}")
    print(f"  W_c = {result.x_c:.4f} ± {result.x_c_err:.4f}  (expected 16.53 ± 0.05)")
    print(f"  ν   = {result.nu:.4f} ± {result.nu_err:.4f}  (expected 1.571 ± 0.007)")
    print(f"  b   = {result.b:.4f} ± {result.b_err:.4f}  (first correction b·M^{{-y}})")
    print(f"  b₂  = {result.b2:.4f} ± {result.b2_err:.4f}  (second correction b₂·M^{{-2y}})")
    print(f"  y   = {result.y_irrel:.4f}  (irrelevant exponent; S&O99: 1.12±0.08)")
    print(f"  χ²/dof = {result.chi2_dof:.3f}  (ideal ≈ 1.0)")
    print(f"  converged: {result.converged}")
    print("="*55)

    nu_ok  = abs(result.nu  - 1.571) < 0.10
    wc_ok  = abs(result.x_c - 16.53) < 0.30   # 6σ on expected ±0.05
    chi_ok = result.chi2_dof < 5.0

    if nu_ok and wc_ok and chi_ok:
        print("\n  PASS: W_c, ν, and χ²/dof all within tolerance. Proceed to sc_dilution.py.")
    else:
        print("\n  FAIL — do NOT proceed to sc_dilution.py:")
        if not nu_ok:
            print(f"    ✗ ν = {result.nu:.4f} deviates from 1.571 by >{abs(result.nu-1.571):.3f}")
        if not wc_ok:
            print(f"    ✗ W_c = {result.x_c:.4f} deviates from 16.53 by {abs(result.x_c-16.53):.3f} (limit 0.30)")
        if not chi_ok:
            print(f"    ✗ χ²/dof = {result.chi2_dof:.2f} >> 1 (limit 5.0)")
        print("  Debug: transfer matrix, L vs ξ_M, W_FIT_RANGE width, poly_order.")

    metadata = dict(lattice="sc", E=E, t=T_HOP, L=L, n_real=n_real,
                    run_type="benchmark", W_c=result.x_c, nu=result.nu)
    save_run(OUTFILE, M_LIST, W_LIST, all_gammas, metadata)
    print(f"\nData saved to {OUTFILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use small N_real for quick sanity check")
    parser.add_argument("--refit", action="store_true",
                        help="Skip TMM; reload existing h5 and refit FSS only")
    parser.add_argument("--w-min", type=float, default=15.0,
                        help="Lower bound of FSS fit window (default 15.0)")
    parser.add_argument("--w-max", type=float, default=18.5,
                        help="Upper bound of FSS fit window (default 18.5)")
    args = parser.parse_args()

    if args.refit:
        from src.io_utils import load_lambda
        import jax.numpy as jnp
        print(f"Refit mode: loading {OUTFILE}, FSS window [{args.w_min}, {args.w_max}]")
        data    = load_lambda(OUTFILE)
        M_list  = [int(m) for m in data["M_list"]]
        x_vals  = data["x_values"]
        M_flat, W_flat, lam_flat, err_flat = [], [], [], []
        for M in M_list:
            for W in x_vals:
                lam, err = data["lambda"][M][W]
                M_flat.append(M); W_flat.append(float(W))
                lam_flat.append(lam); err_flat.append(max(err, 1e-6))
        W_FIT_RANGE = (args.w_min, args.w_max)
        result = fit_fss(M_flat, W_flat, lam_flat, err_flat,
                         x_c_guess=16.5, nu_guess=1.57, poly_order=4,
                         x_range=W_FIT_RANGE, y_irrel=1.12,
                         fit_y_irrel=True, n_corrections=2)
        print(f"\n  W fit range: {W_FIT_RANGE}")
        print(f"  W_c = {result.x_c:.4f} ± {result.x_c_err:.4f}  (expected 16.53)")
        print(f"  ν   = {result.nu:.4f} ± {result.nu_err:.4f}  (expected 1.571)")
        print(f"  b   = {result.b:.4f} ± {result.b_err:.4f}  (first correction b·M^{{-y}})")
        print(f"  b₂  = {result.b2:.4f} ± {result.b2_err:.4f}  (second correction b₂·M^{{-2y}})")
        print(f"  y   = {result.y_irrel:.4f}  (irrelevant exponent; S&O99: 1.12±0.08)")
        print(f"  χ²/dof = {result.chi2_dof:.3f}")
    else:
        main(fast=args.fast)
