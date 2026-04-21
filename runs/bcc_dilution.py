"""
Week 3: Quantum percolation threshold p_q for the 3D BCC site-diluted lattice.

Prerequisites:
  - sc_benchmark.py PASS
  - sc_dilution.py complete (gives us p_q(SC) for comparison)

BCC lattice specifics
---------------------
  Coordination number:  z = 8 (vs z = 6 for SC)
  Classical threshold:  p_c(BCC) = 0.2457
  Bandwidth:            16t (vs 12t for SC)
  Transfer matrix V:    NON-DIAGONAL — 4 off-diagonal shifts per row
  Twisted BC:           Automatically applied when M ∈ {4,8,12} to avoid
                        singular V (see src/lattice.py for derivation)
  Recommended M values: {6, 10} avoid twisted BC; {4,8,12} use it
  We use M ∈ {4, 6, 8, 10} for a 4-point FSS (5 is better; add M=12 if time allows)

Physical expectation
--------------------
BCC has higher coordination → quantum paths between distant sites persist at
lower occupation probability. We expect p_q(BCC) < p_q(SC) in absolute terms,
and the ratio p_q/p_c to be similar (same universality class).

BCC anomaly to watch for: The higher connectivity may delay onset of localization
to smaller p/p_c ratio than SC — this would be a novel and physically interesting result.
"""
from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config
from src.tmm      import make_tmm_fns, lambda_M
from src.fss      import fit_fss
from src.io_utils  import save_run
from src.monitor   import print_system_info, CompileMonitor

CKPT_DIR = Path("data/checkpoints")

# ---------------------------------------------------------------------------
# Configuration
# Note: avoid M ∈ {4,8,12} for BCC if you want to skip twisted BC entirely.
# The code handles twisted BC automatically, but M=6,10 are the "clean" sizes.
# ---------------------------------------------------------------------------
M_LIST = [4, 6, 8, 10]   # will use twisted BC for M=4 and M=8 automatically

P_COARSE = list(np.linspace(0.25, 0.60, 18))
P_FINE   = list(np.linspace(0.30, 0.46, 20))
P_LIST   = sorted(set(P_COARSE + P_FINE))

# L=10k: at p≈p_q, γ_min≈0.3 for M=8 → L×γ_min=3000 >> 1. Same reasoning as SC.
L       = 10_000
N_REAL  = 500
W       = 0.0
E       = 0.0
T_HOP   = 1.0
OUTFILE = Path("data/bcc_dilution.h5")
SEED    = 44


def _check_classical_threshold() -> None:
    """Tolerance ±0.025: L=20 finite-size shift ∝ L^{-1/ν} ≈ +0.010 for BCC."""
    from src.percolation import find_pc
    pc_measured = find_pc(L=20, lattice="bcc", p_min=0.18, p_max=0.35, n_samples=500)
    print(f"Classical p_c(BCC) measured: {pc_measured:.4f}  (expected 0.2457; "
          f"L=20 finite-size value ≈ 0.256)")
    if abs(pc_measured - 0.2457) > 0.025:
        raise RuntimeError(
            f"Classical p_c(BCC) check failed: {pc_measured:.4f} ≠ 0.2457 ± 0.025. "
            "Check percolation.py BCC neighbor function."
        )
    print("  Classical percolation check: PASS\n")


def _compare_with_sc(p_q_bcc: float, nu_bcc: float) -> None:
    """Print comparison table if SC results exist."""
    sc_path = Path("data/sc_dilution.h5")
    if not sc_path.exists():
        print("  (SC results not found; run sc_dilution.py for comparison)")
        return
    from src.io_utils import load_lambda
    sc_data = load_lambda(sc_path)
    p_q_sc  = sc_data["metadata"].get("p_q", "N/A")
    nu_sc   = sc_data["metadata"].get("nu",  "N/A")
    p_c_sc  = 0.3115
    p_c_bcc = 0.2457
    print(f"\n  Comparison:")
    print(f"  {'':20s}  {'SC':>12s}  {'BCC':>12s}")
    print(f"  {'p_c (classical)':20s}  {p_c_sc:12.4f}  {p_c_bcc:12.4f}")
    print(f"  {'p_q (quantum)':20s}  {float(p_q_sc) if isinstance(p_q_sc, float) else p_q_sc:>12}  {p_q_bcc:12.5f}")
    print(f"  {'p_q / p_c':20s}  {float(p_q_sc)/p_c_sc if isinstance(p_q_sc, float) else '?':>12}  {p_q_bcc/p_c_bcc:12.4f}")
    print(f"  {'ν':20s}  {float(nu_sc) if isinstance(nu_sc, float) else nu_sc:>12}  {nu_bcc:12.4f}")


def main() -> None:
    print_system_info()
    _check_classical_threshold()

    print(f"BCC quantum percolation  |  M={M_LIST}  |  {len(P_LIST)} p values  |  "
          f"L={L}  |  N_real={N_REAL}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(SEED)
    all_gammas: dict[int, dict[float, np.ndarray]] = {}

    t0 = time.time()
    for M in M_LIST:
        all_gammas[M] = {}
        N = M * M

        ckpt = CKPT_DIR / f"bcc_dilution_M{M}.npz"
        if ckpt.exists():
            data = np.load(ckpt)
            all_gammas[M] = {P_LIST[i]: data[f"p_{i}"] for i in range(len(P_LIST))}
            print(f"\n  M={M} | loaded from checkpoint {ckpt}  (skipping)")
            continue

        _, tmm_batched = make_tmm_fns(M, "bcc", E=E, t=T_HOP)

        print(f"\n  M={M} | N={N} | BCC complex128 | JIT compile+warmup (batch={N_REAL}, L={L})")
        with CompileMonitor(label=f"M={M} BCC warmup", interval=10):
            _ = tmm_batched(jax.random.split(rng, N_REAL), L, W, P_LIST[len(P_LIST)//2]).block_until_ready()

        ckpt_w_dir = CKPT_DIR / f"bcc_dilution_M{M}_partial"
        ckpt_w_dir.mkdir(exist_ok=True)
        done_p = set()
        for i, p in enumerate(P_LIST):
            wf = ckpt_w_dir / f"p_{i}.npy"
            if wf.exists():
                all_gammas[M][p] = np.load(wf)
                done_p.add(p)
        if done_p:
            print(f"  M={M} | resumed {len(done_p)}/{len(P_LIST)} p values from partial checkpoint")

        remaining_P = [p for p in P_LIST if p not in done_p]
        lam_by_p = {}
        t_M = time.time()
        with tqdm(remaining_P, desc=f"  M={M:2d} BCC", unit="p", ncols=100,
                  initial=len(done_p), total=len(P_LIST)) as pbar:
            for p in pbar:
                rng, subkey = jax.random.split(rng)
                keys = jax.random.split(subkey, N_REAL)
                t_pt = time.time()
                gammas = np.asarray(tmm_batched(keys, L, W, p).block_until_ready())
                dt = time.time() - t_pt
                all_gammas[M][p] = gammas
                lam, err = lambda_M(jnp.array(gammas), M)
                lam_by_p[p] = lam
                pbar.set_postfix(
                    p=f"{p:.3f}",
                    lam=f"{lam:.3f}±{err:.3f}",
                    s_pt=f"{dt:.1f}s",
                    real_s=f"{N_REAL/dt:.0f}",
                )
                np.save(ckpt_w_dir / f"p_{P_LIST.index(p)}.npy", gammas)

        elapsed_M = time.time() - t_M
        lam_vals = list(lam_by_p.values())
        print(f"  M={M} BCC finished | {elapsed_M/60:.1f} min | "
              f"λ_M ∈ [{min(lam_vals):.3f}, {max(lam_vals):.3f}]")

        np.savez(ckpt, **{f"p_{i}": all_gammas[M][P_LIST[i]] for i in range(len(P_LIST))})
        print(f"  ✓ checkpoint saved → {ckpt}")
        for f in ckpt_w_dir.iterdir(): f.unlink()
        ckpt_w_dir.rmdir()

    elapsed = time.time() - t0
    print(f"\nData collected in {elapsed/60:.1f} min")

    M_flat, p_flat, lam_flat, err_flat = [], [], [], []
    for M in M_LIST:
        for p in P_LIST:
            mean, stderr = lambda_M(jnp.array(all_gammas[M][p]), M)
            M_flat.append(M)
            p_flat.append(p)
            lam_flat.append(mean)
            err_flat.append(max(stderr, 1e-6))

    # Rough crossing estimate
    p_arr = np.array(P_LIST)
    lam_by_M = {M: [] for M in M_LIST}
    for M in M_LIST:
        for p in P_LIST:
            lam_by_M[M].append(lambda_M(jnp.array(all_gammas[M][p]), M)[0])
    spread = np.array([np.std([lam_by_M[M][ip] for M in M_LIST]) for ip in range(len(P_LIST))])
    p_q_rough = float(p_arr[np.argmin(spread)])
    print(f"\n  Rough p_q(BCC) estimate: {p_q_rough:.3f}")

    p_fit_hw = 0.08
    p_fit_range = (max(0.22, p_q_rough - p_fit_hw), min(0.65, p_q_rough + p_fit_hw))
    print(f"  FSS fit window: p ∈ [{p_fit_range[0]:.3f}, {p_fit_range[1]:.3f}]")
    # y fixed at 1.12 (S&O99). BCC quantum percolation is in the same 3D orthogonal
    # universality class as SC — same y applies. See RUNBOOK §FSS for identifiability.
    Y_IRREL = 1.12
    result = fit_fss(M_flat, p_flat, lam_flat, err_flat,
                     x_c_guess=p_q_rough, nu_guess=1.57, poly_order=4,
                     x_range=p_fit_range, y_irrel=Y_IRREL,
                     fit_y_irrel=False, n_corrections=1)

    y_lo, y_hi = Y_IRREL - 0.08, Y_IRREL + 0.08
    r_lo = fit_fss(M_flat, p_flat, lam_flat, err_flat,
                   x_c_guess=p_q_rough, nu_guess=1.57, poly_order=4,
                   x_range=p_fit_range, y_irrel=y_lo,
                   fit_y_irrel=False, n_corrections=1)
    r_hi = fit_fss(M_flat, p_flat, lam_flat, err_flat,
                   x_c_guess=p_q_rough, nu_guess=1.57, poly_order=4,
                   x_range=p_fit_range, y_irrel=y_hi,
                   fit_y_irrel=False, n_corrections=1)
    pq_syst = max(abs(r_lo.x_c - result.x_c), abs(r_hi.x_c - result.x_c))
    nu_syst = max(abs(r_lo.nu  - result.nu),  abs(r_hi.nu  - result.nu))

    print("\n" + "="*55)
    print("FSS RESULT (BCC site-diluted quantum percolation)")
    print(f"  y   = {Y_IRREL} fixed (S&O99: 1.12±0.08; 3D orthogonal class)")
    print(f"  p_q = {result.x_c:.5f} ± {result.x_c_err:.5f} (stat)  (first measurement)")
    print(f"  ν   = {result.nu:.4f}  ± {result.nu_err:.4f} (stat)")
    print(f"  b   = {result.b:.4f}  ± {result.b_err:.4f} (stat)")
    print(f"  χ²/dof = {result.chi2_dof:.3f}")
    print(f"  Systematic from y±0.08: p_q syst=±{pq_syst:.5f}  ν syst=±{nu_syst:.4f}")
    print(f"  p_q = {result.x_c:.5f} ± {result.x_c_err:.5f} (stat) ± {pq_syst:.5f} (syst)")
    print(f"  ν   = {result.nu:.4f}  ± {result.nu_err:.4f} (stat) ± {nu_syst:.4f} (syst)")
    print("="*55)

    _compare_with_sc(result.x_c, result.nu)

    metadata = dict(lattice="bcc", E=E, t=T_HOP, L=L, W=W, n_real=N_REAL,
                    run_type="quantum_percolation",
                    p_q=result.x_c, p_q_err=result.x_c_err,
                    nu=result.nu, nu_err=result.nu_err)
    save_run(OUTFILE, M_LIST, P_LIST, all_gammas, metadata)
    print(f"Data saved to {OUTFILE}")


if __name__ == "__main__":
    main()
