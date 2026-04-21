"""
Week 2: Quantum percolation threshold p_q for the 3D SC site-diluted lattice.

Prerequisite: sc_benchmark.py must PASS (W_c and ν within 5% of expected).

What this measures
------------------
At fixed W=0 (no Anderson disorder) and E=0 (band centre), we scan the site
occupation probability p from p_c=0.3115 up to p=0.80 and compute Λ_M for
M ∈ {4,6,8,10,12}. The crossing point of Λ_M(p) curves is the quantum
percolation threshold p_q.

Physical interpretation: p_q is the minimum fraction of occupied sites
for any quantum transport to survive at the band centre. Below p_q, all
eigenstates are localized despite there being a classical connected path
(for p > p_c). The gap p_q - p_c quantifies the "quantum localization cost".

Note on disorder: We set W=0 to isolate the quantum percolation transition.
A separate sweep with W>0 (e.g. W=1) can be done to test whether weak
disorder shifts p_q — this is a secondary result if time allows.

Expected: p_q(SC) ≈ 0.44 ± 0.01 (Soukoulis & Economou 1983; last precise measurement)
Our target precision: ±0.002
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
# ---------------------------------------------------------------------------
M_LIST = [4, 6, 8, 10]   # M=12 deferred: adds ~22h, 4 sizes proven sufficient for FSS
# Dense coverage around the expected p_q ≈ 0.44; sparse far from it.
P_COARSE = list(np.linspace(0.32, 0.65, 20))
P_FINE   = list(np.linspace(0.38, 0.52, 20))
P_LIST   = sorted(set(P_COARSE + P_FINE))

# L=10k: at p≈0.44, γ_min≈0.3 for M=8 → L×γ_min=3000 >> 1. Proven convergent at
# benchmark. L=100k was 10× overkill; N_real=500 sufficient (same as benchmark).
L       = 10_000
N_REAL  = 500
W       = 0.0    # pure geometric disorder (site dilution only)
E       = 0.0    # band centre
T_HOP   = 1.0
OUTFILE = Path("data/sc_dilution.h5")
SEED    = 43

# ---------------------------------------------------------------------------
# Classical percolation sanity check
# ---------------------------------------------------------------------------

def _check_classical_threshold() -> None:
    """Verify that p_c(SC) ≈ 0.311 before the quantum run.

    Tolerance ±0.025: at L=20, finite-size shift is p_c(L)-p_c(∞) ∝ L^{-1/ν}
    ≈ 20^{-1.14} × 0.4 ≈ +0.013, so p_c(L=20) ≈ 0.325. A truly broken
    implementation (e.g. periodic BC bug) gives 0.207 — easily caught.
    """
    from src.percolation import find_pc
    pc_measured = find_pc(L=20, lattice="sc", n_samples=500)
    print(f"Classical p_c(SC) measured: {pc_measured:.4f}  (expected 0.3115; "
          f"L=20 finite-size value ≈ 0.325)")
    if abs(pc_measured - 0.3115) > 0.025:
        raise RuntimeError(
            f"Classical p_c check failed: {pc_measured:.4f} ≠ 0.3115 ± 0.025. "
            "Check percolation.py implementation."
        )
    print("  Classical percolation check: PASS\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print_system_info()
    _check_classical_threshold()

    print(f"SC quantum percolation  |  M={M_LIST}  |  {len(P_LIST)} p values  |  "
          f"L={L}  |  N_real={N_REAL}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(SEED)
    all_gammas: dict[int, dict[float, np.ndarray]] = {}

    t0 = time.time()
    for M in M_LIST:
        all_gammas[M] = {}
        N = M * M

        ckpt = CKPT_DIR / f"sc_dilution_M{M}.npz"
        if ckpt.exists():
            data = np.load(ckpt)
            all_gammas[M] = {P_LIST[i]: data[f"p_{i}"] for i in range(len(P_LIST))}
            print(f"\n  M={M} | loaded from checkpoint {ckpt}  (skipping)")
            continue

        _, tmm_batched = make_tmm_fns(M, "sc", E=E, t=T_HOP)

        print(f"\n  M={M} | N={N} | JIT compile+warmup (batch={N_REAL}, L={L})")
        with CompileMonitor(label=f"M={M} warmup", interval=10):
            _ = tmm_batched(jax.random.split(rng, N_REAL), L, W, P_LIST[len(P_LIST)//2]).block_until_ready()

        ckpt_w_dir = CKPT_DIR / f"sc_dilution_M{M}_partial"
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
        with tqdm(remaining_P, desc=f"  M={M:2d}", unit="p", ncols=100,
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
        print(f"  M={M} finished | {elapsed_M/60:.1f} min | "
              f"λ_M ∈ [{min(lam_vals):.3f}, {max(lam_vals):.3f}]")

        np.savez(ckpt, **{f"p_{i}": all_gammas[M][P_LIST[i]] for i in range(len(P_LIST))})
        print(f"  ✓ checkpoint saved → {ckpt}")
        for f in ckpt_w_dir.iterdir(): f.unlink()
        ckpt_w_dir.rmdir()

    elapsed = time.time() - t0
    print(f"\nData collected in {elapsed/60:.1f} min")

    # Compute Λ_M
    M_flat, p_flat, lam_flat, err_flat = [], [], [], []
    for M in M_LIST:
        for p in P_LIST:
            mean, stderr = lambda_M(jnp.array(all_gammas[M][p]), M)
            M_flat.append(M)
            p_flat.append(p)
            lam_flat.append(mean)
            err_flat.append(max(stderr, 1e-6))

    # Estimate p_q from visual crossing before FSS (rough)
    # Find p where the median Λ_M is flattest across M
    p_arr = np.array(P_LIST)
    lam_by_M = {M: [] for M in M_LIST}
    for M in M_LIST:
        for p in P_LIST:
            lam_by_M[M].append(lambda_M(jnp.array(all_gammas[M][p]), M)[0])
    spread = np.array([np.std([lam_by_M[M][ip] for M in M_LIST]) for ip in range(len(P_LIST))])
    p_q_rough = float(p_arr[np.argmin(spread)])
    print(f"\n  Rough p_q estimate (minimum spread): {p_q_rough:.3f}")

    p_fit_hw = 0.08   # half-width of FSS fit window around rough crossing
    p_fit_range = (max(0.25, p_q_rough - p_fit_hw), min(0.75, p_q_rough + p_fit_hw))
    print(f"  FSS fit window: p ∈ [{p_fit_range[0]:.3f}, {p_fit_range[1]:.3f}]")
    # y fixed at 1.12 (S&O99). With M={4..12}, y is non-identifiable from data:
    # b·M^{-y} degenerates with a_0 as y→0, optimizer exploits this. See RUNBOOK §FSS.
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
    print("FSS RESULT (SC site-diluted quantum percolation)")
    print(f"  y   = {Y_IRREL} fixed (S&O99: 1.12±0.08)")
    print(f"  p_q = {result.x_c:.5f} ± {result.x_c_err:.5f} (stat)  (literature: ~0.44)")
    print(f"  ν   = {result.nu:.4f}  ± {result.nu_err:.4f} (stat)  "
          "(Ujfalusi 2014: 1.62±0.035)")
    print(f"  b   = {result.b:.4f}  ± {result.b_err:.4f} (stat)")
    print(f"  χ²/dof = {result.chi2_dof:.3f}")
    print(f"  Systematic from y±0.08: p_q syst=±{pq_syst:.5f}  ν syst=±{nu_syst:.4f}")
    print(f"  p_q = {result.x_c:.5f} ± {result.x_c_err:.5f} (stat) ± {pq_syst:.5f} (syst)")
    print(f"  ν   = {result.nu:.4f}  ± {result.nu_err:.4f} (stat) ± {nu_syst:.4f} (syst)")
    print("="*55)

    metadata = dict(lattice="sc", E=E, t=T_HOP, L=L, W=W, n_real=N_REAL,
                    run_type="quantum_percolation",
                    p_q=result.x_c, p_q_err=result.x_c_err,
                    nu=result.nu, nu_err=result.nu_err)
    save_run(OUTFILE, M_LIST, P_LIST, all_gammas, metadata)
    print(f"Data saved to {OUTFILE}")


if __name__ == "__main__":
    main()
