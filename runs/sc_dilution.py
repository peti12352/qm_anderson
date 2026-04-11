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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M_LIST = [4, 6, 8, 10, 12]
# Dense coverage around the expected p_q ≈ 0.44; sparse far from it.
P_COARSE = list(np.linspace(0.32, 0.65, 20))
P_FINE   = list(np.linspace(0.38, 0.52, 20))
P_LIST   = sorted(set(P_COARSE + P_FINE))

L       = 100_000
N_REAL  = 2_000
W       = 0.0    # pure geometric disorder (site dilution only)
E       = 0.0    # band centre
T_HOP   = 1.0
OUTFILE = Path("data/sc_dilution.h5")
SEED    = 43

# ---------------------------------------------------------------------------
# Classical percolation sanity check
# ---------------------------------------------------------------------------

def _check_classical_threshold() -> None:
    """Verify that p_c(SC) ≈ 0.311 before the quantum run."""
    from src.percolation import find_pc
    pc_measured = find_pc(L=20, lattice="sc", n_samples=500)
    print(f"Classical p_c(SC) measured: {pc_measured:.4f}  (expected 0.3115)")
    if abs(pc_measured - 0.3115) > 0.01:
        raise RuntimeError(
            f"Classical p_c check failed: {pc_measured:.4f} ≠ 0.3115. "
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

    rng = jax.random.PRNGKey(SEED)
    all_gammas: dict[int, dict[float, np.ndarray]] = {}

    t0 = time.time()
    for M in M_LIST:
        all_gammas[M] = {}
        N = M * M
        _, tmm_batched = make_tmm_fns(M, "sc", E=E, t=T_HOP)

        print(f"\n  M={M} | N={N} | JIT compile+warmup (batch={N_REAL}, L={L})")
        with CompileMonitor(label=f"M={M} warmup", interval=10):
            _ = tmm_batched(jax.random.split(rng, N_REAL), L, W, P_LIST[len(P_LIST)//2]).block_until_ready()

        lam_by_p = {}
        t_M = time.time()
        with tqdm(P_LIST, desc=f"  M={M:2d}", unit="p", ncols=100) as pbar:
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

        elapsed_M = time.time() - t_M
        lam_vals = list(lam_by_p.values())
        print(f"  M={M} finished | {elapsed_M/60:.1f} min | "
              f"λ_M ∈ [{min(lam_vals):.3f}, {max(lam_vals):.3f}]")

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

    result = fit_fss(M_flat, p_flat, lam_flat, err_flat,
                     x_c_guess=p_q_rough, nu_guess=1.57, poly_order=3)

    print("\n" + "="*55)
    print("FSS RESULT (SC site-diluted quantum percolation)")
    print(f"  p_q = {result.x_c:.5f} ± {result.x_c_err:.5f}  (literature: ~0.44)")
    print(f"  ν   = {result.nu:.4f}  ± {result.nu_err:.4f}  "
          "(Ujfalusi 2014: 1.62 ± 0.035; Slevin: 1.571 ± 0.007)")
    print(f"  χ²/dof = {result.chi2_dof:.3f}")
    print("="*55)

    metadata = dict(lattice="sc", E=E, t=T_HOP, L=L, W=W, n_real=N_REAL,
                    run_type="quantum_percolation",
                    p_q=result.x_c, p_q_err=result.x_c_err,
                    nu=result.nu, nu_err=result.nu_err)
    save_run(OUTFILE, M_LIST, P_LIST, all_gammas, metadata)
    print(f"Data saved to {OUTFILE}")


if __name__ == "__main__":
    main()
