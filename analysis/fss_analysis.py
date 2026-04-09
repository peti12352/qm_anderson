"""
Offline FSS analysis and comparison across datasets.

Run after all three data files exist:
    python analysis/fss_analysis.py

Outputs a summary table suitable for the paper, plus refined FSS fits
with corrections to scaling (irrelevant variable u_irr = M^{-y}).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config  # noqa
import jax.numpy as jnp
from src.tmm       import lambda_M
from src.fss       import fit_fss, FSSResult
from src.io_utils   import load_lambda


def _load_and_fit(path: Path, x_c_guess: float, nu_guess: float = 1.57) -> FSSResult | None:
    if not path.exists():
        print(f"  Missing: {path}")
        return None
    data   = load_lambda(path)
    M_list = [int(m) for m in data["M_list"]]
    x_vals = data["x_values"]

    M_flat, x_flat, lam_flat, err_flat = [], [], [], []
    for M in M_list:
        for x in x_vals:
            lam, err = data["lambda"][M][x]
            M_flat.append(M); x_flat.append(float(x))
            lam_flat.append(lam); err_flat.append(max(err, 1e-6))

    return fit_fss(M_flat, x_flat, lam_flat, err_flat,
                   x_c_guess=x_c_guess, nu_guess=nu_guess,
                   poly_order=3, n_boot=1000)


def main() -> None:
    print("="*65)
    print("FSS ANALYSIS SUMMARY")
    print("="*65)

    # SC Anderson benchmark
    r_bench = _load_and_fit(Path("data/sc_benchmark.h5"),  x_c_guess=16.5)
    # SC quantum percolation
    r_sc    = _load_and_fit(Path("data/sc_dilution.h5"),   x_c_guess=0.44)
    # BCC quantum percolation
    r_bcc   = _load_and_fit(Path("data/bcc_dilution.h5"),  x_c_guess=0.38)

    def row(label, r, ref_x_c=None, ref_nu=None):
        if r is None:
            print(f"  {label:45s}  [data missing]")
            return
        x_str  = f"{r.x_c:.5f} ± {r.x_c_err:.5f}"
        nu_str = f"{r.nu:.4f} ± {r.nu_err:.4f}"
        ref_x  = f"(ref: {ref_x_c})" if ref_x_c else ""
        ref_n  = f"(ref: {ref_nu})"  if ref_nu  else ""
        print(f"  {label:45s}")
        print(f"    x_c = {x_str}  {ref_x}")
        print(f"    ν   = {nu_str}  {ref_n}")
        print(f"    χ²/dof = {r.chi2_dof:.2f}  |  converged: {r.converged}")

    row("SC Anderson disorder (W_c, ν)", r_bench, "16.53 t", "1.571")
    print()
    row("SC quantum percolation (p_q, ν)", r_sc, "0.44", "1.57 or 1.62?")
    print()
    row("BCC quantum percolation (p_q, ν)", r_bcc, "FIRST MEASUREMENT", "1.57?")

    if r_sc and r_bcc:
        p_c_sc  = 0.3115
        p_c_bcc = 0.2457
        print()
        print("  UNIVERSALITY CHECK:")
        delta_nu = abs(r_sc.nu - r_bcc.nu)
        sigma    = np.sqrt(r_sc.nu_err**2 + r_bcc.nu_err**2)
        print(f"    |ν_SC - ν_BCC| = {delta_nu:.4f}  ({delta_nu/sigma:.1f}σ)")
        if delta_nu < 2 * sigma:
            print("    → Consistent with same universality class (ν_SC ≈ ν_BCC)")
        else:
            print("    → TENSION: ν values differ at >2σ. Investigate corrections to scaling.")

        print()
        print("  p_q / p_c ratios:")
        print(f"    SC:  p_q/p_c = {r_sc.x_c:.4f} / {p_c_sc:.4f} = {r_sc.x_c/p_c_sc:.3f}")
        print(f"    BCC: p_q/p_c = {r_bcc.x_c:.4f} / {p_c_bcc:.4f} = {r_bcc.x_c/p_c_bcc:.3f}")

    print("="*65)

    # Harris criterion check
    print("\nHARRIS CRITERION CHECK (dν > 2 required for stable universality):")
    for label, r in [("SC benchmark", r_bench), ("SC dilution", r_sc), ("BCC dilution", r_bcc)]:
        if r is None:
            continue
        criterion = 3 * r.nu
        print(f"  {label}: 3ν = 3 × {r.nu:.4f} = {criterion:.4f}  "
              f"{'✓ SATISFIED' if criterion > 2 else '✗ VIOLATED'}")


if __name__ == "__main__":
    main()
