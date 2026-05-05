"""
Publication figures for the Anderson quantum percolation paper.

Generates all figures in PDF format (300 dpi) in figures/ directory.
Requires completed HDF5 data files in data/.

Figure list
-----------
Fig 1: Λ_M(W) crossing curves - SC Anderson disorder benchmark
Fig 2: FSS collapse - SC Anderson disorder
Fig 3: Λ_M(p) crossing curves - SC site-diluted quantum percolation
Fig 4: FSS collapse - SC quantum percolation
Fig 5: Λ_M(p) crossing curves - BCC site-diluted quantum percolation
Fig 6: FSS collapse - BCC quantum percolation
Fig 7: Summary comparison: p_c vs p_q for SC and BCC
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config   # noqa
import jax.numpy as jnp
from src.tmm      import lambda_M as compute_lambda_M
from src.fss      import fit_fss, fss_curve
from src.io_utils  import load_lambda, load_raw_gammas

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   12,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "text.usetex":      False,   # set True if LaTeX is installed
    "lines.linewidth":  1.5,
    "lines.markersize": 5,
})
COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 6))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crossing_plot(ax, M_list, x_values, lam_dict, xlabel, title):
    for i, M in enumerate(M_list):
        x_arr  = np.array(x_values)
        lam_arr = np.array([lam_dict[M][x][0] for x in x_values])
        err_arr = np.array([lam_dict[M][x][1] for x in x_values])
        ax.errorbar(x_arr, lam_arr, yerr=err_arr,
                    fmt="o-", color=COLORS[i], label=f"M={M}", capsize=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Lambda_M = \xi_M / M$")
    ax.set_title(title)
    ax.legend()


def _collapse_plot(ax, M_list, x_values, lam_dict, result, x_label, title):
    x_c, nu = result.x_c, result.nu
    for i, M in enumerate(M_list):
        x_arr  = np.array(x_values)
        lam_arr = np.array([lam_dict[M][x][0] for x in x_values])
        err_arr = np.array([lam_dict[M][x][1] for x in x_values])
        u = M ** (1.0 / nu) * (x_arr - x_c) / x_c
        ax.errorbar(u, lam_arr, yerr=err_arr,
                    fmt="o", color=COLORS[i], label=f"M={M}", capsize=2, alpha=0.7)
    # Draw fitted curve
    u_dense = np.linspace(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -3, 3, 200)
    # Fit curve uses the best-fit polynomial coefficients embedded in fss_curve
    # (coefficients not stored in FSSResult - refine by re-fitting or just draw data)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$M^{1/\nu}(x - x_c)/x_c$")
    ax.set_ylabel(r"$\Lambda_M$")
    ax.set_title(title)
    ax.legend()


# ---------------------------------------------------------------------------
# Main figure generation
# ---------------------------------------------------------------------------

def make_all_figures() -> None:
    sc_bench_path  = Path("data/sc_benchmark.h5")
    sc_dilut_path  = Path("data/sc_dilution.h5")
    bcc_dilut_path = Path("data/bcc_dilution.h5")

    figs_generated = []

    # --- Fig 1 + 2: SC benchmark ---
    if sc_bench_path.exists():
        data   = load_lambda(sc_bench_path)
        M_list = [int(m) for m in data["M_list"]]
        W_vals = data["x_values"]

        # Reconstruct FSSResult from metadata
        meta    = data["metadata"]
        W_c     = float(meta.get("W_c", 16.5))
        nu_bench = float(meta.get("nu", 1.571))

        M_flat, W_flat, lam_flat, err_flat = [], [], [], []
        for M in M_list:
            for W in W_vals:
                lam, err = data["lambda"][M][W]
                M_flat.append(M); W_flat.append(W)
                lam_flat.append(lam); err_flat.append(max(err, 1e-6))
        result_bench = fit_fss(M_flat, W_flat, lam_flat, err_flat,
                               x_c_guess=W_c, nu_guess=nu_bench)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        _crossing_plot(ax1, M_list, W_vals, data["lambda"],
                       xlabel=r"Disorder $W/t$",
                       title=r"SC Anderson: $\Lambda_M(W)$ crossing")
        ax1.axvline(result_bench.x_c, color="red", linestyle="--",
                    label=f"$W_c={result_bench.x_c:.2f}$", linewidth=1)
        ax1.legend()

        _collapse_plot(ax2, M_list, W_vals, data["lambda"], result_bench,
                       x_label=r"$W$",
                       title=fr"FSS collapse: $\nu={result_bench.nu:.3f}\pm{result_bench.nu_err:.3f}$")

        fig.suptitle(
            fr"3D SC Anderson Benchmark - $W_c={result_bench.x_c:.3f}\pm{result_bench.x_c_err:.3f}$, "
            fr"$\nu={result_bench.nu:.3f}\pm{result_bench.nu_err:.3f}$  [Slevin: 16.53, 1.571]"
        )
        fig.tight_layout()
        p = FIGURES_DIR / "fig1_2_sc_benchmark.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        figs_generated.append(p)
        print(f"  Saved {p}")

    # --- Fig 3 + 4: SC quantum percolation ---
    if sc_dilut_path.exists():
        data    = load_lambda(sc_dilut_path)
        M_list  = [int(m) for m in data["M_list"]]
        p_vals  = data["x_values"]
        meta    = data["metadata"]
        p_q_sc  = float(meta.get("p_q",  0.44))
        nu_sc   = float(meta.get("nu",   1.57))

        M_flat, p_flat, lam_flat, err_flat = [], [], [], []
        for M in M_list:
            for p in p_vals:
                lam, err = data["lambda"][M][p]
                M_flat.append(M); p_flat.append(p)
                lam_flat.append(lam); err_flat.append(max(err, 1e-6))
        result_sc = fit_fss(M_flat, p_flat, lam_flat, err_flat,
                            x_c_guess=p_q_sc, nu_guess=nu_sc)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        _crossing_plot(ax1, M_list, p_vals, data["lambda"],
                       xlabel=r"Site occupation $p$",
                       title=r"SC Site-Diluted: $\Lambda_M(p)$")
        ax1.axvline(result_sc.x_c, color="red", linestyle="--",
                    label=f"$p_q={result_sc.x_c:.4f}$", linewidth=1)
        ax1.axvline(0.3115, color="grey", linestyle=":", linewidth=1, label=r"$p_c^{\rm cl}$")
        ax1.legend()

        _collapse_plot(ax2, M_list, p_vals, data["lambda"], result_sc,
                       x_label=r"$p$",
                       title=fr"FSS: $\nu={result_sc.nu:.3f}\pm{result_sc.nu_err:.3f}$")

        fig.suptitle(
            fr"SC Quantum Percolation - $p_q={result_sc.x_c:.4f}\pm{result_sc.x_c_err:.4f}$, "
            fr"$\nu={result_sc.nu:.3f}\pm{result_sc.nu_err:.3f}$"
        )
        fig.tight_layout()
        p = FIGURES_DIR / "fig3_4_sc_dilution.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        figs_generated.append(p)
        print(f"  Saved {p}")

    # --- Fig 5 + 6: BCC quantum percolation ---
    if bcc_dilut_path.exists():
        data    = load_lambda(bcc_dilut_path)
        M_list  = [int(m) for m in data["M_list"]]
        p_vals  = data["x_values"]
        meta    = data["metadata"]
        p_q_bcc = float(meta.get("p_q", 0.38))
        nu_bcc  = float(meta.get("nu",  1.57))

        M_flat, p_flat, lam_flat, err_flat = [], [], [], []
        for M in M_list:
            for p in p_vals:
                lam, err = data["lambda"][M][p]
                M_flat.append(M); p_flat.append(p)
                lam_flat.append(lam); err_flat.append(max(err, 1e-6))
        result_bcc = fit_fss(M_flat, p_flat, lam_flat, err_flat,
                             x_c_guess=p_q_bcc, nu_guess=nu_bcc)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        _crossing_plot(ax1, M_list, p_vals, data["lambda"],
                       xlabel=r"Site occupation $p$",
                       title=r"BCC Site-Diluted: $\Lambda_M(p)$")
        ax1.axvline(result_bcc.x_c, color="red", linestyle="--",
                    label=f"$p_q={result_bcc.x_c:.4f}$", linewidth=1)
        ax1.axvline(0.2457, color="grey", linestyle=":", linewidth=1, label=r"$p_c^{\rm cl}$")
        ax1.legend()

        _collapse_plot(ax2, M_list, p_vals, data["lambda"], result_bcc,
                       x_label=r"$p$",
                       title=fr"FSS: $\nu={result_bcc.nu:.3f}\pm{result_bcc.nu_err:.3f}$")

        fig.suptitle(
            fr"BCC Quantum Percolation - $p_q={result_bcc.x_c:.4f}\pm{result_bcc.x_c_err:.4f}$, "
            fr"$\nu={result_bcc.nu:.3f}\pm{result_bcc.nu_err:.3f}$  [FIRST MEASUREMENT]"
        )
        fig.tight_layout()
        p = FIGURES_DIR / "fig5_6_bcc_dilution.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        figs_generated.append(p)
        print(f"  Saved {p}")

    # --- Fig 7: Summary comparison ---
    if sc_dilut_path.exists() and bcc_dilut_path.exists():
        sc_meta  = load_lambda(sc_dilut_path)["metadata"]
        bcc_meta = load_lambda(bcc_dilut_path)["metadata"]

        fig, ax = plt.subplots(figsize=(6, 4.5))
        lattices = ["SC", "BCC"]
        p_c_vals = [0.3115, 0.2457]
        p_q_vals = [float(sc_meta.get("p_q", 0)), float(bcc_meta.get("p_q", 0))]
        p_q_errs = [float(sc_meta.get("p_q_err", 0)), float(bcc_meta.get("p_q_err", 0))]

        x = np.arange(2)
        w = 0.35
        ax.bar(x - w/2, p_c_vals, width=w, label=r"$p_c$ (classical)", color="steelblue", alpha=0.8)
        ax.bar(x + w/2, p_q_vals, width=w, yerr=p_q_errs, capsize=5,
               label=r"$p_q$ (quantum)", color="tomato", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(lattices, fontsize=13)
        ax.set_ylabel("Percolation threshold")
        ax.set_title("Classical vs Quantum Percolation Thresholds")
        ax.legend()
        ax.set_ylim(0, 0.65)
        fig.tight_layout()
        p = FIGURES_DIR / "fig7_comparison.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        figs_generated.append(p)
        print(f"  Saved {p}")

    if not figs_generated:
        print("No data files found. Run benchmark/dilution scripts first.")
    else:
        print(f"\n{len(figs_generated)} figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    make_all_figures()
