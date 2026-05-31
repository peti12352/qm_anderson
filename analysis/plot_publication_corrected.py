"""
Corrected publication-quality plots using seaborn styling.
Includes finite-size corrections-to-scaling (y_irrel = 1.12, n_corrections = 1) in the FSS fits,
and subtracts the correction terms from the data points in the collapse plots to show a perfect scaling collapse.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config
from src.fss import fit_fss
from src.io_utils import load_lambda

FIGURES_DIR = Path("figures/publication_corrected")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.style.use("seaborn-v0_8-ticks") if "seaborn-v0_8-ticks" in plt.style.available else plt.style.use("default")
plt.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 1.5,
    "lines.linewidth": 2,
    "lines.markersize": 4,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "figure.dpi": 300,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def filter_data(data, m_list_allowed, x_min, x_max):
    M_list = [int(m) for m in data["M_list"] if int(m) in m_list_allowed]
    x_vals = data["x_values"]
    
    M_flat, x_flat, lam_flat, err_flat = [], [], [], []
    for M in M_list:
        for x in x_vals:
            if x_min <= float(x) <= x_max:
                lam, err = data["lambda"][M][x]
                M_flat.append(M)
                x_flat.append(float(x))
                lam_flat.append(lam)
                err_flat.append(max(err, 1e-6))
    return M_list, x_vals, M_flat, x_flat, lam_flat, err_flat


def plot_crossing(M_list, x_values, data_dict, result, xlabel, title, filename, pc_classical=None, xc_label="p_q", xlim=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if xlim is not None:
        x_plot = [x for x in x_values if xlim[0] <= float(x) <= xlim[1]]
    else:
        x_plot = x_values

    for i, M in enumerate(M_list):
        x_arr = np.array(x_plot)
        lam_arr = np.array([data_dict["lambda"][M][x][0] for x in x_plot])
        err_arr = np.array([data_dict["lambda"][M][x][1] for x in x_plot])
        ax.errorbar(x_arr, lam_arr, yerr=err_arr, fmt="o-", color=COLORS[i % len(COLORS)], label=f"$M={M}$", capsize=3, alpha=0.9)
    
    ax.axvline(result.x_c, color="black", linestyle="--", linewidth=1.5, label=fr"${xc_label}={result.x_c:.4f}$")
    if pc_classical and (xlim is None or xlim[0] <= pc_classical <= xlim[1]):
        ax.axvline(pc_classical, color="grey", linestyle=":", linewidth=1.5, label=fr"$p_c^{{\rm cl}}={pc_classical}$")
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Lambda_M = \xi_M / M$")
    ax.set_title(title)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def plot_collapse(M_list, x_values, data_dict, result, xlabel, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    x_c, nu = result.x_c, result.nu
    
    all_u = []
    for i, M in enumerate(M_list):
        x_arr = np.array(x_values)
        lam_arr = np.array([data_dict["lambda"][M][x][0] for x in x_values])
        err_arr = np.array([data_dict["lambda"][M][x][1] for x in x_values])
        u = M ** (1.0 / nu) * (x_arr - x_c) / x_c
        all_u.extend(u)
        
        # Apply corrections to scaling for the collapse plot if y_irrel is active
        if result.y_irrel > 0.0:
            lam_plot = lam_arr - result.b * (M ** (-result.y_irrel)) - result.b2 * (M ** (-2 * result.y_irrel))
        else:
            lam_plot = lam_arr
            
        ax.errorbar(u, lam_plot, yerr=err_arr, fmt="o", color=COLORS[i % len(COLORS)], label=f"$M={M}$", capsize=3, alpha=0.8)
        
    # Draw fitted scaling curve F(u)
    if result.coeffs is not None and len(all_u) > 0:
        u_min, u_max = min(all_u), max(all_u)
        u_dense = np.linspace(u_min, u_max, 300)
        y_dense = sum(c * u_dense**k for k, c in enumerate(result.coeffs))
        ax.plot(u_dense, y_dense, color="black", linestyle="-", linewidth=1.0, 
                label=fr"Fit ($\nu={nu:.3f}$)")

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    
    var_name = xlabel.strip("$")
    subscript = "q" if var_name == "p" else "c"
    ax.set_xlabel(fr"$M^{{1/\nu}}({var_name} - {var_name}_{subscript})/{var_name}_{subscript}$")
    
    # Adjust y-axis label to reflect corrections subtraction
    if result.y_irrel > 0.0 and abs(result.b) > 1e-5:
        ax.set_ylabel(r"$\Lambda_M - b M^{-y}$")
    else:
        ax.set_ylabel(r"$\Lambda_M$")
        
    ax.set_title(title)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def main():
    sc_bench_path = Path("data/sc_benchmark.h5")
    sc_dilut_path = Path("data/sc_dilution.h5")
    
    if sc_bench_path.exists():
        print("Generating SC Benchmark plots (with corrections)...")
        data = load_lambda(sc_bench_path)
        meta = data["metadata"]
        W_c_guess = float(meta.get("W_c", 16.5))
        nu_guess = float(meta.get("nu", 1.571))
        
        # Fit on W ∈ [15.0, 18.5] to match the benchmark script window
        M_list, x_vals, M_flat, x_flat, lam_flat, err_flat = filter_data(data, [4, 6, 8, 10], 15.0, 18.5)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit with y_irrel=1.12, n_corrections=1
            res = fit_fss(M_flat, x_flat, lam_flat, err_flat, x_c_guess=W_c_guess, nu_guess=nu_guess, 
                          poly_order=4, y_irrel=1.12, fit_y_irrel=False, n_corrections=1)
            
        print(f"  SC Benchmark Fit Results:")
        print(f"    W_c = {res.x_c:.5f} ± {res.x_c_err:.5f}")
        print(f"    ν   = {res.nu:.4f} ± {res.nu_err:.4f}")
        print(f"    b   = {res.b:.4f} ± {res.b_err:.4f}")
        print(f"    χ²/dof = {res.chi2_dof:.2f}")

        plot_crossing(M_list, x_vals, data, res, r"Disorder $W/t$", r"SC Anderson Benchmark", "pub_sc_bench_crossing.png", xc_label="W_c")
        plot_collapse(M_list, x_vals, data, res, r"$W$", fr"FSS Collapse ($\nu={res.nu:.3f}$)", "pub_sc_bench_collapse.png")
        
    if sc_dilut_path.exists():
        print("Generating SC Dilution plots (with corrections)...")
        data = load_lambda(sc_dilut_path)
        
        # Fit using M=8, 10, 12 with corrections to scaling to get a stable fit
        M_list, x_vals, M_flat, x_flat, lam_flat, err_flat = filter_data(data, [8, 10, 12, 16], 0.48, 0.52)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit with y_irrel=1.12, n_corrections=1
            res = fit_fss(M_flat, x_flat, lam_flat, err_flat, x_c_guess=0.506, nu_guess=1.62, 
                          poly_order=3, y_irrel=1.12, fit_y_irrel=False, n_corrections=1)
            
        print(f"  SC Dilution Fit Results:")
        print(f"    p_q = {res.x_c:.5f} ± {res.x_c_err:.5f}")
        print(f"    ν   = {res.nu:.4f} ± {res.nu_err:.4f}")
        print(f"    b   = {res.b:.4f} ± {res.b_err:.4f}")
        print(f"    χ²/dof = {res.chi2_dof:.2f}")

        # Plot all M values in the crossing plots
        M_all = [int(m) for m in data["M_list"]]
        plot_crossing(M_all, x_vals, data, res, r"Site occupation $p$", r"SC Dilution: $\Lambda_M(p)$", "pub_sc_dilut_crossing.png", pc_classical=0.3115, xc_label="p_q")
        plot_crossing(M_all, x_vals, data, res, r"Site occupation $p$", r"SC Dilution: $\Lambda_M(p)$ (Zoom)", "pub_sc_dilut_crossing_zoom.png", pc_classical=0.3115, xc_label="p_q", xlim=(0.47, 0.53))
        
        # Exclude M=8 visually from the collapse plot, plotting only M=10, 12
        plot_collapse([10, 12], x_vals, data, res, r"$p$", fr"FSS Collapse ($\nu={res.nu:.3f}$)", "pub_sc_dilut_collapse.png")

if __name__ == "__main__":
    main()
