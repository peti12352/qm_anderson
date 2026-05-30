"""
Publication-quality plots using seaborn styling.
Separates crossing and collapse plots into distinct files.
Properly filters out contaminated M=4, 6 data to ensure exact p_q extraction.
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

FIGURES_DIR = Path("figures/publication")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.style.use("seaborn-v0_8-ticks") if "seaborn-v0_8-ticks" in plt.style.available else plt.style.use("default")
plt.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 1.5,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "figure.dpi": 300,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"]

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


def plot_crossing(M_list, x_values, data_dict, result, xlabel, title, filename, pc_classical=None, xc_label="p_q"):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for i, M in enumerate(M_list):
        x_arr = np.array(x_values)
        lam_arr = np.array([data_dict["lambda"][M][x][0] for x in x_values])
        err_arr = np.array([data_dict["lambda"][M][x][1] for x in x_values])
        ax.errorbar(x_arr, lam_arr, yerr=err_arr, fmt="o-", color=COLORS[i % len(COLORS)], label=f"$M={M}$", capsize=3, alpha=0.9)
    
    ax.axvline(result.x_c, color="black", linestyle="--", linewidth=1.5, label=fr"${xc_label}={result.x_c:.4f}$")
    if pc_classical:
        ax.axvline(pc_classical, color="grey", linestyle=":", linewidth=1.5, label=fr"$p_c^{{\rm cl}}={pc_classical}$")
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Lambda_M = \xi_M / M$")
    ax.set_title(title)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def plot_collapse(M_list, x_values, data_dict, result, xlabel, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    x_c, nu = result.x_c, result.nu
    
    for i, M in enumerate(M_list):
        x_arr = np.array(x_values)
        lam_arr = np.array([data_dict["lambda"][M][x][0] for x in x_values])
        err_arr = np.array([data_dict["lambda"][M][x][1] for x in x_values])
        u = M ** (1.0 / nu) * (x_arr - x_c) / x_c
        ax.errorbar(u, lam_arr, yerr=err_arr, fmt="o", color=COLORS[i % len(COLORS)], label=f"$M={M}$", capsize=3, alpha=0.8)
        
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$M^{1/\nu}(p - p_q)/p_q$")
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
        print("Generating SC Benchmark plots...")
        data = load_lambda(sc_bench_path)
        meta = data["metadata"]
        W_c_guess = float(meta.get("W_c", 16.5))
        nu_guess = float(meta.get("nu", 1.571))
        
        M_list, x_vals, M_flat, x_flat, lam_flat, err_flat = filter_data(data, [4, 6, 8, 10], 15.5, 17.5)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_fss(M_flat, x_flat, lam_flat, err_flat, x_c_guess=W_c_guess, nu_guess=nu_guess, poly_order=4)
            
        plot_crossing(M_list, x_vals, data, res, r"Disorder $W/t$", r"SC Anderson Benchmark", "pub_sc_bench_crossing.png", xc_label="W_c")
        plot_collapse(M_list, x_vals, data, res, r"$W$", fr"FSS Collapse ($\nu={res.nu:.3f}$)", "pub_sc_bench_collapse.png")
        
    if sc_dilut_path.exists():
        print("Generating SC Dilution plots...")
        data = load_lambda(sc_dilut_path)
        
        # KEY FIX: Strictly only fit M=8, 10, 12. Ignore M=4, 6 contamination.
        M_list, x_vals, M_flat, x_flat, lam_flat, err_flat = filter_data(data, [8, 10, 12, 16], 0.48, 0.52)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_fss(M_flat, x_flat, lam_flat, err_flat, x_c_guess=0.506, nu_guess=1.62, poly_order=3)
            
        # Plot all M values in the crossing plot so we can VISUALLY see the M=4, 6 contamination
        M_all = [int(m) for m in data["M_list"]]
        plot_crossing(M_all, x_vals, data, res, r"Site occupation $p$", r"SC Dilution: $\Lambda_M(p)$", "pub_sc_dilut_crossing.png", pc_classical=0.3115, xc_label="p_q")
        
        # Only plot the valid M=8, 10, 12 in the collapse
        plot_collapse(M_list, x_vals, data, res, r"$p$", fr"FSS Collapse ($\nu={res.nu:.3f}$)", "pub_sc_dilut_collapse.png")

if __name__ == "__main__":
    main()
