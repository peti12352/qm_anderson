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
from src.io_utils import load_lambda
from types import SimpleNamespace

FIGURES_DIR = Path("figures/publication")
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
        ax.errorbar(u, lam_arr, yerr=err_arr, fmt="o", color=COLORS[i % len(COLORS)], label=f"$M={M}$", capsize=3, alpha=0.8)
        
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
        W_c = float(meta.get("W_c", 16.53))
        nu = float(meta.get("nu", 1.571))
        
        M_list = [int(m) for m in data["M_list"] if int(m) in [4, 6, 8, 10]]
        x_vals = data["x_values"]
        res = SimpleNamespace(x_c=W_c, nu=nu, coeffs=None)
        
        plot_crossing(M_list, x_vals, data, res, r"Disorder $W/t$", r"SC Anderson Benchmark", "pub_sc_bench_crossing.png", xc_label="W_c")
        plot_collapse(M_list, x_vals, data, res, r"$W$", fr"FSS Collapse ($\nu={res.nu:.3f}$)", "pub_sc_bench_collapse.png")
        
    if sc_dilut_path.exists():
        print("Generating SC Dilution plots...")
        data = load_lambda(sc_dilut_path)
        meta = data["metadata"]
        p_q = float(meta.get("p_q", 0.50628))
        nu = float(meta.get("nu", 1.62))
        
        M_list = [int(m) for m in data["M_list"] if int(m) in [10, 12, 16]]
        x_vals = data["x_values"]
        res = SimpleNamespace(x_c=p_q, nu=nu, coeffs=None)
        
        M_all = [int(m) for m in data["M_list"]]
        plot_crossing(M_all, x_vals, data, res, r"Site occupation $p$", r"SC Dilution: $\Lambda_M(p)$", "pub_sc_dilut_crossing.png", pc_classical=0.3115, xc_label="p_q")
        plot_crossing(M_all, x_vals, data, res, r"Site occupation $p$", r"SC Dilution: $\Lambda_M(p)$ (Zoom)", "pub_sc_dilut_crossing_zoom.png", pc_classical=0.3115, xc_label="p_q", xlim=(0.47, 0.53))
        plot_collapse(M_list, x_vals, data, res, r"$p$", fr"FSS Collapse ($\nu={res.nu:.3f}$)", "pub_sc_dilut_collapse.png")

if __name__ == "__main__":
    main()
