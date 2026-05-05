"""
HDF5 I/O for TMM results.

All raw Lyapunov exponent samples are stored (not just means), so that
bootstrap re-analysis can be done offline without re-running simulations.

File layout
-----------
/metadata/
    lattice, M_list, L, W or p values, E, t, n_real
/data/M_{M}/x_{idx}/
    gammas       : (N_real, 2N) float64 - raw Lyapunov exponents
    x_value      : scalar - control parameter (W or p)
    lambda_mean  : scalar
    lambda_stderr: scalar
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import jax.numpy as jnp


def save_run(
    path: str | Path,
    M_list:   list[int],
    x_values: list[float],
    all_gammas: dict[int, dict[float, np.ndarray]],
    metadata:  dict,
) -> None:
    """
    Save a full TMM dataset to HDF5.

    Parameters
    ----------
    path        : output file path
    M_list      : list of system widths
    x_values    : list of control parameter values (W or p)
    all_gammas  : nested dict [M][x] -> (N_real, 2N) float64 array
    metadata    : dict of scalar metadata (lattice, L, E, t, etc.)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        for k, v in metadata.items():
            meta.attrs[k] = v
        meta.attrs["x_values"] = np.array(x_values)
        meta.attrs["M_list"]   = np.array(M_list)

        data = f.create_group("data")
        for M in M_list:
            grp_M = data.create_group(f"M_{M}")
            N = M * M
            for ix, x in enumerate(x_values):
                grp_x = grp_M.create_group(f"x_{ix:04d}")
                gammas = np.asarray(all_gammas[M][x])  # (N_real, 2N)
                grp_x.create_dataset("gammas",  data=gammas, compression="gzip")
                grp_x.attrs["x_value"] = x
                # Precompute Λ_M statistics for fast retrieval
                gamma_min = gammas[:, N - 1]
                xi_M      = 1.0 / gamma_min
                lam       = xi_M / M
                grp_x.attrs["lambda_mean"]   = float(np.mean(lam))
                grp_x.attrs["lambda_stderr"] = float(np.std(lam) / np.sqrt(len(lam)))


def load_lambda(path: str | Path) -> dict:
    """
    Load Λ_M means and errors from HDF5 file.

    Returns
    -------
    dict with keys:
        'M_list'   : list[int]
        'x_values' : list[float]
        'lambda'   : dict[M][x] -> (mean, stderr)
        'metadata' : dict of scalar metadata
    """
    result = {"lambda": {}}
    with h5py.File(path, "r") as f:
        meta = f["metadata"]
        result["M_list"]   = list(meta.attrs["M_list"])
        result["x_values"] = list(meta.attrs["x_values"])
        result["metadata"] = dict(meta.attrs)

        for M in result["M_list"]:
            result["lambda"][M] = {}
            grp_M = f[f"data/M_{M}"]
            for ix, x in enumerate(result["x_values"]):
                grp_x = grp_M[f"x_{ix:04d}"]
                result["lambda"][M][x] = (
                    float(grp_x.attrs["lambda_mean"]),
                    float(grp_x.attrs["lambda_stderr"]),
                )
    return result


def load_raw_gammas(path: str | Path, M: int) -> tuple[list[float], np.ndarray]:
    """
    Load raw Lyapunov exponents for a given M for bootstrap reanalysis.

    Returns
    -------
    x_values : list of control parameter values
    gammas   : (len(x_values), N_real, 2N) float64
    """
    with h5py.File(path, "r") as f:
        meta     = f["metadata"]
        x_values = list(meta.attrs["x_values"])
        grp_M    = f[f"data/M_{M}"]
        gammas   = np.stack([
            grp_M[f"x_{ix:04d}"]["gammas"][:]
            for ix in range(len(x_values))
        ])  # (n_x, N_real, 2N)
    return x_values, gammas
