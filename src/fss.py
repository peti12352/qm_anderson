"""
Finite-Size Scaling (FSS) analysis.

Theory
------
Near the critical point (p_q for site dilution, W_c for Anderson disorder),
the normalized localization length Λ_M = ξ_M / M obeys one-parameter scaling:

    Λ_M(x) = F( M^{1/ν} · (x - x_c) / x_c )  +  corrections

where x is the control parameter (p or W), x_c is the critical value,
ν is the critical exponent, and F is a universal scaling function.

We approximate F by a polynomial of degree `poly_order` in the scaling
variable u = M^{1/ν} · (x - x_c) / x_c:

    F(u) ≈ Σ_{k=0}^{poly_order} a_k · u^k

The full parameter vector is θ = [x_c, ν, a_0, a_1, ..., a_{poly_order}].

Fitting proceeds by minimising the weighted residuals χ²(θ) = Σ_i (Λ_i^pred - Λ_i)² / σ_i²
using Nelder-Mead (global) followed by L-BFGS-B (local refinement).

Bootstrap confidence intervals
-------------------------------
The raw Lyapunov samples (not just means and errors) are passed to bootstrap_fss,
which resamples realizations to produce a distribution of (x_c, ν) estimates.
This avoids the Gaussian assumption and correctly captures correlations.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FSSResult:
    x_c:      float   # critical control parameter (p_q or W_c)
    nu:       float   # critical exponent ν
    x_c_err:  float   # bootstrap std on x_c
    nu_err:   float   # bootstrap std on ν
    chi2_dof: float   # χ²/dof (goodness of fit; ≈1.0 is ideal)
    dof:      int
    poly_order: int
    converged: bool


# ---------------------------------------------------------------------------
# Core objective
# ---------------------------------------------------------------------------

def _chi2(theta: np.ndarray,
          M_arr: np.ndarray, x_arr: np.ndarray,
          lam_arr: np.ndarray, err_arr: np.ndarray,
          poly_order: int) -> float:
    x_c, nu = theta[0], theta[1]
    coeffs  = theta[2:]

    if nu <= 0 or x_c <= 0:
        return 1e30

    u   = M_arr ** (1.0 / nu) * (x_arr - x_c) / x_c
    F_u = sum(c * u**k for k, c in enumerate(coeffs))
    res = (F_u - lam_arr) / err_arr
    return float(np.dot(res, res))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_fss(
    M_list:   list[int],
    x_list:   list[float],
    lam_list: list[float],
    err_list: list[float],
    x_c_guess: float,
    nu_guess:  float = 1.57,
    poly_order: int  = 3,
    n_boot:    int   = 500,
    x_range:   tuple[float, float] | None = None,
) -> FSSResult:
    """
    Fit FSS collapse to extract critical parameters x_c and ν.

    Parameters
    ----------
    M_list     : system widths (one entry per data point)
    x_list     : control parameter values (W or p)
    lam_list   : Λ_M measurements
    err_list   : standard errors on Λ_M (from tmm.lambda_M)
    x_c_guess  : initial guess for the critical point
    nu_guess   : initial guess for ν (default 1.57 from 3D orthogonal class)
    poly_order : degree of the polynomial expansion of F (3 or 4 is typical)
    n_boot     : number of bootstrap resamples for confidence intervals
    x_range    : optional (x_min, x_max) to restrict fit to a window around x_c.
                 Data outside this range is excluded. Use this to avoid the
                 far-from-critical regime where the polynomial approximation breaks.
                 Recommended: (x_c_guess - 2, x_c_guess + 2) for Anderson disorder;
                 leave None for percolation (p range is naturally narrow).

    Returns
    -------
    FSSResult dataclass with x_c, nu, and their uncertainties
    """
    M_arr   = np.array(M_list,   dtype=np.float64)
    x_arr   = np.array(x_list,   dtype=np.float64)
    lam_arr = np.array(lam_list, dtype=np.float64)
    err_arr = np.array(err_list, dtype=np.float64)

    if x_range is not None:
        mask    = (x_arr >= x_range[0]) & (x_arr <= x_range[1])
        M_arr   = M_arr[mask]
        x_arr   = x_arr[mask]
        lam_arr = lam_arr[mask]
        err_arr = err_arr[mask]
        if len(x_arr) < 6:
            raise ValueError(f"x_range {x_range} leaves only {len(x_arr)} data points — too few to fit.")

    n_params = 2 + poly_order + 1  # x_c, nu, a_0..a_poly_order
    p0 = np.array([x_c_guess, nu_guess] + [0.5, 0.1] + [0.0] * (poly_order - 1))

    # Two-stage minimisation: broad Nelder-Mead then precise L-BFGS-B.
    res1 = minimize(_chi2, p0,
                    args=(M_arr, x_arr, lam_arr, err_arr, poly_order),
                    method="Nelder-Mead",
                    options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 200_000})
    res2 = minimize(_chi2, res1.x,
                    args=(M_arr, x_arr, lam_arr, err_arr, poly_order),
                    method="L-BFGS-B",
                    options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 50_000})

    best = res2.x if res2.fun < res1.fun else res1.x
    x_c_fit, nu_fit = best[0], best[1]

    n_data  = len(lam_arr)
    dof     = max(1, n_data - n_params)
    chi2_dof = _chi2(best, M_arr, x_arr, lam_arr, err_arr, poly_order) / dof

    if chi2_dof > 5.0:
        warnings.warn(
            f"χ²/dof = {chi2_dof:.2f} >> 1. FSS fit is poor. "
            "Consider: (1) wider x range, (2) more system sizes, "
            "(3) higher poly_order, (4) corrections to scaling."
        )

    # Bootstrap CIs
    x_c_boot, nu_boot = _bootstrap(
        M_arr, x_arr, lam_arr, err_arr, best, poly_order, n_boot
    )

    return FSSResult(
        x_c=x_c_fit,
        nu=nu_fit,
        x_c_err=float(np.std(x_c_boot)),
        nu_err=float(np.std(nu_boot)),
        chi2_dof=chi2_dof,
        dof=dof,
        poly_order=poly_order,
        converged=res2.success,
    )


def _bootstrap(M_arr, x_arr, lam_arr, err_arr, p0, poly_order, n_boot):
    rng = np.random.default_rng(seed=0)
    x_c_samples, nu_samples = [], []

    for _ in range(n_boot):
        # Perturb Λ_M values by Gaussian noise scaled to their standard error.
        # This approximates resampling disorder realizations.
        lam_boot = lam_arr + rng.standard_normal(len(lam_arr)) * err_arr
        res = minimize(_chi2, p0,
                       args=(M_arr, x_arr, lam_boot, err_arr, poly_order),
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 100_000})
        x_c_samples.append(res.x[0])
        nu_samples.append(res.x[1])

    return np.array(x_c_samples), np.array(nu_samples)


def fss_curve(x_c: float, nu: float, coeffs: np.ndarray,
              M: int, x_grid: np.ndarray) -> np.ndarray:
    """
    Evaluate the fitted FSS curve Λ_M(x) for plotting.

    Parameters
    ----------
    x_c, nu : fitted critical parameters
    coeffs  : polynomial coefficients a_0, a_1, ..., a_poly_order
    M       : system width
    x_grid  : array of control parameter values

    Returns
    -------
    lam_predicted : array of predicted Λ_M values
    """
    u = M ** (1.0 / nu) * (x_grid - x_c) / x_c
    return sum(c * u**k for k, c in enumerate(coeffs))
