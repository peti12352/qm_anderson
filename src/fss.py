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

Corrections to scaling
-----------------------
For small M the leading irrelevant operator contributes an additive term:

    Λ_M(x) = Σ a_k u^k  +  b · M^{-y}

where y is the irrelevant exponent (y ≈ 1.0 for the 3D orthogonal class;
Slevin & Ohtsuki 1999 measure y = 1.12 ± 0.08). Omitting this term with
M ≤ 10 inflates χ²/dof and biases W_c and ν toward lower values. Pass
y_irrel > 0 to fit_fss to enable the correction (recommended for M < 16).

Parameter vector:
  without corrections: θ = [x_c, ν, a_0, ..., a_p]            (p+3 params)
  with corrections:    θ = [x_c, ν, a_0, ..., a_p, b]         (p+4 params)

Fitting proceeds by minimising χ²(θ) = Σ_i (Λ_i^pred - Λ_i)² / σ_i²
using Nelder-Mead (global) followed by L-BFGS-B (local refinement).

Bootstrap confidence intervals
-------------------------------
Λ_M values are perturbed by Gaussian noise scaled to their standard errors,
producing a distribution of (x_c, ν, b) estimates.
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
    b:        float = 0.0   # correction amplitude (0 if y_irrel=0)
    b_err:    float = 0.0   # bootstrap std on b
    y_irrel:  float = 0.0   # irrelevant exponent used (0 = no correction)


# ---------------------------------------------------------------------------
# Core objective
# ---------------------------------------------------------------------------

def _chi2(theta: np.ndarray,
          M_arr: np.ndarray, x_arr: np.ndarray,
          lam_arr: np.ndarray, err_arr: np.ndarray,
          poly_order: int,
          y_irrel: float = 0.0) -> float:
    x_c, nu = theta[0], theta[1]
    if y_irrel > 0.0:
        coeffs = theta[2:-1]
        b      = theta[-1]
    else:
        coeffs = theta[2:]
        b      = 0.0

    if nu <= 0 or x_c <= 0:
        return 1e30

    u   = M_arr ** (1.0 / nu) * (x_arr - x_c) / x_c
    F_u = sum(c * u**k for k, c in enumerate(coeffs))
    if y_irrel > 0.0:
        F_u = F_u + b * M_arr ** (-y_irrel)
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
    y_irrel:   float = 0.0,
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
    y_irrel    : irrelevant exponent for corrections-to-scaling (Slevin & Ohtsuki 1999).
                 When > 0, adds b·M^{-y_irrel} to the FSS model as a free parameter b.
                 Use y_irrel=1.0 for the 3D orthogonal class when M < 16.
                 Default 0.0 disables corrections (backward compatible).

    Returns
    -------
    FSSResult dataclass with x_c, nu, b, and their uncertainties
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

    # +1 for b when corrections enabled
    n_params = 2 + poly_order + 1 + (1 if y_irrel > 0.0 else 0)
    p0_poly  = [0.5, 0.1] + [0.0] * (poly_order - 1)
    p0_corr  = [0.0] if y_irrel > 0.0 else []   # initial b = 0
    p0       = np.array([x_c_guess, nu_guess] + p0_poly + p0_corr)

    _args = (M_arr, x_arr, lam_arr, err_arr, poly_order, y_irrel)

    # Two-stage minimisation: broad Nelder-Mead then precise L-BFGS-B.
    res1 = minimize(_chi2, p0,
                    args=_args,
                    method="Nelder-Mead",
                    options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 200_000})
    res2 = minimize(_chi2, res1.x,
                    args=_args,
                    method="L-BFGS-B",
                    options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 50_000})

    best     = res2.x if res2.fun < res1.fun else res1.x
    x_c_fit  = best[0]
    nu_fit   = best[1]
    b_fit    = float(best[-1]) if y_irrel > 0.0 else 0.0

    n_data   = len(lam_arr)
    dof      = max(1, n_data - n_params)
    chi2_dof = _chi2(best, M_arr, x_arr, lam_arr, err_arr, poly_order, y_irrel) / dof

    if chi2_dof > 5.0:
        warnings.warn(
            f"χ²/dof = {chi2_dof:.2f} >> 1. FSS fit is poor. "
            "Consider: (1) wider x range, (2) more system sizes, "
            "(3) higher poly_order, (4) corrections to scaling (y_irrel > 0)."
        )

    # Bootstrap CIs
    x_c_boot, nu_boot, b_boot = _bootstrap(
        M_arr, x_arr, lam_arr, err_arr, best, poly_order, n_boot, y_irrel
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
        b=b_fit,
        b_err=float(np.std(b_boot)),
        y_irrel=y_irrel,
    )


def _bootstrap(M_arr, x_arr, lam_arr, err_arr, p0, poly_order, n_boot,
               y_irrel=0.0):
    rng = np.random.default_rng(seed=0)
    x_c_samples, nu_samples, b_samples = [], [], []

    for _ in range(n_boot):
        lam_boot = lam_arr + rng.standard_normal(len(lam_arr)) * err_arr
        res = minimize(_chi2, p0,
                       args=(M_arr, x_arr, lam_boot, err_arr, poly_order, y_irrel),
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 100_000})
        x_c_samples.append(res.x[0])
        nu_samples.append(res.x[1])
        b_samples.append(float(res.x[-1]) if y_irrel > 0.0 else 0.0)

    return np.array(x_c_samples), np.array(nu_samples), np.array(b_samples)


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
