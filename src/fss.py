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

Corrections to scaling (Slevin & Ohtsuki 1999)
-----------------------------------------------
For small M the leading irrelevant operator contributes additive terms:

    Λ_M(x) = Σ a_k u^k  +  b·M^{-y}  +  b₂·M^{-2y}  +  ...

where y is the irrelevant exponent.  For the 3D orthogonal class,
Slevin & Ohtsuki (1999) measure y = 1.12 ± 0.08.  Omitting these terms
with M ≤ 10 inflates χ²/dof and biases W_c and ν toward lower values.

Set y_irrel > 0 to enable corrections.  y can be fixed (faster, fewer
parameters) or fitted freely (more accurate, requires good M coverage).

Parameter vector layout (n_poly = poly_order + 1):
  no corrections:                θ = [x_c, ν, a_0..a_p]
  fixed y, 1 correction:         θ = [x_c, ν, a_0..a_p, b]
  fixed y, 2 corrections:        θ = [x_c, ν, a_0..a_p, b, b₂]
  free  y, 1 correction:         θ = [x_c, ν, a_0..a_p, b,     y]
  free  y, 2 corrections:        θ = [x_c, ν, a_0..a_p, b, b₂, y]

(y is always the last element when free, b values precede it.)

Fitting: Nelder-Mead (global) then L-BFGS-B (local refinement).
Bootstrap: Λ_M perturbed by N(0, σ_M) to propagate statistical errors.
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
    x_c:        float   # critical control parameter (p_q or W_c)
    nu:         float   # critical exponent ν
    x_c_err:    float   # bootstrap std on x_c
    nu_err:     float   # bootstrap std on ν
    chi2_dof:   float   # χ²/dof (goodness of fit; ≈1.0 is ideal)
    dof:        int
    poly_order: int
    converged:  bool
    b:          float = 0.0   # first  correction amplitude  b  (b·M^{-y})
    b_err:      float = 0.0
    b2:         float = 0.0   # second correction amplitude  b₂ (b₂·M^{-2y})
    b2_err:     float = 0.0
    y_irrel:    float = 0.0   # irrelevant exponent (fitted value or fixed input)


# ---------------------------------------------------------------------------
# Core objective
# ---------------------------------------------------------------------------

def _chi2(theta: np.ndarray,
          M_arr: np.ndarray, x_arr: np.ndarray,
          lam_arr: np.ndarray, err_arr: np.ndarray,
          poly_order: int,
          y_irrel: float = 0.0,
          fit_y_irrel: bool = False,
          n_corrections: int = 1) -> float:
    x_c, nu  = theta[0], theta[1]
    n_poly   = poly_order + 1
    coeffs   = theta[2:2 + n_poly]

    if nu <= 0 or x_c <= 0:
        return 1e30

    u   = M_arr ** (1.0 / nu) * (x_arr - x_c) / x_c
    F_u = sum(c * u**k for k, c in enumerate(coeffs))

    if y_irrel > 0.0:
        if fit_y_irrel:
            y      = float(theta[-1])
            b_vals = theta[2 + n_poly:-1]   # [b] or [b, b₂]
        else:
            y      = y_irrel
            b_vals = theta[2 + n_poly:]     # [b] or [b, b₂]

        if y <= 0:
            return 1e30

        F_u = F_u + b_vals[0] * M_arr ** (-y)
        if n_corrections >= 2 and len(b_vals) >= 2:
            F_u = F_u + b_vals[1] * M_arr ** (-2 * y)

    res = (F_u - lam_arr) / err_arr
    return float(np.dot(res, res))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_fss(
    M_list:       list[int],
    x_list:       list[float],
    lam_list:     list[float],
    err_list:     list[float],
    x_c_guess:    float,
    nu_guess:     float = 1.57,
    poly_order:   int   = 3,
    n_boot:       int   = 500,
    x_range:      tuple[float, float] | None = None,
    y_irrel:      float = 0.0,
    fit_y_irrel:  bool  = False,
    n_corrections: int  = 1,
) -> FSSResult:
    """
    Fit FSS collapse to extract critical parameters x_c and ν.

    Parameters
    ----------
    M_list        : system widths (one entry per data point)
    x_list        : control parameter values (W or p)
    lam_list      : Λ_M measurements
    err_list      : standard errors on Λ_M
    x_c_guess     : initial guess for the critical point
    nu_guess      : initial guess for ν
    poly_order    : degree of the polynomial expansion of F
    n_boot        : bootstrap resamples for confidence intervals
    x_range       : (x_min, x_max) window; data outside excluded
    y_irrel       : irrelevant exponent. 0 disables corrections. When > 0
                    and fit_y_irrel=False, y is fixed at this value.
                    When fit_y_irrel=True, used only as initial guess.
    fit_y_irrel   : if True, y is a free parameter in the fit
    n_corrections : 1 → b·M^{-y} only; 2 → b·M^{-y} + b₂·M^{-2y}

    Returns
    -------
    FSSResult with x_c, nu, b, b2, y_irrel (fitted or fixed) and uncertainties
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
            raise ValueError(
                f"x_range {x_range} leaves only {len(x_arr)} points — too few to fit."
            )

    n_poly   = poly_order + 1
    n_corr   = n_corrections if y_irrel > 0.0 else 0
    n_y_free = 1 if (y_irrel > 0.0 and fit_y_irrel) else 0
    n_params = 2 + n_poly + n_corr + n_y_free

    p0 = np.array(
        [x_c_guess, nu_guess, 0.5, 0.1] + [0.0] * (poly_order - 1)  # FSS
        + [0.0] * n_corr                                               # b, b₂
        + ([y_irrel] if n_y_free else [])                              # y
    )

    _args = (M_arr, x_arr, lam_arr, err_arr, poly_order, y_irrel, fit_y_irrel, n_corrections)

    res1 = minimize(_chi2, p0, args=_args, method="Nelder-Mead",
                    options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 200_000})
    res2 = minimize(_chi2, res1.x, args=_args, method="L-BFGS-B",
                    options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 50_000})

    best    = res2.x if res2.fun < res1.fun else res1.x
    x_c_fit = float(best[0])
    nu_fit  = float(best[1])

    if y_irrel > 0.0:
        if fit_y_irrel:
            y_fit  = float(best[-1])
            b_vals = best[2 + n_poly:-1]
        else:
            y_fit  = y_irrel
            b_vals = best[2 + n_poly:]
        b_fit  = float(b_vals[0]) if n_corr >= 1 else 0.0
        b2_fit = float(b_vals[1]) if n_corr >= 2 and len(b_vals) >= 2 else 0.0
    else:
        y_fit  = 0.0
        b_fit  = 0.0
        b2_fit = 0.0

    n_data   = len(lam_arr)
    dof      = max(1, n_data - n_params)
    chi2_dof = _chi2(best, M_arr, x_arr, lam_arr, err_arr,
                     poly_order, y_irrel, fit_y_irrel, n_corrections) / dof

    if chi2_dof > 5.0:
        warnings.warn(
            f"χ²/dof = {chi2_dof:.2f} >> 1. FSS fit is poor. "
            "Consider: (1) wider x range, (2) more system sizes, "
            "(3) higher poly_order, (4) corrections-to-scaling (y_irrel > 0)."
        )

    x_c_boot, nu_boot, b_boot, b2_boot = _bootstrap(
        M_arr, x_arr, lam_arr, err_arr, best, poly_order, n_boot,
        y_irrel, fit_y_irrel, n_corrections
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
        b2=b2_fit,
        b2_err=float(np.std(b2_boot)),
        y_irrel=y_fit,
    )


def _bootstrap(M_arr, x_arr, lam_arr, err_arr, p0, poly_order, n_boot,
               y_irrel=0.0, fit_y_irrel=False, n_corrections=1):
    rng = np.random.default_rng(seed=0)
    x_c_samples, nu_samples = [], []
    b_samples,  b2_samples  = [], []

    n_poly = poly_order + 1
    n_corr = n_corrections if y_irrel > 0.0 else 0

    for _ in range(n_boot):
        lam_boot = lam_arr + rng.standard_normal(len(lam_arr)) * err_arr
        res = minimize(_chi2, p0,
                       args=(M_arr, x_arr, lam_boot, err_arr, poly_order,
                             y_irrel, fit_y_irrel, n_corrections),
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 100_000})
        x_c_samples.append(float(res.x[0]))
        nu_samples.append(float(res.x[1]))
        if y_irrel > 0.0:
            b_vals = res.x[2 + n_poly:-1] if fit_y_irrel else res.x[2 + n_poly:]
            b_samples.append(float(b_vals[0]) if n_corr >= 1 else 0.0)
            b2_samples.append(float(b_vals[1]) if n_corr >= 2 and len(b_vals) >= 2 else 0.0)
        else:
            b_samples.append(0.0)
            b2_samples.append(0.0)

    return (np.array(x_c_samples), np.array(nu_samples),
            np.array(b_samples),   np.array(b2_samples))


def fss_curve(x_c: float, nu: float, coeffs: np.ndarray,
              M: int, x_grid: np.ndarray) -> np.ndarray:
    """Evaluate the fitted FSS curve Λ_M(x) for plotting."""
    u = M ** (1.0 / nu) * (x_grid - x_c) / x_c
    return sum(c * u**k for k, c in enumerate(coeffs))
