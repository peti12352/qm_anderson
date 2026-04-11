"""
Transfer Matrix Method (TMM) with QR stabilization.

Core algorithm
--------------
The Lyapunov exponents of the quasi-1D bar are extracted via the
QR-stabilized product of random transfer matrices (Pichard & Sarma 1981,
MacKinnon & Kramer 1983).

    γ_i = lim_{L→∞} (1/L) Σ_n log|R^{(n)}_{ii}|

where R^{(n)} is the upper-triangular factor from the QR decomposition
at step n.

The quasi-1D localization length is:
    ξ_M = 1 / γ_{N-1}   (N = M², the (N-1)-th entry in 0-indexed descending array)

Modified Gram-Schmidt (MGS) orthogonalization is applied at EVERY step to
prevent overflow. MGS always gives positive diagonal R entries, so no sign
fix is needed. MGS is implemented via lax.scan over columns (pure JAX ops),
avoiding the cuSOLVER custom-call overhead that made jnp.linalg.qr inside
a fori_loop take ~40 ms/step × 100k steps ≈ 4800 s.

GPU parallelism strategy
------------------------
The sequential loop cannot be parallelized within a single bar. Parallelism
comes entirely from running B independent disorder realizations concurrently
via jax.vmap.

Loop implementation: lax.fori_loop(0, L, body, init)
  - L is a regular (non-static) argument — one compiled kernel for ALL L values
  - No (L,) index array allocated (unlike lax.scan over jnp.arange(L))
  - XLA compiles a plain while-loop; compilation takes seconds not minutes
  - lax.scan over arange(L) caused 15-30 min XLA compile hangs because it
    materialized a 100k-element array and triggered full autotuning

Memory note
-----------
Disorder is generated on-the-fly at each fori_loop step via jax.random.fold_in,
NOT pre-allocated as (L, N) arrays. Pre-allocation would require
B × L × N × 8 bytes ≈ 100 GB for B=2000, L=100k, N=64 — far exceeding
GPU VRAM. With fold_in, peak memory per realization is O((2N)²) ≈ 256 KB.

Public API
----------
    make_tmm_fns(M, lattice, E, t)  -> (tmm_single, tmm_batched)

    tmm_batched(keys, L, W, p)      -> (B, 2N) float64 Lyapunov exponents
    lambda_M(gammas, M)             -> (mean, stderr) of Λ_M = ξ_M / M
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from src.lattice import make_transfer_matrix_fn


def make_tmm_fns(
    M: int,
    lattice: str = "sc",
    E: float = 0.0,
    t: float = 1.0,
):
    """
    Factory: create JIT-compiled TMM functions for given (M, lattice, E, t).

    Returns
    -------
    tmm_single  : (key, L, W, p) -> (2N,) Lyapunov exponents (descending)
    tmm_batched : (keys, L, W, p) -> (B, 2N) Lyapunov exponents
    """
    N = M * M
    dim = 2 * N
    build_T = make_transfer_matrix_fn(M, lattice, E, t)

    # Probe the transfer matrix dtype once (at factory time, not inside JIT).
    # SC returns float64; BCC returns complex128. Q must match T's dtype so that
    # T @ Q does not upcast the entire computation back to complex128.
    _probe_dis  = jnp.zeros(N, dtype=jnp.float64)
    _probe_mask = jnp.ones(N, dtype=bool)
    _mat_dtype  = build_T(_probe_dis, _probe_mask).dtype   # float64 or complex128

    def _qr_mgs(A: jnp.ndarray):
        """Modified Gram-Schmidt QR via lax.scan over columns.

        Replaces jnp.linalg.qr to avoid cuSOLVER custom-call overhead inside
        lax.fori_loop. cuSOLVER allocates/frees a workspace buffer on every
        loop iteration, adding ~40-50 ms latency per step × 100k steps = hours.
        Pure-JAX MGS compiles to fused tensor kernels with no custom calls.

        Args:
            A: (dim, dim) float64 or complex128
        Returns:
            Q: (dim, dim) orthonormal columns
            diag_r: (dim,) positive float64 — diagonal of R
        """
        def step(V, j):
            v    = V[:, j]
            norm = jnp.sqrt(jnp.dot(jnp.conj(v), v).real)
            q    = v / jnp.where(norm > 0, norm, jnp.ones_like(norm))
            # Project q out of all remaining columns (mask past columns to zero)
            dots = jnp.conj(q) @ V                  # (dim,)
            mask = jnp.arange(dim) > j              # False for cols ≤ j
            V    = V - q[:, None] * (dots * mask)[None, :]
            return V, (q, norm)

        _, (Q_cols, diag_r) = jax.lax.scan(step, A, jnp.arange(dim))
        # Q_cols[j] = j-th Q column -> transpose to standard (dim, dim) layout
        return Q_cols.T, diag_r.astype(jnp.float64)

    def _tmm_core(key: jax.Array, L: int, W: float, p: float) -> jnp.ndarray:
        """
        Single TMM run. Returns all 2N Lyapunov exponents in descending order.

        Parameters
        ----------
        key : JAX PRNG key
        L   : bar length (number of slices); must be > 10/γ_min for convergence
        W   : disorder strength (on-site energies drawn from Uniform[-W/2, W/2])
        p   : site occupation probability (p=1.0 → no dilution)
        """
        key_dis, key_dil = jax.random.split(key)

        dim = 2 * N
        Q0         = jnp.eye(dim, dtype=_mat_dtype)
        log_norms0 = jnp.zeros(dim, dtype=jnp.float64)

        def body(n, carry):
            Q, log_norms = carry

            # Per-slice keys derived via fold_in — no (L,N) pre-allocation.
            k_dis      = jax.random.fold_in(key_dis, n)
            k_dil      = jax.random.fold_in(key_dil, n)
            dis_slice  = jax.random.uniform(k_dis, (N,), minval=-W / 2.0, maxval=W / 2.0)
            mask_slice = jax.random.uniform(k_dil, (N,)) < p

            T  = build_T(dis_slice, mask_slice)   # (2N,2N) float64 or complex128
            QT = T @ Q

            # MGS QR: diag_r is always positive, no sign fix needed.
            Q, diag_r = _qr_mgs(QT)

            log_norms = log_norms + jnp.log(diag_r)
            return (Q, log_norms)

        # fori_loop compiles as a plain while-loop; L is dynamic (not static),
        # so one kernel covers all bar lengths — no recompilation when L changes.
        _, log_norms = jax.lax.fori_loop(0, L, body, (Q0, log_norms0))

        gamma = log_norms / L
        return jnp.sort(gamma)[::-1]

    # L is no longer static: fori_loop accepts a dynamic upper bound.
    # One compiled kernel handles any L value — no recompilation between
    # test runs (L=5k) and production runs (L=100k).
    tmm_single  = jax.jit(_tmm_core)
    tmm_batched = jax.jit(jax.vmap(_tmm_core, in_axes=(0, None, None, None)))

    return tmm_single, tmm_batched


def lambda_M(gammas: jnp.ndarray, M: int) -> tuple[float, float]:
    """
    Compute Λ_M = ξ_M / M from a batch of Lyapunov exponent arrays.

    Λ_M is the fundamental observable for FSS:
        Λ_M → ∞  in metallic phase   (ξ_M grows faster than M)
        Λ_M → 0  in insulating phase  (ξ_M saturates)
        Λ_M = const at critical point  (all M cross at p_q or W_c)

    Parameters
    ----------
    gammas : (B, 2N) float64 — Lyapunov exponents from tmm_batched, descending
    M      : cross-section width

    Returns
    -------
    mean   : float — mean of Λ_M over B realizations
    stderr : float — standard error = std / sqrt(B)
    """
    N = M * M
    # The smallest positive Lyapunov exponent is at index N-1 (0-indexed).
    # By symplecticity of the transfer matrix, exponents come in ±γ pairs:
    # γ[0] ≥ ... ≥ γ[N-1] > 0 > γ[N] ≥ ... ≥ γ[2N-1]
    gamma_min = gammas[:, N - 1]  # shape (B,)
    xi_M      = 1.0 / gamma_min   # quasi-1D localization length
    lam       = xi_M / M

    mean   = float(jnp.mean(lam))
    stderr = float(jnp.std(lam) / jnp.sqrt(gammas.shape[0]))
    return mean, stderr
