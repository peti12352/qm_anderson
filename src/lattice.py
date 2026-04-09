"""
Transfer matrix construction for SC and BCC tight-binding lattices.

Each lattice provides a factory `make_transfer_matrix_fn(M, lattice, E, t)`
that returns a JIT-compiled function:

    transfer_matrix(disorder: (N,), mask: (N,)) -> (2N, 2N) complex128

where N = M*M is the number of sites in the M×M cross-section.

Transfer matrix form (both lattices):

    T_n = [ V^{-1}(E·I - H_n)   -I ]
          [        I              0  ]

For SC:  V = -t·I  →  V^{-1}(E·I - H_n) = (H_n - E·I)/t
For BCC: V = sum of 4 shift permutations (pre-inverted once at factory time)

Site dilution is encoded solely in H_n: absent sites (mask=False) receive
on-site energy ABSENT_ENERGY >> bandwidth, effectively decoupling them.
The coupling matrix V is held static (independent of mask) for both lattices.
This is the "high-energy impurity" approximation, standard in TMM literature:
the amplitude at absent sites is O(t / ABSENT_ENERGY) → 0 as ABSENT_ENERGY → ∞.

BCC NOTE: The BCC coupling matrix V has eigenvalues
    λ_{kx,ky} = -4t·cos(2πkx/M)·cos(2πky/M)
which are zero when M is divisible by 4 (kx or ky = M/4). For those M
we apply a twisted boundary condition with phase φ = π/(2M) in BOTH x and y
directions, giving:
    λ_{kx,ky} = -4t·cos(π/(2M)+2πkx/M)·cos(π/(2M)+2πky/M)
which is never zero (the zero-condition kx=(M-1)/4+nM/2 has no integer solutions).
M ∈ {6, 10} are non-problematic and recommended for BCC; M ∈ {4, 8, 12} require
twisted BC (applied automatically).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.config import ABSENT_ENERGY


# ---------------------------------------------------------------------------
# Static geometry helpers (called at Python level, not inside JIT)
# ---------------------------------------------------------------------------

def _sc_inplane_adjacency(M: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (row_idx, col_idx) arrays for SC in-plane nearest-neighbor pairs
    in an M×M periodic cross-section.

    Strategy: for each site, emit only its RIGHT and DOWN bond.
    This generates each undirected bond exactly once, including PBC wrap-arounds,
    and is correct for all M ≥ 2.  (Upper-triangle filtering double-counts
    PBC bonds when M = 2, where left-neighbour = right-neighbour.)

    The caller symmetrises with H[rows,cols] = H[cols,rows] = -t.
    """
    rows, cols = [], []
    for i in range(M):
        for j in range(M):
            idx   = i * M + j
            right = i * M + (j + 1) % M     # right in y-direction (PBC)
            down  = ((i + 1) % M) * M + j   # down  in x-direction (PBC)
            rows.extend([idx, idx])
            cols.extend([right, down])
    return np.array(rows), np.array(cols)


def _bcc_coupling_matrix(M: int, t: float, twisted: bool) -> np.ndarray:
    """
    Compute the BCC inter-layer coupling matrix V (N×N, complex128).

    For BCC, site (ix, iy) in slice n connects to sites (ix+dx, iy+dy) mod M
    in slice n+1, for (dx, dy) ∈ {(±1, ±1)}.

    twisted=True applies a phase exp(iπ(dx+dy)/(2M)) per bond, twisting in BOTH
    x and y directions. This shifts ALL k-points: the zero condition becomes
    kx = (M-1)/4 + nM/2, which has no integer solutions for any M.

    Twisting x only (the naive fix) leaves ky = M/4 zero eigenvalues intact
    because the y-direction eigenvalue factor cos(2πky/M) is unchanged.
    """
    N = M * M
    indices = np.arange(N)
    ix = indices // M
    iy = indices % M

    V = np.zeros((N, N), dtype=np.complex128)
    for dx in (-1, +1):
        for dy in (-1, +1):
            jx = (ix + dx) % M
            jy = (iy + dy) % M
            j_idx = jx * M + jy
            # Twist in both x and y: phase = exp(iπ(dx+dy)/(2M))
            # This gives λ = -4t·cos(π/(2M)+2πkx/M)·cos(π/(2M)+2πky/M), never 0.
            phase = np.exp(1j * np.pi * (dx + dy) / (2 * M)) if twisted else 1.0
            V[indices, j_idx] += -t * phase

    return V


def _needs_twisted_bc(M: int) -> bool:
    """BCC V is singular for M divisible by 4 (zero eigenvalue at kx = M/4)."""
    return M % 4 == 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_transfer_matrix_fn(
    M: int,
    lattice: str,
    E: float = 0.0,
    t: float = 1.0,
):
    """
    Return a JIT-compiled function  f(disorder, mask) -> (2N, 2N) complex128
    that builds the transfer matrix for one slice.

    Parameters
    ----------
    M       : cross-section width (bar is M×M × L)
    lattice : "sc" or "bcc"
    E       : energy (Fermi level)
    t       : hopping amplitude (sets the energy unit)
    """
    if lattice not in ("sc", "bcc"):
        raise ValueError(f"lattice must be 'sc' or 'bcc', got '{lattice}'")

    N = M * M
    eye_N = jnp.eye(N, dtype=jnp.complex128)

    if lattice == "sc":
        # V = -t·I  →  V^{-1} = -I/t  →  V^{-1}(E·I - H_n) = (H_n - E·I)/t
        rows_np, cols_np = _sc_inplane_adjacency(M)
        # Build a static (N,N) in-plane hopping matrix (without disorder/mask).
        # We will scale it with the mask at runtime.
        H_hop_np = np.zeros((N, N), dtype=np.float64)
        # rows_np/cols_np encode one direction of each bond; symmetrise explicitly.
        H_hop_np[rows_np, cols_np] = -t
        H_hop_np[cols_np, rows_np] = -t
        H_hop = jnp.array(H_hop_np)  # captured in JIT closure as a compile-time constant

        @jax.jit
        def _transfer_matrix_sc(disorder: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            # disorder : (N,) float64 on-site energies from uniform disorder
            # mask     : (N,) bool, True = site present
            eps = jnp.where(mask, disorder, ABSENT_ENERGY)

            # Zero out hoppings to/from absent sites.
            # mask.astype(float64) converts bool → {0.0, 1.0} safely.
            hop_mask = jnp.outer(mask.astype(jnp.float64), mask.astype(jnp.float64))
            H_n = jnp.diag(eps) + H_hop * hop_mask  # (N,N) float64

            # SC: V = -t·I  →  V^{-1}(E·I − H_n) = (H_n − E·I)/t
            upper_left = (H_n - E * eye_N) / t   # dtype promotes to complex128 via eye_N
            # T = [ upper_left   −I ]
            #     [      I        0 ]
            top    = jnp.concatenate([upper_left, -eye_N], axis=1)
            bottom = jnp.concatenate([eye_N,  jnp.zeros_like(eye_N)], axis=1)
            return jnp.concatenate([top, bottom], axis=0)

        return _transfer_matrix_sc

    else:  # bcc
        twisted = _needs_twisted_bc(M)
        V_np = _bcc_coupling_matrix(M, t, twisted=twisted)
        V_inv_np = np.linalg.inv(V_np)
        V_inv = jnp.array(V_inv_np)  # captured as constant

        @jax.jit
        def _transfer_matrix_bcc(disorder: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            # BCC has no in-plane hoppings: H_n = diag(ε)
            eps = jnp.where(mask, disorder, ABSENT_ENERGY)
            H_n = jnp.diag(eps.astype(jnp.complex128))
            upper_left = V_inv @ (E * eye_N - H_n)
            top    = jnp.concatenate([upper_left, -eye_N], axis=1)
            bottom = jnp.concatenate([eye_N,  jnp.zeros_like(eye_N)], axis=1)
            return jnp.concatenate([top, bottom], axis=0)

        return _transfer_matrix_bcc
