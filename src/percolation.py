"""
Classical site percolation on 3D SC and BCC lattices.

Used to:
  1. Verify p_c(SC)  = 0.3115 and p_c(BCC) = 0.2457 (sanity checks)
  2. Confirm that Λ_M → 0 for p < p_c (no quantum transport without classical path)

Algorithm: Hoshen-Kopelman union-find on a 3D periodic cubic grid.
"""
from __future__ import annotations

import numpy as np


def _find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path compression
        x = parent[x]
    return x


def _union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1


def _neighbors_sc(i: int, j: int, k: int, L: int):
    """6 nearest neighbors — periodic BC."""
    yield (i - 1) % L, j, k
    yield (i + 1) % L, j, k
    yield i, (j - 1) % L, k
    yield i, (j + 1) % L, k
    yield i, j, (k - 1) % L
    yield i, j, (k + 1) % L


def _neighbors_sc_nopbc(i: int, j: int, k: int, L: int):
    """6 nearest neighbors — no periodic BC (for spanning check)."""
    for ni, nj, nk in ((i-1,j,k),(i+1,j,k),(i,j-1,k),(i,j+1,k),(i,j,k-1),(i,j,k+1)):
        if 0 <= ni < L and 0 <= nj < L and 0 <= nk < L:
            yield ni, nj, nk


def _neighbors_bcc(i: int, j: int, k: int, L: int):
    """8 nearest neighbors — periodic BC."""
    for di in (-1, +1):
        for dj in (-1, +1):
            for dk in (-1, +1):
                yield (i + di) % L, (j + dj) % L, (k + dk) % L


def _neighbors_bcc_nopbc(i: int, j: int, k: int, L: int):
    """8 nearest neighbors — no periodic BC (for spanning check)."""
    for di in (-1, +1):
        for dj in (-1, +1):
            for dk in (-1, +1):
                ni, nj, nk = i + di, j + dj, k + dk
                if 0 <= ni < L and 0 <= nj < L and 0 <= nk < L:
                    yield ni, nj, nk


def percolation_probability(
    L: int,
    p: float,
    lattice: str = "sc",
    n_samples: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Estimate the percolation probability P_∞(p, L) via Monte Carlo.

    A sample "percolates" if the giant component in the z-direction wraps
    around the periodic boundary (sites in z=0 and z=L-1 are in the same cluster).

    Parameters
    ----------
    L         : linear system size (L×L×L sites)
    p         : site occupation probability
    lattice   : "sc" or "bcc"
    n_samples : number of independent realisations
    seed      : RNG seed for reproducibility

    Returns
    -------
    (mean_P_inf, stderr)
    """
    rng = np.random.default_rng(seed)
    # Non-periodic neighbors for spanning check: periodic BC makes z=0 directly
    # adjacent to z=L-1, so _wraps with periodic neighbors trivially fires at any p.
    nopbc_fn = _neighbors_sc_nopbc if lattice == "sc" else _neighbors_bcc_nopbc
    count = 0

    for _ in range(n_samples):
        grid  = rng.random((L, L, L)) < p   # (L,L,L) bool
        count += _wraps(grid, L, nopbc_fn)

    mean   = count / n_samples
    stderr = np.sqrt(mean * (1 - mean) / n_samples)
    return mean, stderr


def _wraps(grid: np.ndarray, L: int, neighbors_fn) -> bool:
    """Return True if occupied sites wrap from z=0 to z=L-1."""
    N = L * L * L
    parent = np.arange(N)
    rank   = np.zeros(N, dtype=np.int32)

    def idx(i, j, k):
        return i * L * L + j * L + k

    for i in range(L):
        for j in range(L):
            for k in range(L):
                if not grid[i, j, k]:
                    continue
                for ni, nj, nk in neighbors_fn(i, j, k, L):
                    if grid[ni, nj, nk]:
                        _union(parent, rank, idx(i, j, k), idx(ni, nj, nk))

    # Check if any site in z=0 shares a cluster root with any site in z=L-1
    roots_front = {_find(parent, idx(i, j, 0))
                   for i in range(L) for j in range(L) if grid[i, j, 0]}
    roots_back  = {_find(parent, idx(i, j, L - 1))
                   for i in range(L) for j in range(L) if grid[i, j, L - 1]}
    return bool(roots_front & roots_back)


def find_pc(
    L: int,
    lattice: str = "sc",
    p_min: float = 0.20,
    p_max: float = 0.45,
    n_p: int = 20,
    n_samples: int = 1000,
) -> float:
    """
    Coarse estimate of p_c as the midpoint of the steepest descent in P_∞(p).
    For validation only; not intended as a precision measurement.

    Expected values: SC ≈ 0.311, BCC ≈ 0.246.
    """
    p_vals = np.linspace(p_min, p_max, n_p)
    probs  = [percolation_probability(L, p, lattice, n_samples)[0] for p in p_vals]
    slopes = np.diff(probs) / np.diff(p_vals)
    i_max  = int(np.argmax(np.abs(slopes)))
    return float((p_vals[i_max] + p_vals[i_max + 1]) / 2.0)
