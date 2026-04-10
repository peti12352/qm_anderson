"""
Unit and integration tests for the TMM implementation.

Run with:  pixi run test
           pytest tests/ -v

Tests are ordered from fast/unit to slow/integration.
All tests must pass before any research run is considered valid.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config   # noqa — enable x64 before any jax import
import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[4, 6])
def M_small(request):
    return request.param


# ---------------------------------------------------------------------------
# Transfer matrix tests
# ---------------------------------------------------------------------------

class TestTransferMatrix:
    def test_sc_shape(self, M_small):
        from src.lattice import make_transfer_matrix_fn
        M = M_small
        N = M * M
        build_T = make_transfer_matrix_fn(M, "sc")
        dis = jnp.zeros(N)
        mask = jnp.ones(N, dtype=bool)
        T = build_T(dis, mask)
        assert T.shape == (2*N, 2*N), f"Expected ({2*N},{2*N}), got {T.shape}"

    def test_bcc_shape(self, M_small):
        from src.lattice import make_transfer_matrix_fn
        M = M_small
        N = M * M
        build_T = make_transfer_matrix_fn(M, "bcc")
        dis = jnp.zeros(N)
        mask = jnp.ones(N, dtype=bool)
        T = build_T(dis, mask)
        assert T.shape == (2*N, 2*N)

    def test_sc_dtype(self, M_small):
        """SC transfer matrix must be float64 (real Hamiltonian, no complex needed).
        BCC transfer matrix must be complex128 (twisted BC introduces phases)."""
        from src.lattice import make_transfer_matrix_fn
        M = M_small
        N = M * M
        T_sc  = make_transfer_matrix_fn(M, "sc") (jnp.zeros(N), jnp.ones(N, dtype=bool))
        T_bcc = make_transfer_matrix_fn(M, "bcc")(jnp.zeros(N), jnp.ones(N, dtype=bool))
        assert T_sc.dtype  == jnp.float64,   f"SC: expected float64, got {T_sc.dtype}"
        assert T_bcc.dtype == jnp.complex128, f"BCC: expected complex128, got {T_bcc.dtype}"

    def test_sc_lower_left_is_identity(self, M_small):
        """The lower-left block of the SC transfer matrix must be the identity."""
        from src.lattice import make_transfer_matrix_fn
        M = M_small
        N = M * M
        build_T = make_transfer_matrix_fn(M, "sc")
        T = build_T(jnp.zeros(N), jnp.ones(N, dtype=bool))
        lower_left = T[N:, :N]
        assert jnp.allclose(lower_left, jnp.eye(N, dtype=jnp.complex128)), \
            "Lower-left block of T must be I_N"

    def test_sc_lower_right_is_zero(self, M_small):
        from src.lattice import make_transfer_matrix_fn
        M = M_small
        N = M * M
        build_T = make_transfer_matrix_fn(M, "sc")
        T = build_T(jnp.zeros(N), jnp.ones(N, dtype=bool))
        lower_right = T[N:, N:]
        assert jnp.allclose(lower_right, jnp.zeros((N, N)), atol=1e-12), \
            "Lower-right block of T must be 0"

    def test_sc_absent_sites_decouple(self, M_small):
        """
        Absent sites (mask=False) get ABSENT_ENERGY on-site potential.
        Their presence in the transfer matrix should not affect the
        localization length for the remaining connected sub-lattice
        (verified indirectly: Λ_M decreases with absent-site fraction).
        """
        from src.lattice import make_transfer_matrix_fn
        from src.config  import ABSENT_ENERGY
        M = M_small
        N = M * M
        build_T = make_transfer_matrix_fn(M, "sc")
        # All-absent lattice: upper-left block should be ≈ (ABSENT_ENERGY)/t * I
        mask_absent = jnp.zeros(N, dtype=bool)
        dis = jnp.zeros(N)
        T = build_T(dis, mask_absent)
        upper_left = T[:N, :N]
        # Diagonal entries should be ≈ ABSENT_ENERGY (since (H-E)/t ≈ ABSENT_ENERGY/t)
        diag_vals = jnp.abs(jnp.diag(upper_left))
        assert jnp.all(diag_vals > 1e4), \
            "Absent sites should produce large diagonal entries in upper-left block"

    def test_sc_hermitian_symmetry(self):
        """
        For real E and real t, the transfer matrix T satisfies the symplectic
        condition T^† J T = J, where J = [[0, I], [-I, 0]].
        We verify this weakly: det(T) = ±1 (Liouville theorem).
        """
        from src.lattice import make_transfer_matrix_fn
        M, N = 4, 16
        build_T = make_transfer_matrix_fn(M, "sc")
        rng = np.random.default_rng(0)
        dis  = jnp.array(rng.uniform(-1, 1, N))
        mask = jnp.ones(N, dtype=bool)
        T = build_T(dis, mask)
        det = jnp.abs(jnp.linalg.det(T))
        assert abs(float(det) - 1.0) < 0.01, \
            f"|det(T)| = {float(det):.4f}, expected ≈ 1.0 (Liouville theorem)"


# ---------------------------------------------------------------------------
# TMM core tests
# ---------------------------------------------------------------------------

class TestTMM:
    def test_lyapunov_positive(self):
        """All returned Lyapunov exponents must be finite."""
        from src.tmm import make_tmm_fns
        M = 4
        tmm_single, _ = make_tmm_fns(M, "sc")
        key = jax.random.PRNGKey(0)
        gammas = tmm_single(key, L=1000, W=16.5, p=1.0)
        assert jnp.all(jnp.isfinite(gammas)), "Lyapunov exponents contain NaN/Inf"

    def test_lyapunov_sorted_descending(self):
        from src.tmm import make_tmm_fns
        M = 4
        tmm_single, _ = make_tmm_fns(M, "sc")
        key = jax.random.PRNGKey(1)
        gammas = tmm_single(key, L=1000, W=16.5, p=1.0)
        assert jnp.all(gammas[:-1] >= gammas[1:] - 1e-10), \
            "Lyapunov exponents must be sorted in descending order"

    def test_lyapunov_symmetry(self):
        """
        For Hermitian Hamiltonians, Lyapunov exponents come in ±γ pairs:
        γ[i] + γ[2N-1-i] ≈ 0. Verify this holds within numerical tolerance.
        """
        from src.tmm import make_tmm_fns
        M = 4
        N = M * M
        tmm_single, _ = make_tmm_fns(M, "sc")
        key = jax.random.PRNGKey(2)
        print(f"\n  running M={M} L=5000 single bar...")
        gammas = tmm_single(key, L=5_000, W=16.5, p=1.0)
        print(f"  gammas range: [{float(gammas[-1]):.4f}, {float(gammas[0]):.4f}]")
        pairs = gammas[:N] + gammas[2*N-1:N-1:-1]
        assert jnp.max(jnp.abs(pairs)) < 0.1, \
            f"Lyapunov pairs {pairs} not symmetric (max deviation {jnp.max(jnp.abs(pairs)):.4f})"

    @pytest.mark.slow
    def test_localized_limit(self):
        """At W >> W_c, Λ_M should decrease with M (localized phase)."""
        from src.tmm import make_tmm_fns, lambda_M
        M_small, M_large = 4, 8
        L = 5_000   # short bar sufficient: ξ << L at W=30 (deep insulator)
        W = 30.0    # deep in insulating phase
        n_real = 20

        lam_vals = {}
        for M in [M_small, M_large]:
            print(f"\n  M={M}: compiling + running {n_real} bars × L={L} steps...")
            _, batched = make_tmm_fns(M, "sc", E=0.0)
            keys   = jax.random.split(jax.random.PRNGKey(3), n_real)
            gammas = batched(keys, L, W, 1.0)
            lam_vals[M], _ = lambda_M(gammas, M)
            print(f"  M={M}: Λ_M = {lam_vals[M]:.4f}")

        assert lam_vals[M_small] > lam_vals[M_large], \
            f"At W={W} (localized), Λ_M should decrease with M: " \
            f"Λ_{M_small}={lam_vals[M_small]:.4f}, Λ_{M_large}={lam_vals[M_large]:.4f}"

    @pytest.mark.slow
    def test_metallic_limit(self):
        """At W << W_c, Λ_M should increase with M (metallic phase)."""
        from src.tmm import make_tmm_fns, lambda_M
        M_small, M_large = 4, 8
        L = 5_000   # shorter bar; signal clear because W=5 is far from W_c
        W = 5.0     # deep in metallic phase
        n_real = 20

        lam_vals = {}
        for M in [M_small, M_large]:
            print(f"\n  M={M}: compiling + running {n_real} bars × L={L} steps...")
            _, batched = make_tmm_fns(M, "sc", E=0.0)
            keys   = jax.random.split(jax.random.PRNGKey(4), n_real)
            gammas = batched(keys, L, W, 1.0)
            lam_vals[M], _ = lambda_M(gammas, M)
            print(f"  M={M}: Λ_M = {lam_vals[M]:.4f}")

        assert lam_vals[M_small] < lam_vals[M_large], \
            f"At W={W} (metallic), Λ_M should increase with M: " \
            f"Λ_{M_small}={lam_vals[M_small]:.4f}, Λ_{M_large}={lam_vals[M_large]:.4f}"

    @pytest.mark.slow
    def test_below_classical_threshold_localized(self):
        """
        Below the classical percolation threshold p_c=0.311, ALL states are
        localized (no connected path exists). Λ_M must decrease with M.
        """
        from src.tmm import make_tmm_fns, lambda_M
        M_small, M_large = 4, 8
        p = 0.25   # below p_c(SC)=0.311 — all states localized, strong signal
        L = 5_000
        n_real = 20

        lam_vals = {}
        for M in [M_small, M_large]:
            print(f"\n  M={M}: compiling + running {n_real} bars × L={L} steps (p={p}, below p_c=0.311)...")
            _, batched = make_tmm_fns(M, "sc", E=0.0)
            keys   = jax.random.split(jax.random.PRNGKey(5), n_real)
            gammas = batched(keys, L, 0.0, p)
            lam_vals[M], _ = lambda_M(gammas, M)
            print(f"  M={M}: Λ_M = {lam_vals[M]:.4f}")

        assert lam_vals[M_small] > lam_vals[M_large], \
            f"Below p_c, all states must be localized. " \
            f"Λ_{M_small}={lam_vals[M_small]:.4f}, Λ_{M_large}={lam_vals[M_large]:.4f}"


# ---------------------------------------------------------------------------
# FSS tests
# ---------------------------------------------------------------------------

class TestFSS:
    def test_synthetic_data_recovery(self):
        """
        Generate synthetic Λ_M data from a known (x_c, ν) and verify recovery.
        """
        from src.fss import fit_fss
        true_x_c = 0.440
        true_nu  = 1.57
        # F(u) = 0.5 + 0.1*u  (simple linear scaling function)
        coeffs = [0.5, 0.1]
        rng = np.random.default_rng(seed=0)

        # noise=0.003: small enough that ν uncertainty (∝ noise) is ~0.04,
        # comfortably within the 0.05 tolerance. noise=0.01 gave ν_err≈0.13,
        # making the 0.05 tolerance tighter than 1σ — intermittently failing.
        M_list, x_list, lam_list, err_list = [], [], [], []
        for M in [4, 6, 8, 10, 12]:
            for x in np.linspace(0.35, 0.55, 30):
                u   = M ** (1/true_nu) * (x - true_x_c) / true_x_c
                lam = sum(c * u**k for k, c in enumerate(coeffs))
                noise = 0.003
                lam_noisy = lam + rng.standard_normal() * noise
                M_list.append(M); x_list.append(x)
                lam_list.append(lam_noisy); err_list.append(noise)

        # poly_order=1 matches the true linear model (2 coefficients).
        # Using poly_order=3 overparameterises and biases nu.
        # n_boot=50 is sufficient for a test; production uses n_boot=500.
        result = fit_fss(M_list, x_list, lam_list, err_list,
                         x_c_guess=0.45, nu_guess=1.6,
                         poly_order=1, n_boot=50)
        print(f"\n  FSS result: x_c={result.x_c:.4f} (true {true_x_c}), "
              f"nu={result.nu:.4f} (true {true_nu}), chi2/dof={result.chi2_dof:.3f}")

        assert abs(result.x_c - true_x_c) < 0.003, \
            f"x_c recovery failed: {result.x_c:.4f} vs {true_x_c}"
        assert abs(result.nu - true_nu) < 0.05, \
            f"ν recovery failed: {result.nu:.4f} vs {true_nu}"
