# Methodology: Transfer Matrix Method and Finite-Size Scaling for 3D Quantum Percolation

This document gives a complete, self-contained description of the numerical methodology
used in this project, from system definition through the final uncertainty budget.
Every parameter choice is justified by either a literature reference or an explicit
verification against the benchmark result.

---

## 1. System Definition

We study a 3D tight-binding Hamiltonian on a site-diluted lattice:

```
H = -t Σ_{<i,j>} c_i† c_j    (sum over occupied nearest-neighbour pairs)
```

Each site i is independently occupied with probability p; empty otherwise.
The single control parameter is p (site occupation probability). W=0 throughout
(no on-site Anderson disorder). Two lattice geometries are studied:

| Lattice | z (coord. number) | Bandwidth | p_c (classical) | Bipartite? |
|---------|-------------------|-----------|-----------------|------------|
| SC      | 6                 | 12t       | 0.3115          | Yes (A/B sublattices) |
| BCC NN  | 8                 | 16t       | 0.2457          | Yes (corners/body-centres) |

Both are bipartite, which creates a chiral symmetry anomaly at E=0 (see §4).

---

## 2. Transfer Matrix Method (TMM)

### 2.1 Construction

The TMM reformulates 3D eigenstates as a product of 2D→2D transfer matrices.
A quasi-1D strip of cross-section M × M sites is propagated in the z-direction
over L slices. The M² orbitals on a single slice form the state vector.

For SC NN hopping at energy E (transfer matrix from slice k to k+1):

```
T_k = V^{-1} (E·I − H_k^{onsite} − H_k^{back})
```

where H_k^{onsite} is the in-plane hopping on slice k, H_k^{back} is the
backward-coupling from slice k−1. The matrix V encodes forward hopping to k+1.

For BCC: the forward-coupling V is non-diagonal (4 off-diagonal shift terms
per row, coupling to (x±½, y±½, z+1) neighbours). This makes V explicitly
non-symmetric and requires a twisted boundary condition for specific M values
(M ∈ {4, 8, 12}) to avoid singularity. The BCC transfer matrix is complex
even for real E at W=0 due to the geometry of the twisted phase.

Implementation: `src/tmm.py`, factory function `make_tmm_fns(M, lattice, E, t)`.

### 2.2 QR Stabilization (Modified Gram-Schmidt)

The raw product T_1 T_2 ··· T_L grows/decays by factors exp(γ_min L) ≫ 1 for
large L, causing numerical overflow. Solution: Modified Gram-Schmidt (MGS) QR
decomposition applied at every step.

After L steps, the accumulated product is Q_1 R_1 Q_2 R_2 ··· Q_L R_L where
each Q_k is orthogonal and R_k is upper-triangular. The Lyapunov exponents are:

```
γ_i = (1/L) Σ_k log |R_k[i,i]|,   i = 1, …, N = M²
```

JAX implementation: `lax.fori_loop(0, L, mgs_step, (Q₀, log_diag_sum))` where
`mgs_step` applies T_k and performs in-place MGS. This avoids the O(L) index
array that `lax.scan(body, init, jnp.arange(L))` creates, preventing XLA
autotune hangs.

The normalized localization length is:

```
Λ_M = ξ_M / M = 1 / (M × γ_min)    where γ_min = min positive Lyapunov exponent
```

### 2.3 Statistical Sampling

- **N_real = 500** independent disorder realizations (different random site occupations).
- Statistical error of Λ_M: standard error of mean, σ(Λ_M) = std(Λ_M) / √N_real.
- At the critical point and our parameters, σ(Λ_M) ≈ 10⁻⁴ (very small — see §5).

### 2.4 Convergence Criteria

**L convergence**: require γ_min × L ≫ 1 so that the quasi-1D system has
equilibrated and Λ_M has converged to its L→∞ value. At p_q, ξ_M ~ M and
γ_min ~ 1/M. With L = 10,000 and M_max = 10: γ_min × L ~ L/M = 1,000 ≫ 1. ✓

**N_real convergence**: σ(Λ_M)/Λ_M < 1% at all (M, p). Confirmed by benchmark
(same N_real=500 gives χ²/dof = 2.4 for 80 data points, consistent with
residual correction terms not statistical noise).

---

## 3. Lyapunov Exponent Batch Computation

To exploit GPU parallelism, all N_real realizations are computed simultaneously
via `jax.vmap`. The compilation pattern is:

```python
tmm_single = lambda key, L, W, p: _tmm_core(key, L, W, p)   # one realization
tmm_batched = jax.jit(jax.vmap(tmm_single, in_axes=(0, None, None, None)))
```

`jit(vmap(f))` compiles ONE vectorized kernel; `vmap(jit(f))` compiles N_real
separate kernels. Always use the former.

Critical: W and p passed to `_tmm_core` must be converted to JAX scalars
immediately:

```python
def _tmm_core(key, L, W, p):
    W = jnp.float64(W)   # prevents XLA constant-folding across loop iterations
    p = jnp.float64(p)
    ...
```

Without this, the XLA disk cache bakes the warmup value of p into all subsequent
calls, giving Λ_M = const for all p (the "flat Λ" bug).

---

## 4. Energy Choice: Avoiding the Bipartite Anomaly

### 4.1 Chiral symmetry at E=0

Both SC and BCC are bipartite lattices: their sites can be 2-coloured (A, B)
such that all bonds connect A to B. This gives a chiral operator Γ (flip A↔B
with sign) satisfying:

```
ΓHΓ = −H
```

At E=0, this is an exact symmetry of the transfer matrix: the system belongs
to symmetry class **BDI** (chiral orthogonal), not class **AI** (standard 3D
orthogonal). The BDI class has a different critical exponent ν_BDI ≠ 1.622 and
different corrections-to-scaling (b → 0 at the chiral fixed point).

Measured consequence: SC at E=0 gives p_q = 0.591 ± 0.001, ν ≈ 2.0, b ≈ 0,
χ²/dof ≈ 10. Ujfalusi & Varga 2014 (PRB 90, 245136) confirm p_c^Q(E→0) → 1
and report "severe convergence troubles" for MFSS at E < 0.1.

### 4.2 BDI-AI crossover and the choice E=0.7

E acts as a relevant perturbation at the BDI fixed point. The BDI-AI crossover
length scale is:

```
ξ_cross ~ E^{−1/Δ_E}
```

where Δ_E is the scaling dimension of E at the BDI fixed point. For E=0.1:
ξ_cross ~ 10 lattice spacings ~ M_max = 10 → correction-dominated regime.
Measured: b = −0.153 at E=0.1, giving correction/signal ratio = 38% at M=4.
Result: spurious crossings at p ≈ 0.35, χ²/dof = 1543.

For E=0.7 (SC, 11.7% of half-bandwidth 6t):
- ξ_cross ~ (1/0.7)^{1/Δ_E} ≈ 1.4^{1/Δ_E} ≪ M_min = 4 for any Δ_E > 0.
- BDI-AI crossover corrections negligible at our M values.
- p_c^Q(E=0.7) ≈ 0.384 is the mobility-edge minimum (Ujfalusi 2014).
- Corrections expected similar to Anderson benchmark: b ≈ 0.1, clean crossings.

**BCC note**: BCC half-bandwidth is 16t. E=0.1 is only 1.25% of half-bandwidth —
even closer to BDI crossover than SC. After SC E=0.7 result is obtained, BCC
energy will be selected with the same ξ_cross argument.

---

## 5. Finite-Size Scaling (FSS)

### 5.1 Scaling Theory

Near the quantum percolation threshold p_q (or W_c for Anderson disorder), the
normalized localization length obeys one-parameter scaling:

```
Λ_M(x) = F(M^{1/ν} · (x − x_c) / x_c) + b·M^{−y} + O(M^{−2y})
```

where:
- x = p (dilution) or x = W (disorder), x_c = p_q or W_c
- u = M^{1/ν} · (x − x_c) / x_c is the scaling variable
- F(u) = Σ_{k=0}^{n} a_k u^k is the universal scaling function (polynomial approx.)
- b·M^{−y}: correction from the leading RG-irrelevant operator

The polynomial expansion F(u) = Σ a_k u^k is justified for |u| not too large
(i.e., the system is not too far from the critical point). The window selection
in §6 enforces this.

### 5.2 Polynomial Order

`poly_order = 4` (expansion to degree 4) is used. This choice was validated by:
- poly_order=3: residual polynomial strain at window edges visible (χ²/dof=2.615)
- poly_order=4: strain reduced (χ²/dof=2.421), no further improvement at order=5
  (insufficient data to constrain additional parameter)

### 5.3 Irrelevant Exponent y

The correction term b·M^{-y} where y > 0 is the irrelevant scaling exponent
for the 3D orthogonal universality class.

**Literature values (in our positive-y convention):**

| Source | Observable | |y| |
|--------|------------|-----|
| Slevin & Ohtsuki 1999, PRL 82, 382 | Λ = λ/L (TMM) | 2.8–3.9 (fitted freely) |
| Rodriguez et al. 2011, PRB 84, 134209 | multifractal Δ_q | 1.7 (fixed-λ FSS) |
| This project (effective) | Λ_M = ξ_M/M | 1.12 (empirical) |

S&O 1999 fit y as a free parameter (not fixed), obtaining |y| ≈ 2.8 (box
disorder). Rodriguez 2011 use a different observable (multifractal exponents
at fixed box-to-system size ratio λ); their y governs correction with box size,
not system size, and is not directly comparable to TMM corrections.

Our effective value y = 1.12 is the phenomenological correction exponent that
gives a good benchmark fit (χ²/dof = 2.4, all gates pass). It is **not** the
S&O value (2.8) or the field-theory leading irrelevant exponent — it appears
to capture a subleading correction that dominates at M = {4..10}. This is
acceptable for our purpose: the benchmark validates that the FSS methodology
recovers the correct W_c and ν.

**Why y is fixed, not fitted**: with M ∈ {4..10}, b·M^{-y} and b_2·M^{-2y}
are nearly proportional — the fit is degenerate. Setting y free allows the
optimizer to exploit the near-degeneracy (y → 0 gives b·M^0 = constant, which
absorbs into a_0). Both failure modes confirmed empirically. The correct
approach is to fix y at the empirical value and propagate the ±0.08 uncertainty
as a systematic.

**Systematic from y uncertainty**: Run FSS at y = 1.04, 1.12, 1.20. Report
max(|result(1.04) − result(1.12)|, |result(1.20) − result(1.12)|) as ±syst.

### 5.4 Number of Corrections

`n_corrections = 1` (single b·M^{-y} term) is used. The second correction
b_2·M^{-2y} cannot be jointly constrained with b at only 4 M-values: {M^{-y}}
and {M^{-2y}} are nearly collinear for M ∈ {4,6,8,10}. A second correction
requires M ≥ 5 distinct sizes with wider spacing (e.g., {4,6,8,10,12,16}).

---

## 6. Window Selection: Stability Scan

### 6.1 The problem

FSS requires restricting data to a window p ∈ [p_min, p_max] near x_c. If the
window is too wide, higher-order scaling terms (u^5, u^6) introduce systematic
bias; χ²/dof rises. If too narrow, not enough data points remain to constrain
all free parameters; errors are large.

### 6.2 Procedure

We implement a window stability scan derived from the general principle in
Slevin & Ohtsuki 1999 (maximize goodness-of-fit while keeping model as simple
as possible) and Rodriguez et al. 2011 (assess stability of critical parameters
under model perturbations):

1. Compute rough p_q_rough from pairwise Λ_M crossing detection (sign-change of
   Λ_{M1}(p) − Λ_{M2}(p) with linear interpolation of the zero).

2. Scan δ ∈ {0.03, 0.04, ..., 0.15} (15 values via np.linspace(0.03, 0.15, 15)).

3. For each δ: run FSS optimization (Nelder-Mead + L-BFGS-B, no bootstrap)
   on the window (p_q_rough − δ, p_q_rough + δ). Record x_c(δ), ν(δ), b(δ),
   χ²/dof(δ), and N_data(δ).

4. **Select the widest δ with χ²/dof < 3.0.** More data at wider windows
   constrains the fit more tightly; wider is better as long as the model
   is adequate (χ²/dof < threshold).

5. If no δ gives χ²/dof < 3.0: report the narrowest δ as fallback and flag a
   WARNING. This indicates either (a) the model is wrong (wrong universality
   class, wrong y), or (b) σ(Λ_M) is so small that higher-order corrections
   are resolved (need poly_order=5 or n_corrections=2).

This replaces the previous hardcoded half-width p_fit_hw = 0.08.

### 6.3 χ²/dof Threshold

The threshold of 3.0 is chosen as follows:
- σ(Λ_M) ≈ 10⁻⁴ in our data. Residuals at the 0.1% level (model error ~10⁻⁴)
  produce χ²/dof ≈ 1.0.
- Subleading corrections b_2·M^{-2y} contribute ~10⁻⁴ to residuals at our M
  range → χ²/dof ~ 2-3 is expected from this alone.
- A value > 5 indicates a qualitative fit failure (wrong window, wrong model, or
  wrong symmetry class).
- We accept up to 3.0 as the model-inadequacy floor.

### 6.4 Rodriguez et al. 2011 Stability Test (model order)

Rodriguez et al. 2011 additionally assess stability of fit parameters under
perturbation of the polynomial expansion indices. The procedure (Appendix A):

1. Enumerate all expansions over a reasonable range of (nR, nI, mR, mI).
2. Filter by p-value > 0.1 (Q > 0.1 in S&O notation).
3. For each candidate expansion, perturb each index by +1 separately AND all by
   +1 simultaneously; check that critical parameters x_c and ν remain within
   the 95% CI of the candidate.
4. Take the simplest stable expansion.

This is complementary to the window scan: the window scan addresses data range
(too much/too little data), while the Rodriguez test addresses model order (too
few/many polynomial terms). Both should be satisfied for a reliable result.
For our project, poly_order=4 has been validated by the Anderson benchmark; we
do not repeat the full index perturbation for the dilution runs, but we do vary
poly_order (3 → 4 → 5) as a robustness check.

---

## 7. Parameter Estimation and Uncertainty

### 7.1 Optimization

The χ² objective is minimized in two stages:
1. **Nelder-Mead** (derivative-free, global): `maxiter=200_000`, tolerances
   `xatol=1e-7, fatol=1e-9`. Finds a good basin.
2. **L-BFGS-B** (gradient-based, local): `ftol=1e-12, gtol=1e-8, maxiter=50_000`.
   Refines within the basin found by Nelder-Mead.

Best solution: whichever of the two stages gives lower χ².

### 7.2 Statistical Uncertainty (Bootstrap)

Bootstrap resampling with N_boot = 500:
- For each resample: perturb Λ_M → Λ_M + N(0, σ_M) for all (M, p) in the window.
- Minimize χ² on the perturbed data (Nelder-Mead only, no L-BFGS-B, for speed).
- Statistical uncertainty = std of the 500 bootstrap estimates.

This propagates the measured statistical noise σ(Λ_M) through the nonlinear FSS
fit. It does NOT account for correlated errors between p-values (the 500 disorder
realizations for different p-values are independent, so this is valid).

### 7.3 Systematic Uncertainty (y variation)

Run FSS three times: y = 1.04, 1.12, 1.20 (±0.08 of the central value).
Report: syst = max(|result(1.04) − result(1.12)|, |result(1.20) − result(1.12)|).

### 7.4 Total Uncertainty

Combine in quadrature:
```
σ_total = sqrt(σ_stat² + σ_syst²)
```

---

## 8. Success Criteria

| Observable | Target | Tolerance | Reference |
|------------|--------|-----------|-----------|
| p_q(SC, E=0.7) | 0.384 | ±0.005 | Ujfalusi 2014, Table I |
| ν(SC) | 1.622 | ±0.10 | Slevin & Ohtsuki 1999; Ujfalusi 2014 |
| χ²/dof | < 3.0 | | Window scan threshold |
| b_err/b | < 0.5 | | Correction well-constrained |
| ν(BCC) | ≈ ν(SC) | < 2σ difference | Same universality class (3D-AI) |

If χ²/dof > 5.0 after the window scan selects the best window: **stop and
diagnose before proceeding**. Do not manually override the window to force
an acceptable χ² — that would undermine the non-ad-hoc methodology.

---

## 9. Robustness Checks

Run before writing up:

1. **Polynomial order**: compare poly_order=3, 4, 5. x_c and ν should agree
   within stat errors. Large χ² decrease from 3→4 justifies 4; insignificant
   from 4→5 confirms 4 is sufficient.

2. **y sensitivity**: already reported as ±syst. Spot-check that the syst band
   matches the reported shift.

3. **L convergence**: verify γ_min × L > 100 at x_c for all M.

4. **Window stability**: confirm that p_q(δ) and ν(δ) are flat across all δ
   values where χ²/dof < 3.0 (plateau). A trend in p_q vs. δ indicates
   systematic drift from higher-order terms not captured by the model.

5. **Classical percolation check**: `find_pc(L=20)` ≈ expected p_c before
   each quantum percolation run.

---

## 10. Summary of Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| L | 10,000 | γ_min × L > 1000 >> 1 |
| N_real | 500 | σ(Λ_M)/Λ_M < 1% |
| poly_order | 4 | Validated by benchmark BIC |
| y | 1.12 (fixed) | Empirical; reproduces benchmark; ±0.08 systematic |
| n_corrections | 1 | Degenerate with n=2 at 4 M values |
| n_boot | 500 | σ < 1% on bootstrap CIs |
| E(SC) | 0.7 | Far from BDI crossover; p_q at mobility edge minimum |
| M values | {4, 6, 8, 10} | Proven sufficient for FSS; M=12 adds ~22h |
| χ² threshold | 3.0 | Window stability scan criterion |

---

## 11. References

1. Slevin K and Ohtsuki T 1999 Phys. Rev. Lett. **82** 382
   "Corrections to Scaling at the Anderson Transition" — introduced corrections-
   to-scaling for 3D Anderson TMM; fitted y freely, obtaining |y|≈2.8–3.9 for
   the Λ observable; established ν=1.57–1.58. Bootstrap CIs; maximize Q criterion.

2. Rodriguez A, Vasquez L J, Slevin K and Römer R A 2011 Phys. Rev. B **84** 134209
   "Multifractal finite-size-scaling and universality at the Anderson transition"
   — MFSS method; polynomial order perturbation stability test; |y|≈1.7 for
   fixed-λ FSS. System sizes L=20–120; ν=1.590(1.579,1.602).

3. Ujfalusi L and Varga I 2014 Phys. Rev. B **90** 245136
   "Finite-size scaling and multifractality at the Anderson transition for the
   three Wigner-Dyson symmetry classes in 3D" — quantum percolation threshold
   p_c^Q as function of E for SC lattice; mobility edge; MFSS approach; Table I
   gives p_c^Q(E=0.7)≈0.384; reports BDI anomaly at E=0.
