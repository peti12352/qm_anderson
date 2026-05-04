# Runbook: SC benchmark → SC dilution → BCC dilution

Complete sequence to reproduce the benchmark, validate, and run the novel BCC measurement.
Each step has a clear success criterion. Do not proceed to the next step until it is met.

---

## FSS Model (canonical for this project)

### Equation

    Λ_M(x) = Σ_{k=0}^{4} a_k · u^k  +  b · M^{-y}

    u = M^{1/ν} · (x - x_c) / x_c

where x is the control parameter (W for Anderson disorder, p for dilution).

### Fixed parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `poly_order` | 4 | Reduces polynomial strain vs order 3; see below |
| `y_irrel` | 1.12 | Slevin & Ohtsuki 1999, Phys. Rev. Lett. 82, 382 |
| `fit_y_irrel` | False | See identifiability argument below |
| `n_corrections` | 1 | See identifiability argument below |
| `n_boot` | 500 | Sufficient for σ < 1% on bootstrap CIs |

### Why y is fixed at 1.12, not fitted

y is the irrelevant scaling exponent of the 3D orthogonal universality class. S&O99
measured y = 1.12 ± 0.08 using M ∈ {8..20}. With M = {4..12}, y cannot be identified
from our TMM data. Two explicit failures confirmed this:

**Attempt 1** — `fit_y_irrel=True, n_corrections=2, poly_order=4`:
y = 0.89, b_err/b = 7.0, χ²/dof = 232. Root cause: b·M^{-y} and b₂·M^{-2y} are
nearly linearly dependent at 4 M-values; Nelder-Mead exploited the flat valley.

**Attempt 2** — `fit_y_irrel=True, n_corrections=1, poly_order=4`:
y = 0.05 (collapsed to zero), b_err/b = 7.1, W_c and ν far outside tolerance.
Root cause: b·M^{-y} → b (constant) as y→0 — algebraically degenerate with a_0.
The optimizer found this valley; χ²/dof = 1.931 was achieved by an unphysical solution.

**Conclusion**: Fitting y requires M spanning a decade (e.g. {4..20}). Fix y at the
literature value and propagate its ±0.08 uncertainty as a systematic error by running
at y = 1.04, 1.12, 1.20. This is standard practice (Rodriguez et al. 2011 PRB,
Ujfalusi & Varga 2012).

### Why n_corrections=1, not 2

b₂·M^{-2y} is the second-order correction. With only 4 distinct M-values, {M^{-y}}
and {M^{-2y}} are nearly proportional vectors; b, b₂ cannot be jointly constrained.
b₂ requires M ≥ 5–6 sizes (e.g. {4,6,8,10,12,16}) with wide spacing.

### Why poly_order=4, not 3

poly_order=3 left polynomial strain at window edges (|u|_max ≈ 7). Adding one term
reduced χ²/dof from 2.615 to 2.421 without degrading identifiability.

### Residual χ²/dof ≈ 2.4 — physical origin

With M = {4..10}, M^{-y} at M=4 is 0.232 and at M=10 is 0.076. These are large
corrections (7–23% of Λ_M). The single term b·M^{-y} captures the leading correction,
but subleading terms (b₂·M^{-2y}, analytic corrections M^{-1/ν}) contribute at the
~5% level and are not modelled. This is a fundamental small-M limitation, not a code
bug. It also causes mild positive biases in W_c and ν (established pattern in FSS
literature for small M). All benchmark gates pass.

### Error budget

For each result, the output reports:
- **stat**: bootstrap standard deviation (N=500 resamples)
- **syst**: max(|result(y=1.04) − result(y=1.12)|, |result(y=1.20) − result(y=1.12)|)

Combine as total = sqrt(stat² + syst²) for single-number reporting.

---

## Step 1 — COMPLETE ✓

Benchmark data collected (M={4,6,8,10}, W∈[13,20], L=10k, N_real=500).
Raw TMM runtime: 540 min on RTX 5090.

FSS result (canonical model):
```
y   = 1.12 (fixed)
W_c = 16.6174 ± 0.0156 (stat) ± 0.0148 (syst)
ν   = 1.6205 ± 0.0085 (stat) ± 0.0052 (syst)
b   = 0.0958 ± 0.0045 (stat)
χ²/dof = 2.421
```
Gate: PASS (|W_c−16.53|=0.087 < 0.30; |ν−1.571|=0.049 < 0.10; χ²/dof < 5.0)

To refit at any time (no TMM rerun):
```bash
pixi run refit
# or with non-default y:
pixi run python runs/sc_benchmark.py --refit --y-irrel 1.04
pixi run python runs/sc_benchmark.py --refit --y-irrel 1.20
```

---

## Step 2 — COMPLETE ✓

MGS QR fix applied to tmm.py (removed cuSOLVER dispatch that caused 41× slowdown).
Data regenerated for M=8 and M=10. Results unchanged (same statistics, correct physics).

---

## Step 3 — SC quantum percolation at E=0.1

Validates the quantum percolation setup against Ujfalusi & Varga (2014) Table II: p_c^Q(E=0.1) = 0.45384.
Uses same FSS model as benchmark (y=1.12 fixed, poly_order=4, n_corrections=1).

**Why E=0.1, not E=0:** The SC lattice is bipartite (chiral symmetry: H → −H under sublattice flip).
At E=0, this places the system in symmetry class AIII/BDI, not class AI (3D orthogonal). Measured
consequence: p_q(E=0) ≈ 0.591 for M={4..10}, ν ≈ 2.0, χ²/dof = 10.5 — consistent with Ujfalusi's
observation that p_c^Q(E→0)→1 and MFSS "severely fails to converge" at E<0.1. E=0.1 is the smallest
energy where Ujfalusi's MFSS converges and gives the 3D-AI universality class (ν=1.62±0.035).

Output files: `data/sc_dilution_E0p1.h5`, checkpoints `sc_dilution_E0p1_M{M}.npz`.

```bash
pixi run sc-dilution
```

Kill-safe: per-M and per-p checkpoints written throughout. Resume with same command.

**Expected output:**
```
  Λ_M(p) table (every 4th p):
         p    Λ(M=4)    Λ(M=6)    Λ(M=8)   Λ(M=10)
    0.3200   x.xxxxx   x.xxxxx   x.xxxxx   x.xxxxx
    ...
    0.4540   x.xxxxx   x.xxxxx   x.xxxxx   x.xxxxx   ← all close here
    ...

  Rough p_q (pairwise crossings): 0.45x  [individual: ['0.44x', '0.45x', '0.46x']]
  FSS fit window: p ∈ [0.37x, 0.53x]

  p_q = 0.454xx ± 0.000xx (stat) ± 0.000xx (syst)
  ν   = 1.6xxx  ± 0.0xxx (stat) ± 0.0xxx (syst)
  χ²/dof = x.xxx
```

Crossings at E=0.1 should be staggered by ~0.01–0.04 (corrections-to-scaling at this energy are
smaller than at E=0 where b≈0 but ν≠1.62). Λ_M ordering: insulating (Λ_4>Λ_10 at low p), metallic
(Λ_4<Λ_10 at high p).

**Success criterion:**
- |p_q − 0.454| < 0.01
- |ν − 1.622| < 0.10  (Ujfalusi 2014: 1.622±0.035)
- χ²/dof < 5.0
- b_err/b < 0.5  (correction amplitude well-constrained)

If p_q deviates by > 0.02 from 0.454: **stop, do not run BCC**. Check energy E in sc_dilution.py.

**Refit without re-running TMM:**
```bash
# Refit with default window (crossing-based):
pixi run python runs/sc_dilution.py --refit

# Force window to literature region [0.39, 0.52]:
pixi run python runs/sc_dilution.py --refit --p-min 0.39 --p-max 0.52

# Refit the old E=0 data (crossings at ~0.59, different symmetry class):
pixi run python runs/sc_dilution.py --refit --data-file data/sc_dilution.h5 --p-min 0.54 --p-max 0.64

# Test sensitivity to y:
pixi run python runs/sc_dilution.py --refit --y-irrel 1.04
pixi run python runs/sc_dilution.py --refit --y-irrel 1.20
```

Expected runtime: ~9 hours (same parameters as benchmark: L=10k, N_real=500, M=[4,6,8,10]).

---

### Documented failure modes for Step 3

**Failure A — p_q constant for all p (Λ_M flat, all same value)**

_Symptom_: All p values give the same Λ_M ≈ Λ_warmup. Λ_M table shows identical values in every row.

_Root cause_: JAX XLA disk cache bug. Python float `p` passed to `jax.jit` is treated as a compile-time constant baked into the XLA kernel. The disk cache key is based on abstract shapes/dtypes NOT concrete float values, so all p calls reuse the warmup-compiled kernel (p≈P_LIST[len//2]).

_Fix_: `src/tmm.py` `_tmm_core` must begin with:
```python
W = jnp.float64(W)
p = jnp.float64(p)
```
This converts Python floats to JAX abstract tracers, preventing constant-folding. Verify fix is in place before running.

_Verify_: Check M=10 range. Correct: λ_M ∈ [0.047, 0.435]. Broken: λ_M ∈ [0.435, 0.435].

---

**Failure B — p_q ≈ 0.59–0.65, χ²/dof >> 5, ν > 2**

_Symptom_: "Rough p_q estimate (minimum spread): 0.598" and FSS fit in window [0.52, 0.68] gives p_q=0.589, χ²/dof=27.

_Root cause_: The minimum spread algorithm (`np.argmin(std(Λ_M over M))`) finds the global minimum of spread across all M. With corrections-to-scaling b·M^{-y} (y=1.12, up to 20% at M=4), individual pairwise crossings are staggered:
- M4–M6 crossing: ~p_q + 0.13 (large shift at small M)
- M6–M8 crossing: ~p_q + 0.06
- M8–M10 crossing: ~p_q + 0.03

The global std minimum lands at ~0.60 where corrections accidentally equalize all four Λ_M values — NOT at the actual quantum critical point. The FSS is then fit in the wrong window.

_Fix_: `runs/sc_dilution.py` now uses `_find_crossings` (pairwise sign-change detector with linear interpolation) as the primary estimator. Minimum spread kept only as fallback. If running an old version of the script, use:
```bash
pixi run python runs/sc_dilution.py --refit --p-min 0.38 --p-max 0.52
```

_Distinguish from Failure A_: In Failure B, the Λ_M TABLE shows variation (Λ_M changes with p). In Failure A, every row is identical.

---

**Failure C — Classical p_c check fails: measured 0.20 or 0.32+**

- 0.20: periodic BC bug in `_wraps` — used periodic neighbor function, making z=0 adjacent to z=L-1; any occupied pair straddling that boundary gave a false spanning cluster. Fix: use `_neighbors_sc_nopbc` (checking bounds: `0 ≤ ni < L`).
- 0.32+: tolerance too tight. Finite-size shift p_c(L=20) ≈ p_c(∞) + 0.013. Tolerance must be ±0.025.

Both are fixed in the current `src/percolation.py`.

---

**Failure D — p_q ≈ 0.59, ν ≈ 2.0, χ²/dof ≈ 10, b ≈ 0 (chiral-symmetry fixed point)**

_Symptom_: Crossings tight at [0.589, 0.592, 0.594], p_q=0.591±0.001, ν=1.99±0.03, b_err/b≈7 (b
consistent with zero), χ²/dof≈10.

_Root cause_: This is NOT a code bug — it is correct physics at E=0. The SC lattice is bipartite
(sublattice symmetry: H → −H). At E=0, this places the system in symmetry class AIII/BDI (chiral
orthogonal), not the standard 3D-AI class. Consequences:
- p_c^Q(E=0) > p_c^Q(E=0.1) because band-centre states are anomalously localized in the chiral class
- Ujfalusi (2014) shows p_c^Q(E→0)→1 and reports "severe convergence troubles" for MFSS at E<0.1
- ν≈2.0 is the chiral-class exponent, not 1.62 (AI class)
- b≈0 because the standard AI irrelevant correction is absent at the chiral fixed point
- χ²/dof≈10 has two contributions: (1) P_FINE grid covers [0.38,0.52] not [0.55,0.63]; (2) the
  standard AI scaling function Λ_M = Σa_ku^k + b·M^{-y} is the wrong ansatz for the chiral class

_This result was obtained and documented on 2026-05-04._ The E=0 data lives in `data/sc_dilution.h5`.

_Resolution_: Run at E=0.1 (current default). At E=0.1, Ujfalusi gives p_c^Q=0.454, ν=1.622, 3D-AI
class applies, and χ²/dof should fall below 5.

_Distinguish from Failure B_: In Failure D, the crossings are TIGHT (spread < 0.01) and land near
0.59; in Failure B (old minimum-spread code), the result was also ~0.59 but crossings were NOT found
by the algorithm (it used the wrong method).

---

## Step 4 — BCC quantum percolation (the novel measurement)

Only run after Step 3 passes. Same FSS model, same universality class (3D orthogonal).

```bash
pixi run bcc-dilution
```

Kill-safe: same checkpoint system.

**Expected output:**
```
  Rough p_q (pairwise crossings): 0.3xx  [individual: ['0.3xx', '0.3xx', '0.4xx']]
  FSS fit window: p ∈ [0.2xx, 0.4xx]

  p_q = 0.3xxxx ± 0.000xx (stat) ± 0.000xx (syst)   ← first measurement
  ν   = 1.6xxx  ± 0.0xxx (stat) ± 0.0xxx (syst)
  χ²/dof = x.xxx
  p_q/p_c(BCC) = p_q / 0.2457   ← compare to SC: 0.44/0.312 ≈ 1.41
```

**What to look for:**
- p_q(BCC) < p_q(SC) = 0.44 (expected: BCC is more connected, lower threshold)
- ν consistent with SC result within 2σ total (same universality class)
- χ²/dof < 5.0
- Harris criterion satisfied: 3ν > 2, i.e. ν > 0.667 (always true here)

**Refit without re-running TMM:**
```bash
pixi run python runs/bcc_dilution.py --refit
pixi run python runs/bcc_dilution.py --refit --p-min 0.28 --p-max 0.44
```

Expected runtime: 8–20 hours (complex128 for BCC is ~2× slower per step than SC).

**Failure modes**: Same as Step 3 (Failures A, B, C). Apply the same diagnostics. For Failure B on BCC, use `--p-min 0.28 --p-max 0.44` as the manual window (centered on expected p_q(BCC) ≈ 0.36).

---

## Step 5 — Full analysis and comparison table

```bash
pixi run analyse
```

Prints SC benchmark, SC percolation, BCC percolation side by side with universality
check (|ν_SC − ν_BCC| in σ units) and Harris criterion verification.

---

## Robustness checks (run before writing up)

```bash
# 1. Stability under polynomial order (poly=3 vs 4)
pixi run python runs/sc_benchmark.py --refit
# Compare to result from poly=3; W_c and ν should agree within stat errors.

# 2. Stability of BCC result under FSS window width
# Edit p_fit_hw in bcc_dilution.py: try 0.06 and 0.10, rerun --refit equivalent.

# 3. Stability under y value
pixi run python runs/sc_benchmark.py --refit --y-irrel 1.04
pixi run python runs/sc_benchmark.py --refit --y-irrel 1.20
# Reported automatically as syst error; spot-check that shifts match reported syst.

# 4. Verify L convergence: γ_min × L >> 1 for all M at the critical point.
# At p_q, ξ_M ~ M; γ_min ~ 1/M. L=100k >> M_max=12 — this is fine.
```

---

## Quick reference: what each file does

| File | Purpose |
|------|---------|
| `runs/sc_benchmark.py` | Anderson disorder benchmark; validates TMM+FSS against W_c=16.53, ν=1.571 |
| `runs/sc_dilution.py` | SC quantum percolation; validates against p_q≈0.44 (literature) |
| `runs/bcc_dilution.py` | BCC quantum percolation; **the novel measurement** |
| `analysis/fss_analysis.py` | Post-hoc comparison table and universality check |
| `src/tmm.py` | TMM core; MGS QR; make_tmm_fns factory |
| `src/fss.py` | FSS fitting; χ² minimisation; bootstrap CIs; corrections-to-scaling |
| `src/io_utils.py` | HDF5 save/load; load_lambda for offline refit |
| `data/checkpoints/` | Per-M and per-p .npz/.npy files; delete to force recompute |

---

## Checkpoint management

| Checkpoint | Created by | Survives kill at |
|-----------|-----------|-----------------|
| `data/checkpoints/sc_benchmark_M{M}.npz` | sc_benchmark | M fully complete |
| `data/checkpoints/sc_benchmark_M{M}_partial/W_{i}.npy` | sc_benchmark | after each W point |
| `data/checkpoints/sc_dilution_M{M}.npz` | sc_dilution | M fully complete |
| `data/checkpoints/sc_dilution_M{M}_partial/p_{i}.npy` | sc_dilution | after each p point |
| `data/checkpoints/bcc_dilution_M{M}.npz` | bcc_dilution | M fully complete |
| `data/checkpoints/bcc_dilution_M{M}_partial/p_{i}.npy` | bcc_dilution | after each p point |

To force a full rerun of one M: `rm data/checkpoints/{run}_M{M}.npz`
To force a full rerun of everything: `rm -rf data/checkpoints/ data/*.h5`
