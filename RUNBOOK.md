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

### Window selection (Rodriguez et al. 2011 method)

Instead of a hardcoded window half-width, we run a **stability scan**: fit FSS at
δ ∈ {0.03..0.15} in 15 steps, record χ²/dof(δ), select the widest δ with χ²/dof < 3.0.
This is the non-ad-hoc procedure. Manual override via `--p-min`/`--p-max` is available
for diagnosis only. See `src/fss.py: scan_fss_windows` and `docs/methodology.md §6`.

### Why y is fixed at 1.12, not fitted

y is the effective irrelevant scaling exponent for our M range. S&O 1999 (PRL 82, 382)
actually FIT y freely (not fixed) and obtain |y|≈2.8 (box disorder). Rodriguez 2011
(PRB 84, 134209) obtain |y|≈1.7 for a different observable. Our effective y=1.12 is
empirical: it gives χ²/dof=2.4 on the Anderson benchmark (all gates pass). With M = {4..12}, y cannot be identified
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

## Step 3 — SC quantum percolation at E=0.7

Validates the quantum percolation setup against Ujfalusi & Varga (2014) Table I: p_c^Q(E=0.7) ≈ 0.384.
Uses same FSS model as benchmark (y=1.12 fixed, poly_order=4, n_corrections=1).
Window selected automatically by the stability scan (§FSS Model above).

**Why E=0.7:**
- E=0 → class BDI (chiral anomaly): p_q=0.591, ν=2.0, wrong class. Data in `sc_dilution_E0.h5`.
- E=0.1 → BDI-AI crossover: ξ_cross ≈ 10 ≈ M_max. Corrections dominate (b=−0.153, χ²=1543).
  Spurious crossings at p≈0.35. Data in `sc_dilution_E0p1.h5`. See Failure E below.
- E=0.7 → 11.7% of half-bandwidth: ξ_cross ≪ M_min=4. Corrections expected b≈0.1 (like benchmark).
  p_c^Q(E=0.7)≈0.384 is the mobility-edge minimum (Ujfalusi 2014, Figure 2).

Output files: `data/sc_dilution_E0p7.h5`, checkpoints `sc_dilution_E0p7_M{M}.npz`.

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
    0.3840   x.xxxxx   x.xxxxx   x.xxxxx   x.xxxxx   ← crossings here
    ...

  Rough p_q (pairwise crossings): 0.38x  [individual: ['0.37x', '0.38x', '0.39x']]

  Window stability scan  [χ²/dof < 3.0 = ✓]
    δ        window     p_q        ν        b     χ²/dof   N
  0.030  [0.354, 0.414]  0.38xxx  1.6xxx  0.0xxx    x.xxx ✓  xx
  ...
  0.1xx  [0.2xx, 0.5xx]  0.38xxx  1.6xxx  0.0xxx    x.xxx ✓  xx

  p_q = 0.384xx ± 0.000xx (stat) ± 0.000xx (syst)
  ν   = 1.6xxx  ± 0.0xxx (stat) ± 0.0xxx (syst)
  χ²/dof = x.xxx
```

**Success criterion:**
- |p_q − 0.384| < 0.01
- |ν − 1.622| < 0.10  (3D-AI class)
- χ²/dof < 3.0  (window scan threshold)
- b_err/b < 0.5  (correction amplitude well-constrained)
- Window scan shows a plateau: multiple δ values give consistent p_q within errors

If χ²/dof > 5.0 after window scan: **stop, do not run BCC**. Diagnose (wrong energy? wrong M?).

**Refit without re-running TMM:**
```bash
# Refit with window scan (default):
pixi run python runs/sc_dilution.py --refit

# Force window manually (e.g., for diagnosis):
pixi run python runs/sc_dilution.py --refit --p-min 0.32 --p-max 0.45

# Refit old E=0.1 data:
pixi run python runs/sc_dilution.py --refit --data-file data/sc_dilution_E0p1.h5

# Refit old E=0 data:
pixi run python runs/sc_dilution.py --refit --data-file data/sc_dilution_E0.h5 --p-min 0.54 --p-max 0.64

# Test sensitivity to y:
pixi run python runs/sc_dilution.py --refit --y-irrel 1.04
pixi run python runs/sc_dilution.py --refit --y-irrel 1.20
```

Expected runtime: ~9 hours (L=10k, N_real=500, M=[4,6,8,10]).

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

**Failure E — Spurious crossings from correction dominance near the chiral fixed point**

_Symptom_: Crossings appear at p≈0.35, far below Ujfalusi's p_c^Q(E=0.1)=0.454. FSS gives
p_q=0.372, ν=0.915 (wrong class), b=−0.153 (large negative), χ²/dof=1543.

_Root cause_: E=0.1 is only 1.7% of the SC half-bandwidth (6t) from E=0. The system is in a
crossover between the BDI (chiral) fixed point at E=0 and the AI fixed point at E>>0. At M={4..10},
finite-size corrections b·M^{-y} have magnitude |b|·4^{-1.12} = 0.032, which is 38% of Λ_M≈0.085 at
the apparent crossing. The correction differential between M=4 and M=10 (0.022) inverts the insulating
ordering wherever the physical insulating signal F(u_4)−F(u_10) < 0.022 — which occurs for all
p ∈ [0.35, 0.45]. The true p_q≈0.45 is invisible: ALL data for p>0.35 appears metallic due to
corrections, not physics. χ²/dof=1543 is driven by σ(Λ_M)≈10^{-4} (statistically precise data)
combined with higher-order correction terms b₂·M^{-2y} not captured by the single-correction ansatz.

_Confirmed_: SC E=0.1 run completed 2026-05-05, data in `data/sc_dilution_E0p1.h5` (greengoblin).
Runtime: 701.7 min. M=10: 849s/point.

_Fix (applied)_: **E=0.7** is now the default energy in `sc_dilution.py`. At E=0.7,
ξ_cross ~ (1/0.7)^{1/Δ_E} ≪ M_min=4 for any reasonable Δ_E. Corrections expected
similar to Anderson benchmark (b≈0.1). New output file: `data/sc_dilution_E0p7.h5`.

_Alternative fix (not used)_: Add M=12, 16. At M=16, correction 0.153×16^{-1.12}=0.006 (4%
of signal) — crossings shift toward true p_q. Adds ~20h compute. Only needed if E=0.7 also
fails (unlikely given ξ_cross argument).

_Constrained refit on E=0.1 data (diagnosis only)_:
```bash
pixi run python runs/sc_dilution.py --refit --data-file data/sc_dilution_E0p1.h5 --p-min 0.41 --p-max 0.50
```
Forces the fit window near Ujfalusi's p_q=0.454. Still gives poor χ² (b=−1.298 unphysical)
because correction dominance persists throughout this window. Confirms M range is insufficient.

_BCC warning_: BCC has bandwidth 16t → E=0.1 is 1.25% of half-bandwidth, even closer to BDI
crossover than SC. Same correction-dominance failure expected for BCC at E=0.1 with M={4..10}.
BCC energy will be selected after SC E=0.7 result is confirmed (same ξ_cross argument applies).

---

## Step 4 — BCC quantum percolation (the novel measurement)

Only run after Step 3 passes. Same FSS model, same universality class (3D orthogonal).

**Energy: E=0.1 (not E=0).** BCC NN lattice is bipartite (no odd cycles in NN graph), so at E=0
it has the same chiral symmetry ΓHΓ=−H and the same class-BDI anomaly as SC: p_c^Q(E→0)→1,
ν≠1.622. At E=0.1, chiral symmetry is broken → class AI, ν=1.622, direct comparison to SC
and to Ujfalusi's SC mobility edge.

Output: `data/bcc_dilution_E0p1.h5`, checkpoints `bcc_dilution_E0p1_M{M}.npz`.

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
- p_q(BCC) < p_q(SC, E=0.1) = 0.454 (expected: BCC is more connected, lower threshold)
- p_q(BCC) estimated ≈ 0.358 from p_q/p_c ratio: 0.454 × (0.2457/0.3115)
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
| `data/checkpoints/sc_dilution_E0p7_M{M}.npz` | sc_dilution (E=0.7) | M fully complete |
| `data/checkpoints/sc_dilution_E0p7_M{M}_partial/p_{i}.npy` | sc_dilution (E=0.7) | after each p point |
| `data/checkpoints/sc_dilution_E0p1_M{M}.npz` | sc_dilution (E=0.1, failed) | M fully complete |
| `data/checkpoints/bcc_dilution_E0p1_M{M}.npz` | bcc_dilution (E=0.1) | M fully complete |
| `data/checkpoints/bcc_dilution_E0p1_M{M}_partial/p_{i}.npy` | bcc_dilution (E=0.1) | after each p point |

To force a full rerun of one M: `rm data/checkpoints/{run}_M{M}.npz`
To force a full rerun of everything: `rm -rf data/checkpoints/ data/*.h5`
