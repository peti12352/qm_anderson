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

## Step 3 — SC quantum percolation

Validates the quantum percolation setup against the known SC result p_q ≈ 0.44.
Uses same FSS model as benchmark (y=1.12 fixed, poly_order=4, n_corrections=1).

```bash
cd ~/proj/petya/simulations/qm_anderson
git pull
pixi run sc-dilution
```

Kill-safe: per-M and per-p checkpoints written throughout. Resume with same command.

**Expected output:**
```
y   = 1.12 (fixed)
p_q = 0.44xxx ± 0.000xx (stat) ± 0.000xx (syst)
ν   = 1.6xxx ± 0.0xxx (stat) ± 0.0xxx (syst)
χ²/dof = x.xxx
```

**Success criterion:**
- |p_q − 0.44| < 0.01
- |ν − 1.571| < 0.10  (or consistent with 1.62 ± 0.035 from Ujfalusi 2014)
- χ²/dof < 5.0
- b_err/b < 0.5  (correction amplitude well-constrained)

If p_q deviates by > 0.02: **stop, do not run BCC**. The transfer matrix or RNG is wrong.

Expected runtime: 4–10 hours (L=100k is 10× longer than benchmark, N_real=2000 is 4×).

---

## Step 4 — BCC quantum percolation (the novel measurement)

Only run after Step 3 passes. Same FSS model, same universality class (3D orthogonal).

```bash
pixi run bcc-dilution
```

Kill-safe: same checkpoint system.

**Expected output:**
```
y   = 1.12 (fixed; 3D orthogonal class)
p_q = 0.3xxxx ± 0.000xx (stat) ± 0.000xx (syst)   ← first measurement
ν   = 1.6xxx ± 0.0xxx (stat) ± 0.0xxx (syst)
χ²/dof = x.xxx
p_q/p_c(BCC) = p_q / 0.2457   ← compare to SC: 0.44/0.312 ≈ 1.41
```

**What to look for:**
- p_q(BCC) < p_q(SC) = 0.44 (expected: BCC is more connected, lower threshold)
- ν consistent with SC result within 2σ total (same universality class)
- χ²/dof < 5.0
- Harris criterion satisfied: 3ν > 2, i.e. ν > 0.667 (always true here)

Expected runtime: 8–20 hours (complex128 for BCC is ~2× slower per step than SC).

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
