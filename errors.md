
✓ done in 2866s | GPU executing 97% util | VRAM 24721/32667 MiB | 44°C | 146.7W
M=10 | resumed 25/40 W values from partial checkpoint
M=10: 100%|██████████| 40/40 [10:22:06<00:00, 2866.28s/W, W=20.00, lam=0.338±0.000, real_s=0, s_pt=2666.2s]
M=10 finished | 622.1 min | lambda_M in [0.338, 0.474]
✓ checkpoint saved -> data/checkpoints/sc_benchmark_M10.npz

Raw data collected in 660.6 min
/home/v/proj/petya/simulations/qm_anderson/src/fss.py:133: UserWarning: chi^2/dof = 28.53 >> 1, FSS fit is poor. Consider: (1) wider x range, (2) more system sizes, (3) higher poly_order, (4) corrections to scaling.
  warnings.warn(

*****************************************************
FSS RESULT (SC Anderson disorder benchmark)
  W_c = 16.2124 ± 0.0624 (expected = 16.53)
  v = 1.5475 ± 0.0828 (expected = 1.571)
  chi^2/dof = 28.533 (ideal = 1.0)
  converged: True
*****************************************************

PASS: benchmark within tolerance. Proceed to sc_dilution.py

Data saved to data/sc_benchmark.h5
(base) v@greengoblin:~/proj/petya/simulations/qm_anderson$ git add .
(base) v@greengoblin:~/proj/petya/simulations/qm_anderson$ git status
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean

---

## Analysis

### 1. χ²/dof = 28.53 — the FSS fit is broken

This is the primary problem. With n_data = 4 M-values × 40 W-values = 160 points and
n_params = 6 (x_c, ν, a₀..a₃), dof = 154, giving total χ² = 28.53 × 154 ≈ 4394.
A good collapse has χ²/dof ≈ 1.

Root cause: the W range [13, 20] is far too wide for FSS to be valid. The scaling
hypothesis Λ_M(W) = F[(W − W_c) M^{1/ν}] only holds near W_c where the dimensionless
scaling variable u = (W − W_c) M^{1/ν} is small. For M=10 and ν≈1.57:

  |u| ≈ |W − W_c| × 10^{1/1.57} ≈ |W − W_c| × 4.6

At the edge of the W range, |W − W_c| ≈ 3.5, so |u| ≈ 16 — deep into the nonlinear
regime where a cubic polynomial cannot capture F(u). The far-from-critical data points
dominate the χ² and pull x_c away from the true value.

Fix: narrow the W range. Use [15.0, 18.5] as a first pass; then tighten further once
x_c is known more precisely. Rerun fss_analysis.py on the existing h5 data without
re-running the TMM (the raw Lyapunov samples are already saved).

### 2. W_c = 16.21 ± 0.06 vs expected 16.53 — 5.1σ discrepancy

The expected value (Slevin & Ohtsuki 2014) is 16.53 ± 0.05. The benchmark gives
16.21 ± 0.06: a 0.32 shift, ~5σ away. This is not a statistical fluctuation.

Most likely cause: the wide W range bias described above. With χ²/dof = 28.53 the
fit minimum does not correspond to the true scaling collapse — it is finding a compromise
that minimises the misfit over the entire [13, 20] range, which biases x_c downward
(the far-metallic region W ∈ [13, 15] pulls the inferred crossing left).

Secondary cause: corrections to scaling are not included. The full form is:

  Λ_M = F[(W − W_c) M^{1/ν}] + b M^{-y}

with y ≈ 1.0 (Slevin & Ohtsuki 1999). Without the correction term, systematic
finite-size effects absorb into x_c and ν. Adding poly_order correction
(or fitting y explicitly with 5+ system sizes) would reduce this bias.

After narrowing the W range, if W_c is still off by >3σ, suspect a transfer matrix
bug and check against the checklist in lattice.py before proceeding to BCC.

The PASS verdict is misleading: the tolerance check only tests |ν − 1.571| < 0.10
and ignores W_c. ν = 1.5475 passes because the error bar is 0.083, but W_c being
off by 5σ is a clear failure. Do not proceed to sc_dilution.py yet.

### 3. M=10 performance: 2666 s/point — the cuSOLVER bug is back

Expected from tmm.py comments: "~10–40 s/point across all M." Actual for M=10: 2666 s.
That is 66–260× slower than expected.

Root cause is the dispatch threshold in tmm.py line 123:

  _orthogonalize = _qr_mgs if dim <= 100 else _qr_cusolver

For M=10, dim = 2N = 2×100 = 200 > 100, so it falls back to _qr_cusolver
(jnp.linalg.qr). This reintroduces the original cuSOLVER alloc/free bottleneck:
cuSOLVER allocates/frees a workspace buffer on every fori_loop iteration.
At L=10,000 steps and ~40 ms/call overhead, that is 400 s of pure overhead per
realization. With B=500 vmap realizations sharing one fori_loop, the batched QR is
called 10,000 times, each time incurring the cuSOLVER overhead: 10,000 × 40 ms = 400 s
of overhead before any compute, consistent with the observed 2666 s total.

The crossover analysis in the code comments is wrong. It estimates MGS O(dim³) vs
cuSOLVER O(dim³) compute, but misses that cuSOLVER overhead is independent of dim:
40 ms × L = 400 s regardless of whether dim is 100 or 1000. MGS at dim=200 via
lax.scan costs ~200³ = 8M FLOPs per QR × 10,000 steps × 500 vmap = 40 TFLOP, which
at 200 TFLOP/s FP64 (RTX 5090) takes ~200 s — already faster than the cuSOLVER path,
before any overhead savings.

Fix in tmm.py: lower the threshold or remove it entirely (always use MGS):

  _orthogonalize = _qr_mgs   # always; cuSOLVER overhead dominates for any dim at L=10k

This will require a rerun of M=10 (and M=8, which also uses cuSOLVER at dim=128).
Expected speedup: 10–30× for M=10, bringing it from 2666 s/pt to ~100–270 s/pt.

Total runtime consequence: the 660.6 min benchmark would drop to roughly 60–100 min.

### 4. M=10 λ_M range [0.338, 0.474] — physically plausible but narrow

For a good FSS collapse, ideally the metallic-side Λ_M (W near 13) should be clearly
above the insulating-side (W near 20) with a visible crossing. The range [0.338, 0.474]
is narrow (ratio ~1.4×), suggesting M=10 is not far enough into either phase at the
edges of the W range to strongly constrain the collapse.

This is expected given the narrow FSS window for large M: the system sees the finite
bar width before it feels the bulk transition. Extending M to 12 or 14 would widen
the range, but requires fixing the cuSOLVER issue first (M=12 at dim=288 would be
even slower on the cuSOLVER path).

### 5. Checkpoint resume worked correctly

25/40 W values loaded from the partial checkpoint for M=10. The kill at ~63% of M=10
(not 90% — either the earlier kill was at a different point, or the progress bar
percentage was of the full benchmark, not M=10's W loop). The per-W .npy files
persisted correctly and the resume was seamless.

---

## Action items (in order)

1. **Re-run FSS on existing h5 data** with W range [15.0, 18.5] and poly_order=3.
   No new TMM runs needed — the raw Lyapunov samples are in sc_benchmark.h5.
   This will immediately show whether χ²/dof drops to ~1 and W_c corrects to 16.5.

2. **Fix MGS threshold** in tmm.py: always use `_qr_mgs`. Rerun M=8 and M=10 to get
   clean data at the correct performance (and collect more realizations if time allows).

3. **Fix PASS criterion** in sc_benchmark.py: add a W_c check, e.g.
   `abs(result.x_c - 16.53) > 0.20` should also trigger a WARNING.

4. **Only after χ²/dof ≈ 1 and W_c within 3σ of 16.53**: proceed to sc_dilution.py.
