# Runbook: SC benchmark → SC dilution → BCC dilution

Complete sequence to reproduce the benchmark, validate, and run the novel BCC measurement.
Each step has a clear success criterion. Do not proceed to the next step until it is met.

---

## Step 0 — Pull latest code on the GPU machine

```bash
cd ~/proj/petya/simulations/qm_anderson
git pull
```

Fixes applied in this pull:
- `src/tmm.py`: always use MGS QR (removes cuSOLVER alloc/free bottleneck for M≥8)
- `src/fss.py`: `x_range` parameter to restrict FSS fit to a window near x_c
- `runs/sc_benchmark.py`: narrowed FSS window [15.0, 18.5], fixed PASS criterion (checks W_c AND ν AND χ²/dof), added `--refit` flag, per-W checkpointing
- `runs/sc_dilution.py`: per-M and per-p checkpointing, adaptive FSS window
- `runs/bcc_dilution.py`: same as sc_dilution

---

## Step 1 — Refit FSS on existing benchmark data (no TMM rerun)

The sc_benchmark.h5 file already exists with all raw Lyapunov data. The previous FSS
fit failed because the W range [13, 20] was too wide. This refits using only [15.0, 18.5].

```bash
pixi run refit
```

**Success criterion:**
```
W_c = 16.xx ± 0.xx   (must be within 0.30 of 16.53)
ν   = 1.5xx ± 0.xxx  (must be within 0.10 of 1.571)
χ²/dof = x.xx        (must be < 5.0; ideally < 2.0)
```

If χ²/dof is still >> 1, try a narrower window:
```bash
pixi run python runs/sc_benchmark.py --refit --w-min 15.5 --w-max 18.0
```

If W_c is still off by > 0.30 after narrowing: **stop, do not proceed, investigate tmm.py**.

---

## Step 2 — Re-run sc_benchmark with fixed MGS (cleans M=8 and M=10 data)

The existing h5 data for M=8 and M=10 was computed with the cuSOLVER bug (2666 s/pt
instead of ~100 s/pt). The data itself is statistically correct (just slow to produce),
but re-running is advisable for completeness and for clean runtime estimates.

Delete only the M=8 and M=10 checkpoints so the script re-runs just those:
```bash
rm data/checkpoints/sc_benchmark_M8.npz
rm data/checkpoints/sc_benchmark_M10.npz
pixi run benchmark
```

M=4 and M=6 will load from checkpoint (seconds). M=8 and M=10 will recompute with MGS
(expected ~20–80 s/pt instead of 2666 s/pt — a 30–130× speedup).

Expected total runtime with MGS: **30–90 minutes** (vs 11 hours before).

**Success criterion** (printed at end):
```
PASS: W_c, ν, and χ²/dof all within tolerance. Proceed to sc_dilution.py.
```

If it prints FAIL, read the specific failure lines and debug before continuing.

---

## Step 3 — SC quantum percolation (validation against literature)

Validates the quantum percolation setup against the known SC result p_q ≈ 0.44.

```bash
pixi run sc-dilution
```

This runs M ∈ {4, 6, 8, 10, 12} at L=100,000 and N_real=2,000.
Expected runtime: **4–10 hours** (L is 10× longer than benchmark, N_real is 4×).

Kill-safe: per-M and per-p checkpoints are saved throughout. Resume with same command.

**Success criterion:**
```
p_q(SC) = 0.44xx ± 0.00x   (must be within 0.01 of 0.44)
ν       = 1.5x  ± 0.0x     (consistent with 1.571 or 1.622 within 2σ)
χ²/dof  = x.xx             (must be < 3.0)
```

If p_q deviates by > 0.02: **stop, do not run BCC**. The transfer matrix or RNG is wrong.

---

## Step 4 — BCC quantum percolation (the novel measurement)

Only run this after Step 3 passes.

```bash
pixi run bcc-dilution
```

This runs M ∈ {4, 6, 8, 10} on the BCC lattice (complex128, ~2× slower than SC per step).
Expected runtime: **8–20 hours**.

Kill-safe: same checkpoint system as sc_dilution.

**What to look for in the output:**
- `Rough p_q(BCC) estimate: 0.3xx` — this is the visual crossing. Should be below p_q(SC)=0.44.
- `p_q(BCC) = 0.3xxxx ± 0.00xxx` — the FSS result. This is the novel number.
- `ν = 1.5x ± 0.0x` — should be consistent with SC result (same universality class).
- `χ²/dof < 3.0` — confirms the scaling collapse is good.
- `p_q/p_c(BCC) = p_q / 0.2457` — ratio; compare to p_q/p_c(SC) = 0.44/0.312 ≈ 1.41.

---

## Step 5 — Full FSS analysis and comparison table

After both dilution runs complete:

```bash
pixi run analyse
```

Prints a comparison table: SC benchmark, SC percolation, BCC percolation side by side,
with universality check (|ν_SC - ν_BCC| in σ units) and Harris criterion (3ν > 2).

---

## Robustness checks (run before writing up)

These verify the result is not an artifact of fitting choices:

```bash
# 1. Stability under polynomial order
pixi run python runs/sc_benchmark.py --refit --w-min 15.0 --w-max 18.5
# check that W_c and ν do not shift when poly_order=2 or 4 is used (edit fss.py temporarily)

# 2. Stability of BCC result under FSS window width
# Edit p_fit_hw in bcc_dilution.py from 0.08 to 0.06 and 0.10, rerun analysis/fss_analysis.py

# 3. Verify L convergence (bar length is long enough)
# For M=10 the criterion is L >> 1/γ_min ≈ ξ_M. At p_q, ξ_M ~ M = 10 lattice spacings.
# L = 100,000 >> 10 — this is fine. But check: γ_min × L >> 10 for all M.
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
| `src/fss.py` | FSS fitting; χ² minimisation; bootstrap CIs |
| `src/io_utils.py` | HDF5 save/load; load_lambda for offline refit |
| `data/checkpoints/` | Per-M and per-p .npz/.npy files; delete to force recompute |

---

## Checkpoint management

| Checkpoint | Created by | Survives kill at |
|-----------|-----------|-----------------|
| `data/checkpoints/sc_benchmark_M{M}.npz` | sc_benchmark | M fully complete |
| `data/checkpoints/sc_benchmark_M{M}_partial/p_{i}.npy` | sc_benchmark | after each W point |
| `data/checkpoints/sc_dilution_M{M}.npz` | sc_dilution | M fully complete |
| `data/checkpoints/sc_dilution_M{M}_partial/p_{i}.npy` | sc_dilution | after each p point |
| `data/checkpoints/bcc_dilution_M{M}.npz` | bcc_dilution | M fully complete |
| `data/checkpoints/bcc_dilution_M{M}_partial/p_{i}.npy` | bcc_dilution | after each p point |

To force a full rerun of one M: `rm data/checkpoints/{run}_M{M}.npz`
To force a full rerun of everything: `rm -rf data/checkpoints/ data/*.h5`
