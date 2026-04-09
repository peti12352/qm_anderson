# 3D Anderson Localization via Transfer Matrix Method

TMM + FSS study of the metal-insulator transition in 3D SC and BCC tight-binding
lattices with Anderson disorder and site dilution (quantum percolation).

**Novel results:**
- BCC quantum percolation threshold p_q — first-ever measurement
- SC quantum percolation threshold p_q — precision update of the 1992 result

## Physics in one paragraph

A tight-binding electron on a 3D lattice with random on-site energies undergoes
a metal-insulator transition (Anderson localization) at a critical disorder W_c.
On a site-diluted lattice (fraction 1−p of sites removed), the transition occurs
at a quantum percolation threshold p_q > p_c (classical). We extract p_q and the
universal exponent ν via the transfer matrix method on quasi-1D bars of cross-section
M×M, computing Lyapunov exponents with QR stabilization, then performing
finite-size scaling (Λ_M = ξ_M/M) across multiple M values.

## Requirements

- Linux x86-64, CUDA 12+
- [pixi](https://pixi.sh) ≥ 0.20 — handles all Python/CUDA dependencies

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Quick start

```bash
git clone git@github.com:peti12352/qm_anderson.git
cd qm_anderson
pixi install          # resolves conda-forge + CUDA wheels (~5 min first time)
pixi run test         # all 9 tests must pass before running
```

## Run sequence (RTX 5090)

Run in this order. Each step saves to `data/` and is resumable.

```bash
# 1. SC Anderson benchmark — validates code against known W_c = 16.53 t, ν = 1.571
pixi run benchmark
# Expected: ~2–3 h on RTX 5090 (FP64 limited)

# 2. SC site-diluted quantum percolation
pixi run sc-dilution
# Expected: ~3–4 h

# 3. BCC site-diluted quantum percolation (first measurement)
pixi run bcc-dilution
# Expected: ~4–6 h (BCC has larger V matrix)

# 4. Generate all publication figures
pixi run figures

# 5. Print FSS summary table (W_c, p_q, ν, Harris criterion)
python analysis/fss_analysis.py
```

For a fast sanity check before committing to full runs:

```bash
python runs/sc_benchmark.py --fast
# Runs M=4,6 only, L=10k, N_real=200. Takes ~5 min. Should show crossing near W=16.
```

## Project structure

```
src/
  config.py      — JAX x64 + GPU init, ABSENT_ENERGY constant
  lattice.py     — SC/BCC transfer matrix factory (JIT-compiled)
  tmm.py         — QR-stabilized TMM, lambda_M extractor
  fss.py         — FSS fitting (χ²-minimization + bootstrap CIs)
  percolation.py — Union-find for classical p_c sanity checks
  io_utils.py    — HDF5 save/load with raw gammas
runs/
  sc_benchmark.py   — W scan, M∈{4,6,8,10,12}
  sc_dilution.py    — p scan, SC lattice
  bcc_dilution.py   — p scan, BCC lattice
analysis/
  figures.py        — 7 publication-quality PDF figures
  fss_analysis.py   — offline FSS + universality comparison
tests/
  test_tmm.py       — 9 unit/integration tests
```

## Run time estimates (RTX 5090, FP64)

The RTX 5090 has 32 GB VRAM but FP64 throughput is ~1/64th of FP32 (consumer GPU).
Estimated times assume B=2000 realizations per (M, x) point, L=100k.

| Run | M values | Points | Est. time |
|-----|----------|--------|-----------|
| SC benchmark | 4–12 (5 sizes) | 40 W values | 2–3 h |
| SC dilution  | 4–12 (5 sizes) | 50 p values | 3–4 h |
| BCC dilution | 4–10 (4 sizes) | 50 p values | 4–5 h |
| **Total** | | | **~10–12 h** |

Memory per (M, B) batch: `B × (2M²)² × 16 bytes` (Q matrices only, via fold_in).
For M=12, B=500: 500 × 576 × 16 B ≈ 4.6 MB — fits comfortably.

## Reproduction checklist

- [ ] `pixi run test` — all 9 tests pass
- [ ] SC benchmark: |W_c − 16.53| < 0.3, |ν − 1.571| < 0.05
- [ ] SC dilution: p_q > p_c = 0.3115 (quantum > classical)
- [ ] BCC dilution: p_q > p_c = 0.2457
- [ ] Harris criterion: 3ν > 2 for all runs
- [ ] p_q/p_c ratio consistent between SC and BCC (universality test)
