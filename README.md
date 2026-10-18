# qm_anderson

Transfer-matrix calculation of the 3D Anderson transition and of quantum percolation on cubic lattices.

The sample is a bar, cross section M by M, length L. The wavefunction is stepped from slice to slice by a transfer matrix. A QR factorization at each step keeps the product from overflowing, and the smallest Lyapunov exponent γ_min is the one that decays slowest. The quantity we scale is Λ_M = 1 / (M γ_min). The critical point is the value of the control parameter at which Λ_M for several M fall on one curve.

Anderson disorder means every site is occupied and the on-site energy is drawn uniformly from −W/2 to W/2, at the band centre. Quantum percolation means no on-site disorder: each site is present with probability p, and the energy E is held fixed.

The simple-cubic runs have been done. The body-centered cubic lattice is in the code and has not been run.

## Where things stand

18 October 2026. The simple-cubic Anderson benchmark is inside the gate in RUNBOOK.md. Site dilution at E = 3.1t has been computed through M = 12, and the crossings of Λ_M are still moving toward the published threshold. The May draft quotes a separate two-width fit, p_q = 0.517. The crossings are the measurement. The body-centered run has not been started.

Numbers and the next run: docs/next_steps.md. Papers: docs/literature.md. Failed runs: RUNBOOK.md. docs/manuscript/ is the draft from May 2026.

## Running it

Linux, CUDA 12, and [pixi](https://pixi.sh).

```bash
pixi install
pixi run test
```

The Anderson benchmark is hours on an RTX 5090. The arithmetic is double precision.

```bash
pixi run benchmark
```

Dilution data are already in `data/sc_dilution_E3p1.h5`: E = 3.1, L = 10000, 500 samples, M = 4, 6, 8, 10, 12. Refit without another transfer-matrix run:

```bash
pixi run python runs/sc_dilution.py --refit \
    --data-file data/sc_dilution_E3p1.h5 --m-list 8,10,12
```

A killed job continues from `data/checkpoints/`.

## Files

```
src/         transfer matrix, finite-size scaling, both lattices
runs/        sc_benchmark.py, sc_dilution.py, bcc_dilution.py
analysis/    tables and figures
tests/
docs/methodology.md
docs/next_steps.md
docs/literature.md
RUNBOOK.md
```
