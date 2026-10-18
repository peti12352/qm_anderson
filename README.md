# qm_anderson

Transfer-matrix study of the 3D Anderson transition and of quantum percolation.

A quasi-1D bar, cross section \(M \times M\), length \(L\). The smallest Lyapunov exponent \(\gamma_{\min}\) comes from a QR-stabilized product of slice transfer matrices. The scaling variable is \(\Lambda_M = 1/(M \gamma_{\min})\). Critical points are read from finite-size scaling of \(\Lambda_M\) across several \(M\).

Two disorders. Anderson: every site occupied, on-site energies uniform in \([-W/2, W/2]\), band centre. Quantum percolation: \(W = 0\), each site occupied with probability \(p\), spectrum taken at a fixed energy \(E\).

Simple cubic is running. Body-centered cubic is implemented and has not been measured.

## Status

18 October 2026. The simple-cubic Anderson benchmark passes the gate in `RUNBOOK.md`. Site dilution on the same lattice at \(E = 3.1t\) has been run through \(M = 12\). The crossings move toward the known threshold and are not there yet. A two-width fit that lands at \(p_q = 0.517\) is not a measurement. BCC has no output file. What would be worth publishing, and the run that has to come first, is `docs/next_steps.md`. Papers are `docs/literature.md`. What failed, and why, is `RUNBOOK.md`.

`docs/manuscript/` is a draft from May 2026. Do not cite it.

## Run

Linux, CUDA 12, [pixi](https://pixi.sh).

```bash
pixi install
pixi run test
```

Anderson benchmark, hours on an RTX 5090 in FP64:

```bash
pixi run benchmark
```

The dilution data are already in `data/sc_dilution_E3p1.h5` (\(E = 3.1\), \(L = 10^4\), 500 samples, \(M = 4,6,8,10,12\)). Refit without another transfer-matrix run:

```bash
pixi run python runs/sc_dilution.py --refit \
    --data-file data/sc_dilution_E3p1.h5 --m-list 8,10,12
```

A killed run resumes from `data/checkpoints/`.

## Layout

```
src/         transfer matrix, finite-size scaling, SC and BCC lattices
runs/        sc_benchmark.py, sc_dilution.py, bcc_dilution.py
analysis/    tables and figures
tests/
docs/methodology.md
docs/next_steps.md
docs/literature.md
RUNBOOK.md
```
