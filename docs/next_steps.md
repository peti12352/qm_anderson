# Next steps

The measurement this project is built for is the quantum percolation threshold on the body-centered cubic lattice, at an energy where the bar is wider than the classical percolation length. A literature search on 18 October 2026 did not find that number. What the search did find is in docs/literature.md.

In the 3D orthogonal class the exponent ν is universal. Ujfalusi and Varga average it to 1.622, with the interval 1.587 to 1.658, on the simple cubic lattice. A body-centered run that returns ν near 1.6 is a check that the symmetry class is the ordinary orthogonal one. The quantity that depends on the lattice is p_q.

The ratio of p_q to the classical threshold moves when the coordination number changes. It is a material parameter, in the same way p_q itself is.

## Simple cubic, Anderson disorder

Every site occupied, band centre, M = 4, 6, 8, 10, L = 10000, 500 samples. The fit recorded in RUNBOOK.md, with the irrelevant exponent held at y = 1.12:

```
W_c      = 16.6174 ± 0.0156 (stat) ± 0.0148 (syst)
ν        = 1.6205 ± 0.0085 (stat) ± 0.0052 (syst)
b        = 0.0958 ± 0.0045
χ²/dof   = 2.421
```

The gate, taken from Slevin and Ohtsuki, Phys. Rev. Lett. 82, 382 (1999), is a critical disorder within 0.30 of 16.54 and an exponent within 0.10 of 1.57. This fit is inside the gate. Both numbers sit a little high, which is the small-M bias already described in the runbook. y = 1.12 is an effective exponent for these sizes. Slevin and Ohtsuki fit y on their own data and get a magnitude around 2.8 to 3.9.

## Simple cubic, site dilution

E = 3.1t, no on-site disorder, same length and sample count, M = 4, 6, 8, 10, 12. The file is `data/sc_dilution_E3p1.h5`. Crossings of Λ_M read from that file, for p between 0.40 and 0.62:

```
M = 4 with M = 6     no crossing
M = 6 with M = 8     no crossing
M = 8 with M = 10    0.4822
M = 10 with M = 12   0.5022
```

Ujfalusi and Varga, Phys. Rev. B 90, 174203 (2014), Table II, at this energy: p_c^Q = 0.50628, interval 0.50606 to 0.50647. They use multifractal finite-size scaling on cubes, and the smallest cube they keep at this energy has side 20. Our crossings move from 0.482 to 0.502, toward 0.506. At p = 0.50628 the measured values are Λ_12 = 0.55522 and Λ_10 = 0.55194, so the widest pair has already crossed, about 0.004 below their threshold. Taking the percolation length as (p − 0.3116) to the power −0.876, M = 12 is about 2.9 of those lengths.

A fit on M = 10 and M = 12 only, window 0.46 to 0.54, y fixed, started from a guess of 0.502, gives

```
p_q      = 0.51670 ± 0.00456 (stat) ± 0.00450 (syst)
ν        = 1.599 ± 0.111 (stat) ± 0.112 (syst)
χ²/dof   = 1.068
```

0.517 lies above the M = 10, 12 crossing and above 0.506. The crossings are still rising toward 0.506. A chi-squared near 1, with a fourth-order polynomial and only two widths, means those two curves have been interpolated. No body-centered output file exists.

## Wider simple-cubic bar

Same energy, one more width, M = 14 or M = 16, L = 10000, 500 samples, with p concentrated between 0.48 and 0.54.

`runs/sc_dilution.py` already lists M = 8, 10, 12, 16. A width whose checkpoint is in `data/checkpoints/` is skipped. `scripts/run_m16_sc.sh` launches that script. Look at the checkpoint names before starting it, because a missing checkpoint for M = 8, 10, or 12 will recompute that width.

A new crossing near 0.506, with a three-width fit in the same place, means these sizes are large enough. A crossing that still shifts by about 0.01 means the bar is still too narrow, and a body-centered threshold at the same M would shift by a similar amount.

## Body-centered cubic

This run starts once the simple-cubic crossing has stopped drifting. Energy E = 3.1t. The nearest-neighbour body-centered lattice is bipartite, so the band centre is the chiral point, as it is on the simple cubic lattice. The half-bandwidth is 8t, and 3.1t is 0.39 of that.

The classical site threshold in the code is 0.2457. p_q is the output of the run, so the percolation length at threshold is not known until the crossings are in hand. They have to stop drifting as M grows. The file the script writes is `data/bcc_dilution_E3p1.h5`.

```bash
pixi run bcc-dilution
```

The face-centered cubic lattice is a separate calculation. Triangles in the nearest-neighbour graph put the band centre in the ordinary orthogonal class, with no need to move off E = 0. A slice uses a four-site cell, and at the same M the matrix costs about 64 times a simple-cubic slice. It is worth doing once a body-centered threshold is in hand.

## What a paper from the present data would be claiming

Ujfalusi and Varga mapped the simple-cubic mobility edge on cubes down to side 20. Our widest bar is M = 12. Their exponent, averaged along the edge, is 1.622. A paper whose main number was p_q = 0.517 would be setting a two-width fit, started from a guess, against a published interval of width 0.0004.
