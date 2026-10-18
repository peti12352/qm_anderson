# Next steps

The number worth having is the quantum percolation threshold on the body-centered cubic lattice, at one energy where the bar is wider than the classical correlation length. As of 18 October 2026 that number is not in the literature. See `literature.md`.

\(\nu\) on BCC is not that number. In the 3D orthogonal class \(\nu\) is universal. Ujfalusi and Varga already average it to \(1.622\ (1.587..1.658)\) on the simple cubic lattice. A BCC exponent near \(1.6\) is a check that the run is in the right class. The result is \(p_q\).

\(p_q / p_c\) is not a universality test. Coordination can move the threshold and the classical \(p_c\) by different amounts.

## What is on disk

Anderson benchmark, simple cubic, \(p = 1\), \(E = 0\), \(M \in \{4,6,8,10\}\), \(L = 10^4\), 500 samples. From `RUNBOOK.md`, canonical fit, \(y = 1.12\) fixed:

```
W_c = 16.6174 ± 0.0156 (stat) ± 0.0148 (syst)
ν   = 1.6205 ± 0.0085 (stat) ± 0.0052 (syst)
b   = 0.0958 ± 0.0045
χ²/dof = 2.421
```

Gate against Slevin and Ohtsuki, Phys. Rev. Lett. **82**, 382 (1999): \(|W_c - 16.54| < 0.30\), \(|\nu - 1.57| < 0.10\). It passes. The high side is the small-\(M\) bias already in the runbook. \(y = 1.12\) is empirical. That paper fits \(y\) and gets \(|y| \approx 2.8\)–\(3.9\).

Site dilution, \(E = 3.1t\), \(W = 0\), same \(L\) and sample count, \(M \in \{4,6,8,10,12\}\). File: `data/sc_dilution_E3p1.h5`. Pairwise crossings of \(\Lambda_M\), recomputed from that file:

| pair | crossing in \(p \in [0.40, 0.62]\) |
|---|---|
| 4–6 | none |
| 6–8 | none |
| 8–10 | 0.4822 |
| 10–12 | 0.5022 |

Ujfalusi and Varga, Phys. Rev. B **90**, 174203 (2014), Table II, this energy: \(p_c^Q = 0.50628\ (0.50606..0.50647)\), from multifractal finite-size scaling on cubes with \(L_{\min} = 20\). The crossings move \(0.482 \to 0.502\), toward \(0.506\). At \(p = 0.50628\), \(\Lambda_{12} = 0.55522\) and \(\Lambda_{10} = 0.55194\): the largest pair has already crossed, about \(0.004\) below their value. With \(\xi_{\mathrm{perc}} = (p - 0.3116)^{-0.876}\), \(M = 12\) is still only \(\sim 2.9\,\xi_{\mathrm{perc}}\) there.

The draft fit on \(M \in \{10, 12\}\) only, window \([0.46, 0.54]\), \(y\) fixed, optimizer seeded at \(0.502\), gives

```
p_q = 0.51670 ± 0.00456 (stat) ± 0.00450 (syst)
ν   = 1.599 ± 0.111 (stat) ± 0.112 (syst)
χ²/dof = 1.068
```

That sits above the last crossing and above the literature value. The crossings are climbing toward \(0.506\) from below. The fit stepped over it. \(\chi^2\) near 1 with polynomial order 4 and two widths means the curves can be interpolated. Do not quote \(0.517\).

There is no `data/bcc_*.h5`.

## The run that calibrates the instrument

One more simple-cubic width at the same energy. \(M = 14\) or \(M = 16\), \(E = 3.1t\), \(L = 10^4\), 500 samples, \(p\) tight on \([0.48, 0.54]\).

`runs/sc_dilution.py` already lists \(M \in \{8, 10, 12, 16\}\). Finished widths resume from `data/checkpoints/`. `scripts/run_m16_sc.sh` launches that script. Read the checkpoint names before starting it, or it will redo \(M = 8, 10, 12\).

If the new crossing sits on \(0.506\) and a three-width fit does too, the pipeline measures \(p_q\). If the crossing is still moving by \(\sim 0.01\), it does not, and a BCC number from the same setup would carry the same bias.

## BCC, after that

\(E = 3.1t\). Nearest-neighbour BCC is bipartite, so \(E = 0\) is the chiral point, the same trap as on the simple cubic lattice. Half-bandwidth is \(8t\). \(E = 3.1t\) is \(0.39\) of that.

Classical site threshold used here: \(p_c = 0.2457\). \(p_q\) is the unknown, so \(\xi_{\mathrm{perc}}(p_q)\) is unknown until the crossings stop moving. Output file the code writes: `data/bcc_dilution_E3p1.h5`.

```bash
pixi run bcc-dilution
```

Do not start it on the current simple-cubic calibration.

FCC is the other lattice. Triangles, so it is not bipartite, and the band centre is ordinary class AI. The slice is a four-site basis. Same \(M\) costs about \(64\times\) a simple-cubic slice. That run waits on a BCC number.

## What this is not

A new \(\nu\) for the 3D orthogonal class. A precision update of the simple-cubic mobility edge. Ujfalusi and Varga mapped that edge; our \(M = 12\) bar is smaller than their smallest accepted cube. A paper that leads with \(p_q = 0.517\) would be arguing with a \(0.0002\)-level result using two widths and a seeded fit.
