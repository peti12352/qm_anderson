# Literature

What this calculation is matched against, then what appeared while it was parked (June–October 2026). Quotes below are from the papers or from their abstracts, not from a review.

## The threshold we are trying to reproduce

L. Ujfalusi and I. Varga, Phys. Rev. B **90**, 174203 (2014).
[doi:10.1103/PhysRevB.90.174203](https://doi.org/10.1103/PhysRevB.90.174203)

Simple cubic site quantum percolation. Multifractal finite-size scaling, not transfer matrices. They say the sites "form a regular, usually simple cubic lattice," and they quote the classical threshold \(p_c^C = 0.3116 \pm 0.0002\).

Table II at \(E = 3.1\): \(p_c^Q = 0.50628\ (0.50606..0.50647)\), \(\nu = 1.603\ (1.515..1.695)\), \(L_{\min} = 20\). Averaged over the mobility edge: \(\nu = 1.622\ (1.587..1.658)\). At \(E = 0.7\): \(p_c^Q = 0.38402\ (0.38387..0.38418)\). At \(E = 0.1\): \(p_c^Q = 0.45384\ (0.45365..0.45402)\). Below \(E = 0.1\) they report severe convergence trouble and \(p_c^Q \to 1\) as \(E \to 0\). No body-centered lattice.

OpenAlex lists 13 citations. None of them is a BCC threshold.

## Anderson benchmark

K. Slevin and T. Ohtsuki, Phys. Rev. Lett. **82**, 382 (1999).
[doi:10.1103/PhysRevLett.82.382](https://doi.org/10.1103/PhysRevLett.82.382)

Corrections to scaling at the 3D orthogonal Anderson transition. The box-distribution critical disorder used as the gate in this repo is \(W_c = 16.54\). They fit the irrelevant exponent freely and obtain \(|y| \approx 2.8\)–\(3.9\) for \(\Lambda\). The \(y = 1.12\) fixed in `src/fss.py` is not their number. It is an effective exponent that reproduces this benchmark at \(M \le 10\).

K. Slevin and T. Ohtsuki, New J. Phys. **16**, 015012 (2014).
[doi:10.1088/1367-2630/16/1/015012](https://doi.org/10.1088/1367-2630/16/1/015012)

Later determination of the 3D orthogonal exponent. The gate in `runs/sc_benchmark.py` (\(\nu \approx 1.571\)) is the 1999-era number, not a claim that 2014 reprinted it.

A. Rodriguez, L. J. Vasquez, K. Slevin, and R. A. Römer, Phys. Rev. B **84**, 134209 (2011).
[doi:10.1103/PhysRevB.84.134209](https://doi.org/10.1103/PhysRevB.84.134209)

Multifractal finite-size scaling for the Anderson transition. The stability test in their appendix raises the polynomial order. It is not a scan over window width. The methods note in this repo records \(\nu = 1.590\ (1.579..1.602)\) and \(|y| \approx 1.7\) for fixed-\(\lambda\) scaling, which is a different observable from our \(\Lambda_M\).

E. Abrahams, P. W. Anderson, D. C. Licciardello, and T. V. Ramakrishnan, Phys. Rev. Lett. **42**, 673 (1979).
Scaling theory. No metallic phase in the orthogonal class below two dimensions. This is why a bar with \(M \lesssim \xi_{\mathrm{perc}}\) is not measuring the 3D fixed point: on those scales the cluster is fractal.

## Other lattices, older work

A. Eilmes, A. M. Fischer, and R. A. Römer, Phys. Rev. B **77**, 245117 (2008).
[doi:10.1103/PhysRevB.77.245117](https://doi.org/10.1103/PhysRevB.77.245117)

Critical disorder for the Anderson transition on FCC and BCC. Diagonal disorder, \(p = 1\). Not quantum percolation, and not \(p_q\).

G. Schubert and H. Fehske, arXiv:0905.2537 (2009).
Review of quantum percolation in disordered structures. Simple cubic mobility-edge curves that Ujfalusi and Varga compare against. Not BCC.

Quantum percolation on the face-centered cubic lattice is an old transfer-matrix paper: Phys. Rev. B **44**, 9926 (1991), [doi:10.1103/PhysRevB.44.9926](https://doi.org/10.1103/PhysRevB.44.9926). FCC, not BCC.

## While this was parked

Last commit before the pause: 4 June 2026. Checked again 18 October 2026. arXiv query `BCC` and `"quantum percolation"`: no hits. OpenAlex title search `body-centered cubic anderson`: no works. The four below are the papers from that window that a search for this problem actually returns. None of them measures \(p_q\) on BCC.

1. B. Bhowmik, S. Mitra, R. M. Ziff, and A. Sensharma, arXiv:2602.15163 (2026).
   [https://arxiv.org/abs/2602.15163](https://arxiv.org/abs/2602.15163)

   Classical bond percolation. Their abstract: "We investigate the effect of structural distortion on bond percolation in simple cubic and body-centered cubic lattices using extensive Monte Carlo simulations." Undistorted bond thresholds they quote for reference: \(p_b \approx 0.2488\) (SC) and \(p_b \approx 0.18025\) (BCC). No Schrödinger equation.

2. S. Becker and I. Oltman, arXiv:2609.38017 (2026).
   [https://arxiv.org/abs/2609.38017](https://arxiv.org/abs/2609.38017)

   "For Bernoulli bond percolation on every regular tree of degree at least three, we prove that infinite clusters appear strictly before absolutely continuous spectrum." Trees. \(p_c < p_{ac} < 1\).

3. W. S. Oliveira, J. P. de Lima, and R. R. dos Santos, Phys. Rev. E **111**, 034117 (2025).
   [doi:10.1103/PhysRevE.111.034117](https://doi.org/10.1103/PhysRevE.111.034117), arXiv:2409.04610.

   "We theoretically investigate the quantum percolation problem on Lieb lattices in two and three dimensions." A 3D lattice that has been done. It is not BCC. This is the recent 3D quantum-percolation result, and it predates the pause.

4. W. S. Oliveira, J. Faúndez, R. Arouca, and W. A. M. Morgado, arXiv:2609.04625 (2026).
   [https://arxiv.org/abs/2609.04625](https://arxiv.org/abs/2609.04625)

   "We investigate the interplay between geometric dilution and non-Hermitian disorder in the two-dimensional quantum site-percolation model." Two dimensions, complex on-site potential.

The only paper that cites Ujfalusi and Varga 2014 with a date after 4 June 2026 is the same group's honeycomb work: W. S. Oliveira, J. Faúndez, and W. A. M. Morgado, arXiv:2604.12732, published as [doi:10.1103/yf3h-zdpr](https://doi.org/10.1103/yf3h-zdpr). "We investigate quantum percolation in a honeycomb lattice with site dilution and random spin-orbit coupling." Two-dimensional honeycomb. Not BCC.
