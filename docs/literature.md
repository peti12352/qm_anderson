# Literature

Papers this calculation is compared with, and papers from 2025–2026 that a search for the same problem turns up. The sentences in quotation marks are from the papers.

## Simple-cubic quantum percolation

L. Ujfalusi and I. Varga, Phys. Rev. B 90, 174203 (2014).
https://doi.org/10.1103/PhysRevB.90.174203

Site dilution on the simple cubic lattice, analysed by multifractal finite-size scaling. They write that the sites "form a regular, usually simple cubic lattice," and they use the classical threshold 0.3116 ± 0.0002.

Table II at E = 3.1: p_c^Q = 0.50628, interval 0.50606 to 0.50647, and ν = 1.603, interval 1.515 to 1.695, with smallest cube side 20. Averaged along the mobility edge, ν = 1.622, interval 1.587 to 1.658. At E = 0.7, p_c^Q = 0.38402, interval 0.38387 to 0.38418. At E = 0.1, p_c^Q = 0.45384, interval 0.45365 to 0.45402. Below E = 0.1 the scaling fits converge poorly, and the threshold goes to 1 as E goes to 0. The paper has no body-centered lattice.

OpenAlex lists 13 citations. None of them reports a body-centered quantum percolation threshold.

## Anderson transition, orthogonal class

K. Slevin and T. Ohtsuki, Phys. Rev. Lett. 82, 382 (1999).
https://doi.org/10.1103/PhysRevLett.82.382

Corrections to scaling at the 3D orthogonal Anderson transition. The gate in this repository uses their box-distribution critical disorder, W_c = 16.54. They fit the irrelevant exponent and obtain a magnitude around 2.8 to 3.9 for the observable Λ. The value y = 1.12 fixed in src/fss.py is an effective exponent that reproduces our benchmark at M ≤ 10.

K. Slevin and T. Ohtsuki, New J. Phys. 16, 015012 (2014).
https://doi.org/10.1088/1367-2630/16/1/015012

A later determination of the same exponent. The number ν ≈ 1.571 in runs/sc_benchmark.py is the 1999 gate.

A. Rodriguez, L. J. Vasquez, K. Slevin, and R. A. Römer, Phys. Rev. B 84, 134209 (2011).
https://doi.org/10.1103/PhysRevB.84.134209

Multifractal finite-size scaling of the Anderson transition. The stability check in their appendix increases the order of the polynomial. It is a different test from scanning the fit window. The methods note in this repository records ν = 1.590, interval 1.579 to 1.602, and an irrelevant exponent of magnitude about 1.7. Those numbers are for their fixed-λ multifractal spectrum. Ours are for Λ_M.

E. Abrahams, P. W. Anderson, D. C. Licciardello, and T. V. Ramakrishnan, Phys. Rev. Lett. 42, 673 (1979).

Scaling theory of localization. In the orthogonal class there is no metal below two dimensions. A bar whose width is comparable to the percolation length is sitting on a fractal cluster, and it is not measuring the 3D fixed point.

## Other lattices

A. Eilmes, A. M. Fischer, and R. A. Römer, Phys. Rev. B 77, 245117 (2008).
https://doi.org/10.1103/PhysRevB.77.245117

Critical disorder of the Anderson transition on the face-centered and body-centered cubic lattices, with every site occupied. The control parameter is W.

G. Schubert and H. Fehske, arXiv:0905.2537 (2009).

Review of quantum percolation. The simple-cubic mobility edges that Ujfalusi and Varga compare with are in here.

Quantum percolation on the face-centered cubic lattice: Phys. Rev. B 44, 9926 (1991).
https://doi.org/10.1103/PhysRevB.44.9926

## Papers since the project was paused

Last commit before the pause: 4 June 2026. Literature checked again on 18 October 2026. An arXiv search for BCC together with "quantum percolation" returned nothing. An OpenAlex title search for "body-centered cubic anderson" returned nothing. The four papers below are the neighbouring results those searches return.

1. B. Bhowmik, S. Mitra, R. M. Ziff, and A. Sensharma, arXiv:2602.15163 (2026).
   https://arxiv.org/abs/2602.15163

   Classical bond percolation on lattices whose sites have been randomly displaced. From the abstract: "We investigate the effect of structural distortion on bond percolation in simple cubic and body-centered cubic lattices using extensive Monte Carlo simulations." For the undistorted lattices they quote bond thresholds of about 0.2488 (simple cubic) and 0.18025 (body-centered cubic).

2. S. Becker and I. Oltman, arXiv:2609.38017 (2026).
   https://arxiv.org/abs/2609.38017

   Bond percolation on regular trees. From the abstract: "For Bernoulli bond percolation on every regular tree of degree at least three, we prove that infinite clusters appear strictly before absolutely continuous spectrum." So p_c < p_ac < 1 on those trees.

3. W. S. Oliveira, J. P. de Lima, and R. R. dos Santos, Phys. Rev. E 111, 034117 (2025).
   https://doi.org/10.1103/PhysRevE.111.034117
   arXiv:2409.04610

   Quantum percolation on Lieb lattices in two and three dimensions. This is a 3D threshold, on the Lieb lattice, and it was published before the pause.

4. W. S. Oliveira, J. Faúndez, R. Arouca, and W. A. M. Morgado, arXiv:2609.04625 (2026).
   https://arxiv.org/abs/2609.04625

   Two-dimensional site percolation with a random imaginary on-site potential. From the abstract: "We investigate the interplay between geometric dilution and non-Hermitian disorder in the two-dimensional quantum site-percolation model."

The one paper that cites Ujfalusi and Varga 2014 with a date after 4 June 2026 is by the same group, on the honeycomb lattice: W. S. Oliveira, J. Faúndez, and W. A. M. Morgado, arXiv:2604.12732, journal version https://doi.org/10.1103/yf3h-zdpr. From the abstract: "We investigate quantum percolation in a honeycomb lattice with site dilution and random spin-orbit coupling."
