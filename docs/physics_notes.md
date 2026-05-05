# Physics Notes: Anderson Localization, Quantum Percolation, and the TMM+FSS Method

*A self-contained reference built from first principles. Intended as a reading companion
to the code and RUNBOOK. Concepts are introduced in the order you need them.*

---

## 1. The Physical Problem

### 1.1 Electrons in a disordered solid

Take a crystalline solid and randomly remove atoms - or keep all atoms but randomize
their on-site energies. Quantum mechanics then poses the question: are the eigenstates
of the Hamiltonian **localized** (decaying exponentially in space, like a bound state)
or **extended** (spread across the whole system, like a Bloch wave)?

The Hamiltonian we study is the tight-binding model on a site-diluted lattice:

```
H = −t Σ_{<i,j> both occupied} |i⟩⟨j|
```

Each site i is independently occupied with probability p; empty otherwise. No on-site
disorder (W = 0). The only randomness is geometric: which sites exist.

### 1.2 Classical vs quantum percolation

**Classical percolation** asks: is there a connected path of occupied sites spanning
the system? Below p_c (the classical threshold), all clusters are finite - no spanning
path exists. Above p_c, a giant cluster appears. For SC: p_c = 0.3115.

**Quantum percolation** asks the harder question: even given a connected spanning
cluster, do the quantum eigenstates on that cluster extend across the system, or do
they localize? The answer is not automatic - quantum interference can localize states
even on a connected graph. The quantum threshold p_q > p_c. The gap p_q − p_c is the
"quantum localization cost."

Below p_q: all eigenstates are exponentially localized. No quantum transport.
Above p_q: extended states exist. Quantum transport survives.

This is a **quantum phase transition** driven by a geometric parameter p.

---

## 2. Localized States and the Localization Length

### 2.1 What localization means, rigorously

An eigenstate |ψ_E⟩ is **localized** if its real-space amplitude decays exponentially
from some center r_0:

```
|⟨r|ψ_E⟩|² ~ A · exp(−|r − r_0| / ξ_loc)   as |r − r_0| → ∞
```

The **localization length** ξ_loc is the decay length. It is specific to the eigenstate
and depends on energy E and the disorder realization.

An eigenstate is **extended** if |⟨r|ψ_E⟩|² does not decay exponentially - it spreads
across the whole system, with amplitude ~ 1/V (V = system volume).

The distinction is sharp in the thermodynamic limit. At finite system size, all
eigenstates look somewhat spread out; the difference only becomes clear as L → ∞.

### 2.2 Why localization happens: destructive interference

A classical particle scatters off impurities and diffuses. A quantum particle also
scatters, but its wavefunction accumulates **phase** along each scattering path.
Paths that return to the origin interfere. In 3D with strong enough disorder, the
constructive interference of time-reversed pairs of back-scattering paths suppresses
diffusion - this is **weak localization**. With sufficient disorder, interference
becomes so strong that the wavefunction cannot spread at all: **Anderson localization**.

The mechanism is purely quantum (interference) and has nothing to do with energy
gaps or magnetic fields.

### 2.3 Localization length vs correlation length - they are NOT the same thing

This distinction is crucial and often confused.

**Localization length ξ_loc**: a property of individual eigenstates. It measures how
far the wavefunction of a specific state extends before decaying. States deep in the
localized phase have small ξ_loc. States near the mobility edge have large ξ_loc.

**Correlation length ξ_corr**: a property of the phase transition itself. It measures
the spatial scale over which the critical fluctuations of the system are correlated -
the scale below which the system "looks critical." ξ_corr diverges at the critical
point regardless of which side you approach from.

**How they relate at the Anderson/percolation transition:**

On the localized side (p < p_q): the typical localization length of states near the
mobility edge IS the correlation length:

```
ξ_corr = ξ_loc ~ |p − p_q|^{−ν}   (localized side)
```

On the metallic side (p > p_q): ξ_corr is the scale above which transport becomes
diffusive. Below ξ_corr the system still has critical (fractal) fluctuations. Above
it the system looks like a normal metal.

```
ξ_corr ~ |p − p_q|^{−ν}   (metallic side, same exponent ν)
```

**The key point**: ξ_corr diverges on BOTH sides with the SAME exponent ν. ξ_loc only
makes sense on the localized side. They coincide on the localized side and diverge
with the same ν - which is why people often conflate them. They are different
conceptual objects but the same number near the transition.

At p_q itself: ξ_corr → ∞. The system is scale-invariant. Wavefunctions are fractal
(neither localized nor extended in the usual sense) - they are multifractal.

---

## 3. The Phase Transition and What ν Means Physically

### 3.1 This is a continuous (second-order) phase transition

The transition at p_q is continuous: there is no discontinuous jump in any observable.
Instead, everything changes smoothly, and the localization length diverges as a power
law. This is the hallmark of a second-order transition.

Compare to a first-order transition (e.g. ice melting) where there IS a jump. Here,
as p → p_q from below, ξ_loc grows without bound: the "insulator" becomes less and
less insulating until at exactly p_q it is not insulating at all.

### 3.2 The correlation length exponent ν

ξ_corr ~ |p − p_q|^{−ν}   defines ν.

**Physically**: ν tells you how fast the correlation length grows as you approach
the transition.

- ν = 1: ξ doubles every time you halve |p − p_q|. Linear approach.
- ν = 1.622 (our case): ξ grows faster - ξ ~ |p − p_q|^{−1.62}. Halving the
  distance to p_q multiplies ξ by 2^{1.62} ≈ 3.1.

Concretely: 1% away from p_q (|p − p_q|/p_q = 0.01), the correlation length is
ξ ~ (0.01)^{−1.62} × a ≈ 1800 lattice spacings (a = lattice constant). This is why
finite-size effects are large near the transition - your system of size M = 10 is
tiny compared to ξ_corr = 1800.

### 3.3 The Harris criterion

A fundamental question: if I add weak disorder to a clean critical point, does the
transition survive or get washed out? Harris (1974) showed:

```
If dν > 2  (d = spatial dimension): the transition is stable against weak disorder.
If dν < 2: disorder is relevant; the clean critical point is destroyed.
```

For us: d = 3, ν = 1.622 → dν = 4.87 > 2. The quantum percolation transition is
stable. The exponent ν = 1.622 is self-consistent: disorder at this level of strength
does not destabilize the transition.

(Equivalently: ν > 2/d = 2/3. This is sometimes stated as the "Chayes et al. bound"
for disordered systems: ν must be at least 2/3 in any dimension.)

---

## 4. The Renormalization Group (RG)

### 4.1 zoom out

The RG is a systematic way of asking "what does the system look like at long length
scales?" The operation is: replace every block of b^d sites with one effective site,
re-expressing the Hamiltonian in terms of coarse-grained variables.

At the critical point, the system is **scale-invariant**: the coarse-grained
Hamiltonian looks the same as the original (just with renormalized parameters). This
is the **RG fixed point**.

Away from the critical point, the coarse-graining reveals that some parameters grow
(relevant operators) and others shrink (irrelevant operators) under successive
applications.

### 4.2 Why ξ ~ |x − x_c|^{−ν} emerges from RG

Near the critical point x_c, the deviation u = x − x_c is a **relevant operator**.
Under coarse-graining by factor b, it transforms as:

```
u → b^{1/ν} · u
```

This is just the statement that u is relevant with RG eigenvalue 1/ν > 0. After n
steps of coarse-graining (each by factor b), the effective deviation is:

```
u_eff = b^{n/ν} · u
```

The RG flow stops when u_eff ~ 1 (the system has left the critical region). This
happens at:

```
b^{n/ν} · u ~ 1  →  b^n ~ u^{−ν} ~ |x − x_c|^{−ν}
```

But b^n is the physical length scale reached after n coarse-graining steps. So the
correlation length - the scale at which the system stops looking critical - is:

```
ξ ~ b^n ~ |x − x_c|^{−ν}
```

This is the derivation. ν is an RG eigenvalue; ξ ~ |x − x_c|^{−ν} is the consequence.

### 4.3 Relevant vs irrelevant operators

**Relevant operators** (eigenvalue > 0): grow under coarse-graining. x − x_c is
relevant - it drives the system away from the critical point. You must tune x to x_c
to stay at the fixed point.

**Irrelevant operators** (eigenvalue < 0): shrink under coarse-graining. At infinite
system size they vanish. At finite system size they contribute corrections. The leading
irrelevant operator has eigenvalue −y (y > 0), contributing corrections ~ L^{−y} to
any observable. This is the origin of the b·M^{−y} correction in our FSS ansatz.

**Marginal operators** (eigenvalue = 0): don't change under RG. Rare; lead to
logarithmic corrections.

---

## 5. Finite-Size Scaling (FSS)

### 5.1 The problem

In an infinite system at x_c, ξ → ∞ and the transition is sharp. In a finite system
of size L, ξ is cut off at L: the system cannot know it is at the critical point
more precisely than |x − x_c| ~ L^{−1/ν} (the scale at which ξ ~ L). So the
"transition" in a finite system is a crossover, smeared over a range δx ~ L^{−1/ν}.

FSS turns this from a problem into a tool: the shape of the smearing, controlled by
L, tells you ν.

### 5.2 The FSS hypothesis

Near x_c, ξ is the only relevant length scale. Any dimensionless observable O(L, x)
can only depend on L and x through the dimensionless ratio L/ξ:

```
O(L, x) = f(L/ξ)  =  f( L · |x − x_c|^ν )
```

We can reparametrize the argument to make it linear in x − x_c (convenient for
polynomial approximation):

```
u = L^{1/ν} · (x − x_c)/x_c        so that    u^ν ∝ L/ξ
```

giving the standard FSS form:

```
O(L, x) = f(u)  =  f( L^{1/ν} · (x − x_c)/x_c )
```

If O has a non-trivial scaling dimension η at the fixed point (e.g. it is not
dimensionless), there is an additional prefactor:

```
O(L, x) = L^{η} · f( L^{1/ν} · (x − x_c)/x_c )
```

### 5.3 Why this is powerful

Plot O vs u instead of vs x: all curves for all L collapse onto **one universal
curve** f(u). The collapse works because f is universal - it encodes only the
fixed-point structure, not the specific microscopic details. SC and BCC percolation
(different lattices) share the same f because they are in the same universality class
(3D orthogonal). The data collapse is both a test of the theory and a precision
measurement of ν (vary ν until the collapse is best).

### 5.4 Corrections to scaling

At finite L, irrelevant operators contribute:

```
O(L, x) = f(u) + b·L^{−y} + b_2·L^{−2y} + ...
```

The correction b·L^{−y} shifts the observable away from the asymptotic f(u). For
small L (our M = 4..10 range), corrections can be 10–40% of the signal. Including
the leading correction is essential for unbiased results; including two corrections
requires many more system sizes.

---

## 6. The Transfer Matrix Method (TMM)

### 6.1 The quasi-1D geometry

Direct diagonalization of a 3D Hamiltonian on an L^3 lattice is exponentially
expensive. The TMM maps the 3D problem onto a much more efficient quasi-1D problem.

Take a strip (tube) of cross-section M × M sites, propagated in the z-direction for
L slices:

```
         M sites
    ←────────────→
    □ □ □ □ □ □ □     ↑
    □ □ □ □ □ □ □     M sites
    □ □ □ □ □ □ □     ↓
    ══════════════════════════════ →  L slices
```

The M² sites on each slice form a state vector of dimension N = M².

### 6.2 What the transfer matrix does

The Schrödinger equation H|ψ⟩ = E|ψ⟩ on the strip can be rewritten as a recursion
relating the wavefunction amplitudes on slice k+1 to those on slices k and k−1:

```
ψ_{k+1} = T_k · ψ_k − V^{−1} · V_back · ψ_{k−1}
```

where T_k encodes the on-slice Hamiltonian and the coupling to the next slice. This
can be written as a 2N × 2N transfer matrix acting on (ψ_k, ψ_{k−1}):

```
(ψ_{k+1})   (   T_k    −V^{−1}·V_back ) (ψ_k    )
(ψ_k    ) = (    I              0      ) (ψ_{k−1})
```

After L steps, we have a product of L random matrices (random because each slice has
a different random occupation pattern).

### 6.3 Lyapunov exponents

By Oseledec's theorem (1968), for almost every disorder realization, the product
T_L T_{L−1} ··· T_1 has well-defined growth rates: the **Lyapunov exponents** γ_i
(i = 1, ..., 2N), ordered γ_1 ≥ γ_2 ≥ ··· ≥ γ_N > 0 > −γ_N > ··· ≥ −γ_1.

The positive Lyapunov exponents describe the growth of the matrix product in L. The
**smallest positive Lyapunov exponent γ_min** controls the longest length scale in
the problem: the localization length of the quasi-1D strip:

```
ξ_M = 1/γ_min
```

This converges to a definite self-averaging value as L → ∞ - the law of large numbers
for random matrix products. In practice L = 10,000 is "infinity": the Lyapunov
exponents are converged to ~0.1%.

### 6.4 The normalized localization length Λ_M

We define:

```
Λ_M = ξ_M / M = 1 / (M · γ_min)
```

Dividing by M makes this dimensionless. This is the key observable for FSS.

**Why γ_min and not some other Lyapunov exponent?** γ_min controls the slowest
decay / fastest growth in the system - the most extended mode. In the metallic phase
this mode extends across the cross-section; in the localized phase it decays. It is
the mode closest to the mobility edge.

### 6.5 What M and L are - the critical distinction

| | M (cross-section) | L (strip length) |
|---|---|---|
| Role | **Finite-size variable** (the "L" in FSS) | Statistical averaging device |
| Physical meaning | The system size that competes with ξ_3D | Must be >> ξ_M to converge |
| Varies in FSS? | Yes - {4, 6, 8, 10} | No - fixed at 10,000 |
| Limit of interest | M → ∞ recovers 3D physics | L → ∞ taken first (converges) |

**Why M plays the role of L in FSS**: the TMM has already taken L → ∞ (Oseledec's
theorem). What remains is a well-defined number ξ_M for each M. Now M is the only
finite length in the problem, and it competes with the 3D correlation length ξ_3D.
The ratio M/ξ_3D controls whether the strip looks localized or metallic.

---

## 7. How Λ_M Encodes the Phase Transition

### 7.1 The three regimes

**p << p_q (deep localized phase):**
ξ_3D is small (few lattice spacings). The strip width M >> ξ_3D. The strip "contains"
many localization volumes. Λ_M = ξ_M/M is small and decreases as M grows - larger
strips show more localization, not less.

**p >> p_q (metallic phase):**
ξ_3D is the scale above which transport is diffusive. M < ξ_3D (strip is narrower than
the correlation length). The strip looks like a quantum wire. ξ_M scales with M
(the localization in a quasi-1D metal scales with its cross-section). Λ_M = ξ_M/M is
approximately constant or grows slightly with M.

**p = p_q exactly:**
ξ_3D → ∞. The strip is always narrower than ξ_3D for any finite M. ξ_M ~ M exactly
(a known result for the critical point of the 3D orthogonal class). Therefore Λ_M =
ξ_M/M = const, independent of M. **All Λ_M(p_q) curves agree.** This is the crossing.

### 7.2 The crossing and why it's at p_q

The ordering of Λ_M curves reverses at p_q:
- For p < p_q: Λ_4 > Λ_6 > Λ_8 > Λ_10 (smaller M → larger Λ, because smaller
  strip = less localization relative to strip width)
- For p > p_q: Λ_4 < Λ_6 < Λ_8 < Λ_10 (larger M → larger Λ, metallic scaling)

The reversal happens at p_q. The crossing point of any pair (M1, M2) gives a rough
estimate of p_q. With corrections-to-scaling (b·M^{−y}), different pairs cross at
slightly different p; the FSS fit extracts the true p_q from all data simultaneously.

---

## 8. FSS Ansatz for Λ_M

### 8.1 Applying FSS to Λ_M

Λ_M is dimensionless (η = 0), so there is no L^η prefactor. The FSS form is:

```
Λ_M(p) = F( M^{1/ν} · (p − p_q) / p_q )  +  b·M^{−y}
```

F is the universal scaling function; b·M^{−y} is the leading correction.

We approximate F by a polynomial in the scaling variable u = M^{1/ν}(p−p_q)/p_q:

```
F(u) ≈ a_0 + a_1·u + a_2·u² + a_3·u³ + a_4·u⁴
```

**Notation: a_0 = Λ_M(p_q) is the value at the critical point (same for
all M by definition). a_1 controls the slope - positive for the metallic side,
negative for the localized side. Higher terms account for the curvature of F away
from u=0.

The polynomial is valid for small |u| - i.e., close to p_q. This is why we need a
fitting window.

### 8.2 Free params and what the fit extracts

| Parameter | Meaning | Constrained by |
|-----------|---------|----------------|
| p_q | Critical point location | Position of the crossing |
| ν | Correlation length exponent | How fast curves separate away from crossing |
| a_0 | Λ_M at the critical point | Height of the crossing |
| a_1..a_4 | Shape of F(u) | Curvature away from crossing |
| b | Amplitude of corrections | Mismatch between M-dependent curves at fixed p |

The fit uses all (M, p) data simultaneously, weighted by the statistical error σ_{M,p}
on each Λ_M measurement. This is χ² minimization:

```
χ² = Σ_{M,p} [ (Λ_M^{meas} − F(u) − b·M^{−y}) / σ_{M,p} ]²
```

### 8.3 Irrelevant exponent y

y is the scaling dimension of the leading irrelevant operator. It controls how fast
corrections decay with M. In the 3D orthogonal class, the literature reports
|y| ≈ 1.7–2.8 depending on the observable and method.

**In our implementation**: y = 1.12 is an empirical effective value that reproduces
the Anderson benchmark (W_c, ν within 5% of literature; χ²/dof = 2.4). It captures
subleading corrections dominant at M = {4..10}. It is NOT the leading RG irrelevant
exponent as computed by Slevin & Ohtsuki 1999 (who fit y freely and get |y| ≈ 2.8).
The origin of y = 1.12 in the broader literature is unclear. It is treated as a fixed
empirical parameter with ±0.08 uncertainty propagated as a systematic error.

---

## 9. From Fit to Results: the Full Procedure

### 9.1 Step 1: Rough p_q from pairwise crossings

For each adjacent pair (M_i, M_{i+1}), find where Λ_{M_i}(p) − Λ_{M_{i+1}}(p)
changes sign. Linear interpolation gives the crossing p. Take the median of all
crossings as p_q_rough.

This is robust against the minimum-spread estimator, which fails when corrections
are large (the global minimum of std(Λ_M over M) at each p is not at p_q when
b·M^{−y} differentials between M values are comparable to the physical signal).

### 9.2 Step 2: Window adequacy scan (select δ)

The polynomial F(u) is valid only for small |u|. To find the right window:

1. Scan δ ∈ {0.03, ..., 0.15} (15 values).
2. For each δ: fit FSS in [p_rough − δ, p_rough + δ] (no bootstrap; fast scan).
3. Record χ²/dof(δ) and Q(δ) = P(χ² ≥ χ²_obs | model is true).
4. Select the **widest δ with χ²/dof < 3.0**.

Logic: wider is better (more data, tighter constraints), but when the window is
too wide, polynomial truncation error enters and χ²/dof rises. The threshold 3.0 is
calibrated against our benchmark floor (χ²/dof = 2.4 from unmodelled b_2·M^{−2y}
corrections). S&O 1999 use Q > 0.1; with our σ(Λ_M) ~ 10^{−4}, Q ≈ 0 for all
windows due to unmodelled subleading corrections, so the Q criterion cannot be
applied directly.

### 9.3 Step 3: Full FSS fit with bootstrap

On the selected window: full Nelder-Mead + L-BFGS-B minimization of χ².
Bootstrap CIs: perturb each Λ_M by N(0, σ_{M,p}) independently, refit 500 times,
take std of the 500 estimates as the statistical uncertainty.

### 9.4 Step 4: Rodriguez polynomial stability test

Fit FSS at poly_order ∈ {3, 4, 5}. Check that p_q and ν from each order lie within
the 95% CI (±2σ) of the reference order-4 result. If they do: the polynomial is
adequate and the results are stable. This is the actual Rodriguez et al. 2011
Appendix A stability criterion (polynomial ORDER perturbation, not window-width scan).

### 9.5 Step 5 - Systematic from y

Repeat the full fit at y = 1.04, 1.12, 1.20. Report:
```
σ_syst = max(|result(1.04) − result(1.12)|, |result(1.20) − result(1.12)|)
σ_total = sqrt(σ_stat² + σ_syst²)
```

---

## 10. The Symmetry Classes and Why Energy Matters

### 10.1 The three Wigner-Dyson classes

Random matrix theory classifies disordered quantum systems by their symmetries:

| Class | Time-reversal | Spin-rotation | Example |
|-------|--------------|---------------|---------|
| AI (orthogonal) | Yes | Yes | Non-magnetic, spinless, no SO coupling |
| AII (symplectic) | Yes | No | Strong spin-orbit coupling |
| A (unitary) | No | - | Magnetic field present |

Each class has its own Anderson transition (in 3D) with its own ν. Our system: no
magnetic field, no spin - class AI. ν = 1.622 (Slevin & Ohtsuki 1999; Rodriguez 2011).

### 10.2 Bipartite lattices and the chiral anomaly at E=0

SC and BCC are **bipartite**: sites can be 2-colored A/B with all bonds connecting
A to B. The bipartite structure implies a chiral operator Γ (flip A↔B, change sign
of A-sublattice amplitudes) that anticommutes with H:

```
ΓHΓ = −H
```

At E = 0 exactly, this is an exact symmetry: if |ψ, E⟩ is an eigenstate, so is
Γ|ψ, −E⟩. The transfer matrix at E = 0 has chiral symmetry. This places the system
in **class BDI** (chiral orthogonal), not class AI.

Class BDI has a different fixed point, different ν, different corrections. Ujfalusi &
Varga 2014 show p_c^Q(E→0) → 1 for SC, and MFSS "severely fails to converge" below
E = 0.1 due to the anomalous DOS at E = 0 from sublattice imbalance zero modes.

**Consequence**: run at E = 0.7 (11.7% of SC half-bandwidth 6t). The BDI→AI
crossover length scale ξ_cross ~ E^{−1/Δ_E} ≪ M_min = 4 at E = 0.7, so BDI
corrections are negligible. At E = 0.1, ξ_cross ≈ 10 ≈ M_max - correction
dominance: b = −0.153, χ²/dof = 1543, spurious crossings at p ≈ 0.35.

---

## 11. Summary of Concepts

```
Physical system
│
├── Site-diluted lattice (control parameter p)
│   ├── p < p_c: no connected cluster (trivially localized)
│   ├── p_c < p < p_q: connected cluster but all states localized (quantum percolation)
│   └── p > p_q: extended states exist → quantum transport
│
├── Key quantity: correlation length ξ ~ |p − p_q|^{−ν}
│   ├── = localization length ξ_loc on the localized side
│   └── = transport coherence length on the metallic side
│
└── Critical exponent ν = 1.622 (3D-AI class; same for SC and BCC)

Method
│
├── TMM: quasi-1D strip M × M × L → Lyapunov exponents → ξ_M → Λ_M = ξ_M/M
│   ├── L = 10,000: statistical averaging (Oseledec limit)
│   └── M = {4,6,8,10}: the finite-size variable for FSS
│
└── FSS: Λ_M(p) = F(M^{1/ν}·(p−p_q)/p_q) + b·M^{−y}
    ├── F: universal scaling function (polynomial approx.)
    ├── b·M^{−y}: leading irrelevant correction
    ├── Fit: 8 parameters {p_q, ν, a_0..a_4, b}
    ├── Window: δ adequacy scan (widest δ with χ²/dof < 3.0)
    ├── Stability: Rodriguez polynomial-order perturbation test
    └── Uncertainties: bootstrap (stat) + y-variation (syst)
```

---

## References

1. **Anderson P W 1958** Phys. Rev. **109** 1492 - original localization paper

2. **Fisher M E and Barber M N 1972** Phys. Rev. Lett. **28** 1516 - finite-size scaling
   introduced

3. **MacKinnon A and Kramer B 1983** Z. Phys. B **53** 1 - transfer matrix method for
   Anderson localization; original Λ_M observable

4. **Slevin K and Ohtsuki T 1999** Phys. Rev. Lett. **82** 382 - corrections-to-scaling
   in 3D Anderson TMM; y fitted freely, |y|≈2.8; ν=1.57–1.58; bootstrap CIs

5. **Rodriguez A, Vasquez L J, Slevin K and Römer R A 2011** Phys. Rev. B **84** 134209
   - multifractal FSS; polynomial-order stability test (Appendix A); ν=1.590

6. **Ujfalusi L and Varga I 2014** Phys. Rev. B **90** 245136 - quantum percolation
   on SC lattice; mobility edge p_c^Q(E); BDI anomaly at E=0; p_c^Q(E=0.7)≈0.384
