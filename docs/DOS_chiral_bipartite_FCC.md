# DOS Peaks, Chiral Symmetry, and Bipartite vs FCC — From Ground Up

---

## First: what is the DOS and why does it have peaks at all?

The density of states ρ(E) counts how many eigenstates exist per unit energy interval. For a periodic crystal with band structure E(k), it is:

```
ρ(E) = ∫_{BZ} δ(E - E(k)) d³k / (2π)³
```

This integral gets large wherever many k-points have the same energy — i.e., wherever the energy surface E(k) = const has a large area, or where the band is flat. The gradient ∇_k E(k) appears in the denominator when you evaluate the integral. When ∇_k E(k) = 0, the integrand diverges. These are **Van Hove singularities**, and they occur at high-symmetry k-points in every lattice.

In 3D, Van Hove singularities produce kinks in the DOS (not divergences — that only happens in 1D and 2D). In 2D, they produce logarithmic divergences. In 1D, they produce 1/√|E−E₀| poles.

This mechanism — flat band regions in k-space — is universal. Every lattice has it. It has nothing to do with bipartiteness.

---

## Three Distinct Mechanisms for E=0 Features

Before going further, it is essential to distinguish three completely separate reasons why E=0 can be special:

| Mechanism | Cause | Lattices | Disorder needed? |
|-----------|-------|----------|------------------|
| 1. Van Hove singularity | ∇_k E = 0 happens to be at k-point with E=0 | Depends on specific band | No (clean crystal) |
| 2. Bipartite band symmetry | ρ(E) = ρ(−E) forces dρ/dE = 0 at E=0 | All bipartite | No (clean crystal) |
| 3. Zero modes from sublattice imbalance | Random dilution → N_A ≠ N_B → localized E=0 states pile up | Bipartite + diluted | Yes |

These operate independently and can coexist. Each is explained below.

---

## Mechanism 1: Van Hove Singularities

For the SC NN tight-binding model:

```
E(k) = -2t(cos kx + cos ky + cos kz)
```

The band runs from −6t (all cosines = +1, k = 0) to +6t (all cosines = −1, k = π). The gradient vanishes at high-symmetry k-points. Working them out:

- k = (0,0,0): band minimum, E = −6t
- k = (π,0,0) and 5 permutations: saddle point, E = −2t
- k = (π,π,0) and 11 permutations: saddle point, E = +2t
- k = (π,π,π): band maximum, E = +6t

These give the well-known SC DOS with kink features at E = ±2t and ±6t. Notice: **E=0 is NOT a Van Hove singularity for the SC lattice.** The DOS at E=0 is finite and smooth — it's just the midpoint between the two saddle-point kinks.

For FCC NN tight-binding (12 neighbors):

```
E(k) = -4t[cos(kx/2)cos(ky/2) + cos(ky/2)cos(kz/2) + cos(kz/2)cos(kx/2)]
```

The Van Hove singularities occur at the Γ, X, L, W, K points of the FCC Brillouin zone. Their energies are determined by the dispersion — there's no symmetry reason any of them must be at E=0. The FCC DOS is asymmetric (heavily weighted toward negative energies for typical NN hopping).

---

## Mechanism 2: Bipartite Symmetry Forces ρ(E) = ρ(−E)

Recall the proof: for any bipartite H, ΓHΓ = −H. If |ψ, E⟩ is an eigenstate, so is Γ|ψ, −E⟩. The eigenstates come in ±E pairs. Therefore ρ(E) = ρ(−E) exactly for every disorder realization.

This forces: **dρ/dE|_{E=0} = 0** by symmetry (the derivative of an even function vanishes at zero). So E=0 is always a local extremum of the DOS in any bipartite lattice.

Whether it's a maximum, minimum, or saddle depends on the specific dispersion:

**SC:** E=0 is between the two saddle-point kinks at ±2t. The DOS at E=0 is finite, unremarkable — just constrained to have zero slope. The symmetry ρ(E) = ρ(−E) is maintained, but E=0 is not a peak.

**Honeycomb (graphene):** The dispersion near the K and K' points has Dirac cones:

```
E(k) ≈ ±ℏv_F |k − K|
```

In 2D, a linear cone gives ρ(E) ~ |E|. The DOS vanishes at E=0. E=0 is a minimum, not a peak. This is why pristine graphene is a semimetal, not a metal: zero DOS at the Fermi level.

**BCC NN only (bipartite by A=corners, B=body-centers):**

```
E(k) = ±8t|cos(kx/2)cos(ky/2)cos(kz/2)|
```

The E=0 surface is where at least one cosine factor vanishes — i.e., kx = π, OR ky = π, OR kz = π. This is the entire boundary of the BCC Brillouin zone (the N-point and surrounding faces). The zero-energy surface occupies a macroscopic fraction of the BZ boundary, creating an anomalously large DOS at E=0. For BCC NN tight-binding, E=0 has a genuine peak (logarithmic divergence or sharp cusp depending on the exact k-space geometry).

**Summary of Mechanism 2:** bipartiteness guarantees symmetry but the specific shape — peak, dip, or flat — depends on the band structure details. SC has no feature, honeycomb has a dip, BCC NN has a peak.

---

## Mechanism 3: Zero Modes from Sublattice Imbalance (Disorder)

This only exists in diluted bipartite lattices and has nothing to do with the clean band structure.

When you randomly remove sites:
- N_A occupied A-sites, N_B occupied B-sites
- By the central limit theorem: |N_A − N_B| ~ √N (random fluctuation)

**The theorem:** for any bipartite graph, the Hamiltonian has at least |N_A − N_B| eigenvalues exactly at E=0.

**Proof sketch:** Write H in block form [0, T; T†, 0] where T is the N_A × N_B hopping matrix between sublattices. If N_A > N_B, then T maps an N_A-dimensional space to an N_B-dimensional space. By rank-nullity: dim(null T) ≥ N_A − N_B. Vectors in null(T) with zero B-components satisfy H|ψ⟩ = 0. These are zero modes localized on the A-sublattice. ∎

For a system with N occupied sites, you get ~√N localized zero modes. These pile up in the DOS creating a delta-function-like peak exactly at E=0. In the thermodynamic limit, the zero-mode density per site is ~1/√N → 0, but their contribution to the DOS is singular because they're all at E=0 exactly.

**FCC does NOT have this.** With no bipartite structure, there's no T matrix, no sublattice imbalance, no zero-mode counting theorem. Randomly diluted FCC has no guaranteed E=0 zero modes. The DOS near E=0 is just whatever the random disorder produces — no special pile-up.

---

## The Gade-Wegner Singularity (Mechanism 3 continued, weak disorder limit)

In the limit of weak disorder in a bipartite lattice, the Gade-Wegner theory (1991) gives an anomalous DOS near E=0 that is not just a delta function but a true thermodynamic singularity:

```
In 2D class AIII:   ρ(E) ~ exp(−c|ln E|^{1/2})
```

This diverges faster than any power law as E→0 but slower than a pole. The clean exponential form comes from the chiral RG structure — it's not a simple Van Hove singularity.

In 3D, the Gade-Wegner result doesn't directly apply (it's derived in 2D), but the qualitative physics persists: disorder in a bipartite lattice creates anomalous spectral weight piling up at E=0. The 3D form is an active research question. The Ujfalusi paper hit this directly — their MFSS "severely failed to converge" at E=0 precisely because the anomalous zero-mode spectral weight corrupts the multifractal statistics.

---

## FCC vs Bipartite: Side-by-Side

| Property | SC (bipartite) | BCC NN (bipartite) | Honeycomb (bipartite) | FCC (non-bipartite) |
|-----------|---------------|-------------------|----------------------|---------------------|
| ΓHΓ = −H | ✓ | ✓ | ✓ | ✗ |
| ρ(E) = ρ(−E) | ✓ | ✓ | ✓ | ✗ |
| E=0 DOS feature | Flat (unremarkable) | Sharp peak | Dip (Dirac, ρ→0) | No special feature |
| Zero modes under dilution | ✓ (~√N) | ✓ (~√N) | ✓ (~√N) | ✗ (none guaranteed) |
| Class at E=0 (diluted) | BDI | BDI | AIII | AI |
| Class at E≠0 (diluted) | AI | AI | AI | AI |
| ν at E=0 | Unknown (BDI) | Unknown (BDI) | Known (AIII 2D) | 1.622 (AI) |
| ν at E≠0 | 1.622 | 1.622 | 1.622 | 1.622 |

For FCC quantum percolation: the physics at E=0 is **identical** to E≠0. Class AI throughout. No chiral anomaly, no zero modes, no universality class change. You could run TMM at E=0 on FCC and expect to get the same ν=1.622 as any other energy. p_q(FCC) would follow the standard AI mobility edge shape — no divergence as E→0.

---

## Odd Cycles and Why They Break Everything

FCC has triangular faces → odd cycles. Here is why this kills the bipartite structure precisely:

**Bipartite ↔ no odd cycles:** if you try to 2-color the FCC lattice starting from one atom (say, color it A), you must color all its neighbors B, then all their neighbors A, and so on. But when you follow a triangular face (three atoms mutually connected: i-j-k-i), you start at A, go to B, go to A... and must return to i with color A — but the path i→j→k→i is a 3-cycle, and the last step k→i requires i to be colored B (opposite of k which is A). Contradiction. The 2-coloring is impossible.

The triangles in FCC come from its 12 nearest neighbors forming a cuboctahedral shell: any two neighbors that are themselves neighbors form a triangle with the central atom. FCC is maximally non-bipartite in some sense.

**BCC NN avoids this:** the 8 nearest-neighbor vectors (±½, ±½, ±½) form a cube, and no two nearest neighbors of an atom are themselves nearest neighbors. No triangles in the NN graph. So BCC NN is bipartite.
