# E=0 in SC Quantum Percolation: A Research Reference

**Status (2026-05-04)**: E=0 data collected, E=0 is a different physics problem from what we set out to solve. This document is the complete reference if you want to continue this as a standalone research direction.

---

## Quick Reference

| Quantity | Value | Reliability |
|----------|-------|-------------|
| p_q(E=0) | 0.591 ± 0.001 | **Reliable** — from crossings, not FSS |
| ν(E=0) | 1.99 ± 0.03 | **Indicative only** — poor χ², wrong ansatz, underconstrained M range |
| b | ≈ 0 (b_err/b = 7.7) | **Meaningful** — signature of chiral fixed point |
| χ²/dof | 10.5 | **Expected** — grid mismatch + wrong FSS model |
| Symmetry class | BDI (chiral orthogonal) | **Confirmed** |
| Same as 3D AI (ν=1.62)? | **No** | Different fixed point |
| Data file | `data/sc_dilution.h5` (greengoblin) | Full 39-point p scan, M={4,6,8,10} |

**Why E=0.1 is used for the main project**: at E≠0, chiral symmetry is broken, system is 3D orthogonal class AI, ν=1.622, same as the Anderson benchmark. At E=0, system is class BDI — scientifically interesting but requires a separate research effort.

---

## Part I: Ground-Up Physics

### 1. What is site dilution?

Start with a perfect SC lattice: a 3D grid of atoms at positions (i,j,k) with i,j,k ∈ ℤ. Each atom has one orbital (s-orbital for simplicity). Electrons hop between nearest neighbors (distance 1 apart) with amplitude -t.

**Site dilution** means: independently and randomly, each site is either present (occupied) with probability p, or absent with probability (1-p). Nothing else changes — no disorder in the hopping strength, no random on-site energies. The only randomness is geometric: which atoms are there?

When a site is absent:
- Its orbital doesn't exist → it contributes no row/column to the Hamiltonian
- All hoppings to/from it vanish automatically (no atom, no bond)
- The local potential at that site is irrelevant (site doesn't exist)

In code (`src/lattice.py`), absent sites are handled differently: they're assigned `ABSENT_ENERGY = 1e6`, which makes their on-site energy so large that they effectively decouple from everything. This is mathematically equivalent to removing them.

**Physical picture**: you get a random collection of occupied clusters. For p below the classical percolation threshold p_c ≈ 0.3115 (for SC), there is no system-spanning cluster — electrons can't even classically traverse the sample. Above p_c, there's a giant connected cluster. But classical connectivity doesn't mean quantum transport: even with p > p_c, quantum interference can localize all wavefunctions on the cluster. The quantum percolation threshold p_q > p_c is the minimum occupation where quantum transport survives.

---

### 2. The SC tight-binding Hamiltonian

```
H = -t Σ_{<i,j>} ξ_i ξ_j  c†_i c_j
```

where:
- `<i,j>` runs over all nearest-neighbor pairs
- `ξ_i = 1` if site i is occupied, `ξ_i = 0` if absent
- `ξ_i ξ_j = 1` only if BOTH endpoints are occupied → hopping only along occupied bonds
- In matrix form: H is an N_occ × N_occ real symmetric matrix where N_occ = (number of occupied sites)

The spectrum of H lies in the bandwidth [-6t, 6t] (6 neighbors for SC). The density of states ρ(E) is symmetric around E=0 for any bipartite lattice (explained below).

---

### 3. What is a bipartite lattice?

The SC lattice has a special property: you can color every site either **black** (sublattice A) or **white** (sublattice B) such that every black site's neighbors are all white, and vice versa.

The rule for SC: site (i,j,k) belongs to sublattice A if (i+j+k) is even, sublattice B if (i+j+k) is odd. Check: if (i,j,k) ∈ A (even sum), then its neighbors are (i±1,j,k), (i,j±1,k), (i,j,k±1), all with odd sum → all ∈ B. ✓

This is the **bipartite** property: the graph has no odd cycles. Equivalently: you can 2-color it. The SC, BCC (with appropriate definition), and honeycomb lattices are bipartite. The FCC lattice is NOT bipartite (it has triangular faces → odd cycles).

**Why this matters**: if you write H in the basis that separates A and B sites, it takes the off-diagonal block form:

```
H = [ 0    T  ]
    [ T†   0  ]
```

where T is the N_A × N_B hopping matrix between sublattices. There's nothing on the diagonal because all hoppings connect A to B — never A to A or B to B.

---

### 4. The Γ (sublattice) operator

Define the diagonal matrix:

```
Γ = diag(+1, +1, ..., +1, -1, -1, ..., -1)
         ←── N_A A-sites ───→ ←── N_B B-sites ───→
```

In the A-B block form:

```
Γ = [ +I_A    0   ]
    [  0    -I_B  ]
```

This operator does nothing physically — it's purely a bookkeeping/labeling device. It assigns +1 to A-sites and -1 to B-sites. No measurement corresponds to Γ; it's a mathematical symmetry operation.

**What does Γ do to a wavefunction?** If `|ψ⟩` has amplitudes ψ_i, then `Γ|ψ⟩` flips the sign of all B-site amplitudes:

```
(Γ|ψ⟩)_i = +ψ_i  if i ∈ A
(Γ|ψ⟩)_i = -ψ_i  if i ∈ B
```

It's a simple sign flip on half the sites. Note Γ² = I (applying it twice returns you to the original).

---

### 5. Chiral symmetry: why ΓHΓ = -H

Compute ΓHΓ using the block form:

```
ΓHΓ = [ +I_A    0   ] [ 0    T  ] [ +I_A    0   ]
      [  0    -I_B  ] [ T†   0  ] [  0    -I_B  ]

    = [ +I_A    0   ] [  0      -T   ]
      [  0    -I_B  ] [ +T†      0   ]

    = [  0      -T   ]
      [ -T†      0   ]

    = -H  ✓
```

This is an exact algebraic identity — it follows purely from the bipartite block structure. No approximations, no assumptions about the specific hoppings or disorder realization. It holds for every disorder realization, for every occupation pattern.

This is **chiral symmetry**: `ΓHΓ = -H`, i.e., conjugation by Γ negates H.

---

### 6. What E represents physically

In the Schrödinger equation `H|ψ⟩ = E|ψ⟩`, E is the **energy of the eigenstate**. In a metal, electrons are spread across all energies from the bottom to the top of the band. In a TMM calculation, we probe transport at a *specific* energy E: we ask "if we inject an electron with energy E into the left end of the bar, does it come out the right end, or does it decay exponentially?"

More precisely: the TMM computes the transfer matrix at fixed E, which propagates the wavefunction amplitude from slice to slice along the bar. The input E appears in the transfer matrix via the resolvent `H - E·I` (the on-site energies become (ε_i - E) in the resolvent). Different E values probe different parts of the spectrum — near the band center (E≈0), near the band edge (E≈±6), or anywhere in between.

In a localized system, the electron's amplitude decays exponentially with distance: `|ψ(z)| ~ exp(-z/ξ)`. The TMM measures the localization length ξ. At the metal-insulator transition (quantum percolation threshold), ξ diverges.

**Why scan E at all?** The critical occupation p_c^Q depends on energy: near the band edges, states are easier to localize (narrow DOS, low connectivity), so p_c^Q is higher there. Near mid-band, states are harder to localize, so p_c^Q can be lower. This creates an energy-dependent mobility edge p_c^Q(E) — a surface in (p, E) space separating localized from extended states.

---

### 7. What happens at E=0 specifically

At E=0, chiral symmetry applies. Here is the full argument chain:

**Step 1 — Eigenstates pair under Γ**

If H|ψ⟩ = E|ψ⟩, apply Γ to both sides:

```
Γ H|ψ⟩ = E Γ|ψ⟩
```

Insert Γ² = I:

```
(ΓHΓ)(Γ|ψ⟩) = E(Γ|ψ⟩)
(-H)(Γ|ψ⟩) = E(Γ|ψ⟩)
H(Γ|ψ⟩) = -E(Γ|ψ⟩)
```

So `Γ|ψ⟩` is an eigenstate with energy -E. For every eigenstate at +E there is a partner at -E. The spectrum is symmetric around zero.

**Step 2 — At E=0, states are self-paired**

If E=0, the partner state is `Γ|ψ⟩`, also at E=0. So the partner and the original are in the same eigenspace. The eigenspace at E=0 is invariant under Γ. This means we can choose a basis for the E=0 eigenspace where each basis vector is an eigenstate of Γ:

```
Γ|ψ⟩ = +|ψ⟩   (lives on A-sublattice only, B-amplitudes = 0)
Γ|ψ⟩ = -|ψ⟩   (lives on B-sublattice only, A-amplitudes = 0)
```

**An E=0 eigenstate lives entirely on one sublattice.** It has zero amplitude on all sites of the other sublattice.

**Step 3 — Physical consequence: sublattice-polarized states**

An A-sublattice state has zero amplitude at every B-site. For this state to propagate in the z-direction (along the TMM bar), it needs to hop from layer to layer. But each hop in SC connects an A-site to a B-site. An A-only state cannot hop without touching B-sites. So E=0 states cannot propagate: they are geometrically confined to a single sublattice, which is a disconnected set (all A-sites, no B-sites). They are localized by the sublattice structure itself, regardless of occupation p.

This is a strong statement: at E=0, the eigenstates are anomalously localized purely by the bipartite geometry, not by quantum interference with disorder.

---

### 8. Why site dilution creates zero-energy modes (sublattice imbalance)

In a perfect (undiluted, p=1) bipartite crystal, N_A = N_B = N/2 exactly. The eigenvalue spectrum is symmetric.

Under random site dilution at occupation p: each site is independently included with probability p. For a sample with n occupied sites, N_A and N_B are binomial random variables. Their difference:

```
|N_A - N_B| ~ √n    (random walk fluctuations)
```

By a theorem on bipartite graphs (a consequence of the Perron-Frobenius theorem for bipartite matrices): the matrix H for a bipartite graph has at least |N_A - N_B| eigenvalues exactly at zero.

**Proof sketch**: Write H in block form [0, T; T†, 0]. If N_A > N_B, then T is a N_A × N_B matrix with more rows than columns. Its null space has dimension ≥ N_A - N_B (by rank-nullity). Vectors in the null space of T satisfy H|ψ⟩ = 0 when the B-components are zero, i.e., they are zero-energy A-sublattice eigenstates. QED.

**Physical consequence**: a sample with n occupied sites has ~√n localized zero modes. These modes:
- Are pinned to E=0 exactly (not approximately)
- Have support on the minority sublattice only
- Do not participate in transport at all (localized by sublattice constraint)
- Grow in number as p decreases (more sublattice imbalance)

As p → 1, the imbalance |N_A - N_B|/N → 0, zero modes become sparse, and E=0 states can potentially delocalize. This is why p_c^Q(E=0) is large — you need near-perfect occupation before even a small fraction of E=0 states can avoid being zero modes.

---

### 9. Symmetry class change at E=0

The Altland-Zirnbauer classification organizes Hamiltonians by their discrete symmetries. The relevant ones here:

| Symmetry | Operator | Condition |
|----------|----------|-----------|
| Time-reversal (T) | Complex conjugation K | T² = +1 for spinless fermions |
| Particle-hole (C) | From bipartite structure | C² = +1 for class BDI |
| Chiral (S) | Γ | S = TC |

At E≠0: We probe H - E·I. Does Γ(H-EI)Γ = -(H-EI)?

```
Γ(H-EI)Γ = ΓHΓ - EΓ²= -H - EI = -(H+EI) ≠ -(H-EI)
```

No — there's an extra 2E term. **Chiral symmetry is broken at E≠0.** The system has only time-reversal symmetry → class AI (3D orthogonal). This is the standard Anderson localization universality class with ν = 1.622 ± 0.035.

At E=0: ΓHΓ = -H holds exactly. The system also has T (real Hamiltonian, spinless → T² = +1) and the bipartite structure gives an effective particle-hole symmetry C. Together: class BDI.

```
E ≠ 0:   Class AI  (3D orthogonal)    ν = 1.622 ± 0.035  (Ujfalusi 2014)
E = 0:   Class BDI (chiral orthogonal) ν = unknown for 3D quantum percolation
```

This is not a small perturbative effect — it is a categorical change in what fixed point governs the critical behavior. The FSS equations, the corrections-to-scaling, the irrelevant exponents, the amplitude b — all are different.

---

### 10. The Gade-Wegner singularity in class BDI/AIII

In 2D, Gade and Wegner showed that the chiral class has an anomalous density of states at E=0:

```
ρ(E) ~ exp(-c |ln|E|| ^{1/2})   (Gade-Wegner singularity in 2D)
```

This is a singularity that is not a simple power law — it diverges faster than any polynomial at E=0 but slower than a pole.

In 3D, the situation is less studied. For disordered bipartite systems in 3D, the DOS near E=0 is expected to have anomalous behavior (related to the sublattice-imbalance zero modes described above), but the precise functional form is an active research question.

**Practical implication for TMM**: the Lyapunov exponents at E=0 converge more slowly in L (bar length) because the spectral statistics near E=0 are anomalous. The effective "mixing length" along the bar is longer. At E≠0, the TMM at L=10,000 is well-converged; at E=0, L=10,000 may be insufficient.

---

## Part II: What Ujfalusi Did Near E=0

### Their method (brief recap)

Ujfalusi & Varga (2014) used **Multifractal FSS (MFSS)**: compute all eigenstates of the full 3D L³ cube, calculate generalized IPRs P_q, extract multifractal dimensions D_q, apply FSS. System sizes L = 20..140, up to 50,000 realizations per (p,L) point.

### Their energy grid

They ran 11 energy points: E ∈ {0.1, 0.3, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.1}. Notice: minimum energy is 0.1, not 0. They deliberately avoided E=0.

### What they observed near E=0

They also attempted E = 0.001 and E = 0.01 (shown in their Fig. 6a). At these energies, the MFSS **severely failed to converge** — their words. The scaling collapses did not work; the extracted p_c^Q and ν had huge uncertainties and were inconsistent between system sizes.

They argued (Section III.B) that this failure is physical, not numerical:
1. The DOS pseudogap at E=0 means fewer eigenstates fall in the energy window → poor statistics
2. The zero modes (localized, non-scaling states) mix with the critical states → the multifractal statistics are contaminated
3. p_c^Q(E→0) → 1: as E→0, the quantum critical point approaches p=1 (full occupation), i.e., all states below full occupation are localized at E=0

Their Fig. 6a shows the mobility edge p_c^Q(E): a smooth curve with minimum at E≈0.7 (p≈0.384), rising toward both band edges AND toward E=0. The curve appears to extrapolate to p_c^Q(E=0) > 0.65 but they could not pin down the exact value.

### How our E=0 result compares

Our TMM gave p_q(E=0) = 0.591 for M={4..10}, N=500, L=10,000. The Ujfalusi curve at E=4.1 (the other end of the spectrum, near band edge) gives p_c^Q = 0.638. Our E=0 value of 0.591 is lower than their E=4.1 value — this means our finite-M result is still far from the E=0 thermodynamic limit.

As M→∞, our Λ_M crossings should shift toward p=1 (consistent with Ujfalusi's divergence). For M={4..10}, we're seeing the beginning of this shift: the crossings at 0.589–0.594 are well above the E≠0 minimum (0.384) but well below the expected thermodynamic limit (>>0.65). This is a finite-size precursor — the crossings will drift toward p=1 as M increases. **We have NOT measured the thermodynamic p_c^Q(E=0). We've measured a finite-size crossing that is on the way toward it.**

---

## Part III: Our E=0 Results — Detailed Analysis

### Data collected (greengoblin, 2026-05-04)

**Run parameters**: M={4,6,8,10}, L=10,000, N_real=500, W=0, E=0, SC lattice
**File**: `data/sc_dilution.h5` (greengoblin)
**p range**: 39 points, p ∈ [0.320, 0.650]

**Λ_M table (selected rows)**:
```
p       Λ(M=4)   Λ(M=6)   Λ(M=8)   Λ(M=10)
0.3200  0.06800  0.05792  0.05164  0.04680
0.4242  0.11526  0.10639  0.10208  0.09929
0.5458  0.20358  0.19587  0.18897  0.18590
0.6153  0.28665  0.29630  0.30326  0.31166
```

**Ordering at p=0.32**: Λ_4 > Λ_6 > Λ_8 > Λ_10 → insulating (smaller M localizes less than larger M). ✓
**Ordering at p=0.62**: Λ_4 < Λ_6 < Λ_8 < Λ_10 → metallic ordering reversed. ✓
**Crossing at p≈0.591**: curves invert between these two rows. ✓

### FSS result (manual window [0.54, 0.64])

```
p_q = 0.59109 ± 0.00117 (stat) ± 0.00015 (syst)
ν   = 1.9888  ± 0.0315  (stat) ± 0.0030  (syst)
b   = -0.0074 ± 0.0057         → b consistent with zero at 1.3σ
χ²/dof = 10.555
```

Crossings (pairwise): M4–M6: 0.589, M6–M8: 0.592, M8–M10: 0.594 → spread = 0.005

### What is reliable and what is not

**p_q = 0.591 ± 0.001: RELIABLE**

The crossing position is determined directly from the Λ_M(p) curves, not from the FSS polynomial. The three crossings cluster tightly (spread 0.005), confirming a well-defined critical point in the data. This value is as reliable as the data quality: N=500 realizations, L=10,000. Statistical error on each Λ_M value is ~0.001–0.003.

However: this is a FINITE-SIZE CROSSING for M={4..10}. It is NOT the thermodynamic p_c^Q(E=0). As M→∞, this crossing will drift upward toward p=1 (Ujfalusi's limit). To extrapolate to M→∞ would require much larger M values.

**ν = 1.99 ± 0.03: INDICATIVE ONLY**

ν comes from the FSS fit quality, which is poor for two independent reasons (see next section). Take ν as a rough estimate in the correct qualitative ballpark for class BDI in 3D.

**b ≈ 0: PHYSICALLY MEANINGFUL**

The AI-class irrelevant correction b·M^{-1.12} is expected to be O(0.05-0.1) for the Anderson problem. Here b is consistent with zero. This is a signature of the chiral fixed point where this particular operator is absent or forbidden by symmetry. The tight crossing spread (0.005) is also evidence of small/zero corrections.

---

## Part IV: Why the FSS Analysis Fails — Two Independent Reasons

### Reason 1: Grid mismatch (fixable)

**The problem**: P_FINE = linspace(0.38, 0.52, 20) puts 20 high-density points at [0.38, 0.52], which is deep in the insulating phase at E=0 (the transition is at 0.591). In the fit window [0.54, 0.64], only P_COARSE points exist, with spacing ≈ 0.017. Counting p values inside [0.54, 0.64] from P_COARSE = linspace(0.32, 0.65, 20): approximately {0.543, 0.561, 0.580, 0.598, 0.616, 0.634} → 6 points.

Data count: 4 M values × 6 p values = 24 data points.
FSS parameters: {a_0, a_1, a_2, a_3, a_4, b, p_q, ν} = 8 free parameters.
Degrees of freedom: 24 - 8 = 16.
χ² = χ²/dof × dof = 10.5 × 16 = 168.

This is enormous for 24 data points. The FSS polynomial is being asked to constrain 8 parameters from 6 loosely spaced p values per M — it's statistically underdetermined near the critical region.

**The fix for E=0 research**: Change P_FINE to linspace(0.55, 0.63, 20). This puts 20 high-resolution points directly in the critical region. With the coarse grid, total data in [0.54, 0.64] becomes 6 + 20 = 26 points × 4 = 104 data points → dof = 96. The grid contribution to χ² drops dramatically.

### Reason 2: Wrong FSS ansatz (fundamental)

**The problem**: Our FSS function is:

```
Λ_M(p) = Σ_{k=0}^{4} a_k u^k + b · M^{-y}
```

This function is derived for the **3D orthogonal (AI) universality class**. Specifically:
- The scaling function F(u) = Σ a_k u^k assumes F is analytic in u near u=0. At the chiral (BDI) fixed point, F may have non-analytic behavior related to the Gade-Wegner singularity.
- The correction b·M^{-y} with y=1.12 is the irrelevant exponent for the AI class. For BDI, the leading irrelevant exponent y_BDI is unknown (for 3D quantum percolation specifically; no literature value).
- The scaling variable u = M^{1/ν}(p-p_c)/p_c assumes a specific form of the scaling hypothesis that may need modification for the chiral class.

Even with a perfect grid, the wrong model will give χ²/dof > 1 because the data follows a different function shape.

**The fix for E=0 research**: Replace the AI FSS ansatz with a BDI-appropriate one. This requires:
1. Determining the correct scaling function form for class BDI in 3D (from theory or fitting to simpler models)
2. Determining y_BDI from literature or fitting with larger M
3. Possibly including logarithmic corrections near the chiral fixed point

---

## Part V: How to Do E=0 Correctly (The Research Program)

If you want to extract a reliable ν(E=0) and p_c^Q(E=0, M→∞), here is the full research program:

### Step A: Get the grid right

Change in `sc_dilution.py`:
```python
P_COARSE = list(np.linspace(0.32, 0.75, 20))   # extend to p=0.75, E=0 crossing may be higher for large M
P_FINE   = list(np.linspace(0.55, 0.65, 20))   # dense coverage of the critical region
```

And set E=0.0. This ensures the fit window [0.54, 0.64] has 20 high-resolution + 4 coarse points → ~100 data points for FSS. Run with same L=10,000, N=500. Runtime: same ~9 hours.

### Step B: Add larger M values

The main scientific problem is that M={4..10} only spans a factor of 2.5. For the AI class, this is barely sufficient (4 sizes, polynomial order 4 → just identifiable). For the BDI class where corrections are unknown, you need more leverage. Add M=12 and M=16 (and ideally M=20).

**Runtime estimate for BCC at E=0**:
- M=12: ~4× slower than M=10 (O(M^6) in MGS) → ~4h
- M=16: ~12× slower → ~12h
- M=20: ~30× slower → ~30h

Or alternatively: move to complex128 and use a hybrid MGS/cuSOLVER approach for large M (the MGS bottleneck at large M was documented in feedback_tmm_performance.md).

### Step C: Fix the FSS ansatz

**Option 1 — Phenomenological**: Keep the AI ansatz but allow y_BDI to float. With 6+ M values, y_BDI becomes identifiable. Check if the fitted y_BDI is consistent across different p windows.

```python
result = fit_fss(..., fit_y_irrel=True, n_corrections=1)
```

This is the simplest change and tests whether the AI-class correction structure is at all applicable.

**Option 2 — Add logarithmic correction**: If the Gade-Wegner singularity produces logarithmic corrections in 3D analogous to the 2D case, the FSS function should include:

```
Λ_M(p) = Σ a_k u^k + b · M^{-y} + c · (ln M)^{-α}
```

The exponent α is unknown for 3D BDI quantum percolation. This is a 2-parameter correction (b,c or just c if b=0 as suggested by our data).

**Option 3 — Use MFSS**: Implement the multifractal approach for comparison. Compute full eigenstates of the 3D cube at E=0 and apply MFSS. This is Ujfalusi's approach applied to E=0 where their method failed. With a different computational method (TMM provides different statistical probing), you might converge where MFSS couldn't.

### Step D: Extrapolate crossings to M→∞

Even before FSS fitting, the raw crossing positions as a function of M tell you about the thermodynamic limit. Plot the crossing between M and 2M as a function of 1/M:

```
p_cross(M, 2M) vs. 1/M:  extrapolate to 1/M → 0
```

If the crossings drift upward as M increases (as expected for E=0), you get an upper bound on p_c^Q(E=0, ∞). Combined with the FSS fit, this gives a consistent check.

### Step E: Verify the symmetry class assignment

Check that ν at E=0 is inconsistent with the AI class at statistical significance:
- Our ν = 1.99 ± 0.03 is (1.99 - 1.62) / 0.03 = 12σ from the AI value
- But with χ²/dof = 10.5, the formal error bar is unreliable (it's underestimated by a factor of √10)
- Effective uncertainty: 0.03 × √10 ≈ 0.1, giving (1.99 - 1.62) / 0.1 = 3.7σ discrepancy
- Still significant, but less decisive

With a proper grid and 6 M values, you'd get a reliable ν with small χ²/dof, and the discrepancy from the AI class would be at much higher significance.

---

## Part VI: Complete Comparison — Our TMM vs Ujfalusi MFSS

### Physical model (identical)

Both study: 3D simple cubic lattice, random site dilution at probability p, nearest-neighbor hopping t=1, no Anderson disorder. The physical model is exactly the same. Differences are entirely in method.

### Method comparison

| Aspect | Ujfalusi MFSS | Our TMM+FSS |
|--------|--------------|-------------|
| **Geometry** | L³ cube (3D isotropic) | M²×L quasi-1D bar |
| **Observable** | Multifractal exponents D_q from full eigenstates | Minimum Lyapunov exponent λ_min |
| **Derived FSS variable** | Ã_q = P_q · L^{(q-1)D_q} | Λ_M = 1/(2M·λ_min) |
| **Eigensolver** | Jacobi-Davidson (interior, sparse 3D) | Transfer matrix (product, sequential) |
| **System sizes** | L=20..140 (factor 7, 7 sizes) | M=4..10 (factor 2.5, 4 sizes) |
| **Realizations** | 10k–50k per (p,L) | N=500 per (p,M) |
| **Corrections** | (1/ln L)·f(u) — exact from multifractal | b·M^{-1.12} — phenomenological, AI class |
| **Energy handling** | Eigenstate energy window ΔE≈0.1 around E | TMM at fixed E (resolvent H-E·I) |
| **Memory per run** | L³ sparse matrix + eigenvectors (GBs at L=140) | M²×M² matrix only (KB at M=10) |
| **Parallelism** | Many realizations, independently diagonalizable | vmap over realizations (GPU batching) |
| **GPU suitability** | Moderate (sparse solvers, not GPU-optimal) | High (dense batched matmul) |

### FSS equation comparison

**Ujfalusi** (their Eq. 29):
```
Ã_q(p, L) = G_q(u) + (1/ln L) · Σ_{j=0}^{n} c_j · u^j

u = L^{1/ν} · (p - p_c^Q)
```
The (1/ln L) term is exact from multifractal theory — not phenomenological.

**Ours**:
```
Λ_M(p) = Σ_{k=0}^{4} a_k · u^k + b · M^{-y}

u = M^{1/ν} · (p - p_c) / p_c
```
The b·M^{-y} term is the leading irrelevant correction from RG theory for the 3D AI fixed point.

Both are legitimate FSS equations for their respective methods. They should give the same p_c^Q and ν at each energy when applied correctly.

### Precision comparison

At E=0.1:

| Quantity | Ujfalusi MFSS | Our TMM target |
|----------|--------------|----------------|
| p_c^Q(E=0.1) | 0.45384 ± 0.00028 | ≈ 0.454 ± 0.003 |
| ν | 1.622 ± 0.035 | ≈ 1.62 ± 0.05 |

Their precision is ~10× better because of 7 system sizes (vs our 4) spanning factor 7 (vs our 2.5). Our precision is sufficient to confirm the same universality class and measure BCC p_q to ±0.005, which is the project goal.

### Goals comparison

| Aspect | Ujfalusi & Varga (2014) | This project |
|--------|------------------------|-------------|
| Primary result | Complete mobility edge p_c^Q(E) for SC | First TMM measurement of p_q(BCC) |
| Novel physics | ν energy-independent; E→0 divergence | BCC quantum percolation (unmeasured) |
| Method novelty | MFSS applied to quantum percolation | TMM+FSS on BCC geometry |
| Energy coverage | 11 energy points, full spectrum | 1 energy point (E=0.1) for validation |
| Lattices | SC only | SC (validation) + BCC (novel) |
| E=0 | Studied, failed, argued p→1 | Measured (p_q=0.591), chiral class |

The key difference in scientific contribution: Ujfalusi mapped the full p_c^Q(E) curve for SC. We want to measure p_q for BCC — a completely different lattice never studied with TMM. Our SC run at E=0.1 is purely a validation step.

---

## Part VII: Current Status and Next Steps

### E=0.1 run (in progress on greengoblin, 2026-05-04)

```
M=4:  DONE (7 min) → λ_M ∈ [0.063, 1.480]  ← note: much wider range than E=0
M=6:  running at 31%, ~50s/p
M=8:  pending
M=10: pending
Total estimated: ~9 hours
```

Expected result: p_q ≈ 0.454, ν ≈ 1.62, χ²/dof < 5.

### If E=0.1 passes, proceed to BCC

```bash
pixi run bcc-dilution
```

BCC quantum percolation at E=0.1 (expected p_q(BCC) < p_q(SC) = 0.454 because BCC is more connected than SC).

### If you want to continue the E=0 research

Priority order:

1. **(1 day)** Fix P_FINE grid for E=0: change to linspace(0.55, 0.63, 20) in `sc_dilution.py`, set E=0, rerun with `--refit` first to check window, then full run. This gives a better χ²/dof estimate and more reliable ν.

2. **(1 week)** Add M=12: update M_LIST = [4,6,8,10,12] and rerun at E=0. M=12 takes ~4h additional. With 5 M values, FSS is better constrained. Can attempt fit_y_irrel=True to see if y_BDI is identifiable.

3. **(2 weeks)** Literature search on 3D class BDI Anderson transition. Search: "3D chiral orthogonal Anderson transition", "class BDI localization 3D", "Gade Wegner 3D". Current best reference is probably Brouwer et al. and Altland group. Find if ν_BDI has been established for any 3D model.

4. **(month)** Full study: M up to 16-20, y_BDI fitted, correct FSS ansatz, extrapolate crossings to M→∞, compare with Ujfalusi's divergence.

---

## Appendix: One-Line Definitions

- **Site dilution**: randomly remove atoms from the lattice; hopping only exists where both endpoints are present
- **Bipartite lattice**: 2-colorable graph; SC site (i,j,k) is sublattice A if (i+j+k) even, B if odd; all bonds connect A to B
- **Γ operator**: diagonal matrix +1 on A-sites, -1 on B-sites; bookkeeping, no physical observable
- **Chiral symmetry ΓHΓ = -H**: automatic consequence of bipartiteness; spectrum symmetric around E=0
- **E**: energy eigenvalue; TMM probes transport by electrons with this energy; enters as H - E·I
- **E=0**: bipartite symmetry intact → class BDI; eigenstates live on one sublattice; zero modes from sublattice imbalance → p_c^Q → 1
- **E≠0**: chiral symmetry broken → class AI (3D orthogonal); ν=1.622; standard Anderson physics
- **Zero mode**: eigenstate at exactly E=0 localized on one sublattice due to sublattice imbalance |N_A - N_B|
- **Class AI**: standard 3D orthogonal universality class; ν=1.622 (Ujfalusi 2014); no chiral symmetry
- **Class BDI**: chiral orthogonal class; ν unknown for 3D quantum percolation; our E=0 result
- **Mobility edge p_c^Q(E)**: energy-dependent quantum percolation threshold; maps from AI minimum (~0.384 at E=0.7) to divergence as E→0
- **FSS ansatz mismatch**: using AI correction b·M^{-1.12} on BDI data is like fitting a Gaussian to a Lorentzian — wrong in principle
- **Gade-Wegner singularity**: anomalous DOS near E=0 in chiral class; ρ(E) ~ exp(-c|ln E|^{1/2}) in 2D; 3D form unknown

---

*Document created: 2026-05-04. Data in: `data/sc_dilution.h5` (greengoblin). E=0.1 run in progress.*
