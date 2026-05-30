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

$$
H = -t \sum_{\langle i,j \rangle} |i\rangle\langle j|
$$

where the sum runs over nearest-neighbour pairs $\langle i,j \rangle$ with both sites occupied.
Each site is independently occupied with probability $p$; empty otherwise. No on-site
disorder ($W = 0$). The only randomness is geometric (which sites are occupied).

### 1.2 Classical vs quantum percolation

**Classical percolation**: existence of connected path of occupied sites spanning
the system? Below $p_c$ (the classical threshold), all clusters are finite - no spanning
path exists. Above $p_c$, a giant cluster appears. For SC: $p_c = 0.3115$.

**Quantum percolation**: even given a connected spanning
cluster ($p > p_c$), do the quantum eigenstates on that cluster extend across the system, or do
they localize? The answer is not automatic - quantum interference can localize states
even on a connected graph. The quantum threshold $p_q > p_c$. The gap $p_q - p_c$ is the
quantum localization cost.

Below $p_q$: all eigenstates are exponentially localized. No quantum transport.
Above $p_q$: extended states exist. Quantum transport survives.

This is a **quantum phase transition** driven by a geometric parameter $p$.

---

## 2. Localized States and the Localization Length

### 2.1 What localization means

An eigenstate $|\psi_E\rangle$ is **localized** if its real-space amplitude decays exponentially
from some center $\mathbf{r}_0$:

$$
|\langle \mathbf{r}|\psi_E\rangle|^2 \sim A\, e^{-|\mathbf{r} - \mathbf{r}_0|/\xi_\text{loc}},
\qquad |\mathbf{r} - \mathbf{r}_0| \to \infty
$$

The **localization length** $\xi_\text{loc}$ is the decay length. It is specific to the eigenstate
and depends on energy $E$ and the disorder realization.

An eigenstate is **extended** if $|\langle\mathbf{r}|\psi_E\rangle|^2$ does not decay exponentially - it spreads
across the whole system, with amplitude $\sim 1/V$ ($V$ = system volume).

The distinction is sharp in the thermodynamic limit. At finite system size, all
eigenstates look somewhat spread out; the difference only becomes clear as $L \to \infty$.

### 2.2 Destructive interference

A classical particle scatters off impurities and diffuses. A quantum particle also
scatters, but its wavefunction accumulates **phase** along each scattering path.
Paths that return to the origin interfere. In 3D with strong enough disorder, the
constructive interference of time-reversed pairs of back-scattering paths suppresses
diffusion - this is **weak localization**. With sufficient disorder, interference
becomes so strong that the wavefunction cannot spread at all: **Anderson localization**.

The mechanism is purely quantum (interference) and has nothing to do with energy
gaps or magnetic fields.

### 2.3 Localization length vs correlation length

This distinction is crucial and often confused.

**Localization length $\xi_\text{loc}$**: a property of individual eigenstates. It measures how
far the wavefunction of a specific state extends before decaying. States deep in the
localized phase have small $\xi_\text{loc}$. States near the mobility edge have large $\xi_\text{loc}$.

**Correlation length $\xi_\text{corr}$**: a property of the phase transition itself. It measures
the spatial scale over which critical fluctuations are correlated. $\xi_\text{corr}$ diverges at the critical
point regardless of which side you approach from.

**How they relate at the Anderson or percolation transition:**

On the localized side ($p < p_q$): the typical localization length of states near the
mobility edge IS the correlation length:

$$
\xi_\text{corr} = \xi_\text{loc} \sim |p - p_q|^{-\nu} \qquad \text{(localized side)}
$$

On the metallic side ($p > p_q$): $\xi_\text{corr}$ is the scale above which transport becomes
diffusive. Below $\xi_\text{corr}$ the system still has critical (fractal) fluctuations. Above
it the system looks like a normal metal.

$$
\xi_\text{corr} \sim |p - p_q|^{-\nu} \qquad \text{(metallic side, same exponent } \nu\text{)}
$$

**Key point**: $\xi_\text{corr}$ diverges on BOTH sides with the SAME exponent $\nu$. $\xi_\text{loc}$ only
makes sense on the localized side. They coincide on the localized side and diverge
with the same $\nu$. They are different conceptual quantities but the same number near the transition.

At $p_q$ itself: $\xi_\text{corr} \to \infty$. The system is scale-invariant. Wavefunctions are fractal
(neither localized nor extended in the usual sense - multifractal).

---

## 3. Phase Transition and What $\nu$ Means Physically

### 3.1 Continuous (2nd-order) phase transition

The transition at $p_q$ is continuous: there is no discontinuous jump in any observable.
Instead, everything changes smoothly and the localization length diverges as a power
law. This is the hallmark of a second-order transition.

Compare to a first-order transition (e.g. ice melting) where there IS a jump. Here,
as $p \to p_q$ from below, $\xi_\text{loc}$ grows without bound: the insulation decreases until at
exactly $p_q$ it is not insulating at all.

### 3.2 Correlation length exponent $\nu$

$$
\xi_\text{corr} \sim |p - p_q|^{-\nu}
$$

defines $\nu$. Physically $\nu$ tells you how fast the correlation length grows as you approach
the transition.

- $\nu = 1$: $\xi$ doubles every time you halve $|p - p_q|$. Linear approach.
- $\nu = 1.622$ (our case): $\xi$ grows faster. Halving the distance to $p_q$ multiplies $\xi$ by $2^{1.62} \approx 3.1$.

Concretely: 1% away from $p_q$ ($|p - p_q|/p_q = 0.01$), the correlation length is

$$
\xi \sim (0.01)^{-1.62} \times a \approx 1800 \text{ lattice spacings}
$$

This is why finite-size effects are large near the transition. The system of size $M = 10$ is
very small compared to $\xi_\text{corr} = 1800$.

### 3.3 Harris criterion

A fundamental question: if I add weak disorder to a clean critical point, does the
transition survive or get washed out? Harris (1974) showed:

$$
\begin{cases}
d\nu > 2 & \text{transition stable against weak disorder} \\
d\nu < 2 & \text{disorder relevant; clean critical point destroyed}
\end{cases}
$$

For us: $d = 3$, $\nu = 1.622$ $\Rightarrow$ $d\nu = 4.87 > 2$. The quantum percolation transition is
stable. The exponent $\nu = 1.622$ is self-consistent: disorder at this level of strength
does not destabilize the transition.

(Equivalently: $\nu > 2/d = 2/3$. This is sometimes stated as the "Chayes et al. bound"
for disordered systems: $\nu$ must be at least $2/3$ in any dimension.)

---

## 4. Renormalization Group (RG)

### 4.1 RG operation

**What problem does RG solve?** Near a phase transition, fluctuations occur on all
length scales from the lattice spacing $a$ up to $\xi$. Direct calculation is
intractable. RG disentangles this by systematically eliminating the short-distance
degrees of freedom, producing an effective theory for the long-distance physics.

**The blocking step.** Group every $b^d$ sites into a block (here $d = 3$; typically $b = 2$):

<svg xmlns="http://www.w3.org/2000/svg" width="272" height="138"
     viewBox="0 0 272 138" style="display:block;margin:1em auto;">
  <style>text{font-family:"Times New Roman",Times,serif;}</style>
  <defs>
    <marker id="la" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="black"/>
    </marker>
  </defs>

  <!-- original bonds -->
  <g stroke="black" stroke-width="0.8">
    <!-- horizontal -->
    <line x1="22" y1="24" x2="46" y2="24"/><line x1="46" y1="24" x2="70" y2="24"/><line x1="70" y1="24" x2="94" y2="24"/>
    <line x1="22" y1="48" x2="46" y2="48"/><line x1="46" y1="48" x2="70" y2="48"/><line x1="70" y1="48" x2="94" y2="48"/>
    <line x1="22" y1="72" x2="46" y2="72"/><line x1="46" y1="72" x2="70" y2="72"/><line x1="70" y1="72" x2="94" y2="72"/>
    <line x1="22" y1="96" x2="46" y2="96"/><line x1="46" y1="96" x2="70" y2="96"/><line x1="70" y1="96" x2="94" y2="96"/>
    <!-- vertical -->
    <line x1="22" y1="24" x2="22" y2="48"/><line x1="22" y1="48" x2="22" y2="72"/><line x1="22" y1="72" x2="22" y2="96"/>
    <line x1="46" y1="24" x2="46" y2="48"/><line x1="46" y1="48" x2="46" y2="72"/><line x1="46" y1="72" x2="46" y2="96"/>
    <line x1="70" y1="24" x2="70" y2="48"/><line x1="70" y1="48" x2="70" y2="72"/><line x1="70" y1="72" x2="70" y2="96"/>
    <line x1="94" y1="24" x2="94" y2="48"/><line x1="94" y1="48" x2="94" y2="72"/><line x1="94" y1="72" x2="94" y2="96"/>
  </g>

  <!-- block boundaries -->
  <g fill="none" stroke="black" stroke-width="1" stroke-dasharray="4,3">
    <rect x="10" y="12" width="48" height="48"/>
    <rect x="58" y="12" width="48" height="48"/>
    <rect x="10" y="60" width="48" height="48"/>
    <rect x="58" y="60" width="48" height="48"/>
  </g>

  <!-- original sites -->
  <g fill="black">
    <circle cx="22" cy="24" r="3"/><circle cx="46" cy="24" r="3"/><circle cx="70" cy="24" r="3"/><circle cx="96" cy="24" r="3"/>
    <circle cx="22" cy="48" r="3"/><circle cx="46" cy="48" r="3"/><circle cx="70" cy="48" r="3"/><circle cx="96" cy="48" r="3"/>
    <circle cx="22" cy="72" r="3"/><circle cx="46" cy="72" r="3"/><circle cx="70" cy="72" r="3"/><circle cx="96" cy="72" r="3"/>
    <circle cx="22" cy="96" r="3"/><circle cx="46" cy="96" r="3"/><circle cx="70" cy="96" r="3"/><circle cx="96" cy="96" r="3"/>
  </g>

  <!-- label -->
  <text x="58" y="124" text-anchor="middle" font-size="12" font-style="italic">spacing a</text>

  <!-- arrow -->
  <line x1="112" y1="60" x2="148" y2="60" stroke="black" stroke-width="1.2" marker-end="url(#la)"/>

  <!-- blocked bonds -->
  <g stroke="black" stroke-width="1">
    <line x1="178" y1="36" x2="226" y2="36"/>
    <line x1="178" y1="84" x2="226" y2="84"/>
    <line x1="178" y1="36" x2="178" y2="84"/>
    <line x1="226" y1="36" x2="226" y2="84"/>
  </g>

  <!-- blocked sites (larger) -->
  <g fill="black">
    <circle cx="178" cy="36" r="5.5"/>
    <circle cx="226" cy="36" r="5.5"/>
    <circle cx="178" cy="84" r="5.5"/>
    <circle cx="226" cy="84" r="5.5"/>
  </g>

  <!-- label -->
  <text x="202" y="124" text-anchor="middle" font-size="12" font-style="italic">spacing ba</text>
</svg>

Within each block there are $b^d$ sites with their internal Hamiltonian $H_\text{block}$.
Identify the lowest-energy eigenstate of each block - this becomes the **effective
site** (the "super-site"). Compute the effective hopping between adjacent super-sites
as the matrix element of the inter-block coupling in this low-energy subspace. This
gives a new Hamiltonian $H'$ that:

- acts on a lattice with spacing $ba$ (coarser by $b$)
- has the same tight-binding form as $H$
- has renormalized parameters: effective hopping $t'$, effective occupation $p'$

Eliminating the intra-block degrees of freedom to produce $H'$ is what **integrating
out** means: the $b^d - 1$ internal modes per block are removed by projecting onto the
low-energy subspace, not discarded - their effect is encoded in the renormalized $t'$, $p'$.

**The RG map.** Let $\{g\} = \{p, t, E, \ldots\}$ denote all parameters of $H$. One
blocking step defines a map:

$$\{g\} \;\xrightarrow{\;R_b\;}\; \{g'\} = R_b\bigl[\{g\}\bigr]$$

Repeated application traces a **trajectory** in the space of Hamiltonians: the RG flow.

**Fixed points.** A fixed point $\{g^*\}$ satisfies $R_b[\{g^*\}] = \{g^*\}$: coarse-graining
leaves $H$ unchanged. This is **scale invariance** - the system looks identical at all
length scales. Scale invariance requires no characteristic length scale, i.e. $\xi \to \infty$.
Therefore **every critical point is an RG fixed point**.

For our system there are three fixed points:

- **Insulating** ($p \to 0$): localized systems flow here under repeated blocking.
- **Metallic** ($p \to 1$): extended systems flow here.
- **Critical** ($p = p_q$): the transition. Unstable - any $\delta p \neq 0$ flows away.

<svg xmlns="http://www.w3.org/2000/svg" width="480" height="112"
     viewBox="0 0 480 112" style="display:block;margin:1em auto;">
  <style>text{font-family:"Times New Roman",Times,serif;}</style>
  <defs>
    <marker id="ra" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="black"/>
    </marker>
  </defs>

  <!-- insulating node -->
  <circle cx="72" cy="56" r="42" fill="white" stroke="black" stroke-width="1.5"/>
  <text x="72" y="51" text-anchor="middle" font-size="13">p = 0</text>
  <text x="72" y="68" text-anchor="middle" font-size="12" font-style="italic">insulating</text>

  <!-- critical node (double ring) -->
  <circle cx="240" cy="56" r="42" fill="white" stroke="black" stroke-width="1.5"/>
  <circle cx="240" cy="56" r="35" fill="white" stroke="black" stroke-width="0.7"/>
  <text x="240" y="51" text-anchor="middle" font-size="13">p = p<tspan baseline-shift="sub" font-size="10">q</tspan></text>
  <text x="240" y="68" text-anchor="middle" font-size="12" font-style="italic">critical</text>

  <!-- metallic node -->
  <circle cx="408" cy="56" r="42" fill="white" stroke="black" stroke-width="1.5"/>
  <text x="408" y="51" text-anchor="middle" font-size="13">p = 1</text>
  <text x="408" y="68" text-anchor="middle" font-size="12" font-style="italic">metallic</text>

  <!-- arrow: critical → insulating -->
  <line x1="196" y1="43" x2="117" y2="43" stroke="black" stroke-width="1.2" marker-end="url(#ra)"/>
  <text x="156" y="36" text-anchor="middle" font-size="12" font-style="italic">&#948;p &lt; 0</text>

  <!-- arrow: critical → metallic -->
  <line x1="284" y1="43" x2="363" y2="43" stroke="black" stroke-width="1.2" marker-end="url(#ra)"/>
  <text x="323" y="36" text-anchor="middle" font-size="12" font-style="italic">&#948;p &gt; 0</text>
</svg>

**Linearization.** Near $p_q$, write $\delta p = p - p_q$ and linearize $R_b$:

$$\delta p \;\to\; \delta p' = b^{1/\nu} \cdot \delta p$$

The exponent $1/\nu$ is the RG eigenvalue at the fixed point - it comes from
diagonalizing the linearized map $\partial R_b / \partial g\big|_{g^*}$, not from assumption.
Since $b^{1/\nu} > 1$, $\delta p$ grows under coarse-graining: $p - p_q$ is a **relevant
operator** that drives the system away from criticality. This is the input to section 4.2.

### 4.2 Why $\xi \sim |x - x_c|^{-\nu}$ emerges from RG

Near the critical point $x_c$, the deviation $u = x - x_c$ is a **relevant operator**.
Under coarse-graining by factor $b$, it transforms as:

$$
u \;\to\; b^{1/\nu} \cdot u
$$

This is just the statement that $u$ is relevant with RG eigenvalue $1/\nu > 0$. After $n$
steps of coarse-graining (each by factor $b$), the effective deviation is:

$$
u_\text{eff} = b^{n/\nu} \cdot u
$$

The RG flow stops when $u_\text{eff} \sim 1$ (the system has left the critical region). This
happens at:

$$
b^{n/\nu} \cdot u \sim 1 \;\implies\; b^n \sim u^{-\nu} \sim |x - x_c|^{-\nu}
$$

But $b^n$ is the physical length scale reached after $n$ coarse-graining steps. So the
correlation length - the scale at which the system stops looking critical - is:

$$
\xi \sim b^n \sim |x - x_c|^{-\nu}
$$

This is the derivation. $\nu$ is an RG eigenvalue; $\xi \sim |x - x_c|^{-\nu}$ is the consequence.

### 4.3 Relevant vs irrelevant operators

**Relevant operators** (eigenvalue $> 0$): grow under coarse-graining. $x - x_c$ is
relevant - it drives the system away from the critical point. We must tune $x$ to $x_c$
to stay at the fixed point.

**Irrelevant operators** (eigenvalue $< 0$): shrink under coarse-graining. At infinite
system size they vanish. At finite system size they contribute corrections. The leading
irrelevant operator has eigenvalue $-y$ ($y > 0$), contributing corrections $\sim L^{-y}$ to
any observable. This is the origin of the $b \cdot M^{-y}$ correction in our FSS ansatz.

**Marginal operators** (eigenvalue $= 0$): don't change under RG. Rare; lead to
logarithmic corrections.

---

## 5. Finite-Size Scaling (FSS)

### 5.1 Problem

In an infinite system, all observables are non-analytic (singular) at $x_c$: $\Lambda_M$
jumps, $\xi$ diverges, etc. The transition is a sharp point.

In a finite system of linear size $L$, all observables are smooth analytic functions
of $x$ everywhere. A finite system has no phase transitions in the strict sense.
The divergence of $\xi$ is cut off: once $\xi$ reaches $L$, the system cannot develop
further correlations. Equivalently, the system cannot resolve the difference between
being exactly at $x_c$ and being at a distance $|x - x_c| \sim L^{-1/\nu}$ away (the
distance at which $\xi(x) \sim L$). The sharp singularity of the infinite-system
transition is therefore replaced by a smooth crossover of width $\delta x \sim L^{-1/\nu}$.

FSS turns this from a problem into a tool: the width of the crossover, and the
shape of how observables vary across it, both depend on $L$ through $L^{-1/\nu}$, which
is how $\nu$ is extracted.

### 5.2 FSS hypothesis

Near $x_c$, $\xi$ is the only relevant length scale. Any dimensionless observable $O(L, x)$
can only depend on $L$ and $x$ through the dimensionless ratio $L/\xi$:

$$
O(L, x) = f(L/\xi) = f\!\left(L \cdot |x - x_c|^{\nu}\right)
$$

We can reparametrize the argument to make it linear in $x - x_c$ (convenient for
polynomial approximation):

$$
u = L^{1/\nu} \cdot \frac{x - x_c}{x_c} \qquad \text{so that } u^\nu \propto L/\xi
$$

giving the standard FSS form:

$$
O(L, x) = f(u) = f\!\left(L^{1/\nu} \cdot \frac{x - x_c}{x_c}\right)
$$

If $O$ has a non-trivial scaling dimension $\eta$ at the fixed point (e.g. it is not
dimensionless), there is an additional prefactor:

$$
O(L, x) = L^{\eta} \cdot f\!\left(L^{1/\nu} \cdot \frac{x - x_c}{x_c}\right)
$$

### 5.3 Why this is powerful

Plot $O$ vs $u$ instead of vs $x$: all curves for all $L$ collapse onto **one universal
curve** $f(u)$. The collapse works because $f$ is universal - it encodes only the
fixed-point structure, not the specific microscopic details. SC and BCC percolation
(different lattices) share the same $f$ because they are in the same universality class
(3D orthogonal). The data collapse is both a test of the theory and a precision
measurement of $\nu$ (vary $\nu$ until the collapse is best).

### 5.4 Corrections to scaling

At finite $L$, irrelevant operators contribute:

$$
O(L, x) = f(u) + b \cdot L^{-y} + b_2 \cdot L^{-2y} + \cdots
$$

The correction $b \cdot L^{-y}$ shifts the observable away from the asymptotic $f(u)$. For
small $L$ (our $M = 4\ldots10$ range), corrections can be 10-40% of the signal. Including
the leading correction is essential for unbiased results; including two corrections
requires many more system sizes.

---

## 6. Transfer Matrix Method (TMM)

### 6.1 Quasi-1D geometry

Direct diagonalization of a 3D Hamiltonian on an $L^3$ lattice is exponentially
large. The TMM maps the 3D problem onto a much more efficient quasi-1D problem.

Take a strip (tube) of cross-section $M \times M$ sites, propagated in the $z$-direction for
$L$ slices:


The $M^2$ sites on each slice form a state vector of dimension $M^2$.

### 6.2 What the transfer matrix does

**Hamiltonian block structure.** $H$ on the strip decomposes into three types of blocks:

$$
H = \sum_k C_k + \sum_k \bigl(V_k + V_k^\dagger\bigr)
$$

- $C_k$: $M^2 \times M^2$ intra-slice matrix for slice $k$ — off-diagonal entries $-t$ for nearest-neighbour pairs within the slice; diagonal entries zero ($W = 0$, see below).
- $V_k$: $M^2 \times M^2$ forward inter-slice hopping from slice $k$ to $k+1$. **Diagonal**: $(V_k)_{ii} = -t$ if site $i$ is occupied on both slices $k$ and $k+1$, else $0$.
- $V_k^\dagger = V_\text{back}$ : backward hopping from slice $k+1$ to $k$. This is $V_\text{back}$ in the recursion.

$V_k$ and $V_{k-1}^\dagger$ are both diagonal but drawn from different slice-pair occupation patterns, so $V^{-1}V_\text{back} \neq I$ in general.

**Derivation.** Project $H|\psi\rangle = E|\psi\rangle$ onto slice $k$:

$$
C_k\,\psi_k + V_{k-1}^\dagger\,\psi_{k-1} + V_k\,\psi_{k+1} = E\,\psi_k
$$

Isolate $\psi_{k+1}$ (left-multiply by $V_k^{-1}$):

$$
\psi_{k+1} = \underbrace{V_k^{-1}(E\cdot I - C_k)}_{T_k}\,\psi_k
           - \underbrace{V_k^{-1}\,V_{k-1}^\dagger}_{V^{-1}V_\text{back}}\,\psi_{k-1}
$$

$V_k$ is diagonal with $-t$ on connected entries and $0$ on disconnected ones; the computation is
restricted to the subspace where both slices are occupied, giving $V_k^{-1} = (-1/t)\cdot I_\text{active}$
there. The $V^{-1}V_\text{back}$ term is not merely a correction. It is required to preserve the
symplectic structure of the product, which forces Lyapunov exponents to come in $\pm$ pairs.

Written as a $2M^2 \times 2M^2$ matrix acting on $(\psi_k,\, \psi_{k-1})$:

$$
\begin{pmatrix} \psi_{k+1} \\ \psi_k \end{pmatrix}
=
\begin{pmatrix} T_k & -V^{-1}V_\text{back} \\ I & 0 \end{pmatrix}
\begin{pmatrix} \psi_k \\ \psi_{k-1} \end{pmatrix}
$$

After $L$ steps we have a product of $L$ random $2M^2\times 2M^2$ matrices, one per slice.

**Why $W = 0$ (no on-site disorder).** We study quantum percolation, not the Anderson model.
The only randomness is geometric — which sites exist. Setting $W = 0$ isolates localization
from topology alone; on-site energies are uniformly zero. The probe energy $E$ still enters
through $(E\cdot I - C_k)$ in $T_k$: scanning $E$ probes different parts of the spectrum,
which is why $E = 0.7$ matters for the BDI/AI crossover (\S10.2).

### 6.3 Lyapunov exponents

By Oseledec's theorem (1968), for almost every disorder realization, the product
$T_L T_{L-1} \cdots T_1$ has well-defined growth rates: the **Lyapunov exponents** $\gamma_i$
($i = 1, \ldots, 2M^2$), ordered

$$
\gamma_1 \geq \gamma_2 \geq \cdots \geq \gamma_{M^2} > 0 > -\gamma_{M^2} \geq \cdots \geq -\gamma_1
$$

The positive Lyapunov exponents describe the growth of the matrix product in $L$. The
**smallest positive Lyapunov exponent $\gamma_\text{min}$** controls the longest length scale in
the problem - the localization length of the quasi-1D strip:

$$
\xi_M = 1/\gamma_\text{min}
$$

This converges to a definite self-averaging value as $L \to \infty$ - the law of large numbers
for random matrix products. In practice $L = 10{,}000$ is "infinity": the Lyapunov
exponents are converged to ~0.1%.

### 6.3b Oseledec as the matrix analog of the Law of Large Numbers

**Scalar case (LLN).** Let $a_1, a_2, \ldots$ be i.i.d. positive scalars with $\mathbb{E}[|\log a|] < \infty$.
The product $P_n = a_1 a_2 \cdots a_n$ satisfies:

$$
\frac{1}{n}\log P_n = \frac{1}{n}\sum_{i=1}^n \log a_i \;\xrightarrow{\;\text{a.s.}\;}\; \mathbb{E}[\log a] \equiv \gamma
$$

by the ordinary LLN applied to $\log a_i$. So $P_n \sim e^{n\gamma}$ almost surely: one
Lyapunov exponent, given directly by averaging the log.

**Matrix obstacle.** For matrices $A_i \in GL(d)$, the same step fails:

$$
\log(A_n \cdots A_1) \neq \sum_{i=1}^n \log A_i \quad \text{when } [A_i, A_j] \neq 0
$$

Non-commutativity breaks additivity. LLN cannot be applied directly to the matrix logarithm.

**-> Kingman's subadditive ergodic theorem.** Define $f_n = \log\|A_n \cdots A_1\|$.
Submultiplicativity of the matrix norm gives:

$$
\|A_{m+n} \cdots A_1\| \leq \|A_{m+n} \cdots A_{m+1}\|\cdot\|A_m \cdots A_1\|
\;\implies\; f_{m+n} \leq f_m + f_n \circ \theta^m
$$

where $\theta$ is the shift. This is **subadditivity** — the matrix analog of additivity,
weakened by non-commutativity. Kingman (1968): if $(f_n)$ is subadditive and $\mathbb{E}[f_1^+] < \infty$,
then $f_n/n \to \gamma_1$ almost surely. For $d=1$, subadditivity is equality and Kingman
reduces to the ordinary LLN exactly.

**Full Lyapunov spectrum.** Apply the same argument to the $k$-th exterior power
$\bigwedge^k(A_n \cdots A_1)$, whose norm is $\sigma_1 \cdots \sigma_k$ (product of top $k$
singular values). Kingman gives:

$$
\frac{1}{n}\log \sigma_k(A_n \cdots A_1) \;\xrightarrow{\;\text{a.s.}\;}\; \gamma_k,
\qquad k = 1, \ldots, d
$$

This is Oseledec's theorem. The full Lyapunov spectrum $\gamma_1 \geq \cdots \geq \gamma_d$
replaces the single mean $\mathbb{E}[\log a]$ of the scalar case.

| | Scalar (LLN) | Matrix (Oseledec) |
|---|---|---|
| Objects | $a_i \in \mathbb{R}^+$ | $A_i \in GL(d)$ |
| Operation | addition (via $\log$) | matrix product |
| Obstacle | none | non-commutativity |
| Bridge | LLN directly | Kingman subadditivity |
| Limit | one value $\mathbb{E}[\log a]$ | $d$ values $\gamma_1 \geq \cdots \geq \gamma_d$ |

**In our context.** Each transfer matrix $T_k$ is random (different slice occupation).
Non-commutativity is physical: the order in which wavefunction amplitudes propagate
through slices matters. Oseledec guarantees that despite this, all $2M^2$ growth rates
converge to definite values $\gamma_i$ for almost every disorder realization — this is
why $\xi_M = 1/\gamma_\text{min}$ is well-defined and self-averaging.x

### 6.4 Normalized localization length $\Lambda_M$

We define:

$$
\Lambda_M = \frac{\xi_M}{M} = \frac{1}{M \cdot \gamma_\text{min}}
$$

Dividing by $M$ makes this dimensionless. This is the key observable for FSS.

**Why $\gamma_\text{min}$ and not some other Lyapunov exponent?** $\gamma_\text{min}$ controls the slowest
decay / fastest growth in the system - the most extended mode. In the metallic phase
this mode extends across the cross-section; in the localized phase it decays. It is
the mode closest to the mobility edge.

### 6.5 What $M$ and $L$ are (critical distinction)

|                   | $M$ (cross-section)                       | $L$ (strip length)                     |
| ----------------- | ----------------------------------------- | -------------------------------------- |
| Role              | **Finite-size variable** (the $L$ in FSS) | Statistical averaging                  |
| Physical meaning  | system size that competes with $\xi_{3D}$ | must be $\gg \xi_M$ to converge        |
| Varies in FSS     | Yes -> $\{4, 6, 8, 10\}$                  | No -> fixed at 10,000                  |
| Limit of interest | $M \to \infty$ recovers 3D physics        | $L \to \infty$ taken first (converges) |

**Why $M$ plays the role of $L$ in FSS**: the TMM has already taken $L \to \infty$ (Oseledec's
theorem). What remains is a well-defined number $\xi_M$ for each $M$. Now $M$ is the only
finite length in the problem, and it competes with the 3D correlation length $\xi_{3D}$.
The ratio $M/\xi_{3D}$ controls whether the strip looks localized or metallic.

---

## 7. How $\Lambda_M$ Encodes the Phase Transition

### 7.1 Three regimes

**$p \ll p_q$ (deep localized phase):**
$\xi_{3D}$ is small (few lattice spacings). The strip width $M \gg \xi_{3D}$. The strip "contains"
many localization volumes. $\Lambda_M = \xi_M/M$ is small and decreases as $M$ grows - larger
strips show more localization, not less.

**$p \gg p_q$ (metallic phase):**
$\xi_{3D}$ is the scale above which transport is diffusive. $M < \xi_{3D}$ (strip is narrower than
the correlation length). The strip looks like a quantum wire. $\xi_M$ scales with $M$
(the localization in a quasi-1D metal scales with its cross-section). $\Lambda_M = \xi_M/M$ is
approximately constant or grows slightly with $M$.

**$p = p_q$ exactly:**
$\xi_{3D} \to \infty$. The strip is always narrower than $\xi_{3D}$ for any finite $M$. $\xi_M \sim M$ exactly
(a known result for the critical point of the 3D orthogonal class). Therefore
$\Lambda_M = \xi_M/M = \text{const}$, independent of $M$. **All $\Lambda_M(p_q)$ curves agree.** This is the crossing.

### 7.2 Crossing and why it's at $p_q$

The ordering of $\Lambda_M$ curves reverses at $p_q$:
- For $p < p_q$: $\Lambda_4 > \Lambda_6 > \Lambda_8 > \Lambda_{10}$ (smaller $M$ $\to$ larger $\Lambda$, because smaller
  strip = less localization relative to strip width)
- For $p > p_q$: $\Lambda_4 < \Lambda_6 < \Lambda_8 < \Lambda_{10}$ (larger $M$ $\to$ larger $\Lambda$, metallic scaling)

The reversal happens at $p_q$. The crossing point of any pair $(M_1, M_2)$ gives a rough
estimate of $p_q$. With corrections-to-scaling ($b \cdot M^{-y}$), different pairs cross at
slightly different $p$; the FSS fit extracts the true $p_q$ from all data simultaneously.

---

## 8. FSS Ansatz for $\Lambda_M$

### 8.1 Applying FSS to $\Lambda_M$

$\Lambda_M$ is dimensionless ($\eta = 0$), so there is no $L^\eta$ prefactor. The FSS form is:

$$
\Lambda_M(p) = F\!\left(M^{1/\nu} \cdot \frac{p - p_q}{p_q}\right) + b \cdot M^{-y}
$$

$F$ is the universal scaling function; $b \cdot M^{-y}$ is the leading correction.

We approximate $F$ by a polynomial in the scaling variable $u = M^{1/\nu}(p - p_q)/p_q$:

$$
F(u) \approx a_0 + a_1 u + a_2 u^2 + a_3 u^3 + a_4 u^4
$$

$a_0 = \Lambda_M(p_q)$ is the value at the critical point (same for all $M$ by definition).
$a_1$ controls the slope - positive for the metallic side, negative for the localized side.
Higher terms account for the curvature of $F$ away from $u = 0$.

The polynomial is valid for small $|u|$ (close to $p_q$). This is why we need a
fitting window.

### 8.2 Free parameters and what the fit extracts

| Parameter | Meaning | Constrained by |
|-----------|---------|----------------|
| $p_q$ | Critical point location | Position of the crossing |
| $\nu$ | Correlation length exponent | How fast curves separate away from crossing |
| $a_0$ | $\Lambda_M$ at the critical point | Height of the crossing |
| $a_1 \ldots a_4$ | Shape of $F(u)$ | Curvature away from crossing |
| $b$ | Amplitude of corrections | Mismatch between $M$-dependent curves at fixed $p$ |

The fit uses all $(M, p)$ data simultaneously, weighted by the statistical error $\sigma_{M,p}$
on each $\Lambda_M$ measurement. This is $\chi^2$ minimization:

$$
\chi^2 = \sum_{M,p} \left[\frac{\Lambda_M^\text{meas} - F(u) - b \cdot M^{-y}}{\sigma_{M,p}}\right]^2
$$

### 8.3 Irrelevant exponent $y$

$y$ is the scaling dimension of the leading irrelevant operator. It controls how fast
corrections decay with $M$. In the 3D orthogonal class, the literature reports
$|y| \approx 1.7\text{--}2.8$ depending on the observable and method.

**In our implementation**: $y = 1.12$ is an empirical effective value that reproduces
the Anderson benchmark ($W_c$, $\nu$ within 5% of literature; $\chi^2/\text{dof} = 2.4$). It captures
subleading corrections dominant at $M = \{4\ldots10\}$. It is NOT the leading RG irrelevant
exponent as computed by Slevin & Ohtsuki 1999 (who fit $y$ freely and get $|y| \approx 2.8$).
The origin of $y = 1.12$ in the broader literature is unclear. It is treated as a fixed
empirical parameter with $\pm 0.08$ uncertainty propagated as a systematic error.

---

## 9. From Fit to Results: the Full Procedure

### 9.1 Step 1: Rough $p_q$ from pairwise crossings

For each adjacent pair $(M_i, M_{i+1})$, find where $\Lambda_{M_i}(p) - \Lambda_{M_{i+1}}(p)$
changes sign. Linear interpolation gives the crossing $p$. Take the median of all
crossings as $p_{q,\text{rough}}$.

This is robust against the minimum-spread estimator, which fails when corrections
are large (the global minimum of $\mathrm{std}(\Lambda_M)$ over $M$ at each $p$ is not at $p_q$ when
$b \cdot M^{-y}$ differentials between $M$ values are comparable to the physical signal).

### 9.2 Step 2: Window adequacy scan (select $\delta$)

The polynomial $F(u)$ is valid only for small $|u|$. To find the right window:

1. Scan $\delta \in \{0.03, \ldots, 0.15\}$ (15 values).
2. For each $\delta$: fit FSS in $[p_\text{rough} - \delta,\; p_\text{rough} + \delta]$ (no bootstrap; fast scan).
3. Record $\chi^2/\text{dof}(\delta)$ and $Q(\delta) = P(\chi^2 \geq \chi^2_\text{obs} \mid \text{model true})$.
4. Select the **widest $\delta$ with $\chi^2/\text{dof} < 3.0$**.

Logic: wider is better (more data, tighter constraints), but when the window is
too wide, polynomial truncation error enters and $\chi^2/\text{dof}$ rises. The threshold $3.0$ is
calibrated against our benchmark floor ($\chi^2/\text{dof} = 2.4$ from unmodelled $b_2 \cdot M^{-2y}$
corrections). S&O 1999 use $Q > 0.1$; with our $\sigma(\Lambda_M) \sim 10^{-4}$, $Q \approx 0$ for all
windows due to unmodelled subleading corrections, so the $Q$ criterion cannot be
applied directly.

### 9.3 Step 3: Full FSS fit with bootstrap

On the selected window: full Nelder-Mead + L-BFGS-B minimization of $\chi^2$.
Bootstrap CIs: perturb each $\Lambda_M$ by $\mathcal{N}(0, \sigma_{M,p})$ independently, refit 500 times,
take std of the 500 estimates as the statistical uncertainty.

### 9.4 Step 4: Rodriguez polynomial stability test

Fit FSS at poly_order $\in \{3, 4, 5\}$. Check that $p_q$ and $\nu$ from each order lie within
the 95% CI ($\pm 2\sigma$) of the reference order-4 result. If they do: the polynomial is
adequate and the results are stable. This is the actual Rodriguez et al. 2011
Appendix A stability criterion (polynomial ORDER perturbation, not window-width scan).

### 9.5 Step 5: Systematic from $y$

Repeat the full fit at $y = 1.04, 1.12, 1.20$. Report:

$$
\sigma_\text{syst} = \max\!\left(|\text{result}(1.04) - \text{result}(1.12)|,\; |\text{result}(1.20) - \text{result}(1.12)|\right)
$$

$$
\sigma_\text{total} = \sqrt{\sigma_\text{stat}^2 + \sigma_\text{syst}^2}
$$

---

## 10. Symmetry Classes and Why Energy Matters

### 10.1 Three Wigner-Dyson classes

Random matrix theory classifies disordered quantum systems by their symmetries:

| Class | Time-reversal | Spin-rotation | Example |
|-------|--------------|---------------|---------|
| AI (orthogonal) | Yes | Yes | Non-magnetic, spinless, no SO coupling |
| AII (symplectic) | Yes | No | Strong spin-orbit coupling |
| A (unitary) | No | - | Magnetic field present |

Each class has its own Anderson transition (in 3D) with its own $\nu$. Our system: no
magnetic field, no spin - class AI. $\nu = 1.622$ (Slevin & Ohtsuki 1999; Rodriguez 2011).

### 10.2 Bipartite lattices and the chiral anomaly at $E = 0$

SC and BCC are **bipartite**: sites can be 2-colored A/B with all bonds connecting
A to B. The bipartite structure implies a chiral operator $\Gamma$ (flip A$\leftrightarrow$B, change sign
of A-sublattice amplitudes) that anticommutes with $H$:

$$
\Gamma H \Gamma = -H
$$

At $E = 0$ exactly, this is an exact symmetry: if $|\psi, E\rangle$ is an eigenstate, so is
$\Gamma|\psi, -E\rangle$. The transfer matrix at $E = 0$ has chiral symmetry. This places the system
in **class BDI** (chiral orthogonal), not class AI.

Class BDI has a different fixed point, different $\nu$, different corrections. Ujfalusi &
Varga 2014 show $p_c^Q(E \to 0) \to 1$ for SC, and MFSS "severely fails to converge" below
$E = 0.1$ due to the anomalous DOS at $E = 0$ from sublattice imbalance zero modes.

**Consequence**: run at $E = 0.7$ (11.7% of SC half-bandwidth $6t$). The BDI$\to$AI
crossover length scale $\xi_\text{cross} \sim E^{-1/\Delta_E} \ll M_\text{min} = 4$ at $E = 0.7$, so BDI
corrections are negligible. At $E = 0.1$, $\xi_\text{cross} \approx 10 \approx M_\text{max}$ - correction
dominance: $b = -0.153$, $\chi^2/\text{dof} = 1543$, spurious crossings at $p \approx 0.35$.

---

## 11. Summary of Concepts

### Physical system

The Hamiltonian is

$$H = -t \sum_{\langle i,j \rangle} c_i^\dagger c_j,$$

with site occupancy drawn i.i.d. with probability $p$ (site-diluted lattice, $W = 0$).
Three regimes are separated by two thresholds:

| Regime | Condition | Physics |
|---|---|---|
| Insulating (no cluster) | $p < p_c$ | No percolating backbone; all states trivially localized |
| Quantum-percolation | $p_c < p < p_q$ | Percolating cluster exists; quantum interference localizes every state |
| Metallic (extended) | $p > p_q$ | Extended eigenstates appear; quantum transport is possible |

The classical percolation threshold $p_c$ (SC: 0.3115, BCC: 0.2457) is a geometric fact.
The quantum percolation threshold $p_q > p_c$ is dynamical: interference localizes states
even on a connected cluster. The diverging length scale near $p_q$ is

$$\xi \sim |p - p_q|^{-\nu},$$

where $\nu$ is the correlation-length exponent. Both sides share the same $\nu$:
$\xi_\text{loc}$ diverges from below, the transport coherence length diverges from above.

**Universality class.** The transition belongs to the 3D Anderson (orthogonal, AI) class,
giving $\nu = 1.622 \pm 0.010$ (our fit; consistent with Slevin-Ohtsuki 1999 $\nu = 1.57$-$1.58$
and Rodriguez 2011 $\nu = 1.590$). The Harris criterion requires $\nu \geq 2/d = 2/3$;
the measured value satisfies this comfortably.

---

### Method: Transfer Matrix

The Hamiltonian is sliced along one axis. Slice $k$ has $M^2$ sites with on-site matrix
$C_k$ and inter-slice hopping matrices $V_k$, $V_k^\dagger$:

$$H = \sum_k C_k + \sum_k \bigl(V_k + V_k^\dagger\bigr).$$

Projecting the Schrodinger equation onto slice $k$ and solving for $\psi_{k+1}$ gives the
two-component recursion

$$\begin{pmatrix}\psi_{k+1}\\\psi_k\end{pmatrix}
= \underbrace{\begin{pmatrix}T_k & -V^{-1}V_\text{back}\\I & 0\end{pmatrix}}_{\mathcal{T}_k}
\begin{pmatrix}\psi_k\\\psi_{k-1}\end{pmatrix},
\quad T_k = V_k^{-1}(E\cdot I - C_k).$$

The product $P_L = \mathcal{T}_L \cdots \mathcal{T}_1$ is a $2M^2 \times 2M^2$ symplectic matrix.
By Oseledec's theorem (the matrix analog of the law of large numbers, proved via Kingman's
subadditive ergodic theorem), the singular values of $P_L$ grow exponentially:
$\sigma_i(P_L)^{1/L} \to e^{\gamma_i}$ almost surely. The smallest positive Lyapunov exponent
gives the localization length $\xi_M = 1/\gamma_\text{min}$, and the dimensionless ratio

$$\Lambda_M = \xi_M / M$$

is the FSS observable. It is $O(1)$ at criticality and $\to\infty$ ($\to 0$) in the metallic
(localized) phase as $M \to \infty$.

| Parameter | Role | Value used |
|---|---|---|
| $L$ | Strip length (statistical averaging) | 10,000 |
| $M$ | Cross-section width (FSS variable) | 4, 6, 8, 10 |
| $E$ | Energy (sets $T_k$, must avoid $E=0$ BDI anomaly) | 0.7 |

---

### Method: Finite-Size Scaling

Near $p_q$, the one-parameter scaling hypothesis gives

$$\Lambda_M(p) = F\!\left(M^{1/\nu}\cdot\frac{p - p_q}{p_q}\right) + b\cdot M^{-y} + \cdots$$

$F$ is the universal scaling function (approximated by a polynomial $\sum_n a_n u^n$ where
$u = M^{1/\nu}(p-p_q)/p_q$), $b\cdot M^{-y}$ is the leading irrelevant correction (exponent
$y > 0$), and $\eta = 0$ so no $M^\eta$ prefactor appears. All $\Lambda_M$ curves cross near
$p_q$ because $F(0)$ is constant.

**Fitting procedure.**

1. **Window scan.** Restrict data to $|p - p_q^\text{rough}| < \delta$ and scan $\delta$
   downward. Accept the widest $\delta$ giving $\chi^2/\text{dof} < 3.0$.

2. **Minimize** $\chi^2 = \sum_{M,p}\left[\frac{\Lambda_M^\text{meas} - F(u) - b\cdot M^{-y}}{\sigma_{M,p}}\right]^2$
   over $\{p_q,\,\nu,\,a_0,\ldots,a_{n_p},\,b\}$ ($\sim 8$ free parameters for $n_p = 4$).

3. **Statistical uncertainty.** Bootstrap over $L$-blocks of the Lyapunov sequence.

4. **Systematic uncertainty.** Vary $y$ over $[y_\text{min}, y_\text{max}]$ (or fix $y = 1.12$
   empirically) and take the envelope of $p_q$ and $\nu$ as the systematic error.

The stability of the fit is checked by perturbing the polynomial order by $\pm 1$
(Rodriguez 2011 Appendix A) and confirming that $p_q$ and $\nu$ shift by less than
one statistical $\sigma$.

---

## References

1. **Anderson P W 1958** Phys. Rev. **109** 1492 - original localization paper
2. **Fisher M E and Barber M N 1972** Phys. Rev. Lett. **28** 1516 - finite-size scaling
   introduced
3. **MacKinnon A and Kramer B 1983** Z. Phys. B **53** 1 - transfer matrix method for
   Anderson localization; original $\Lambda_M$ observable
4. **Slevin K and Ohtsuki T 1999** Phys. Rev. Lett. **82** 382 - corrections-to-scaling
   in 3D Anderson TMM; $y$ fitted freely, $|y| \approx 2.8$; $\nu = 1.57\text{--}1.58$; bootstrap CIs
5. **Rodriguez A, Vasquez L J, Slevin K and Römer R A 2011** Phys. Rev. B **84** 134209
   - multifractal FSS; polynomial-order stability test (Appendix A); $\nu = 1.590$
1. **Ujfalusi L and Varga I 2014** Phys. Rev. B **90** 245136 - quantum percolation
   on SC lattice; mobility edge $p_c^Q(E)$; BDI anomaly at $E = 0$; $p_c^Q(E = 0.7) \approx 0.384$
