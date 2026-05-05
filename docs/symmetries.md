# Quantum Percolation and Tight-Binding Lattice Physics

## 1. Energy ($E$) and the Kinetic Hamiltonian
*   **Definition of $E$:** $E$ is the dimensionless single-particle energy eigenvalue. In this tight-binding model, the on-site potential energy $\epsilon$ for filled sites is explicitly set to a baseline of zero ($\epsilon=0$).
*   **Kinetic Term:** The Hamiltonian for non-interacting spinless electrons is defined entirely by the kinetic nearest-neighbor (NN) hopping operator: $\mathcal{H} = -t\sum_{\langle i,j \rangle} (a_i^\dagger a_j + a_j^\dagger a_i)$, with hopping energy $t=1$. 
*   **Negative Kinetic Energy:** The expected kinetic energy $\langle\mathcal{H}\rangle$ evaluates to a negative value for bonding states. This occurs when the electron's wavefunction exhibits identical (in-phase) amplitudes on adjacent lattice sites, lowering the total energy expectation below the zero baseline.

## 2. Single-Electron Interference and Localization
*   **Single-Particle Physics:** The model strictly governs non-interacting electrons. Therefore, quantum interference occurs entirely within the spatial distribution of a single electron's wavefunction interfering with itself, not between multiple interacting electrons.
*   **Exact Cancellation ($E=0$):** For an electron to possess exactly zero kinetic energy expectation, its wavefunction must establish perfectly destructive interference. For any unoccupied site $k$ adjacent to the electron's localized boundary, the sum of the probability amplitudes from all occupied nearest-neighbor sites $j$ within the cluster must equal zero ($\sum_{j \in C, j \in NN(k)} \Psi_j = 0$).
*   **Localization and Hopping:** Due to the random removal of sites (percolation disorder), isolated finite clusters are formed. An electron in an $E=0$ state *can* hop between the specific nearest-neighbor sites entirely *within* its isolated finite cluster (it exists as a superposition over these sites). However, it cannot propagate outside the cluster into the bulk lattice; perfectly destructive interference at the boundary nodes rigidly confines the wavefunction to the cluster.

## 3. Bipartite Symmetry and the Chiral Operator
*   **Bipartite Lattices:** A lattice is bipartite if its sites can be partitioned into two distinct sublattices (A and B) such that all nearest-neighbor hopping strictly connects A to B.
*   **Chiral Operator ($C$):** For bipartite lattices, the Hamiltonian anticommutes with the chiral operator $C$ ($C\mathcal{H} = -\mathcal{H}C$), which reverses the wavefunction sign on one sublattice.
*   **DOS Symmetry:** This chiral symmetry mathematically guarantees that every eigenstate at energy $+E$ possesses an exact partner state at $-E$. Consequently, the total Density of States (DOS) is perfectly symmetric around the band center ($E=0$).

## 4. Geometric Frustration: SC, BCC vs. FCC
*   **Unfrustrated Lattices (SC, BCC):** Simple Cubic (SC) and Body-Centered Cubic (BCC) are strictly bipartite lattices. All closed NN hopping loops contain an even number of steps. There is no geometric topological constraint preventing the assignment of alternating $+1$ and $-1$ phases to adjacent sites, allowing exact $E=0$ cancellation.
*   **Frustrated Lattices (FCC):** The Face-Centered Cubic (FCC) lattice is non-bipartite due to the presence of 3-step loops (triangles). 
    *   **The FCC Triangle:** Consider a Corner atom at $(0,0,0)$, a Face-Center atom in the xy-plane at $(a/2, a/2, 0)$, and an adjacent Face-Center atom in the xz-plane at $(a/2, 0, a/2)$.
    *   The distance from the Corner to the first Face-Center is $\sqrt{(a/2)^2 + (a/2)^2} = a\sqrt{2}/2$.
    *   The distance from the first Face-Center to the adjacent Face-Center is $\sqrt{(a/2-a/2)^2 + (a/2-0)^2 + (0-a/2)^2} = a\sqrt{2}/2$.
    *   Because all three distances are identical ($a\sqrt{2}/2$), they are all mutual nearest neighbors, forming an equilateral triangle. This odd-length loop topologically forbids alternating $+1/-1$ phase assignment without a sign clash, inducing geometric quantum frustration.

## 5. Degeneracy, the Dirac-Delta Peak, and Numerical Consequences
*   **Unfrustrated DOS (SC, BCC):** Because alternating phases are unfrustrated, the isolated finite clusters created by percolation trivially satisfy the $\sum \Psi_j = 0$ boundary condition. This creates a macroscopic degeneracy of spatially distinct, strictly localized "molecular states" with exactly zero kinetic energy. This massive accumulation of states manifests as a pathological Dirac-delta singularity in the DOS precisely at $E=0$.
*   **Frustrated DOS (FCC):** The triangular geometry of the FCC lattice topologically forbids exact $E=0$ cancellation. States must adopt complex phases or amplitude imbalances to satisfy boundary conditions, which inherently shifts their energy eigenvalues away from zero. The macroscopic degeneracy is destroyed, the Dirac-delta peak vanishes, and the band center exhibits a finite, continuous DOS.
*   **Numerical Solvers:** Iterative eigensolvers (e.g., Jacobi-Davidson) require finite spectral gaps ($\Delta E$) between eigenvalues to achieve vector orthogonality. In unfrustrated SC and BCC lattices, the infinite density of degenerate $E=0$ states destroys this spectral gap, causing severe algorithmic convergence failure. In FCC lattices, frustration suppresses the singularity, restoring the spectral gaps and ensuring stable numerical convergence.