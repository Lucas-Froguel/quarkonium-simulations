# Thesis Ideas: Quarkonium Dissociation via Quantum Computing

## Tier 1: Screened potential + real-time evolution (builds directly on existing code)

### You're right, it's somewhat trivial

The screened Cornell potential Hamiltonian in the HO basis would be computed numerically — you'd evaluate matrix elements <n|V(r,T)|m> by numerical integration over the HO wavefunctions, not analytically. The hypergeometric functions in the Gallimore code handle the unscreened case; adding exp(-r/rD) kills the closed-form solutions, but numerical quadrature is straightforward.

The time evolution itself is just Trotterized exp(-iH(T(t))dt) with H changing at each step as T ramps. It's implementable, but it's thin — the medium is still a classical parameter, and the whole thing could be done faster and better with a classical ODE solver. The quantum computer adds nothing fundamental here. It would serve mainly as a stepping stone / validation for Tier 2.

## Tier 2: Lindblad dynamics with an ancilla (the real prize)

### What are we physically calculating?

**The system**: a quarkonium state (J/psi, psi(2S), chi_c, or bottomonium equivalents) inside a QGP at temperature T.

**What evolves**: the **density matrix** rho(t) of the quarkonium subsystem. Not a pure state — a mixed state, because the quarkonium is entangled with/decohering into the thermal medium. The density matrix encodes:
- Populations: diagonal elements rho_ii = probability of being in state |i>
- Coherences: off-diagonal elements rho_ij = quantum superposition between states

**The evolution equation** (Lindblad master equation):

```
drho/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})
```

- First term: unitary evolution under the quarkonium Hamiltonian (existing H)
- Second term: **dissipation** from the thermal medium — transitions, decoherence, energy exchange with the QGP

**The jump operators L_k** encode specific physical processes:
- **Thermal dissociation**: L_1 takes |bound> -> |unbound> with rate proportional to T (gluon absorption breaks the pair apart)
- **Recombination**: L_2 takes |unbound> -> |bound> (a c and c-bar find each other and rebind)
- **Decoherence between eigenstates**: L_3 dephases superpositions of 1S/2S/1P (thermal fluctuations randomize the relative phase)
- **Transitions between bound states**: L_4 takes |2S> -> |1S> + gluon emission, etc.

The rates come from QCD calculations — specifically from chromoelectric correlators computed in pNRQCD by Brambilla, Vairo, and collaborators.

### What you measure (the physics output)

| Observable | What it tells you | Experimentally measured? |
|---|---|---|
| **Survival probability** P_1S(t) = rho_1S,1S(t) | Fraction of J/psi that survive at time t | Yes — R_AA at RHIC/LHC |
| **Sequential suppression** P_1S(T) vs P_2S(T) vs P_1P(T) | Which states melt first as T increases | Yes — CMS Upsilon(nS) measurements |
| **Entropy** S(t) = -Tr(rho ln rho) | How fast quarkonium thermalizes | Not directly, but informs transport models |
| **Coherence decay** |rho_ij(t)| for i != j | Timescale of quantum decoherence in QGP | Open question — no experimental access yet |
| **Recombination dynamics** | How unbound c,c-bar pairs reform into J/psi | Yes — explains J/psi enhancement at LHC vs RHIC |

### Why this isn't trivially classical

The Lindblad equation for a 4-state system (2 qubits) is a 16x16 matrix ODE — yes, classically solvable. But:
1. The point is demonstrating the **quantum algorithm** on a physically motivated problem, as a proof-of-concept for scaling to larger Hilbert spaces where classical Lindblad simulation becomes exponential
2. With a larger basis (8, 16, 32 quarkonium states -> 3, 4, 5 qubits), the density matrix is 2^n x 2^n and classical simulation grows as 4^n
3. You're producing the **first implementation with real pNRQCD Lindblad operators** rather than toy models — that's the novelty over De Jong et al.

**Clarification on Du & Qian**: their paper is about heavy quark **diffusion** (Brownian motion of a single charm quark in the QGP), not quarkonium bound states. Related but different physics. What we'd be doing is closer to De Jong et al.'s framework but with Brambilla et al.'s quarkonium-specific Lindblad operators.

## Tier 3: Explicit quantum bath (most ambitious)

### What this means concretely

```
|Psi_total> = |quarkonium> x |bath>
               2 qubits      2-3 qubits
```

You evolve the full system unitarily under H_total = H_system + H_bath + H_interaction, then trace out the bath to get rho_system.

### What we need to understand

1. **What do the bath qubits represent physically?** This is the hard, open-ended question. Options:
   - **Gluon field modes**: discretize the thermal gluon field into a few modes. Each mode has occupation number -> truncate to 0/1 -> one qubit per mode. The interaction is chromoelectric dipole coupling: H_int ~ r . E, where r is the Q-Qbar separation operator and E is the gluon electric field.
   - **Thermal noise field**: model the QGP as a few harmonic oscillators (Caldeira-Leggett model) in a thermal initial state. Well-studied in quantum optics; transferable to QCD via the mapping in Akamatsu's framework.
   - **Discrete collision model**: each bath qubit represents one "collision" with a thermal gluon. After interacting, it's discarded (traced out) and replaced with a fresh thermal qubit. This is the repeated-interaction framework — mathematically equivalent to Lindblad in the limit of many collisions.

2. **How do you prepare the bath in a thermal state?** A thermal state at temperature T is rho_bath = exp(-H_bath/T)/Z. On a quantum computer, you can't directly prepare a mixed state. Options:
   - **Thermofield double**: purify the thermal state by doubling the bath qubits (expensive in qubits)
   - **Random pure state sampling**: Davoudi, Mueller & Powers (2023) showed that random pure states approximate thermal expectations — "thermal pure quantum states"
   - **Just use |0> bath and post-select**: in the collision model, each fresh bath qubit starts in a T-dependent state cos(theta)|0> + sin(theta)|1> where theta encodes the temperature. Simple and physical.

3. **What's hard / open-ended**:

| Challenge | Difficulty | Status |
|---|---|---|
| Choosing the right bath model (what do qubits represent?) | Conceptual — requires physics judgment | Open; no established "best" approach for quarkonium |
| Deriving H_interaction from QCD | Moderate — use pNRQCD dipole coupling | Known theoretically (Brambilla, Vairo) |
| Bath size needed for physical results | Unknown — 2 bath qubits might be too few | Open research question |
| Connecting to Lindblad limit | Tractable — show Tier 3 reduces to Tier 2 when bath is traced out | Would be a strong consistency check |
| Extracting genuinely non-classical observables | The real prize — system-bath entanglement, non-Markovian effects | Open; this is where you'd be doing something classically impossible |

### What would make Tier 3 genuinely new

If you can show that the quarkonium-bath **entanglement** or **non-Markovian memory effects** change the physics predictions compared to Tier 2 (Lindblad = Markovian approximation). The Lindblad equation assumes the bath has no memory — each interaction is independent. If the QGP has finite correlation time comparable to the quarkonium evolution timescale, the Markovian approximation breaks down. A small explicit bath captures this. That's a result no classical open-quantum-system calculation produces, because they all assume Markovianity.

## Thesis arc assessment

- **Tier 1**: warm-up / validation (1-2 weeks of coding)
- **Tier 2**: the core result — first physically motivated Lindblad quarkonium simulation on a QC (the bulk of the thesis)
- **Tier 3**: the ambitious extension — if you can show non-Markovian effects matter, that's a genuine contribution to quarkonium-in-QGP physics

The Tier 2 -> Tier 3 transition is where the thesis goes from "nice quantum computing exercise" to "new physics insight."

## References

### Quarkonium in QGP — theory
- [Matsui & Satz, "J/psi suppression by quark-gluon plasma formation" (1986)](https://doi.org/10.1016/0370-2693(86)91404-8)
- [Akamatsu, "Quarkonium in QGP: Open quantum system approaches re-examined" (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0146641021000934)
- [Yao, "Open Quantum Systems for Quarkonia" (2021)](https://arxiv.org/abs/2102.01736)
- [Rothkopf, "Heavy quarkonium in extreme conditions" (2020)](https://www.sciencedirect.com/science/article/pii/S0370157320300600)
- [Brambilla et al., "Bottomonium suppression using quantum trajectories" (2021)](https://link.springer.com/article/10.1007/JHEP05(2021)136)
- [QTraj solver code (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0010465521003787)
- [Quarkonium dynamics in the quantum Brownian regime (2024)](https://link.springer.com/article/10.1007/JHEP06(2024)060)

### Quantum computing for quarkonium / QGP
- [De Jong et al., "Quantum simulation of open quantum systems in heavy-ion collisions" (2021)](https://doi.org/10.1103/PhysRevD.104.L051501)
- [Du & Qian, "Accelerated quantum circuit Monte Carlo for heavy quark thermalization" (2024)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.076025)
- [Barata et al., "Quantum simulation of in-medium QCD jets" (2023)](https://doi.org/10.1103/physrevd.108.056023)
- [Davoudi, Mueller, Powers, "Thermal pure quantum states for gauge theories" (2023)](https://dx.doi.org/10.1103/PhysRevLett.131.081901)

### Quantum algorithms for Lindblad / open systems
- [Bartolomeo et al., "Single ancilla qubit for Lindblad simulation" (2024)](https://link.aps.org/doi/10.1103/PhysRevResearch.6.043321)
- [Peng et al., "Zero-ancilla Lindblad simulation" (2025)](https://link.aps.org/doi/10.1103/PhysRevResearch.7.023076)
- [Review: Quantum algorithms for open quantum systems (2024)](https://arxiv.org/html/2406.05219v1)
- [Lindblad simulation via repeated interactions (2023)](https://arxiv.org/abs/2312.05371)

### Community roadmaps
- [Bauer et al., "Quantum simulation for HEP," PRX Quantum (2023)](https://link.aps.org/pdf/10.1103/PRXQuantum.4.027001)
- [Quantum computing for heavy-ion physics: prospects (2025)](https://arxiv.org/html/2510.04207)
