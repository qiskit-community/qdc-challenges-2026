"""
Dissipative thermal-state preparation by statevector trajectory simulation.

Protocol (one reset cycle):
    1. initialize the bath in a thermal computational-basis state ~ exp(-beta H_E),
    2. evolve system+bath under a Trotterized H_tot = H_S + H_E + alpha*(O_S X_E),
    3. measure and reset the bath; keep the system state.
Iterating this channel converges the system toward the Gibbs state of H_S.

"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator


# ============================================================================
# A coupling term is  coeff * X_bath (otimes) A_S, where A_S is a product of system
# Paulis. `sample` returns the (random) terms for one reset cycle as a list of
#   (paulis, bath_index, coeff)
# with paulis = [(pauli_char, system_site), ...] giving A_S. The coeffcient is
# chosen such that ||O_S||_2=1
@dataclass
class Coupling:
    name: str
    sample: Callable[[int, int, np.random.Generator],
                     List[Tuple[List[Tuple[str, int]], int, float]]]
    size: Optional[Callable[[int], int]] = None
    # size(NS) -> number of distinct jump operators this coupling draws from, given
    # NS system qubits. Used by `mix_couplings` to sample uniformly over the union.


# ============================================================================
# Model: the Hamiltonian-specific object.
# ============================================================================
@dataclass
class Model:
    name: str
    NS: int                                  # number of system qubits
    hamiltonian: SparsePauliOp              # H_S (defines the energy observable)
    trotter_step: Callable[[QuantumCircuit, float], None]
    # trotter_step(circ, dt): apply one symmetric 2nd-order Trotter step of H_S
    # to system qubits 0..NS-1, for time dt. Must be consistent with `hamiltonian`.
    coupling: Optional[Coupling] = None     # default (None) -> pauli_coupling()
    initial_state: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None
    # initial_state(NS, rng) -> 2^N statevector; default (None) -> random product state.


# ============================================================================
# MODELS  --  add a factory here to switch Hamiltonians; nothing else changes.
# ============================================================================
def kagome_edges(Lx, Ly, pbc=False):
    """Bond list of the (Lx, Ly) kagome lattice."""
    neigh = {0: [(0, 0, 1), (0, 0, 2), (-1, 0, 1), (0, -1, 2)],
             1: [(0, 0, 0), (0, 0, 2), (1, 0, 0), (1, -1, 2)],
             2: [(0, 0, 0), (0, 0, 1), (0, 1, 0), (-1, 1, 1)]}
    sid = lambda i, j, s: i + Lx * j + (Lx * Ly) * s
    edges = set()
    for i in range(Lx):
        for j in range(Ly):
            for s in range(3):
                for di, dj, sp in neigh[s]:
                    ii, jj = i + di, j + dj
                    if pbc:
                        ii, jj = ii % Lx, jj % Ly
                    elif not (0 <= ii < Lx and 0 <= jj < Ly):
                        continue
                    a, b = sid(i, j, s), sid(ii, jj, sp)
                    edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def ising_model(NS, edges, gx, gz, name="ising"):
    """Ising model on an arbitrary topology:
        H = sum_<ij in edges> Z_i Z_j  +  gx sum_i X_i  +  gz sum_i Z_i.

    Every Ising model Trotterizes the same way, so all you supply is the qubit
    count, the bond list, and the two field strengths -- this returns a ready
    `Model`. `kagome_ising` / `tfim_chain` just hand it different edge lists.
    """
    terms = ([("ZZ", [a, b], 1.0) for a, b in edges]
             + [("X", [i], gx) for i in range(NS)]
             + [("Z", [i], gz) for i in range(NS)])
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=NS)

    def trotter_step(circ, dt):
        # symmetric split of H_S for time dt:  e^{-i gx X dt/2} e^{-i ZZ dt} e^{-i gz Z dt} e^{-i gx X dt/2}
        circ.rx(gx * dt, range(NS))
        for a, b in edges:
            circ.rzz(2 * dt, a, b)          # rzz(theta) = e^{-i theta/2 ZZ} -> e^{-i dt ZZ}
        circ.rz(2 * gz * dt, range(NS))
        circ.rx(gx * dt, range(NS))

    return Model(name, NS, H, trotter_step)


def kagome_ising(Lx, Ly, gx, gz, pbc=False):
    return ising_model(3 * Lx * Ly, kagome_edges(Lx, Ly, pbc), gx, gz,
                       name=f"kagome_ising_{Lx}x{Ly}")



# ============================================================================
# COUPLINGS  --  the system-bath jump operators.
# ============================================================================
def pauli_coupling():
    """Default single-site jump operators  O = {X, Y, Z}: each
    bath qubit couples via  O_site (otimes) X_bath, with O a random single-qubit Pauli
    on a distinct random site."""
    def sample(NS, NE, rng):
        kinds = rng.choice(["X", "Y", "Z"], size=NE)
        sites = rng.choice(NS, size=NE, replace=False)
        return [([(kinds[n], int(sites[n]))], n, 1.0) for n in range(NE)]
    return Coupling("pauli", sample, size=lambda NS: 3 * NS)   # 3 Paulis x N sites



def mix_couplings(couplings, weights=None):
    """Combine several couplings: each bath qubit independently picks one of
    `couplings` and uses its jump operator. With weights=None (default) the pick
    is weighted by each coupling's operator count `size(NS)`, i.e. uniform over the
    *union* of all individual jump operators
    Pass `weights` (summing to 1) to override with fixed per-coupling probabilities."""
    def sample(NS, NE, rng):
        w = weights
        if w is None and all(c.size is not None for c in couplings):
            sizes = np.array([c.size(NS) for c in couplings], dtype=float)
            w = sizes / sizes.sum()                          # uniform over all operators
        terms = []
        for n in range(NE):
            c = couplings[rng.choice(len(couplings), p=w)]
            for paulis, _, coeff in c.sample(NS, 1, rng):     # sample 1 bath qubit
                terms.append((paulis, n, coeff))             # relabel to bath qubit n
        return terms
    return Coupling("mix(" + "+".join(c.name for c in couplings) + ")", sample)


# ============================================================================
# INITIAL STATES
# ============================================================================



# ============================================================================
# Generic machinery (Hamiltonian-independent).
# ============================================================================
def _gaussian_envelope(nTrot, t_tot, sigma=1e2):
    dt = t_tot / nTrot
    xs = np.arange(-t_tot / 2 + t_tot / (2 * nTrot), t_tot / 2, t_tot / nTrot)
    g = np.exp(-xs ** 2 / t_tot ** 2 / (4 * sigma ** 2))
    return g / np.sqrt(dt * np.sum(g ** 2))


def _pauli_rotation(circ, pauli_qubits, theta):
    """exp(-i theta/2 * P), P = product of the given (pauli_char, qubit) factors.
    Standard basis-change + CX-ladder + Rz construction (a generalized rzz)."""
    qs = [q for _, q in pauli_qubits]
    for p, q in pauli_qubits:                        # rotate each factor into the Z basis
        if p == "X":
            circ.h(q)
        elif p == "Y":
            circ.sdg(q); circ.h(q)
    for i in range(len(qs) - 1):
        circ.cx(qs[i], qs[i + 1])
    circ.rz(theta, qs[-1])                            # rz(theta) = exp(-i theta/2 Z)
    for i in reversed(range(len(qs) - 1)):
        circ.cx(qs[i], qs[i + 1])
    for p, q in pauli_qubits:                         # undo the basis change
        if p == "X":
            circ.h(q)
        elif p == "Y":
            circ.h(q); circ.s(q)


def _add_evolution(circ, model, omega, alpha, nTrot, t_tot, rng, coupling, sigma=1e2):
    """One full Trotterized H_tot: coupling | H_E | H_S | H_E | coupling, per Trotter step.

    The coupling terms (which Paulis on which sites) are drawn once for the whole
    cycle; only the Gaussian-envelope amplitude changes between sub-steps."""
    NS, NE = model.NS, len(omega)
    dt = t_tot / nTrot
    g = _gaussian_envelope(nTrot, t_tot, sigma)
    terms = coupling.sample(NS, NE, rng)           # [(paulis, bath_index, coeff), ...]

    def coupling_layer(st):
        amp = g[st] * dt * alpha                       # = g[st] * dt * 2 * alpha / 2
        for paulis, b, coeff in terms:
            # term = coeff * X_bath (otimes) prod(paulis), applied as one Pauli rotation
            _pauli_rotation(circ, [("X", NS + b)] + list(paulis), coeff * amp)

    for st in range(nTrot):
        coupling_layer(st)
        for n in range(NE):
            circ.rz(-dt * omega[n] / 2, NS + n)         # H_E = sum omega_n/2 Z_n
        model.trotter_step(circ, dt)                   # <-- the only model-specific call
        for n in range(NE):
            circ.rz(-dt * omega[n] / 2, NS + n)
        coupling_layer(st)


def _sample_thermal_bath(omega, beta, rng):
    """Bath as a thermal computational-basis sample; returns the bath integer index.
    P(qubit n excited) = 1/(1+exp(beta*omega_n))."""
    p_exc = 1.0 / (1.0 + np.exp(np.clip(beta * omega, -700, 700)))
    excited = rng.random(len(omega)) < p_exc
    bath_int = 0
    for n, e in enumerate(excited):
        if e:
            bath_int |= (1 << n)                              # bath qubit n -> bit n
    return bath_int


def _random_product_state(NS, rng):
    bits = rng.integers(0, 2, size=NS)
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    vec = np.zeros(1 << NS, dtype=complex)
    vec[idx] = 1.0
    return vec


def _embed(sys_state, bath_int, NS, NE):
    """Full state = |bath_int>_bath (otimes) |sys_state>_system  (bath = most significant qubits)."""
    full = np.zeros(1 << (NS + NE), dtype=complex)
    full[bath_int << NS: (bath_int + 1) << NS] = sys_state
    return full


def _collapse_system(full_sv, NS, NE, rng):
    """Sample a bath measurement outcome and return the normalized post-measurement system state."""
    m = full_sv.reshape((1 << NE, 1 << NS))               # rows = bath, cols = system
    probs = (np.abs(m) ** 2).sum(axis=1)
    probs /= probs.sum()
    o = rng.choice(1 << NE, p=probs)
    s = m[o].copy()
    return s / np.linalg.norm(s)


def _expval(H, state):
    return float(np.vdot(state, H @ state).real)


def thermal_ensemble(H, beta, cutoff_over_beta=8):
    """Low-energy Gibbs ensemble of H_S.

    Returns (eigvecs, weights): the eigenstates within cutoff_over_beta/beta of
    the ground state (columns of eigvecs, shape (2^N, K)) and their normalized
    Boltzmann weights w_i ~ exp(-beta E_i) (shape (K,)). Used to build the
    target state sigma = sum_i w_i |e_i><e_i| for the fidelity.
    """
    H = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
    evals, evecs = np.linalg.eigh(H)                  
    mask = (evals - evals[0]) < cutoff_over_beta / beta
    w = np.exp(-beta * (evals[mask] - evals[0]))
    w /= w.sum()
    return evecs[:, mask], w



def _auto_batch(n_total_qubits, nSamples, budget_bytes=8_000_000_000):
    """Largest batch of circuits whose statevectors fit the memory budget. Default 8Gb"""
    per = (1 << n_total_qubits) * 16 * 2                      # ~ stored state + result, complex128
    return max(1, min(nSamples, budget_bytes // per))



def run_thermal_prep(model, *, alpha, beta, t_tot, nTrot, nSteps, nSamples,
                     NE, omega_max, sigma=1e2, coupling=None, init_state=None,
                     batch_size=None, seed=0, progress=True):
    """
    Returns energies, shape (nSteps+1, nSamples): row 0 is the initial energy,
    row k the energy after k reset cycles, per trajectory. Divide by model.N for density.

    Each cycle draws a fresh bath frequency per qubit, omega ~ U[0, omega_max].
    sigma      : width of the Gaussian filter f(t) = N exp(-t^2/(4 sigma^2 T^2))
                 (Eq. 10). Large sigma (default 1e2) -> effectively constant
                 filter; sigma ~ 1/4 smoothly turns the coupling on/off, which
                 lowers the fixed-point error at longer T.
    coupling   : system-bath jump operators; default falls back to
                 model.coupling, else pauli_coupling().
    init_state : initial-state callable (N, rng) -> statevector; default falls
                 back to model.initial_state, else a random product state.
    batch_size : circuits per Aer job (default: memory-budgeted auto value).

    """
    rng = np.random.default_rng(seed)
    NS = model.NS
    H = model.hamiltonian.to_matrix(sparse=True)             # built once
    backend = AerSimulator(method="statevector")
    if batch_size is None:
        batch_size = _auto_batch(NS + NE, nSamples)
    coupling = coupling or model.coupling or pauli_coupling()
    init_state = init_state or model.initial_state or _random_product_state


    states = [init_state(NS, rng) for _ in range(nSamples)]
    energies = np.zeros((nSteps + 1, nSamples))
    energies[0] = [_expval(H, s) for s in states]

    for step in range(nSteps):
        circuits = []
        for s in range(nSamples):
            omega = rng.uniform(0, omega_max, size=NE)
            bath_int = _sample_thermal_bath(omega, beta, rng)
            circ = QuantumCircuit(NS + NE)
            circ.set_statevector(Statevector(_embed(states[s], bath_int, NS, NE)))
            _add_evolution(circ, model, omega, alpha, nTrot, t_tot, rng, coupling, sigma)
            circ.save_statevector()
            circuits.append(circ)

        # submit independent trajectories in memory-bounded batches (one job each)
        svs = [None] * nSamples
        for lo in range(0, nSamples, batch_size):
            chunk = circuits[lo:lo + batch_size]
            res = backend.run(chunk).result()
            for k in range(len(chunk)):
                svs[lo + k] = np.asarray(res.get_statevector(k))

        for s in range(nSamples):
            states[s] = _collapse_system(svs[s], NS, NE, rng)
            energies[step + 1, s] = _expval(H, states[s])


        if progress:
            msg = f"step {step + 1}/{nSteps}  <HS>/NS = {energies[step + 1].mean() / NS:+.4f}"
            print(msg, flush=True)


    return energies


# ============================================================================
# Example run for (2,2) Kagome Ising model
# ============================================================================
if __name__ == "__main__":
    gx, gz = 0.5, 2.0
    model = kagome_ising(Lx=2, Ly=2, gx=gx, gz=gz, pbc=False)  # <-- swap this line for another Model
    energies = run_thermal_prep(
        model, alpha=1.75, beta=100.0, t_tot=0.75, nTrot=3,
        nSteps=100, nSamples=100, NE=1, omega_max=max(4 * gx, 4 * gz, 4),
        seed=0, 
    )
    Ess = energies[-1].mean() / model.NS
    sem = energies[-1].std() / np.sqrt(energies.shape[1]) / model.NS
    print(f"\n{model.name}: steady <HS>/NS = {Ess:.4f} +/- {sem:.4f}")
