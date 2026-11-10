"""Logical Gibbs-circuit construction on kagome/heavy-hex.

The notebook owns backend-ISA transpilation, Samplomatic boxing/build, post-selection,
dynamical decoupling and Samplex parameter sweeps. This module ONLY constructs the
logical circuits (including reset windows and ZNE amplification) for those stages.

The physical layout, randomization/seed and ZNE conventions match v1.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.transpiler import PassManager

from .gibbs_transpilation_utilities import noise_amplification
from .declifford_rzz import DeCliffordRZZ


def system_size(Lx, Ly):
    if Lx == 2 and Ly == 2: return 12
    elif Lx == 3 and Ly == 3: return 36
    else: raise ValueError('invalid layout dimensions')



def _apply_twirl(circuits, twirl_string, N):
    for n in range(N):
        if str(twirl_string[n]) == 'x':
            [qc.x(n) for qc in circuits]
        if str(twirl_string[n]) == 'y':
            [qc.y(n) for qc in circuits]
        if str(twirl_string[n]) == 'z':
            [qc.z(n) for qc in circuits]



def build_logical_circuits(Lx, Ly, nSteps, beta, noise, seed, reset_duration, init_environment, H_tot_trot, hardware_layouts, ALPHA, NTROT, T_TOT, GX, GZ, init_system=None):
    """One randomization: returns len(noise) logical circuits, one per ZNE noise factor."""
    NS = system_size(Lx, Ly)
    layout = hardware_layouts(Lx, Ly)
    NE = len(layout) - NS
    omega_max = max(4 * GX, 4 * GZ, 8)

    declifford_pm = PassManager([DeCliffordRZZ()])

    post_rot = Parameter('post_rot')
    idle_twirl_rng = np.random.default_rng(seed=seed)
    gate_twirl_rng = np.random.default_rng(seed=seed + 1000)
    sd = int(gate_twirl_rng.integers(0, 1e6))

    # The instantiation randomness -- omega_n here, plus sample_boltzmann_zn, SE_Coupling and
    # random_coupling inside H_tot_trot -- is drawn from the GLOBAL numpy state, not from a
    # seeded Generator. Seed it explicitly so each instantiation is distinct and reproducible.
    # (CZReplacementPass also calls np.random.seed, so the global state must be re-seeded here
    # rather than once by the caller.)
    np.random.seed(seed + 500000)

    temp = [QuantumCircuit(NS + NE) for _ in range(len(noise))]

    # Parameterized beta=0 product state: init[n] in {0, pi} selects |0> or |1>. Leaving it
    # parameterized exposes initialization as a samplex sweep dimension, so one job
    # covers multiple initial states while keeping each fixed across its twirls.
    
    if init_system is None:
        init = ParameterVector('init', NS)
        for qc in temp:
            for n in range(NS):
                qc.rx(init[n], n)
    else:
        for qc in temp:
            init_system(qc, NS)
    


    for step in range(nSteps):
        omega_n = np.random.uniform(0, omega_max, size=NE)

        # IMPORTANT FOR ZNE: all noise-amplification variants must use the same
        # sampled Environment state. Replay the RNG state for each variant, then
        # advance the global RNG only once.
        rng_state_before_env = np.random.get_state()
        init_environment(temp[0], NS, omega_n, beta)
        rng_state_after_env = np.random.get_state()

        for qc in temp[1:]:
            np.random.set_state(rng_state_before_env)
            init_environment(qc, NS, omega_n, beta)

        np.random.set_state(rng_state_after_env)
                
        trot = transpile(H_tot_trot(omega_n, ALPHA, NTROT, T_TOT, GX, GZ, Lx, Ly, NS),
                         basis_gates=["rx", "rz", "cz", "rzz", "x"], optimization_level=2)
        trot = declifford_pm.run(trot)
        [qc.compose(trot, inplace=True) for qc in temp]

        if step != nSteps - 1:
            # twirl the system qubits across the reset window, outside the barriers
            twirl_string = idle_twirl_rng.choice(['i', 'x', 'y', 'z'], size=NS)
            _apply_twirl(temp, twirl_string, NS)
            for qc in temp:
                qc.barrier(label='pre_reset_{:d}'.format(step))
            for n in range(NE):
                [qc.reset(qubit=n + NS) for qc in temp]
                for i, prob in enumerate(noise):
                    if prob > 0:
                        temp[i].delay(int(2 * prob * reset_duration), [n + NS], unit='dt')
            for qc in temp:
                qc.barrier(label='post_reset_{:d}'.format(step))
            _apply_twirl(temp, twirl_string, NS)

    for i, prob in enumerate(noise):
        if prob > 0:
            temp[i] = noise_amplification(temp[i], prob, sd)

    for i in range(len(temp)):
        temp[i].rz(np.pi / 2, range(NS))
        temp[i].rx(post_rot, range(NS))
    [qc.measure_all() for qc in temp]

    return temp