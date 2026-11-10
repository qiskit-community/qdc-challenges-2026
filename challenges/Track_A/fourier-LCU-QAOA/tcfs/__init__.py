"""Validated tooling for the QDC 2026 "Trade Circuits for Shots" challenge.

Implements the constructions of A. Carrera Vazquez, D. J. Egger, S. Woerner,
"Efficient Fourier-Based Linear Combination of Unitaries and Applications in
Quantum Optimization", arXiv:2605.18985 (2026):

- ``instances``     densest-k-subgraph instances, exact tables, baselines
- ``diagonal_lcu``  Fourier-based LCU for diagonal unitaries (Sec. III)
- ``schur_weyl``    permutation-invariant LCU / XY-mixer (Sec. IV, App. B-D)
- ``circuits``      Qiskit circuits for all constructions (Figs. 3 and 5)
- ``exact``         fast exact statevector engine + distribution metrics
- ``cvar``          weighted CVaR estimators (upper/lower tail)
- ``optimize``      the paper's experiment protocols (Sec. VI A-B)
- ``hardware``      utility-scale pathway (backend-native instances, transfer,
                    handouts with their ``instance_id``)
- ``policy``        axis-T training toolbox (objectives, restarts, surrogates)
- ``fingerprint``   canonical circuit fingerprint + structural summary
- ``event``         the training instance, the paper's Table I angles, the
                    demo instance (``demo_instance``, id "demo106") and the
                    QPU-time estimate; ``connect`` -> ``Campaign`` is legacy
- ``jobs``          run a job once per tag (``run_once``, ``forget``)
- ``nb``            the notebook's tables and figures
- ``mapping``       the Map step with qiskit-addon-opt-mapper

Legacy (the prototype grader of notebook v0.4, kept with its tests; the
notebook is graded by the ``qc_grader`` functions of the grading platform):

- ``grader``        mock grader with tolerance-banded checks (challenge spec)
- ``ledger``        append-only, hash-chained JSON-lines ledger
- ``oracle``        the authors' released metrics code as a scoring oracle
- ``review``        review-channel check: submitted circuits vs declared LCU
- ``commit``        counts commitment: canonical shots, Merkle chunks, reveal
- ``anchors``       job anchors: light job metadata (runtime, mock, reported)
- ``grader_server`` remote event grader: register / reveal / cards / part1 /
                    views / close
- ``grader_client`` team side of the remote grader (``RemoteCampaign``)
- ``checks``        the exercise checks of notebook v0.4 (Parts 1 and 2)

Conventions (fixed across the package, see each module for details):
- Qubit i encodes decision variable x_i; statevector index b has
  x_i = (b >> i) & 1  (Qiskit little-endian).
- Maximization throughout: f_lam(x) = f(x) - lam*(1^T x - k)^2.
- QAOA phase separator is exp(-i*gamma*H) with H|x> = f_lam(x)|x>.
"""

__version__ = "0.1.0"
