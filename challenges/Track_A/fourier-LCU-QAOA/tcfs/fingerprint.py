"""Canonical circuit fingerprints and structural summaries (grader v2).

The event grader never receives a circuit: circuits go to the external
circuit store ("Insights"), and the grader only holds one fingerprint per
circuit plus a small structural summary. The fingerprint must therefore
identify a *bound circuit*, not a file: two QPY dumps of the same laid-out
circuit can differ byte for byte (``qpy.dump`` is not byte-deterministic
for circuits that carry a ``TranspileLayout``), so the raw sha256 of the
bytes cannot be the identity that the store and the grader compare.

Canonical form (``FINGERPRINT_VERSION``):

- the layout is ignored: qubit / clbit indices are the circuit's own
  indices (``qc.layout`` never enters; a DAG carries none);
- the op nodes of ``circuit_to_dag(qc)`` are listed in the lexicographic
  topological order keyed by each node's own token, so the order in which
  commuting instructions were appended does not matter, only the per-wire
  sequences do (input nodes sort first, output nodes last, so the op order
  is a function of the DAG and the tokens alone);
- a node's token is ``name | qargs | cargs | params`` with every real
  parameter rounded to ``params_decimals`` (ints and floats of equal value
  give the same token), plus ``unit`` for ``delay`` / ``barrier``;
- the header carries ``num_qubits``, ``num_clbits`` and the global phase
  mod 2*pi rounded the same way.

The fingerprint is exact for ISA-level circuits (basis gates, measure,
reset, barrier, delay). A composite gate is identified by name and
parameters only, and control-flow conditions do not enter - decompose
before fingerprinting if that matters. Metadata and names never enter.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.converters import circuit_to_dag

FINGERPRINT_VERSION = "tcfs-fp-v1"
_TWO_PI = 2.0 * math.pi
_NOT_2Q = frozenset({"measure", "barrier", "reset", "delay"})
_IDLE = frozenset({"barrier", "delay"})


def _num(x: float, d: int) -> str:
    v = round(float(x), d) + 0.0          # + 0.0 folds -0.0 into 0.0
    return f"{v:.{d}f}"


def _param_token(p, d: int) -> str:
    if isinstance(p, ParameterExpression):
        if p.parameters:
            raise ValueError("circuit has unbound Parameters "
                             f"{sorted(str(x) for x in p.parameters)}: bind "
                             "them before fingerprinting")
        z = complex(p)
        return _num(z.real, d) if z.imag == 0 else \
            f"({_num(z.real, d)},{_num(z.imag, d)})"
    if isinstance(p, QuantumCircuit):
        return "qc:" + circuit_fingerprint(p, d)
    if isinstance(p, (bool, np.bool_)):
        return str(bool(p))
    if isinstance(p, (int, float, np.integer, np.floating)):
        return _num(p, d)
    if isinstance(p, (complex, np.complexfloating)):
        return f"({_num(p.real, d)},{_num(p.imag, d)})"
    if isinstance(p, np.ndarray):
        flat = np.asarray(p).ravel()
        return "a" + str(p.shape) + "[" + ",".join(
            _param_token(complex(x) if np.iscomplexobj(flat) else x, d)
            for x in flat) + "]"
    return repr(p)


def _global_phase_token(qc: QuantumCircuit, d: int) -> str:
    gp = qc.global_phase
    if isinstance(gp, ParameterExpression):
        if gp.parameters:
            raise ValueError("global phase depends on unbound Parameters: "
                             "bind them before fingerprinting")
        gp = float(gp)
    r = round(float(gp) % _TWO_PI, d)
    if r >= round(_TWO_PI, d):                # 2*pi - epsilon rounds up
        r = 0.0
    return _num(r, d)


def circuit_fingerprint(qc: QuantumCircuit, params_decimals: int = 9) -> str:
    """sha256 hex of the canonical form of a bound circuit (module doc).

    Equal for a circuit and its QPY round trip (with or without a layout),
    whatever the bytes of the dump; different as soon as one parameter moves
    by more than 10**-params_decimals. Raises ``ValueError`` on unbound
    ``Parameter``s."""
    if qc.parameters:
        raise ValueError(f"circuit has {len(qc.parameters)} unbound "
                         "Parameters: bind them before fingerprinting")
    d = int(params_decimals)
    dag = circuit_to_dag(qc)
    qidx = {q: i for i, q in enumerate(dag.qubits)}
    cidx = {c: i for i, c in enumerate(dag.clbits)}
    tokens: dict = {}

    def wire(w) -> str:
        return f"q{qidx[w]:07d}" if w in qidx else f"c{cidx[w]:07d}"

    def key(node) -> str:
        op = getattr(node, "op", None)
        if op is None:                       # DAGInNode / DAGOutNode
            tag = "\x00" if type(node).__name__ == "DAGInNode" else "\x7f\x7f"
            return tag + wire(node.wire)
        parts = [op.name,
                 ",".join(str(qidx[q]) for q in node.qargs),
                 ",".join(str(cidx[c]) for c in node.cargs),
                 ";".join(_param_token(p, d) for p in op.params)]
        if op.name in _IDLE:
            parts.append(str(getattr(op, "unit", "") or ""))
        tok = "|".join(parts)
        tokens[node._node_id] = tok
        return tok

    h = hashlib.sha256()
    h.update(f"{FINGERPRINT_VERSION}\nq={qc.num_qubits}\nc={qc.num_clbits}\n"
             f"gp={_global_phase_token(qc, d)}\n".encode())
    for node in dag.topological_op_nodes(key=key):
        tok = tokens.get(node._node_id)
        if tok is None:                      # key not consulted for this node
            tok = key(node)
        h.update(tok.encode())
        h.update(b"\n")
    return h.hexdigest()


def _is_2q(ci) -> bool:
    return len(ci.qubits) >= 2 and ci.operation.name not in _NOT_2Q


def structural_summary(qc: QuantumCircuit) -> dict:
    """Small JSON-able structure record of one circuit: ``n_qubits`` (qubits
    touched by any operation other than barrier / delay), ``n_clbits``,
    ``two_qubit_gates`` (measure / barrier / reset / delay excluded),
    ``depth_2q``, ``depth`` and ``ops`` (``count_ops``)."""
    active = set()
    two_q = 0
    for ci in qc.data:
        if ci.operation.name in _IDLE:
            continue
        active.update(ci.qubits)
        if _is_2q(ci):
            two_q += 1
    return {"n_qubits": len(active), "n_clbits": int(qc.num_clbits),
            "two_qubit_gates": int(two_q), "depth_2q": int(qc.depth(_is_2q)),
            "depth": int(qc.depth()),
            "ops": {str(k): int(v) for k, v in qc.count_ops().items()}}
