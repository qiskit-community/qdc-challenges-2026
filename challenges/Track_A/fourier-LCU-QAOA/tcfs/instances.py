"""Densest-k-subgraph instances (paper Sec. VI, challenge spec Sec. 2).

Problem:  max_x  f(x) = sum_{(i,j) in E} w_ij x_i x_j   s.t.  1^T x = k,
folded into the unconstrained  f_lam(x) = f(x) - lam*(1^T x - k)^2  with
lam = 1 + max_i sum_{(i,j) in E} |w_ij|, which makes constraint violation
never beneficial (paper Sec. VI A).

Exact tables are indexed by the Qiskit statevector index b, with
x_i = (b >> i) & 1.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class DensestKSubgraph:
    """A densest-k-subgraph instance on n nodes."""

    n: int
    k: int
    edges: tuple  # tuple of (i, j, w_ij)
    name: str = "instance"
    _cache: dict = field(default_factory=dict, compare=False, repr=False)
    # The id of a published instance: the "instance_id" of a handout file
    # (``hardware.load_handout``), "demo106" for the notebook's demo instance
    # (``event.demo_instance``), "" for every other instance. The grading
    # functions send it; the grading server holds the instance's references
    # under it. It does not take part in equality or hashing.
    instance_id: str = field(default="", compare=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def random_regular(cls, n: int = 12, d: int = 3, k: int | None = None,
                       seed: int = 0, name: str | None = None):
        """Seeded random d-regular unweighted instance (paper Sec. VI A/B)."""
        g = nx.random_regular_graph(d, n, seed=seed)
        edges = tuple(sorted((min(i, j), max(i, j), 1.0) for i, j in g.edges()))
        return cls(n=n, k=k if k is not None else n // 3, edges=edges,
                   name=name or f"regular{d}-n{n}-seed{seed}")

    @classmethod
    def from_graph(cls, graph: nx.Graph, k: int, name: str = "instance"):
        edges = []
        for i, j, data in graph.edges(data=True):
            w = float(data.get("weight", 1.0))
            edges.append((min(i, j), max(i, j), w))
        return cls(n=graph.number_of_nodes(), k=k, edges=tuple(sorted(edges)),
                   name=name)

    # ------------------------------------------------------------- properties
    @property
    def graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        g.add_weighted_edges_from(self.edges)
        return g

    @property
    def lam(self) -> float:
        """Penalty factor lam = 1 + max_i sum_j |w_ij| (paper Sec. VI A)."""
        deg = np.zeros(self.n)
        for i, j, w in self.edges:
            deg[i] += abs(w)
            deg[j] += abs(w)
        return 1.0 + float(deg.max())

    @property
    def theta_init(self) -> float:
        """Warm-start angle theta_init = 2*arcsin(sqrt(k/n))."""
        return 2.0 * np.arcsin(np.sqrt(self.k / self.n))

    @property
    def p_feasible_warm_start(self) -> float:
        """P(1^T x = k) for the warm-start product state: the binomial
        probability C(n, k) p^k (1 - p)^(n - k) with p = k / n."""
        n, k = self.n, self.k
        p = k / n
        return float(math.comb(n, k) * p ** k * (1.0 - p) ** (n - k))

    # ----------------------------------------------------------- exact tables
    def _bits(self) -> np.ndarray:
        """(2^n, n) matrix of bits; column i is x_i for every index."""
        if "bits" not in self._cache:
            idx = np.arange(1 << self.n, dtype=np.int64)
            self._cache["bits"] = ((idx[:, None] >> np.arange(self.n)) & 1
                                   ).astype(np.int8)
        return self._cache["bits"]

    def hamming_table(self) -> np.ndarray:
        """1^T x for every statevector index."""
        if "ham" not in self._cache:
            self._cache["ham"] = self._bits().sum(axis=1).astype(np.int64)
        return self._cache["ham"]

    def objective_table(self) -> np.ndarray:
        """f(x) for every statevector index."""
        if "f" not in self._cache:
            bits = self._bits()
            f = np.zeros(1 << self.n)
            for i, j, w in self.edges:
                f += w * bits[:, i] * bits[:, j]
            self._cache["f"] = f
        return self._cache["f"]

    def penalized_table(self) -> np.ndarray:
        """f_lam(x) = f(x) - lam*(1^T x - k)^2 for every statevector index."""
        if "flam" not in self._cache:
            w = self.hamming_table()
            self._cache["flam"] = (self.objective_table()
                                   - self.lam * (w - self.k) ** 2)
        return self._cache["flam"]

    def objective_of_bitstring(self, x: np.ndarray) -> float:
        """f(x) for a single 0/1 vector x (index order: x[i] = node i)."""
        return float(sum(w * x[i] * x[j] for i, j, w in self.edges))

    def evaluate_bits(self, bits: np.ndarray):
        """(f, w) for a batch of 0/1 rows bits[m, n] (bits[:, i] = x_i) -
        works at any n (no 2^n tables)."""
        bits = np.asarray(bits)
        if "edge_arrays" not in self._cache:
            e = np.array([(i, j, w) for i, j, w in self.edges], dtype=float)
            self._cache["edge_arrays"] = (e[:, 0].astype(int),
                                          e[:, 1].astype(int), e[:, 2])
        ei, ej, ew = self._cache["edge_arrays"]
        f = (bits[:, ei] * bits[:, ej]).astype(float) @ ew
        return f, bits.sum(axis=1).astype(int)

    # ------------------------------------------------------------- references
    def brute_force(self) -> tuple[float, list]:
        """(OPT, argmax list) of the constrained problem, by enumeration."""
        if "opt" not in self._cache:
            best, arg = -np.inf, []
            for sub in itertools.combinations(range(self.n), self.k):
                x = np.zeros(self.n, dtype=np.int8)
                x[list(sub)] = 1
                v = self.objective_of_bitstring(x)
                if v > best + 1e-12:
                    best, arg = v, [sub]
                elif abs(v - best) <= 1e-12:
                    arg.append(sub)
            self._cache["opt"] = (best, arg)
        return self._cache["opt"]

    def warm_start_probs(self) -> np.ndarray:
        """Exact warm-start product-state probabilities for every index."""
        w = self.hamming_table().astype(float)
        p = self.k / self.n
        # direct powers (not exp/log) so the k = 0 and k = n edges are exact
        return p ** w * (1.0 - p) ** (self.n - w)

    def baseline_B(self, alpha: float = 0.10) -> float:
        """B = CVaR_alpha(f_lam) of the warm-start state (challenge spec Sec. 2).

        Exact enumeration; only valid for small n (the spec computes B at
        generation time; at utility scale it is estimated by seeded Monte
        Carlo over the product distribution instead).
        """
        from .cvar import cvar

        return cvar(self.penalized_table(), self.warm_start_probs(), alpha)
