"""The hierarchy, defined as data, and the summing matrix S derived from it.

The whole point of reconciliation is that the aggregation structure is a hard
constraint, not a modelling preference: every coherent forecast vector y (length
n, all nodes) must equal S b for some bottom-level vector b (length m).  S is
built from the parent/child declarations so that adding a substation cannot
leave the constraint matrix stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .linalg import Matrix, matvec, zeros


@dataclass(frozen=True)
class Node:
    name: str
    parent: str | None
    level: str


@dataclass
class Hierarchy:
    """A tree of demand nodes, ordered aggregates-first then bottom level."""

    nodes: list[Node]
    names: list[str] = field(init=False)
    index: dict[str, int] = field(init=False)
    bottom: list[str] = field(init=False)
    aggregates: list[str] = field(init=False)

    def __post_init__(self) -> None:
        by_name = {n.name: n for n in self.nodes}
        if len(by_name) != len(self.nodes):
            raise ValueError("duplicate node names in hierarchy")
        children: dict[str, list[str]] = {n.name: [] for n in self.nodes}
        roots = []
        for n in self.nodes:
            if n.parent is None:
                roots.append(n.name)
            else:
                if n.parent not in by_name:
                    raise ValueError("node %s has unknown parent %s" % (n.name, n.parent))
                children[n.parent].append(n.name)
        if len(roots) != 1:
            raise ValueError("hierarchy must have exactly one root, found %d" % len(roots))
        self._children = children
        self._by_name = by_name
        self.bottom = [n.name for n in self.nodes if not children[n.name]]
        self.aggregates = [n.name for n in self.nodes if children[n.name]]
        # Canonical ordering: aggregates first, bottom last.  Every matrix in
        # this project (S, the covariance, the reconciled vectors) uses it, so
        # it is fixed here once rather than re-derived per call site.
        self.names = list(self.aggregates) + list(self.bottom)
        self.index = {name: i for i, name in enumerate(self.names)}

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def m(self) -> int:
        return len(self.bottom)

    def children(self, name: str) -> list[str]:
        return list(self._children[name])

    def leaves_under(self, name: str) -> list[str]:
        kids = self._children[name]
        if not kids:
            return [name]
        out: list[str] = []
        for k in kids:
            out.extend(self.leaves_under(k))
        return out

    def summing_matrix(self) -> Matrix:
        """S, shape (n, m): row i selects the leaves that make up node i."""
        S = zeros(self.n, self.m)
        bottom_index = {name: j for j, name in enumerate(self.bottom)}
        for i, name in enumerate(self.names):
            for leaf in self.leaves_under(name):
                S[i][bottom_index[leaf]] = 1.0
        return S

    def coherence_residuals(self, y: list[float]) -> dict[str, float]:
        """Per-aggregate-node |parent - sum(children)| for a full node vector."""
        out = {}
        for name in self.aggregates:
            kids = self._children[name]
            got = sum(y[self.index[k]] for k in kids)
            out[name] = abs(y[self.index[name]] - got)
        return out

    def max_incoherence(self, y: list[float]) -> float:
        r = self.coherence_residuals(y)
        return max(r.values()) if r else 0.0

    def bottom_up(self, b: list[float]) -> list[float]:
        return matvec(self.summing_matrix(), b)


def default_hierarchy(regions: int = 4, per_region: int = 3) -> Hierarchy:
    """1 national + `regions` regions + regions*per_region sub-regions.

    Defaults give 1/4/12 = 17 nodes.  Deliberately modest: MinT needs an
    n x n solve and, more importantly, a covariance estimated from residuals,
    and with a realistic backtest window a 180-node hierarchy would be
    rank-deficient - the honest small version demonstrates the same maths.
    """
    nodes = [Node("NAT", None, "national")]
    for r in range(regions):
        rname = "R%d" % (r + 1)
        nodes.append(Node(rname, "NAT", "region"))
        for s in range(per_region):
            nodes.append(Node("%s_%s" % (rname, "abcdefgh"[s]), rname, "sub"))
    return Hierarchy(nodes)
