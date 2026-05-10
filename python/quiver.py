"""
quiver.py
=========
Quiver-with-relations data structures for bond algebras.

A quiver Q = (Q_0, Q_1, s, t) consists of:
  - Q_0: set of vertices (atoms in the unit cell)
  - Q_1: set of arrows (bonds, oriented)
  - s, t: source and target maps Q_1 -> Q_0

The path algebra kQ has basis = all paths in Q (including length-0
paths e_i at each vertex). Multiplication = path concatenation
(zero if not composable).

The bond algebra is kQ / I where I is an admissible ideal encoding:
  - Triangle commutativity: if a-b-c and a-c are both paths, the
    composition equals the direct arrow (up to scalar).
  - Symmetry relations: arrows related by point-group symmetry
    carry equal coefficients.

For Koszul algebras (quadratic relations), the Koszul complex gives
HH^* efficiently. For general path algebras, we use the bar resolution.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class Arrow:
    """A directed edge (bond) in the quiver."""
    source: int
    target: int
    label: str = ""
    weight: float = 1.0


@dataclass
class Quiver:
    """
    A quiver (directed graph) with vertices and arrows.

    Vertices are indexed 0, ..., n_vertices - 1.
    """
    n_vertices: int
    arrows: List[Arrow] = field(default_factory=list)

    def add_arrow(self, source: int, target: int,
                  label: str = "", weight: float = 1.0) -> int:
        idx = len(self.arrows)
        self.arrows.append(Arrow(source, target, label, weight))
        return idx

    def arrows_from(self, v: int) -> List[int]:
        return [i for i, a in enumerate(self.arrows) if a.source == v]

    def arrows_to(self, v: int) -> List[int]:
        return [i for i, a in enumerate(self.arrows) if a.target == v]

    def adjacency_matrix(self) -> np.ndarray:
        A = np.zeros((self.n_vertices, self.n_vertices), dtype=int)
        for arrow in self.arrows:
            A[arrow.source, arrow.target] += 1
        return A

    @property
    def n_arrows(self) -> int:
        return len(self.arrows)


@dataclass
class Relation:
    """
    A relation in the path algebra: a linear combination of paths
    that equals zero.

    Each term is (coefficient, path) where path is a list of arrow indices.
    """
    terms: List[Tuple[float, List[int]]]

    @property
    def degree(self) -> int:
        if not self.terms:
            return 0
        return len(self.terms[0][1])

    def is_quadratic(self) -> bool:
        return all(len(path) == 2 for _, path in self.terms)


@dataclass
class QuiverWithRelations:
    """
    A quiver with an admissible ideal of relations.

    The quotient kQ / I is the bond algebra A.
    """
    quiver: Quiver
    relations: List[Relation] = field(default_factory=list)

    def is_koszul(self) -> bool:
        return all(r.is_quadratic() for r in self.relations)

    @property
    def n_vertices(self) -> int:
        return self.quiver.n_vertices

    @property
    def n_arrows(self) -> int:
        return self.quiver.n_arrows

    @property
    def n_relations(self) -> int:
        return len(self.relations)


def paths_of_length(quiver: Quiver, length: int) -> List[List[int]]:
    """
    Enumerate all paths of a given length in the quiver.

    A path of length n is a sequence of n composable arrows.
    """
    if length == 0:
        return [[]]

    if length == 1:
        return [[i] for i in range(quiver.n_arrows)]

    shorter = paths_of_length(quiver, length - 1)
    result = []
    for path in shorter:
        last_arrow = quiver.arrows[path[-1]]
        for ext_idx in quiver.arrows_from(last_arrow.target):
            result.append(path + [ext_idx])
    return result


def path_source(quiver: Quiver, path: List[int]) -> int:
    if not path:
        raise ValueError("Empty path has no source")
    return quiver.arrows[path[0]].source


def path_target(quiver: Quiver, path: List[int]) -> int:
    if not path:
        raise ValueError("Empty path has no target")
    return quiver.arrows[path[-1]].target


def composable(quiver: Quiver, path1: List[int], path2: List[int]) -> bool:
    """Check if path1 followed by path2 is composable."""
    if not path1 or not path2:
        return True
    return quiver.arrows[path1[-1]].target == quiver.arrows[path2[0]].source
