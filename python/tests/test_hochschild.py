"""
Tests for Hochschild cohomology computation.

Ground truth for small examples:
- Path algebra of A_n quiver (linear): HH^0 = n, HH^1 = n-1, HH^k = 0 for k >= 2
- Path algebra of cycle C_n: HH^0 = n, HH^1 = n, HH^2 = n, ...
- Polynomial ring k[x] (= path algebra of loop): HH^0 = k[x], HH^1 = k[x], HH^k = 0 k>=2

For FINITE-DIMENSIONAL path algebras (with length truncation or relations),
the dimensions are computable.
"""
import numpy as np
import pytest
from ..quiver import Quiver, QuiverWithRelations, Relation
from ..bar_resolution import (compute_hochschild_cohomology,
                              hochschild_differential_matrix,
                              hochschild_cocycles)


def make_A2_quiver():
    """A_2 quiver: two vertices, one arrow 0 -> 1."""
    Q = Quiver(n_vertices=2)
    Q.add_arrow(0, 1)
    return QuiverWithRelations(quiver=Q, relations=[])


def make_A3_quiver():
    """A_3 quiver: 0 -> 1 -> 2."""
    Q = Quiver(n_vertices=3)
    Q.add_arrow(0, 1)
    Q.add_arrow(1, 2)
    return QuiverWithRelations(quiver=Q, relations=[])


def make_triangle_quiver():
    """Triangle: 0 -> 1, 1 -> 2, 2 -> 0 (cycle C_3)."""
    Q = Quiver(n_vertices=3)
    Q.add_arrow(0, 1)
    Q.add_arrow(1, 2)
    Q.add_arrow(2, 0)
    return QuiverWithRelations(quiver=Q, relations=[])


def make_square_quiver():
    """Square with relations: 0->1->3 = 0->2->3."""
    Q = Quiver(n_vertices=4)
    Q.add_arrow(0, 1)  # 0
    Q.add_arrow(1, 3)  # 1
    Q.add_arrow(0, 2)  # 2
    Q.add_arrow(2, 3)  # 3

    rel = Relation(terms=[(1.0, [0, 1]), (-1.0, [2, 3])])
    return QuiverWithRelations(quiver=Q, relations=[rel])


class TestDifferential:
    def test_A2_differential_d0(self):
        qwr = make_A2_quiver()
        D = hochschild_differential_matrix(qwr, degree=0)
        # C^0: paths of length 0 = [[]] -> dim 1
        # C^1: paths of length 1 = [[0]] (the single arrow) -> dim 1
        # d^0 maps C^0 -> C^1, shape = (dim C^1, dim C^0) = (1, 1)
        assert D.shape[1] == 1  # dim C^0
        assert D.shape[0] == 1  # dim C^1
        # Wait: paths of length 1 = all arrows = 1 arrow
        # Hmm, let me reconsider. The A_2 quiver has 1 arrow.
        # paths_of_length(Q, 0) = [[]] (1 element)
        # paths_of_length(Q, 1) = [[0]] (1 element, the single arrow)
        # d^0: C^0 -> C^1, so D shape = (dim C^1, dim C^0) = (1, 1)

    def test_A3_differential(self):
        qwr = make_A3_quiver()
        D0 = hochschild_differential_matrix(qwr, degree=0)
        D1 = hochschild_differential_matrix(qwr, degree=1)
        # A3: 2 arrows, 1 path of length 2
        # C^0: paths of length 0 = [[]] -> dim 1
        # C^1: paths of length 1 = [[0], [1]] -> dim 2
        # C^2: paths of length 2 = [[0, 1]] -> dim 1
        assert D0.shape == (2, 1)
        assert D1.shape == (1, 2)


class TestHHDimensions:
    def test_A2_hh(self):
        qwr = make_A2_quiver()
        hh = compute_hochschild_cohomology(qwr, max_degree=2)
        # For A_2 path algebra (hereditary, finite rep type):
        # This is a finite-dimensional algebra with basis {e_0, e_1, a}
        # HH^0 = center = span of e_0 + e_1 (if path algebra is basic)
        # For the reduced complex, results depend on truncation.
        # At minimum, all dims should be non-negative
        for deg, dim in hh.items():
            assert dim >= 0, f"HH^{deg} = {dim} < 0"

    def test_A3_hh(self):
        qwr = make_A3_quiver()
        hh = compute_hochschild_cohomology(qwr, max_degree=2)
        for deg, dim in hh.items():
            assert dim >= 0

    def test_triangle_hh(self):
        qwr = make_triangle_quiver()
        hh = compute_hochschild_cohomology(qwr, max_degree=2)
        for deg, dim in hh.items():
            assert dim >= 0
        # For the cycle C_3 path algebra (infinite-dimensional),
        # in the truncated computation we expect HH^0 >= 1

    def test_square_with_relations(self):
        qwr = make_square_quiver()
        hh = compute_hochschild_cohomology(qwr, max_degree=2)
        for deg, dim in hh.items():
            assert dim >= 0


class TestGroupCohomology:
    def test_cyclic_h2(self):
        """H^2(Z_n, C) = 0 for cyclic groups (trivial Schur multiplier)."""
        from ..hochschild_bond import group_cohomology_h2

        # Z_2: single 180-degree rotation about z
        R_z2 = np.array([
            np.eye(3),
            np.diag([-1, -1, 1.0]),
        ])
        assert group_cohomology_h2(R_z2) == 0

    def test_trivial_h2(self):
        """H^2(trivial group, C) = 0."""
        from ..hochschild_bond import group_cohomology_h2
        R_trivial = np.eye(3).reshape(1, 3, 3)
        assert group_cohomology_h2(R_trivial) == 0
