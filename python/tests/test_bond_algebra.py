"""
Tests for bond algebra construction from crystal structures.
"""
import numpy as np
import pytest
from ..hochschild_bond import (bond_algebra, detect_bonds, build_bond_quiver,
                               compute_hochschild, hh2_decomposition)


def make_simple_molecule():
    """Water-like: O at origin, two H at ~1A."""
    coords = np.array([
        [0.0, 0.0, 0.0],    # O
        [0.96, 0.0, 0.0],   # H
        [-0.24, 0.93, 0.0], # H
    ])
    Z = np.array([8, 1, 1])
    return coords, Z


def make_linear_chain():
    """Linear chain: A-B-A-B (4 atoms, alternating)."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ])
    Z = np.array([29, 8, 29, 8])  # Cu-O-Cu-O
    return coords, Z


def make_square_planar():
    """Square planar: central atom + 4 neighbors."""
    coords = np.array([
        [0.0, 0.0, 0.0],    # center (Cu)
        [2.0, 0.0, 0.0],    # O right
        [-2.0, 0.0, 0.0],   # O left
        [0.0, 2.0, 0.0],    # O up
        [0.0, -2.0, 0.0],   # O down
    ])
    Z = np.array([29, 8, 8, 8, 8])
    return coords, Z


class TestBondDetection:
    def test_water_bonds(self):
        coords, Z = make_simple_molecule()
        bonds = detect_bonds(coords, Z)
        # Should find 2 O-H bonds
        assert len(bonds) == 2
        # Both should involve atom 0 (oxygen)
        for i, j, d in bonds:
            assert 0 in (i, j)

    def test_linear_chain_bonds(self):
        coords, Z = make_linear_chain()
        bonds = detect_bonds(coords, Z)
        # Should find 3 nearest-neighbor bonds
        assert len(bonds) == 3

    def test_no_self_bonds(self):
        coords, Z = make_simple_molecule()
        bonds = detect_bonds(coords, Z)
        for i, j, d in bonds:
            assert i != j
            assert d > 0.4


class TestBondQuiver:
    def test_water_quiver(self):
        coords, Z = make_simple_molecule()
        Q = build_bond_quiver(coords, Z)
        assert Q.n_vertices == 3
        # 2 bonds * 2 directions = 4 arrows
        assert Q.n_arrows == 4

    def test_square_planar_quiver(self):
        coords, Z = make_square_planar()
        Q = build_bond_quiver(coords, Z)
        assert Q.n_vertices == 5
        # 4 bonds * 2 directions = 8 arrows (center to each neighbor)
        assert Q.n_arrows >= 8


class TestBondAlgebra:
    def test_water_algebra(self):
        coords, Z = make_simple_molecule()
        A = bond_algebra(coords, Z, include_triangles=False,
                        include_symmetry=False)
        assert A.n_vertices == 3
        assert A.n_arrows == 4
        assert A.n_relations == 0

    def test_water_with_symmetry(self):
        coords, Z = make_simple_molecule()
        A = bond_algebra(coords, Z, include_triangles=False,
                        include_symmetry=True)
        # The two O-H bonds should be equivalent
        assert A.n_relations > 0

    def test_compute_hochschild_water(self):
        coords, Z = make_simple_molecule()
        A = bond_algebra(coords, Z, include_triangles=False,
                        include_symmetry=False)
        result = compute_hochschild(A, max_degree=2)
        assert 'hh_dims' in result
        assert 0 in result['hh_dims']
        assert 1 in result['hh_dims']
        assert 2 in result['hh_dims']
        # All dimensions non-negative
        for deg, dim in result['hh_dims'].items():
            assert dim >= 0


class TestHH2Decomposition:
    def test_trivial_group_decomposition(self):
        """With trivial point group, H^2(G) = 0, so all HH^2 is phonon."""
        coords, Z = make_simple_molecule()
        R_trivial = np.eye(3).reshape(1, 3, 3)
        A = bond_algebra(coords, Z, point_group_mats=R_trivial,
                        include_triangles=False)
        decomp = hh2_decomposition(A, coords, Z, R_trivial)
        assert decomp['dim_anomaly'] == 0
        assert decomp['dim_phonon'] == decomp['dim_HH2']
