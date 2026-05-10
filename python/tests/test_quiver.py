"""
Tests for quiver data structures and path enumeration.
"""
import numpy as np
import pytest
from ..quiver import (Quiver, Arrow, Relation, QuiverWithRelations,
                      paths_of_length, path_source, path_target, composable)


def test_quiver_basic():
    Q = Quiver(n_vertices=3)
    Q.add_arrow(0, 1, "a")
    Q.add_arrow(1, 2, "b")
    Q.add_arrow(0, 2, "c")

    assert Q.n_vertices == 3
    assert Q.n_arrows == 3
    assert Q.arrows_from(0) == [0, 2]
    assert Q.arrows_to(2) == [1, 2]


def test_adjacency_matrix():
    Q = Quiver(n_vertices=3)
    Q.add_arrow(0, 1)
    Q.add_arrow(1, 2)
    Q.add_arrow(0, 2)

    A = Q.adjacency_matrix()
    assert A[0, 1] == 1
    assert A[1, 2] == 1
    assert A[0, 2] == 1
    assert A[2, 0] == 0


def test_paths_of_length():
    Q = Quiver(n_vertices=3)
    Q.add_arrow(0, 1)  # arrow 0
    Q.add_arrow(1, 2)  # arrow 1
    Q.add_arrow(0, 2)  # arrow 2

    # Length 0: empty paths
    p0 = paths_of_length(Q, 0)
    assert p0 == [[]]

    # Length 1: all arrows
    p1 = paths_of_length(Q, 1)
    assert len(p1) == 3

    # Length 2: composable pairs
    p2 = paths_of_length(Q, 2)
    # Arrow 0 (0->1) followed by arrow 1 (1->2) is composable
    assert [0, 1] in p2
    # Arrow 2 (0->2) has no continuation (nothing from 2)
    assert len(p2) == 1


def test_paths_cycle():
    Q = Quiver(n_vertices=3)
    Q.add_arrow(0, 1)  # 0
    Q.add_arrow(1, 2)  # 1
    Q.add_arrow(2, 0)  # 2

    p2 = paths_of_length(Q, 2)
    assert [0, 1] in p2  # 0->1->2
    assert [1, 2] in p2  # 1->2->0
    assert [2, 0] in p2  # 2->0->1
    assert len(p2) == 3

    p3 = paths_of_length(Q, 3)
    assert [0, 1, 2] in p3  # full cycle
    assert len(p3) == 3


def test_relation_quadratic():
    rel = Relation(terms=[(1.0, [0, 1]), (-1.0, [2, 3])])
    assert rel.is_quadratic()
    assert rel.degree == 2


def test_relation_non_quadratic():
    rel = Relation(terms=[(1.0, [0, 1, 2]), (-1.0, [3])])
    assert not rel.is_quadratic()


def test_quiver_with_relations():
    Q = Quiver(n_vertices=3)
    Q.add_arrow(0, 1)
    Q.add_arrow(1, 2)
    Q.add_arrow(0, 2)

    rel = Relation(terms=[(1.0, [0, 1]), (-1.0, [2])])
    qwr = QuiverWithRelations(quiver=Q, relations=[rel])

    assert qwr.n_vertices == 3
    assert qwr.n_arrows == 3
    assert qwr.n_relations == 1
    assert not qwr.is_koszul()  # relation has mixed degrees


def test_koszul_quiver():
    Q = Quiver(n_vertices=4)
    Q.add_arrow(0, 1)
    Q.add_arrow(1, 2)
    Q.add_arrow(0, 2)
    Q.add_arrow(2, 3)

    rel = Relation(terms=[(1.0, [0, 1]), (-1.0, [2, 3])])
    # This isn't quite right (composability), but tests is_koszul logic
    qwr = QuiverWithRelations(quiver=Q, relations=[rel])
    assert qwr.is_koszul()


def test_composable():
    Q = Quiver(n_vertices=3)
    Q.add_arrow(0, 1)  # 0
    Q.add_arrow(1, 2)  # 1
    Q.add_arrow(0, 2)  # 2

    assert composable(Q, [0], [1])   # 0->1, 1->2: yes
    assert not composable(Q, [0], [2])  # 0->1, 0->2: target 1 != source 0
    # Arrow 2 is 0->2, arrow 1 is 1->2. Target of [2] is 2, source of [1] is 1. Not composable.
    assert not composable(Q, [2], [1])
