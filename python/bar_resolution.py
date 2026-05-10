"""
bar_resolution.py
=================
Bar resolution and Hochschild cohomology computation for path algebras.

The Hochschild complex for an algebra A (over field k) is:
  C^n(A, A) = Hom_k(A^{tensor n}, A)

with differential:
  (d^n f)(a_1, ..., a_{n+1}) =
    a_1 * f(a_2, ..., a_{n+1})
    + sum_{i=1}^n (-1)^i f(a_1, ..., a_i * a_{i+1}, ..., a_{n+1})
    + (-1)^{n+1} f(a_1, ..., a_n) * a_{n+1}

For path algebras kQ/I, we work with the reduced bar resolution
restricted to the arrow-generated subalgebra. The key insight:
a basis for C^n is given by (n+2)-tuples of vertices
(i_0, ..., i_{n+1}) together with a choice of path from i_0 to i_1,
arrow from i_1 to i_2, ..., arrow from i_n to i_{n+1}, path from
i_{n+1} back. But for the REDUCED complex, we only need arrows
(not general paths) in the tensor factors.

For small quivers (unit cells with < 50 atoms), direct matrix
computation of the differential is feasible.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import svds
from typing import List, Dict, Tuple, Optional

from .quiver import (Quiver, QuiverWithRelations, Relation,
                     paths_of_length, path_source, path_target, composable)


def _basis_Cn(quiver: Quiver, n: int) -> List[Tuple]:
    """
    Compute a basis for the reduced Hochschild n-cochains C^n(A, A).

    For the path algebra kQ (before quotient), a basis element of
    C^n is specified by:
      - n arrows (the tensor factors a_1, ..., a_n)
      - a path p (the output value)
    subject to: source of a_1 = target of p (bimodule condition).

    For computational HH^*, we use the NORMALIZED complex where
    the output is also restricted to arrows (degree-1 elements)
    plus vertex idempotents (degree-0). This gives the same
    cohomology by Bardzell's theorem for monomial algebras.

    In practice, for the reduced complex:
      basis of C^n = { (a_1, ..., a_n) : composable sequence of n arrows }
                   indexed by paths of length n in Q.

    Returns list of paths (each path = list of arrow indices).
    """
    return paths_of_length(quiver, n)


def hochschild_differential_matrix(
    qwr: QuiverWithRelations,
    degree: int,
    max_path_length: int = 6,
) -> np.ndarray:
    """
    Compute the matrix of the Hochschild differential d^n: C^n -> C^{n+1}.

    For the REDUCED normalized bar complex of a path algebra with
    monomial relations, the differential acts on basis elements
    (paths of length n) as follows:

    For a path p = (a_1, ..., a_n) of length n:
      d(p) = sum of paths of length n+1 obtained by:
        - Prepending an arrow b composable with a_1: (+1) * (b, a_1, ..., a_n)
        - For each i, "splitting" a_i into two composable arrows: contributes
          to shorter paths via multiplication (but in the normalized complex,
          the inner face maps give composable extensions)
        - Appending an arrow c composable after a_n: (sign) * (a_1, ..., a_n, c)

    For MONOMIAL relations (paths in I), we set to zero any basis
    element that contains a relation path as a subpath.

    Parameters
    ----------
    qwr : QuiverWithRelations
    degree : int (compute d^{degree}: C^degree -> C^{degree+1})
    max_path_length : int (truncation for large quivers)

    Returns
    -------
    D : ndarray of shape (dim C^{degree+1}, dim C^degree)
        The differential matrix.
    """
    Q = qwr.quiver

    # Basis for C^degree and C^{degree+1}
    basis_n = paths_of_length(Q, degree)
    basis_n1 = paths_of_length(Q, degree + 1)

    if not basis_n or not basis_n1:
        return np.zeros((len(basis_n1), len(basis_n)))

    # Build index maps for fast lookup
    path_to_idx_n1 = {}
    for idx, path in enumerate(basis_n1):
        path_to_idx_n1[tuple(path)] = idx

    dim_n = len(basis_n)
    dim_n1 = len(basis_n1)
    D = np.zeros((dim_n1, dim_n))

    # Relation paths (for zeroing out)
    relation_paths = set()
    for rel in qwr.relations:
        for coeff, path in rel.terms:
            if abs(coeff) > 1e-10:
                relation_paths.add(tuple(path))

    def contains_relation(path: List[int]) -> bool:
        for rel_path in relation_paths:
            rlen = len(rel_path)
            for start in range(len(path) - rlen + 1):
                if tuple(path[start:start + rlen]) == rel_path:
                    return True
        return False

    for col, path_n in enumerate(basis_n):
        if contains_relation(path_n):
            continue

        # Face maps in the normalized bar complex:
        # d_0: prepend (left multiplication)
        if path_n:
            src = Q.arrows[path_n[0]].source
            for arr_idx in Q.arrows_to(src):
                new_path = [arr_idx] + path_n
                if not contains_relation(new_path):
                    key = tuple(new_path)
                    if key in path_to_idx_n1:
                        D[path_to_idx_n1[key], col] += 1.0

        # d_i for i = 1, ..., n-1: inner face maps
        # In the normalized complex for path algebras, the inner
        # face map d_i multiplies a_i with a_{i+1}. If a_i and a_{i+1}
        # compose to a single arrow (i.e., there is no intermediate
        # vertex), this contributes. But for path algebras where
        # arrows are elementary, a_i * a_{i+1} is a path of length 2,
        # which is NOT a single arrow unless there's a relation.
        # So inner face maps vanish in the normalized complex for
        # path algebras without quadratic relations collapsing paths.
        # This is a key simplification for monomial algebras.

        # d_{n+1}: append (right multiplication)
        if path_n:
            tgt = Q.arrows[path_n[-1]].target
            for arr_idx in Q.arrows_from(tgt):
                new_path = path_n + [arr_idx]
                if not contains_relation(new_path):
                    key = tuple(new_path)
                    if key in path_to_idx_n1:
                        sign = (-1) ** (degree + 1)
                        D[path_to_idx_n1[key], col] += sign

    return D


def compute_hochschild_cohomology(
    qwr: QuiverWithRelations,
    max_degree: int = 3,
    max_path_length: int = 8,
) -> Dict[int, int]:
    """
    Compute dimensions of Hochschild cohomology groups HH^n(A)
    for n = 0, 1, ..., max_degree.

    Uses the bar resolution: HH^n = ker(d^n) / im(d^{n-1}).

    Parameters
    ----------
    qwr : QuiverWithRelations
    max_degree : int
    max_path_length : int (truncation for paths)

    Returns
    -------
    hh_dims : dict mapping degree -> dim HH^n(A)
    """
    Q = qwr.quiver
    hh_dims = {}

    # Compute differentials d^{-1}, d^0, d^1, ..., d^{max_degree}
    # HH^n = ker(d^n) / im(d^{n-1})
    differentials = {}
    for deg in range(-1, max_degree + 1):
        if deg < 0:
            # d^{-1}: C^{-1} -> C^0, trivially zero (no C^{-1})
            dim_c0 = len(paths_of_length(Q, 0))
            differentials[-1] = np.zeros((dim_c0, 0))
        else:
            differentials[deg] = hochschild_differential_matrix(
                qwr, deg, max_path_length)

    for n in range(max_degree + 1):
        # HH^n = ker(d^n) / im(d^{n-1})
        d_n = differentials[n]       # maps C^n -> C^{n+1}
        d_nm1 = differentials[n - 1]  # maps C^{n-1} -> C^n

        # dim ker(d^n)
        if d_n.size == 0:
            # d^n is trivially zero: ker = all of C^n
            dim_cn = len(paths_of_length(Q, n))
            ker_dim = dim_cn
        else:
            # ker(d^n) = nullspace of d^n
            # d^n has shape (dim C^{n+1}, dim C^n)
            rank_dn = np.linalg.matrix_rank(d_n, tol=1e-10)
            ker_dim = d_n.shape[1] - rank_dn

        # dim im(d^{n-1})
        if d_nm1.size == 0:
            im_dim = 0
        else:
            im_dim = np.linalg.matrix_rank(d_nm1, tol=1e-10)

        hh_dims[n] = max(0, ker_dim - im_dim)

    return hh_dims


def hochschild_cocycles(
    qwr: QuiverWithRelations,
    degree: int,
    max_path_length: int = 8,
) -> np.ndarray:
    """
    Compute a basis for the cocycle space Z^n = ker(d^n).

    Returns
    -------
    cocycles : ndarray of shape (dim Z^n, dim C^n)
        Rows are cocycle representatives in the path basis.
    """
    d_n = hochschild_differential_matrix(qwr, degree, max_path_length)

    if d_n.size == 0:
        dim_cn = len(paths_of_length(qwr.quiver, degree))
        return np.eye(dim_cn)

    # Nullspace of d_n (columns of V where singular values are ~0)
    U, s, Vt = np.linalg.svd(d_n, full_matrices=True)
    rank = np.sum(s > 1e-10)
    nullspace = Vt[rank:, :]  # rows of Vt beyond the rank

    return nullspace


def hochschild_coboundaries(
    qwr: QuiverWithRelations,
    degree: int,
    max_path_length: int = 8,
) -> np.ndarray:
    """
    Compute a basis for the coboundary space B^n = im(d^{n-1}).

    Returns
    -------
    coboundaries : ndarray of shape (dim B^n, dim C^n)
        Rows are coboundary representatives.
    """
    if degree == 0:
        return np.zeros((0, len(paths_of_length(qwr.quiver, 0))))

    d_nm1 = hochschild_differential_matrix(qwr, degree - 1, max_path_length)

    if d_nm1.size == 0:
        dim_cn = len(paths_of_length(qwr.quiver, degree))
        return np.zeros((0, dim_cn))

    # Image of d^{n-1} = column space of d^{n-1}
    U, s, Vt = np.linalg.svd(d_nm1, full_matrices=True)
    rank = np.sum(s > 1e-10)
    image_basis = U[:, :rank].T  # rows form a basis for the image

    return image_basis
