"""
decomposition.py
================
Decompose HH^2 into anomaly (group-cohomological) and phonon-deformation
parts, with projection operators and explicit cocycle representatives.

The key theorem (conjectural, to be validated computationally):

  HH^2(A_M) = H^2(G, C) + V_phonon

where:
  - A_M is the bond algebra of material M
  - G is the point group of M
  - H^2(G, C) is the Schur multiplier (anomaly part, already in v6)
  - V_phonon is the orthogonal complement (novel phonon-deformation content)

The phonon-deformation modes in V_phonon correspond to infinitesimal
deformations of the bond algebra that are NOT induced by the group
structure. These connect to soft phonon modes and structural instabilities
relevant for superconductivity.

Validation strategy:
  - Materials with large dim(V_phonon) should correlate with
    low phonon minimum frequency (near-instability)
  - Materials with V_phonon = 0 should be structurally rigid
    (high phonon frequencies, unlikely phonon-mediated SC)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from .quiver import QuiverWithRelations, paths_of_length
from .bar_resolution import (hochschild_cocycles, hochschild_coboundaries,
                             hochschild_differential_matrix)


def group_cocycles_in_bond_algebra(
    qwr: QuiverWithRelations,
    point_group_mats: np.ndarray,
    coords: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """
    Embed group 2-cocycles H^2(G, C) into the Hochschild 2-cocycle
    space Z^2(A, A).

    The embedding works as follows: a group 2-cocycle f: G x G -> C
    can be "spread" over the quiver by assigning to each pair of
    composable arrows (a, b) the value f(g_a, g_b) where g_a is the
    symmetry element mapping source(a) to target(a).

    This requires a choice of orbit representatives, which we fix
    using the atom permutation action of G.

    Parameters
    ----------
    qwr : QuiverWithRelations
    point_group_mats : (ng, 3, 3)
    coords : (na, 3)
    Z : (na,)

    Returns
    -------
    embedding : (dim_H2_group, dim_C2) matrix
        Rows are the embedded group cocycles in the C^2 basis.
    """
    Q = qwr.quiver
    ng = point_group_mats.shape[0]
    na = Q.n_vertices

    # Build multiplication table
    mult_table = np.zeros((ng, ng), dtype=int)
    for i in range(ng):
        for j in range(ng):
            prod = point_group_mats[i] @ point_group_mats[j]
            for k in range(ng):
                if np.max(np.abs(prod - point_group_mats[k])) < 1e-6:
                    mult_table[i, j] = k
                    break

    # Find atom permutations under group action
    centered = coords - coords.mean(axis=0)
    atom_perms = np.zeros((ng, na), dtype=int)
    for g in range(ng):
        R = point_group_mats[g]
        rotated = (R @ centered.T).T
        for i in range(na):
            dists = np.linalg.norm(centered - rotated[i:i+1], axis=1)
            candidates = np.where((dists < 0.15) & (Z == Z[i]))[0]
            if len(candidates) > 0:
                atom_perms[g, i] = candidates[np.argmin(dists[candidates])]
            else:
                atom_perms[g, i] = i

    # For each arrow, find which group element maps source to target
    # (if any; otherwise assign identity)
    arrows = Q.arrows
    arrow_group_elem = np.zeros(len(arrows), dtype=int)
    for arr_idx, arrow in enumerate(arrows):
        src, tgt = arrow.source, arrow.target
        for g in range(ng):
            if atom_perms[g, src] == tgt:
                arrow_group_elem[arr_idx] = g
                break

    # Basis of C^2 = paths of length 2
    basis_c2 = paths_of_length(Q, 2)
    dim_c2 = len(basis_c2)

    # Compute group H^2 cocycles via bar resolution
    # Z^2_group: f(g,h) + f(gh,k) = f(g,hk) + f(h,k)
    n_gvars = ng * ng
    constraints = []
    for i in range(ng):
        for j in range(ng):
            for k in range(ng):
                row = np.zeros(n_gvars)
                gh = mult_table[i, j]
                hk = mult_table[j, k]
                row[i * ng + j] += 1
                row[gh * ng + k] += 1
                row[i * ng + hk] -= 1
                row[j * ng + k] -= 1
                constraints.append(row)

    C = np.array(constraints)
    rank_C = np.linalg.matrix_rank(C, tol=1e-10)
    U, s, Vt = np.linalg.svd(C, full_matrices=True)
    group_cocycle_basis = Vt[rank_C:]  # (dim_Z2_group, ng^2)

    # Remove coboundaries
    cobnd = np.zeros((n_gvars, ng))
    for i in range(ng):
        for j in range(ng):
            gh = mult_table[i, j]
            idx = i * ng + j
            cobnd[idx, i] += 1
            cobnd[idx, j] += 1
            cobnd[idx, gh] -= 1

    rank_cobnd = np.linalg.matrix_rank(cobnd, tol=1e-10)

    # Project out coboundaries from Z^2
    if rank_cobnd > 0 and group_cocycle_basis.shape[0] > 0:
        Uc, sc, Vct = np.linalg.svd(cobnd, full_matrices=True)
        cobnd_image = Uc[:, :rank_cobnd]  # (ng^2, rank_cobnd)

        # Project each cocycle out of the coboundary space
        proj = np.eye(n_gvars) - cobnd_image @ cobnd_image.T
        group_h2_reps = group_cocycle_basis @ proj.T

        # Re-extract linearly independent rows
        if group_h2_reps.shape[0] > 0:
            rank_h2 = np.linalg.matrix_rank(group_h2_reps, tol=1e-10)
            U2, s2, V2t = np.linalg.svd(group_h2_reps, full_matrices=False)
            group_h2_reps = V2t[:rank_h2]
        else:
            group_h2_reps = np.zeros((0, n_gvars))
    else:
        group_h2_reps = group_cocycle_basis

    dim_h2_group = group_h2_reps.shape[0]

    if dim_h2_group == 0 or dim_c2 == 0:
        return np.zeros((0, dim_c2))

    # Embed into C^2(A, A): for each path (a1, a2) of length 2,
    # assign value f(g_{a1}, g_{a2})
    embedding = np.zeros((dim_h2_group, dim_c2))
    for path_idx, path in enumerate(basis_c2):
        g1 = arrow_group_elem[path[0]]
        g2 = arrow_group_elem[path[1]]
        for h2_idx in range(dim_h2_group):
            embedding[h2_idx, path_idx] = group_h2_reps[h2_idx, g1 * ng + g2]

    return embedding


def decompose_hh2(
    qwr: QuiverWithRelations,
    coords: np.ndarray,
    Z: np.ndarray,
    point_group_mats: np.ndarray,
) -> Dict[str, any]:
    """
    Full decomposition of HH^2 into anomaly and phonon parts.

    Returns
    -------
    result : dict with:
        'dim_HH2': int
        'dim_anomaly': int (from H^2(G))
        'dim_phonon': int (novel content)
        'cocycle_basis': ndarray (full HH^2 basis)
        'anomaly_basis': ndarray (anomaly part)
        'phonon_basis': ndarray (phonon-deformation part)
    """
    Q = qwr.quiver

    # Full HH^2 cocycles and coboundaries
    cocycles = hochschild_cocycles(qwr, degree=2)
    coboundaries = hochschild_coboundaries(qwr, degree=2)

    # HH^2 = Z^2 / B^2: get representatives
    dim_z2 = cocycles.shape[0] if cocycles.size > 0 else 0
    dim_b2 = coboundaries.shape[0] if coboundaries.size > 0 else 0
    dim_hh2 = dim_z2 - dim_b2

    if dim_hh2 <= 0:
        dim_c2 = len(paths_of_length(Q, 2))
        return {
            'dim_HH2': 0,
            'dim_anomaly': 0,
            'dim_phonon': 0,
            'cocycle_basis': np.zeros((0, dim_c2)),
            'anomaly_basis': np.zeros((0, dim_c2)),
            'phonon_basis': np.zeros((0, dim_c2)),
        }

    # Get group-cocycle embedding
    group_embedding = group_cocycles_in_bond_algebra(
        qwr, point_group_mats, coords, Z)

    dim_c2 = cocycles.shape[1] if cocycles.size > 0 else 0

    # Project group embedding onto Z^2 (intersect with cocycles)
    if group_embedding.shape[0] > 0 and cocycles.shape[0] > 0:
        # Find the part of group_embedding that lies in Z^2
        # Project each row of group_embedding onto span(cocycles)
        if cocycles.shape[0] > 0:
            # cocycles rows form a basis for Z^2
            # Project: proj = cocycles.T @ (cocycles @ cocycles.T)^-1 @ cocycles
            CtC = cocycles @ cocycles.T
            if np.linalg.matrix_rank(CtC) > 0:
                proj_coeffs = np.linalg.lstsq(
                    CtC.T, (group_embedding @ cocycles.T).T, rcond=None)[0].T
                anomaly_in_z2 = proj_coeffs @ cocycles
            else:
                anomaly_in_z2 = np.zeros((group_embedding.shape[0], dim_c2))
        else:
            anomaly_in_z2 = np.zeros((group_embedding.shape[0], dim_c2))

        # Further project out B^2 to get anomaly in HH^2
        if coboundaries.shape[0] > 0:
            BtB = coboundaries @ coboundaries.T
            if np.linalg.matrix_rank(BtB) > 0:
                proj_B = coboundaries.T @ np.linalg.pinv(BtB) @ coboundaries
                anomaly_in_hh2 = anomaly_in_z2 - anomaly_in_z2 @ proj_B.T
            else:
                anomaly_in_hh2 = anomaly_in_z2
        else:
            anomaly_in_hh2 = anomaly_in_z2

        # Count independent anomaly directions
        rank_anomaly = np.linalg.matrix_rank(anomaly_in_hh2, tol=1e-10)
        dim_anomaly = min(rank_anomaly, dim_hh2)
    else:
        anomaly_in_hh2 = np.zeros((0, dim_c2))
        dim_anomaly = 0

    dim_phonon = dim_hh2 - dim_anomaly

    # Build explicit bases (using SVD to clean up)
    if dim_anomaly > 0:
        U_a, s_a, Vt_a = np.linalg.svd(anomaly_in_hh2, full_matrices=False)
        anomaly_basis = Vt_a[:dim_anomaly]
    else:
        anomaly_basis = np.zeros((0, dim_c2))

    # Phonon basis = cocycles orthogonal to both coboundaries and anomaly
    if dim_phonon > 0 and cocycles.shape[0] > 0:
        # Remove coboundary and anomaly directions from cocycles
        remove = []
        if coboundaries.shape[0] > 0:
            remove.append(coboundaries)
        if anomaly_basis.shape[0] > 0:
            remove.append(anomaly_basis)

        if remove:
            remove_mat = np.vstack(remove)
            # Gram-Schmidt orthogonalization
            Q_mat, R_mat = np.linalg.qr(remove_mat.T, mode='reduced')
            proj_remove = Q_mat @ Q_mat.T
            phonon_candidates = cocycles - cocycles @ proj_remove
            # Extract non-zero rows
            norms = np.linalg.norm(phonon_candidates, axis=1)
            phonon_candidates = phonon_candidates[norms > 1e-10]
            if phonon_candidates.shape[0] > 0:
                rank_ph = np.linalg.matrix_rank(phonon_candidates, tol=1e-10)
                U_p, s_p, Vt_p = np.linalg.svd(
                    phonon_candidates, full_matrices=False)
                phonon_basis = Vt_p[:min(dim_phonon, rank_ph)]
            else:
                phonon_basis = np.zeros((0, dim_c2))
        else:
            phonon_basis = cocycles[:dim_phonon]
    else:
        phonon_basis = np.zeros((0, dim_c2))

    return {
        'dim_HH2': dim_hh2,
        'dim_anomaly': dim_anomaly,
        'dim_phonon': dim_phonon,
        'cocycle_basis': cocycles[:dim_hh2] if cocycles.shape[0] >= dim_hh2 else cocycles,
        'anomaly_basis': anomaly_basis,
        'phonon_basis': phonon_basis,
    }
