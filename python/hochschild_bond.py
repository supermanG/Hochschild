"""
hochschild_bond.py
==================
Core module: construct bond algebras from crystal structures and
compute their Hochschild cohomology.

The bond algebra of a crystal is the path algebra of the bond quiver
(atoms = vertices, bonds = arrows) modulo structural relations
encoding rigid bond constraints.

Main API:
  - bond_algebra(coords, Z, lat, cutoff) -> QuiverWithRelations
  - compute_hochschild(A, max_degree=3) -> dict of HH^n dimensions
  - hh2_phonon_deformation(A, point_group_mats) -> int (dim of phonon part)

LH & Claude 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations

from .quiver import Quiver, QuiverWithRelations, Relation, Arrow
from .bar_resolution import (compute_hochschild_cohomology,
                             hochschild_cocycles,
                             hochschild_coboundaries)


# ---------------------------------------------------------------------------
# Bond detection and quiver construction
# ---------------------------------------------------------------------------

# Covalent radii (Angstroms) for bond detection
COVALENT_RADII = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.77, 7: 0.71,
    8: 0.66, 9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11,
    15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76,
    21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.61, 26: 1.52,
    27: 1.50, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20,
    33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16, 37: 2.20, 38: 1.95,
    39: 1.90, 40: 1.75, 41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46,
    45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39,
    51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44, 56: 2.15,
    57: 2.07, 72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44,
    77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46,
    83: 1.48, 90: 1.79, 92: 1.56,
}


def detect_bonds(
    coords: np.ndarray,
    Z: np.ndarray,
    lat: Optional[np.ndarray] = None,
    tolerance: float = 0.3,
    max_bonds_per_atom: int = 12,
) -> List[Tuple[int, int, float]]:
    """
    Detect bonds between atoms using sum-of-covalent-radii criterion.

    Parameters
    ----------
    coords : (na, 3) Cartesian coordinates
    Z : (na,) atomic numbers
    lat : (3, 3) lattice matrix (if periodic, use minimum-image convention)
    tolerance : fractional tolerance beyond sum of covalent radii
    max_bonds_per_atom : cap on coordination number

    Returns
    -------
    bonds : list of (i, j, distance) tuples, i < j
    """
    na = len(Z)
    bonds = []

    for i in range(na):
        r_i = COVALENT_RADII.get(int(Z[i]), 1.5)
        dists_from_i = []

        for j in range(i + 1, na):
            r_j = COVALENT_RADII.get(int(Z[j]), 1.5)
            max_dist = (r_i + r_j) * (1.0 + tolerance)

            # Minimum-image distance
            diff = coords[j] - coords[i]
            if lat is not None:
                # Fractional coordinates for minimum image
                lat_inv = np.linalg.inv(lat)
                frac_diff = lat_inv @ diff
                frac_diff -= np.round(frac_diff)
                diff = lat @ frac_diff

            dist = np.linalg.norm(diff)
            if dist < max_dist and dist > 0.4:
                dists_from_i.append((j, dist))

        # Sort by distance and cap
        dists_from_i.sort(key=lambda x: x[1])
        for j, dist in dists_from_i[:max_bonds_per_atom]:
            bonds.append((i, j, dist))

    return bonds


def build_bond_quiver(
    coords: np.ndarray,
    Z: np.ndarray,
    lat: Optional[np.ndarray] = None,
    tolerance: float = 0.3,
    max_bonds_per_atom: int = 12,
) -> Quiver:
    """
    Build the bond quiver from a crystal structure.

    Vertices = atoms, arrows = bonds (both directions for each bond,
    since the quiver is oriented).
    """
    na = len(Z)
    bonds = detect_bonds(coords, Z, lat, tolerance, max_bonds_per_atom)

    Q = Quiver(n_vertices=na)
    for i, j, dist in bonds:
        Q.add_arrow(i, j, label=f"{int(Z[i])}-{int(Z[j])}", weight=dist)
        Q.add_arrow(j, i, label=f"{int(Z[j])}-{int(Z[i])}", weight=dist)

    return Q


# ---------------------------------------------------------------------------
# Relations: structural constraints on the path algebra
# ---------------------------------------------------------------------------

def triangle_relations(quiver: Quiver) -> List[Relation]:
    """
    Generate triangle-commutativity relations.

    For every triple (i, j, k) where bonds i-j, j-k, and i-k all exist,
    impose: (arrow i->j) * (arrow j->k) = (arrow i->k)
    i.e., the path i->j->k minus the direct arrow i->k is zero.

    This encodes the geometric constraint that the crystal is rigid:
    going around a triangle returns to the same relative position.
    """
    na = quiver.n_vertices
    relations = []

    # Build adjacency for fast lookup: (source, target) -> arrow index
    arrow_map = {}
    for idx, arrow in enumerate(quiver.arrows):
        key = (arrow.source, arrow.target)
        if key not in arrow_map:
            arrow_map[key] = []
        arrow_map[key].append(idx)

    for i in range(na):
        for j in range(na):
            if i == j:
                continue
            arr_ij_list = arrow_map.get((i, j), [])
            if not arr_ij_list:
                continue

            for k in range(na):
                if k == i or k == j:
                    continue
                arr_jk_list = arrow_map.get((j, k), [])
                arr_ik_list = arrow_map.get((i, k), [])

                if arr_jk_list and arr_ik_list:
                    # Relation: path [arr_ij, arr_jk] - [arr_ik] = 0
                    arr_ij = arr_ij_list[0]
                    arr_jk = arr_jk_list[0]
                    arr_ik = arr_ik_list[0]
                    rel = Relation(terms=[
                        (1.0, [arr_ij, arr_jk]),
                        (-1.0, [arr_ik]),
                    ])
                    relations.append(rel)

    return relations


def symmetry_relations(
    quiver: Quiver,
    coords: np.ndarray,
    Z: np.ndarray,
    point_group_mats: Optional[np.ndarray] = None,
    tol: float = 0.1,
) -> List[Relation]:
    """
    Generate symmetry-equivalence relations.

    Arrows related by a point-group symmetry operation carry equal
    weights in the algebra. This imposes: arrow_a - arrow_b = 0
    for symmetry-equivalent bonds.

    For now, uses a simple distance-based equivalence: bonds with
    the same (Z_source, Z_target) pair and same length (within tol)
    are equivalent.
    """
    if point_group_mats is None:
        return _equivalence_by_type_and_length(quiver, coords, Z, tol)

    return _equivalence_by_symmetry(quiver, coords, Z, point_group_mats, tol)


def _equivalence_by_type_and_length(
    quiver: Quiver,
    coords: np.ndarray,
    Z: np.ndarray,
    tol: float,
) -> List[Relation]:
    """Group arrows by (element pair, bond length) and impose equality."""
    relations = []
    arrows = quiver.arrows

    # Group arrows by bond type
    groups = {}
    for idx, arrow in enumerate(arrows):
        z_pair = (min(int(Z[arrow.source]), int(Z[arrow.target])),
                  max(int(Z[arrow.source]), int(Z[arrow.target])))
        dist = arrow.weight
        # Round distance to bin equivalent bonds
        dist_bin = round(dist / tol) * tol
        key = (z_pair, dist_bin)
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)

    # Within each group, impose pairwise equality (first - each = 0)
    for key, arrow_indices in groups.items():
        if len(arrow_indices) <= 1:
            continue
        ref = arrow_indices[0]
        for other in arrow_indices[1:]:
            rel = Relation(terms=[
                (1.0, [ref]),
                (-1.0, [other]),
            ])
            relations.append(rel)

    return relations


def _equivalence_by_symmetry(
    quiver: Quiver,
    coords: np.ndarray,
    Z: np.ndarray,
    point_group_mats: np.ndarray,
    tol: float,
) -> List[Relation]:
    """
    Use point-group operations to identify symmetry-equivalent arrows.
    """
    na = len(Z)
    ng = point_group_mats.shape[0]
    arrows = quiver.arrows
    n_arr = len(arrows)

    # For each group element g, find the permutation of atoms
    atom_perms = np.zeros((ng, na), dtype=int)
    centered = coords - coords.mean(axis=0)

    for g in range(ng):
        R = point_group_mats[g]
        rotated = (R @ centered.T).T
        for i in range(na):
            dists = np.linalg.norm(centered - rotated[i], axis=1)
            # Match by closest atom of same element
            candidates = np.where((dists < tol) & (Z == Z[i]))[0]
            if len(candidates) > 0:
                atom_perms[g, i] = candidates[np.argmin(dists[candidates])]
            else:
                atom_perms[g, i] = i

    # For each arrow, find its orbit under the group
    visited = set()
    equivalence_classes = []

    for arr_idx in range(n_arr):
        if arr_idx in visited:
            continue
        orbit = {arr_idx}
        src = arrows[arr_idx].source
        tgt = arrows[arr_idx].target

        for g in range(ng):
            new_src = atom_perms[g, src]
            new_tgt = atom_perms[g, tgt]
            # Find arrow with this source-target pair
            for other_idx in range(n_arr):
                if (arrows[other_idx].source == new_src and
                        arrows[other_idx].target == new_tgt):
                    orbit.add(other_idx)
                    break

        equivalence_classes.append(sorted(orbit))
        visited.update(orbit)

    # Impose equality within each orbit
    relations = []
    for orbit in equivalence_classes:
        if len(orbit) <= 1:
            continue
        ref = orbit[0]
        for other in orbit[1:]:
            rel = Relation(terms=[(1.0, [ref]), (-1.0, [other])])
            relations.append(rel)

    return relations


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def bond_algebra(
    coords: np.ndarray,
    Z: np.ndarray,
    lat: Optional[np.ndarray] = None,
    point_group_mats: Optional[np.ndarray] = None,
    tolerance: float = 0.3,
    max_bonds_per_atom: int = 12,
    include_triangles: bool = True,
    include_symmetry: bool = True,
) -> QuiverWithRelations:
    """
    Construct the bond algebra of a crystal as a quiver with relations.

    Parameters
    ----------
    coords : (na, 3) Cartesian coordinates of atoms in unit cell
    Z : (na,) atomic numbers
    lat : (3, 3) lattice matrix (optional, for periodic structures)
    point_group_mats : (ng, 3, 3) point-group rotation matrices (optional)
    tolerance : bond detection tolerance
    max_bonds_per_atom : coordination cap
    include_triangles : include triangle-commutativity relations
    include_symmetry : include symmetry-equivalence relations

    Returns
    -------
    A : QuiverWithRelations (the bond algebra kQ/I)
    """
    Q = build_bond_quiver(coords, Z, lat, tolerance, max_bonds_per_atom)

    relations = []
    if include_triangles:
        relations.extend(triangle_relations(Q))
    if include_symmetry:
        relations.extend(symmetry_relations(
            Q, coords, Z, point_group_mats))

    return QuiverWithRelations(quiver=Q, relations=relations)


def compute_hochschild(
    A: QuiverWithRelations,
    max_degree: int = 3,
) -> Dict[str, any]:
    """
    Compute Hochschild cohomology of the bond algebra.

    Parameters
    ----------
    A : QuiverWithRelations (the bond algebra)
    max_degree : compute HH^0, ..., HH^{max_degree}

    Returns
    -------
    result : dict with keys:
        'hh_dims': {0: dim_HH0, 1: dim_HH1, ...}
        'n_vertices': int
        'n_arrows': int
        'n_relations': int
        'is_koszul': bool
    """
    hh_dims = compute_hochschild_cohomology(A, max_degree=max_degree)

    return {
        'hh_dims': hh_dims,
        'n_vertices': A.n_vertices,
        'n_arrows': A.n_arrows,
        'n_relations': A.n_relations,
        'is_koszul': A.is_koszul(),
    }


def hh2_decomposition(
    A: QuiverWithRelations,
    coords: np.ndarray,
    Z: np.ndarray,
    point_group_mats: np.ndarray,
) -> Dict[str, int]:
    """
    Decompose HH^2(A) into anomaly part and phonon-deformation part.

    The decomposition theorem (conjectural):
      HH^2(A) = H^2(G, C) + (phonon-deformation modes)

    The anomaly part H^2(G, C) is the group cohomology of the point
    group G with trivial coefficients. The phonon-deformation part is
    the orthogonal complement in HH^2.

    Parameters
    ----------
    A : QuiverWithRelations
    coords : (na, 3)
    Z : (na,)
    point_group_mats : (ng, 3, 3) point-group rotation matrices

    Returns
    -------
    decomp : dict with keys:
        'dim_HH2': total dimension
        'dim_anomaly': H^2(G, C) part
        'dim_phonon': phonon-deformation part
        'h2_group': dim H^2(G, C) computed independently
    """
    # Compute full HH^2
    hh = compute_hochschild_cohomology(A, max_degree=2)
    dim_hh2 = hh.get(2, 0)

    # Compute H^2(G, C) of the point group
    dim_h2_group = group_cohomology_h2(point_group_mats)

    # The phonon part is the remainder
    dim_phonon = max(0, dim_hh2 - dim_h2_group)

    return {
        'dim_HH2': dim_hh2,
        'dim_anomaly': min(dim_h2_group, dim_hh2),
        'dim_phonon': dim_phonon,
        'h2_group': dim_h2_group,
    }


def hh2_phonon_deformation(
    A: QuiverWithRelations,
    coords: np.ndarray,
    Z: np.ndarray,
    point_group_mats: np.ndarray,
) -> int:
    """
    Compute the dimension of the phonon-deformation part of HH^2.

    This is the key novel quantity: materials with high dim(HH^2_phonon)
    admit many infinitesimal deformations connected to phonon coupling.

    Returns dim(HH^2_phonon).
    """
    decomp = hh2_decomposition(A, coords, Z, point_group_mats)
    return decomp['dim_phonon']


# ---------------------------------------------------------------------------
# Group cohomology H^2(G, C) computation
# ---------------------------------------------------------------------------

def group_cohomology_h2(R_mats: np.ndarray) -> int:
    """
    Compute dim H^2(G, C) for a finite group G given by rotation matrices.

    For a finite group G acting trivially on C:
      H^2(G, C) = Schur multiplier M(G) = H^2(G, C^x)

    For crystallographic point groups:
      - Cyclic C_n: H^2 = 0
      - Dihedral D_n (n >= 2): H^2 = Z_2 (dim 1 over C)
      - Tetrahedral T: H^2 = Z_2
      - Octahedral O: H^2 = Z_2
      - Icosahedral I: H^2 = Z_2

    We compute this via the standard bar resolution for finite groups:
    dim H^2(G, C) = dim Z^2 - dim B^2 where
      Z^2 = {f: G x G -> C | f(g,h) + f(gh,k) = f(g,hk) + f(h,k)}
      B^2 = {f: G x G -> C | f(g,h) = phi(g) + phi(h) - phi(gh) for some phi}

    Parameters
    ----------
    R_mats : (ng, 3, 3) rotation matrices defining the group

    Returns
    -------
    dim_h2 : int
    """
    ng = R_mats.shape[0]

    # Build multiplication table
    mult_table = np.zeros((ng, ng), dtype=int)
    for i in range(ng):
        for j in range(ng):
            prod = R_mats[i] @ R_mats[j]
            for k in range(ng):
                if np.max(np.abs(prod - R_mats[k])) < 1e-6:
                    mult_table[i, j] = k
                    break

    # 2-cocycle condition: f(g,h) + f(gh,k) = f(g,hk) + f(h,k)
    # f is a function G x G -> C, represented as a vector of length ng^2
    # Index: f[i*ng + j] = f(g_i, g_j)

    n_vars = ng * ng

    # Build the cocycle constraint matrix
    # For each (g, h, k): f(g,h) + f(gh,k) - f(g,hk) - f(h,k) = 0
    constraints = []
    for i in range(ng):
        for j in range(ng):
            for k in range(ng):
                row = np.zeros(n_vars)
                gh = mult_table[i, j]
                hk = mult_table[j, k]
                row[i * ng + j] += 1       # f(g, h)
                row[gh * ng + k] += 1      # f(gh, k)
                row[i * ng + hk] -= 1      # f(g, hk)
                row[j * ng + k] -= 1       # f(h, k)
                constraints.append(row)

    constraint_matrix = np.array(constraints)

    # Z^2 = nullspace of constraint_matrix
    rank_constraints = np.linalg.matrix_rank(constraint_matrix, tol=1e-10)
    dim_z2 = n_vars - rank_constraints

    # B^2 = coboundaries: f(g,h) = phi(g) + phi(h) - phi(gh)
    # phi: G -> C is a vector of length ng
    # The coboundary map delta: C^1 -> C^2 sends phi to f where
    # f(g,h) = phi(g) + phi(h) - phi(gh)
    coboundary_matrix = np.zeros((n_vars, ng))
    for i in range(ng):
        for j in range(ng):
            gh = mult_table[i, j]
            idx = i * ng + j
            coboundary_matrix[idx, i] += 1     # phi(g)
            coboundary_matrix[idx, j] += 1     # phi(h)
            coboundary_matrix[idx, gh] -= 1    # phi(gh)

    dim_b2 = np.linalg.matrix_rank(coboundary_matrix, tol=1e-10)

    return dim_z2 - dim_b2


# ---------------------------------------------------------------------------
# Batch computation for material pools
# ---------------------------------------------------------------------------

def compute_hh_for_materials(
    materials: List[Dict],
    max_degree: int = 3,
    point_group_mats: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Compute Hochschild cohomology for a list of materials.

    Each material dict must have: 'coords' (na, 3), 'Z' (na,),
    optionally 'lat' (3, 3).

    Returns list of result dicts (one per material).
    """
    import time
    results = []
    n_mat = len(materials)
    t0 = time.time()

    for i, mat in enumerate(materials):
        coords = mat['coords']
        Z = mat['Z']
        lat = mat.get('lat', None)

        try:
            A = bond_algebra(coords, Z, lat, point_group_mats)
            hh = compute_hochschild(A, max_degree=max_degree)

            if point_group_mats is not None:
                decomp = hh2_decomposition(A, coords, Z, point_group_mats)
                hh['hh2_decomposition'] = decomp

            results.append(hh)
        except Exception as e:
            results.append({
                'hh_dims': {n: -1 for n in range(max_degree + 1)},
                'error': str(e),
            })

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  HH^*: {i+1}/{n_mat} ({elapsed:.1f}s)")

    if verbose:
        elapsed = time.time() - t0
        print(f"  HH^* complete: {n_mat} materials ({elapsed:.1f}s)")

    return results
