#!/usr/bin/env python3
"""
integrate_rtsc.py
=================
Integrate Hochschild cohomology as a new qualifier for the RTSC v15
predictor framework.

This script:
1. Loads the v14 candidate pool from the parent RTSC project
2. Computes HH^* for each material
3. Adds HH^1, HH^2, and (where available) HH^2_phonon as features
4. Outputs a feature file compatible with the RTSC pipeline

Key findings from the JARVIS analysis:
- LOW HH^1/HH^2 correlates with HIGH lambda (structural rigidity)
- HH^2_phonon serves as a NEGATIVE screen: very high values indicate
  materials too flexible for conventional SC

Integration as v15 qualifier:
- hh_rigidity = 1 / (1 + dim_HH2): measures structural stiffness
- hh_phonon_flag = 1 if dim_HH2_phonon > threshold else 0
  (flag for exclusion from conventional SC candidates)

LH & Claude 2026
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.quiver import Quiver, QuiverWithRelations
from python.hochschild_bond import (bond_algebra, compute_hochschild,
                                     hh2_decomposition, group_cohomology_h2)


ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
    'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Th': 90, 'U': 92,
}


def load_dft_3d_cache(cache_path: str) -> dict:
    """Load the JARVIS DFT-3D cache used by the RTSC pipeline."""
    with open(cache_path, 'r') as f:
        data = json.load(f)
    return data


def compute_hh_qualifiers(
    materials: list,
    max_atoms: int = 20,
    max_degree: int = 2,
) -> list:
    """
    Compute Hochschild-derived qualifiers for a list of materials.

    Returns list of dicts with keys:
      - jid: material identifier
      - hh_rigidity: 1 / (1 + dim_HH2), higher = more rigid = better for SC
      - hh1_dim: dim HH^1 (derivations)
      - hh2_dim: dim HH^2 (deformations)
      - hh_phonon_flag: 1 if HH^2 > threshold (too flexible warning)
      - computable: True if computation succeeded
    """
    results = []
    n_mat = len(materials)
    t0 = time.time()

    # Determine threshold from distribution (will be set after first pass)
    hh2_values = []

    for i, mat in enumerate(materials):
        jid = mat.get('jid', f'mat_{i}')
        n_atoms = len(mat.get('elements', []))

        if n_atoms > max_atoms or n_atoms == 0:
            results.append({
                'jid': jid,
                'hh_rigidity': None,
                'hh1_dim': None,
                'hh2_dim': None,
                'hh_phonon_flag': None,
                'computable': False,
                'reason': f'n_atoms={n_atoms} exceeds max={max_atoms}',
            })
            continue

        try:
            atoms = mat.get('atoms', mat)
            if 'lattice_mat' in atoms:
                lat = np.array(atoms['lattice_mat'])
                frac_coords = np.array(atoms['coords'])
                cart_coords = frac_coords @ lat
                elements = atoms['elements']
            else:
                cart_coords = np.array(mat['coords'])
                elements = mat.get('elements', [])
                lat = mat.get('lat', np.eye(3))

            Z = np.array([ELEMENT_TO_Z.get(el, 6) for el in elements])

            A = bond_algebra(
                cart_coords, Z, lat,
                tolerance=0.4,
                max_bonds_per_atom=8,
                include_triangles=True,
                include_symmetry=True,
            )

            hh = compute_hochschild(A, max_degree=max_degree)
            hh1 = hh['hh_dims'].get(1, 0)
            hh2 = hh['hh_dims'].get(2, 0)

            rigidity = 1.0 / (1.0 + float(hh2))
            hh2_values.append(hh2)

            results.append({
                'jid': jid,
                'hh_rigidity': rigidity,
                'hh1_dim': int(hh1),
                'hh2_dim': int(hh2),
                'hh_phonon_flag': None,  # set after threshold determination
                'computable': True,
                'n_arrows': hh.get('n_arrows', 0),
                'n_relations': hh.get('n_relations', 0),
            })

        except Exception as e:
            results.append({
                'jid': jid,
                'hh_rigidity': None,
                'hh1_dim': None,
                'hh2_dim': None,
                'hh_phonon_flag': None,
                'computable': False,
                'reason': str(e)[:100],
            })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_done = sum(1 for r in results if r['computable'])
            print(f"  [{i+1}/{n_mat}] {n_done} computed ({elapsed:.1f}s)")

    # Set phonon flag based on 90th percentile threshold
    if hh2_values:
        threshold = np.percentile(hh2_values, 90)
        for r in results:
            if r['computable'] and r['hh2_dim'] is not None:
                r['hh_phonon_flag'] = 1 if r['hh2_dim'] > threshold else 0

    elapsed = time.time() - t0
    n_done = sum(1 for r in results if r['computable'])
    print(f"\nComplete: {n_done}/{n_mat} computed ({elapsed:.1f}s)")
    if hh2_values:
        print(f"  HH^2 threshold (90th %ile): {threshold:.0f}")
        print(f"  Flagged as too-flexible: {sum(1 for r in results if r.get('hh_phonon_flag') == 1)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute HH qualifiers for RTSC candidate pool')
    parser.add_argument('--dft3d_cache', type=str,
                        default='../rtsc/data/jarvis/dft_3d_2021_cache.json')
    parser.add_argument('--supercon_data', type=str,
                        default='../rtsc/data/jarvis/supercon_3d.json')
    parser.add_argument('--output', type=str,
                        default='results/hh_qualifiers.json')
    parser.add_argument('--max_atoms', type=int, default=20)
    parser.add_argument('--n_max', type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 60)
    print("  Hochschild HH Qualifier Computation for RTSC v15")
    print("=" * 60)

    # Load materials
    print(f"\n[1] Loading data from {args.supercon_data}...")
    with open(args.supercon_data, 'r') as f:
        raw = json.load(f)

    materials = raw[:args.n_max]
    print(f"  Loaded {len(materials)} materials")

    # Compute qualifiers
    print(f"\n[2] Computing HH qualifiers (max_atoms={args.max_atoms})...")
    results = compute_hh_qualifiers(materials, max_atoms=args.max_atoms)

    # Save
    print(f"\n[3] Saving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    computed = [r for r in results if r['computable']]
    print(f"\n  Summary:")
    print(f"    Total: {len(results)}")
    print(f"    Computed: {len(computed)}")
    print(f"    Skipped: {len(results) - len(computed)}")
    if computed:
        rigidities = [r['hh_rigidity'] for r in computed]
        print(f"    Rigidity range: [{min(rigidities):.4f}, {max(rigidities):.4f}]")
        print(f"    Mean rigidity: {np.mean(rigidities):.4f}")

    print("\n  Done.")


if __name__ == '__main__':
    main()
