#!/usr/bin/env python3
"""
run_on_jarvis.py
================
Run Hochschild cohomology computation on the JARVIS supercon_3d dataset
(1058 superconducting materials with known Tc, lambda, phonon data).

This script:
1. Loads JARVIS supercon_3d materials
2. Constructs bond algebras for each
3. Computes HH^0, HH^1, HH^2, HH^3
4. Decomposes HH^2 into anomaly + phonon-deformation parts
5. Cross-validates against known Tc and lambda
6. Outputs results to results/ directory

Usage:
  python -m python.run_on_jarvis --data_path ../rtsc/data/jarvis/supercon_3d.json
  python -m python.run_on_jarvis --data_path ../rtsc/data/jarvis/supercon_3d.json --n_max 100 --quick

LH & Claude 2026
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from scipy import stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.quiver import Quiver, QuiverWithRelations
from python.hochschild_bond import (bond_algebra, compute_hochschild,
                                     hh2_decomposition, group_cohomology_h2)
from python.bar_resolution import compute_hochschild_cohomology


# Element symbol to Z mapping
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


def load_jarvis_supercon(data_path: str, n_max: int = 10000) -> List[Dict]:
    """Load JARVIS supercon_3d dataset."""
    with open(data_path, 'r') as f:
        raw = json.load(f)

    materials = []
    for entry in raw[:n_max]:
        atoms = entry['atoms']
        lat = np.array(atoms['lattice_mat'])
        frac_coords = np.array(atoms['coords'])
        elements = atoms['elements']

        # Convert fractional to Cartesian
        cart_coords = frac_coords @ lat

        Z = np.array([ELEMENT_TO_Z.get(el, 6) for el in elements])

        materials.append({
            'coords': cart_coords,
            'Z': Z,
            'lat': lat,
            'elements': elements,
            'jid': entry.get('jid', ''),
            'Tc': entry.get('Tc', np.nan),
            'lamb': entry.get('lamb', np.nan),
            'stability': entry.get('stability', ''),
            'n_atoms': len(Z),
        })

    return materials


def run_hochschild_analysis(
    materials: List[Dict],
    max_degree: int = 3,
    max_atoms: int = 30,
    verbose: bool = True,
) -> List[Dict]:
    """
    Compute HH^* for each material.

    Skips materials with > max_atoms (computation becomes expensive).
    """
    results = []
    n_mat = len(materials)
    t0 = time.time()
    n_skipped = 0

    for i, mat in enumerate(materials):
        if mat['n_atoms'] > max_atoms:
            results.append({
                'jid': mat['jid'],
                'skipped': True,
                'reason': f"n_atoms={mat['n_atoms']} > {max_atoms}",
            })
            n_skipped += 1
            continue

        try:
            A = bond_algebra(
                mat['coords'], mat['Z'], mat['lat'],
                tolerance=0.4,
                max_bonds_per_atom=8,
                include_triangles=True,
                include_symmetry=True,
            )

            hh = compute_hochschild(A, max_degree=max_degree)
            hh['jid'] = mat['jid']
            hh['Tc'] = mat['Tc']
            hh['lamb'] = mat['lamb']
            hh['n_atoms'] = mat['n_atoms']
            hh['skipped'] = False
            results.append(hh)

        except Exception as e:
            results.append({
                'jid': mat['jid'],
                'skipped': True,
                'reason': f"error: {str(e)[:100]}",
            })
            n_skipped += 1

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_done = i + 1 - n_skipped
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{n_mat}] {n_done} computed, {n_skipped} skipped "
                  f"({elapsed:.1f}s, {rate:.1f} mat/s)")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nComplete: {n_mat - n_skipped} computed, "
              f"{n_skipped} skipped ({elapsed:.1f}s)")

    return results


def analyze_results(results: List[Dict], verbose: bool = True) -> Dict:
    """Analyze correlation between HH^* dimensions and Tc/lambda."""

    # Extract computed results
    computed = [r for r in results if not r.get('skipped', True)]
    n_computed = len(computed)

    if n_computed < 10:
        return {'n_computed': n_computed, 'insufficient_data': True}

    # Collect arrays
    hh0 = np.array([r['hh_dims'][0] for r in computed])
    hh1 = np.array([r['hh_dims'][1] for r in computed])
    hh2 = np.array([r['hh_dims'][2] for r in computed])
    hh3 = np.array([r['hh_dims'].get(3, 0) for r in computed])
    tc = np.array([r['Tc'] for r in computed])
    lamb = np.array([r['lamb'] for r in computed])
    n_atoms = np.array([r['n_atoms'] for r in computed])
    n_arrows = np.array([r.get('n_arrows', 0) for r in computed])

    # Filter valid (non-nan) Tc and lambda
    valid_tc = ~np.isnan(tc) & (tc > 0)
    valid_lamb = ~np.isnan(lamb) & (lamb > 0)

    analysis = {
        'n_computed': n_computed,
        'n_valid_tc': int(np.sum(valid_tc)),
        'n_valid_lamb': int(np.sum(valid_lamb)),
        'hh_statistics': {
            'HH0': {'mean': float(np.mean(hh0)), 'std': float(np.std(hh0)),
                    'min': int(np.min(hh0)), 'max': int(np.max(hh0))},
            'HH1': {'mean': float(np.mean(hh1)), 'std': float(np.std(hh1)),
                    'min': int(np.min(hh1)), 'max': int(np.max(hh1))},
            'HH2': {'mean': float(np.mean(hh2)), 'std': float(np.std(hh2)),
                    'min': int(np.min(hh2)), 'max': int(np.max(hh2))},
            'HH3': {'mean': float(np.mean(hh3)), 'std': float(np.std(hh3)),
                    'min': int(np.min(hh3)), 'max': int(np.max(hh3))},
        },
    }

    # Correlations with Tc
    if np.sum(valid_tc) >= 10:
        tc_valid = tc[valid_tc]
        for name, arr in [('HH0', hh0), ('HH1', hh1), ('HH2', hh2), ('HH3', hh3)]:
            arr_valid = arr[valid_tc]
            if np.std(arr_valid) > 0:
                r, p = stats.spearmanr(arr_valid, tc_valid)
                analysis[f'{name}_vs_Tc'] = {'spearman_r': float(r), 'p_value': float(p)}
            else:
                analysis[f'{name}_vs_Tc'] = {'spearman_r': 0.0, 'p_value': 1.0}

    # Correlations with lambda
    if np.sum(valid_lamb) >= 10:
        lamb_valid = lamb[valid_lamb]
        for name, arr in [('HH0', hh0), ('HH1', hh1), ('HH2', hh2), ('HH3', hh3)]:
            arr_valid = arr[valid_lamb]
            if np.std(arr_valid) > 0:
                r, p = stats.spearmanr(arr_valid, lamb_valid)
                analysis[f'{name}_vs_lambda'] = {'spearman_r': float(r), 'p_value': float(p)}
            else:
                analysis[f'{name}_vs_lambda'] = {'spearman_r': 0.0, 'p_value': 1.0}

    # HH^2 stratification: split by high/low HH^2
    if np.sum(valid_tc) >= 20:
        tc_v = tc[valid_tc]
        hh2_v = hh2[valid_tc]
        median_hh2 = np.median(hh2_v)
        high_hh2 = hh2_v > median_hh2
        low_hh2 = hh2_v <= median_hh2

        if np.sum(high_hh2) >= 5 and np.sum(low_hh2) >= 5:
            tc_high = tc_v[high_hh2]
            tc_low = tc_v[low_hh2]
            u_stat, p_mw = stats.mannwhitneyu(tc_high, tc_low, alternative='greater')
            analysis['hh2_stratification'] = {
                'median_hh2': float(median_hh2),
                'n_high': int(np.sum(high_hh2)),
                'n_low': int(np.sum(low_hh2)),
                'mean_Tc_high_hh2': float(np.mean(tc_high)),
                'mean_Tc_low_hh2': float(np.mean(tc_low)),
                'mannwhitney_p': float(p_mw),
            }

    if verbose:
        print("\n" + "=" * 60)
        print("  HOCHSCHILD COHOMOLOGY ANALYSIS RESULTS")
        print("=" * 60)
        print(f"  Materials computed: {n_computed}")
        print(f"  Valid Tc: {analysis['n_valid_tc']}")
        print(f"  Valid lambda: {analysis['n_valid_lamb']}")
        print()
        print("  HH^* statistics:")
        for name in ['HH0', 'HH1', 'HH2', 'HH3']:
            s = analysis['hh_statistics'][name]
            print(f"    {name}: mean={s['mean']:.2f}, std={s['std']:.2f}, "
                  f"range=[{s['min']}, {s['max']}]")
        print()
        print("  Correlations with Tc (Spearman):")
        for name in ['HH0', 'HH1', 'HH2', 'HH3']:
            key = f'{name}_vs_Tc'
            if key in analysis:
                r = analysis[key]['spearman_r']
                p = analysis[key]['p_value']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {name} vs Tc: r={r:.4f}, p={p:.4f} {sig}")
        print()
        print("  Correlations with lambda (Spearman):")
        for name in ['HH0', 'HH1', 'HH2', 'HH3']:
            key = f'{name}_vs_lambda'
            if key in analysis:
                r = analysis[key]['spearman_r']
                p = analysis[key]['p_value']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {name} vs lambda: r={r:.4f}, p={p:.4f} {sig}")

        if 'hh2_stratification' in analysis:
            s = analysis['hh2_stratification']
            print(f"\n  HH^2 stratification (median split at {s['median_hh2']:.0f}):")
            print(f"    High HH^2 (n={s['n_high']}): mean Tc = {s['mean_Tc_high_hh2']:.2f} K")
            print(f"    Low HH^2 (n={s['n_low']}): mean Tc = {s['mean_Tc_low_hh2']:.2f} K")
            print(f"    Mann-Whitney p = {s['mannwhitney_p']:.4f}")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description='Run Hochschild cohomology on JARVIS superconductors')
    parser.add_argument('--data_path', type=str,
                        default='../rtsc/data/jarvis/supercon_3d.json')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--n_max', type=int, default=10000)
    parser.add_argument('--max_atoms', type=int, default=30)
    parser.add_argument('--max_degree', type=int, default=3)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.n_max = min(args.n_max, 100)
        args.max_atoms = 20

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("  Hochschild HH^* Computation on JARVIS Superconductors")
    print("=" * 60)
    print(f"  Data: {args.data_path}")
    print(f"  Max materials: {args.n_max}")
    print(f"  Max atoms per material: {args.max_atoms}")
    print(f"  Max cohomology degree: {args.max_degree}")
    print()

    # Load data
    print("[1] Loading JARVIS supercon_3d...")
    materials = load_jarvis_supercon(args.data_path, args.n_max)
    print(f"  Loaded {len(materials)} materials")
    print(f"  Tc range: [{min(m['Tc'] for m in materials):.2f}, "
          f"{max(m['Tc'] for m in materials):.2f}] K")
    print(f"  Atom count range: [{min(m['n_atoms'] for m in materials)}, "
          f"{max(m['n_atoms'] for m in materials)}]")

    # Compute HH^*
    print(f"\n[2] Computing Hochschild cohomology (max_degree={args.max_degree})...")
    results = run_hochschild_analysis(
        materials, max_degree=args.max_degree,
        max_atoms=args.max_atoms)

    # Analyze
    print("\n[3] Analyzing correlations...")
    analysis = analyze_results(results)

    # Save results
    print(f"\n[4] Saving to {args.results_dir}/...")

    # Save raw results (dimensions only, not full cocycle matrices)
    save_results = []
    for r in results:
        save_r = {k: v for k, v in r.items()
                  if k not in ('cocycle_basis', 'anomaly_basis', 'phonon_basis')}
        # Convert numpy types
        if 'hh_dims' in save_r:
            save_r['hh_dims'] = {str(k): int(v) for k, v in save_r['hh_dims'].items()}
        save_results.append(save_r)

    with open(os.path.join(args.results_dir, 'hh_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    with open(os.path.join(args.results_dir, 'hh_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print("\n  Done.")
    return analysis


if __name__ == '__main__':
    main()
