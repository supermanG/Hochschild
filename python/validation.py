"""
validation.py
=============
Cross-validation of Hochschild cohomology results against known
physical observables:

1. HH^2_phonon vs phonon minimum frequency (JARVIS dft_3d.modes)
2. HH^2_anomaly vs v6 H^2(G, U(1)) Schur multiplier
3. HH^1 (derivations) vs number of soft optical modes

Expected correlations:
- High dim(HH^2_phonon) <-> low phonon min frequency (near instability)
- dim(HH^2_anomaly) == dim(H^2(G, U(1))) from v6 (consistency check)
- HH^1 correlates with number of IR-active modes

LH & Claude 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def validate_anomaly_consistency(
    hh_results: List[Dict],
    h2_v6_values: np.ndarray,
) -> Dict[str, any]:
    """
    Check that dim(HH^2_anomaly) matches the v6 H^2(G, U(1)) values.

    Parameters
    ----------
    hh_results : list of dicts from compute_hh_for_materials
    h2_v6_values : (n_mat,) array of v6 H^2 dimensions

    Returns
    -------
    result : dict with agreement statistics
    """
    n_mat = len(hh_results)
    n_agree = 0
    n_disagree = 0
    disagreements = []

    for i in range(n_mat):
        hh = hh_results[i]
        if 'hh2_decomposition' not in hh:
            continue

        anomaly_dim = hh['hh2_decomposition'].get('dim_anomaly', -1)
        v6_dim = int(h2_v6_values[i])

        if anomaly_dim == v6_dim:
            n_agree += 1
        else:
            n_disagree += 1
            disagreements.append({
                'index': i,
                'hh2_anomaly': anomaly_dim,
                'v6_h2': v6_dim,
            })

    total = n_agree + n_disagree
    agreement_rate = n_agree / total if total > 0 else 0.0

    return {
        'n_materials': total,
        'n_agree': n_agree,
        'n_disagree': n_disagree,
        'agreement_rate': agreement_rate,
        'disagreements': disagreements[:20],
    }


def validate_phonon_correlation(
    hh_results: List[Dict],
    phonon_min_freq: np.ndarray,
    phonon_stable: np.ndarray,
) -> Dict[str, any]:
    """
    Test correlation between dim(HH^2_phonon) and phonon instability.

    Hypothesis: materials with non-trivial HH^2_phonon are closer to
    phonon instability (lower minimum phonon frequency).

    Parameters
    ----------
    hh_results : list of dicts
    phonon_min_freq : (n_mat,) minimum phonon frequency (cm^-1)
    phonon_stable : (n_mat,) boolean, True if dynamically stable

    Returns
    -------
    result : dict with correlation statistics
    """
    hh2_phonon = []
    min_freq = []
    stable = []

    for i, hh in enumerate(hh_results):
        if 'hh2_decomposition' not in hh:
            continue
        if np.isnan(phonon_min_freq[i]):
            continue

        dim_ph = hh['hh2_decomposition'].get('dim_phonon', 0)
        hh2_phonon.append(dim_ph)
        min_freq.append(phonon_min_freq[i])
        stable.append(bool(phonon_stable[i]))

    hh2_phonon = np.array(hh2_phonon)
    min_freq = np.array(min_freq)
    stable = np.array(stable)

    n = len(hh2_phonon)
    if n < 10:
        return {'n_materials': n, 'insufficient_data': True}

    # Pearson correlation: HH^2_phonon vs min_freq
    r_pearson, p_pearson = stats.pearsonr(hh2_phonon, min_freq)

    # Spearman rank correlation (more robust)
    r_spearman, p_spearman = stats.spearmanr(hh2_phonon, min_freq)

    # Point-biserial: HH^2_phonon > 0 vs phonon stability
    has_phonon_modes = hh2_phonon > 0
    if np.sum(has_phonon_modes) > 0 and np.sum(~has_phonon_modes) > 0:
        # Mann-Whitney U test: min_freq for HH2_phonon > 0 vs == 0
        freq_with = min_freq[has_phonon_modes]
        freq_without = min_freq[~has_phonon_modes]
        u_stat, p_mannwhitney = stats.mannwhitneyu(
            freq_with, freq_without, alternative='less')
    else:
        u_stat, p_mannwhitney = 0.0, 1.0

    # Instability enrichment: fraction of unstable materials in
    # HH^2_phonon > 0 vs HH^2_phonon == 0
    if np.sum(has_phonon_modes) > 0:
        frac_unstable_with = np.mean(~stable[has_phonon_modes])
    else:
        frac_unstable_with = 0.0

    if np.sum(~has_phonon_modes) > 0:
        frac_unstable_without = np.mean(~stable[~has_phonon_modes])
    else:
        frac_unstable_without = 0.0

    return {
        'n_materials': n,
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman),
        'mannwhitney_u': float(u_stat),
        'mannwhitney_p': float(p_mannwhitney),
        'frac_unstable_with_phonon_modes': float(frac_unstable_with),
        'frac_unstable_without_phonon_modes': float(frac_unstable_without),
        'enrichment_ratio': float(
            frac_unstable_with / max(frac_unstable_without, 1e-10)),
        'mean_hh2_phonon': float(np.mean(hh2_phonon)),
        'mean_min_freq_with': float(np.mean(min_freq[has_phonon_modes]))
            if np.sum(has_phonon_modes) > 0 else None,
        'mean_min_freq_without': float(np.mean(min_freq[~has_phonon_modes]))
            if np.sum(~has_phonon_modes) > 0 else None,
    }


def validate_hh1_vs_soft_modes(
    hh_results: List[Dict],
    n_ir_active: np.ndarray,
) -> Dict[str, any]:
    """
    Test correlation between dim(HH^1) and number of IR-active modes.

    HH^1(A) = outer derivations of the bond algebra = first-order
    soft-mode directions. Should correlate with the number of
    experimentally observable IR-active phonon modes.

    Parameters
    ----------
    hh_results : list of dicts
    n_ir_active : (n_mat,) number of IR-active modes per material

    Returns
    -------
    result : dict with correlation statistics
    """
    hh1_dims = []
    ir_modes = []

    for i, hh in enumerate(hh_results):
        if 'hh_dims' not in hh:
            continue
        dim_hh1 = hh['hh_dims'].get(1, -1)
        if dim_hh1 < 0 or np.isnan(n_ir_active[i]):
            continue

        hh1_dims.append(dim_hh1)
        ir_modes.append(n_ir_active[i])

    hh1_dims = np.array(hh1_dims)
    ir_modes = np.array(ir_modes)
    n = len(hh1_dims)

    if n < 10:
        return {'n_materials': n, 'insufficient_data': True}

    r_spearman, p_spearman = stats.spearmanr(hh1_dims, ir_modes)
    r_pearson, p_pearson = stats.pearsonr(hh1_dims, ir_modes)

    return {
        'n_materials': n,
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman),
        'mean_hh1': float(np.mean(hh1_dims)),
        'mean_ir_modes': float(np.mean(ir_modes)),
    }


def full_validation_report(
    hh_results: List[Dict],
    h2_v6_values: Optional[np.ndarray] = None,
    phonon_min_freq: Optional[np.ndarray] = None,
    phonon_stable: Optional[np.ndarray] = None,
    n_ir_active: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    Run all validation checks and compile a report.
    """
    report = {}

    if h2_v6_values is not None:
        report['anomaly_consistency'] = validate_anomaly_consistency(
            hh_results, h2_v6_values)

    if phonon_min_freq is not None and phonon_stable is not None:
        report['phonon_correlation'] = validate_phonon_correlation(
            hh_results, phonon_min_freq, phonon_stable)

    if n_ir_active is not None:
        report['hh1_soft_modes'] = validate_hh1_vs_soft_modes(
            hh_results, n_ir_active)

    return report
