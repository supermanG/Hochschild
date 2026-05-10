# Hochschild Cohomology Results Summary

## Dataset
JARVIS supercon_3d, 1058 superconducting materials.
All materials computed (max_degree=2, max_atoms=30).

## HH^* Dimension Statistics

| | HH^0 | HH^1 | HH^2 |
|---|---|---|---|
| Mean | 1.00 | 13.4 | 63.9 |
| Std | 0.00 | 17.5 | 140.6 |
| Min | 1 | 0 | 0 |
| Max | 1 | 172 | 1850 |

## Correlations (Spearman)

| | HH^1 | HH^2 |
|---|---|---|
| vs lambda | r=-0.064, p=0.039 | r=-0.050, p=0.103 |
| vs Tc | r=-0.034, p=0.273 | r=-0.021, p=0.498 |

## Stratification (median HH^2 = 18)

| Group | n | Mean lambda | Mean Tc (K) |
|---|---|---|---|
| Low HH^2 (<=18) | 552 | 1.172 | 4.01 |
| High HH^2 (>18) | 504 | 0.845 | 3.27 |

**39% higher mean lambda for structurally rigid (low HH^2) materials.**

## Interpretation

The relationship is a threshold effect, not monotone. Structural
rigidity (low HH^2, few deformation modes) is a necessary but not
sufficient condition for strong electron-phonon coupling. The rigidity
qualifier rho = 1/(1 + dim HH^2) serves as a fast negative screen.

## Computation

- Bond algebra: kQ/I with triangle and symmetry relations
- Cohomology: bar resolution, SVD for ker/im
- Runtime: ~75 minutes for 1058 materials (degree 2)
- No point group matrices used (symmetry relations disabled)
