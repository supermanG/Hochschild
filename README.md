# Hochschild HH^2 Obstruction to Phonon-Mediated Superconductivity

Compute the Hochschild cohomology HH^*(A) of the bond algebra A for
crystalline materials. HH^2 classifies infinitesimal deformations of
the algebra structure, hence soft-mode/phonon-coupling potential,
providing an obstruction-class qualifier for superconductor candidates.

## Key idea

For a crystal M with bond algebra A_M (path algebra of the bond
quiver modulo structural relations):

- HH^0(A) = center = global symmetry-invariant scalars
- HH^1(A) = outer derivations = first-order soft modes
- HH^2(A) = infinitesimal deformations / obstructions
- HH^3(A) = obstructions to higher-order deformations

Materials with non-trivial HH^2(bond-algebra) admit infinitesimal
deformations connected to phonon coupling and structural soft modes.

The decomposition theorem (conjectural, to be validated):
```
HH^2(A_M) = H^2(G, C) + (phonon-deformation modes)
```
where the first factor is the group-cohomological anomaly (already
captured in the RTSC framework as v6 H^2) and the second factor is
the novel phonon-deformation content.

## Structure

```
python/
  hochschild_bond.py    -- Core module: bond algebra + HH^* computation
  quiver.py             -- Quiver-with-relations data structures
  bar_resolution.py     -- Bar resolution for Hochschild complex
  decomposition.py      -- HH^2 decomposition into anomaly + phonon
  validation.py         -- Cross-validation against phonon stability
  tests/                -- Unit tests
latex/
  paper.tex             -- Publication manuscript
  figures/              -- Generated figures
data/                   -- Material data (JARVIS cache, candidate pools)
results/                -- Computation outputs
```

## Dependencies

- numpy, scipy (numerical linear algebra)
- networkx (quiver / bond-graph operations)
- matplotlib (visualization)
- JARVIS data cache (from parent RTSC project)

## References

- Connes, "Noncommutative Geometry" (1994)
- Cibils, "Cohomology of incidence algebras and simplicial complexes" (1989)
- Loday, "Cyclic Homology" (2nd ed, 1998)
- Gerstenhaber, "The cohomology structure of an associative ring" (1963)

## Relation to RTSC framework

This is task T1.4 in the RTSC overnight plan. The Hochschild HH^2
computation replaces the v13 path-counting proxy (`hh_proxy_dim_HH0`,
`hh_proxy_dim_HH1`, `hh_proxy_paths_2`, `hh_proxy_paths_3`) with
a mathematically rigorous deformation-class computation.

Target venue: J Noncomm Geom, Comm Math Phys, or Adv Math.

## Authors

LH and Claude, 2026.
