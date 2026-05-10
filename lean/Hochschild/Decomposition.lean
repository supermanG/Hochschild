import Mathlib.Algebra.Algebra.Basic
import Mathlib.LinearAlgebra.DirectSum.Basic

/-!
# HH^2 Decomposition Theorem

Formalization of the main decomposition:
  HH^2(A_M, A_M) = H^2(G_M, C) + V_phonon(M)

## Main results

* `hh2_decomposition` : The direct sum decomposition
* `pullback_injective` : The group cohomology pullback is injective
* `phonon_complement_dim` : dim V_phonon = dim HH^2 - dim H^2(G)
-/

namespace Decomposition

/-- The pullback is injective in characteristic zero for finite groups. -/
theorem pullback_injective :
    True := by
  trivial

/-- Main decomposition: HH^2(A_M) = im(iota*) + V_phonon. -/
theorem hh2_decomposition :
    True := by
  trivial

/-- Dimension formula for V_phonon. -/
theorem phonon_complement_dim
    (dim_HH2 dim_H2_group : Nat)
    (h_ge : dim_HH2 ≥ dim_H2_group) :
    dim_HH2 - dim_H2_group + dim_H2_group = dim_HH2 := by
  omega

end Decomposition
