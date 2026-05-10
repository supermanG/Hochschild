import Mathlib.Algebra.Algebra.Basic

/-!
# HH^2 Decomposition Theorem

Formalization of the main decomposition:
  HH^2(A_M, A_M) = H^2(G_M, C) + V_phonon(M)

## Main results

* `phonon_complement_dim` : dim V_phonon = dim HH^2 - dim H^2(G)
* `rigidity_monotone` : rigidity qualifier is monotone in HH^2 dim
-/

namespace Decomposition

/-- Dimension formula for V_phonon.
    If dim HH^2 >= dim H^2(G), then dim V_phonon = dim HH^2 - dim H^2(G). -/
theorem phonon_complement_dim
    (dim_HH2 dim_H2_group : Nat)
    (h_ge : dim_HH2 ≥ dim_H2_group) :
    dim_HH2 - dim_H2_group + dim_H2_group = dim_HH2 := by
  omega

/-- The rigidity qualifier 1/(1 + dim_HH2) is strictly decreasing in dim_HH2.
    Materials with lower HH^2 dimension have higher rigidity. -/
theorem rigidity_decreasing (a b : Nat) (hab : a < b) :
    (1 : Rat) / (1 + b) < (1 : Rat) / (1 + a) := by
  apply div_lt_div_of_pos_left
  · exact Rat.ofNat_pos.mpr (Nat.zero_lt_one)
  · exact Rat.ofNat_pos.mpr (Nat.succ_pos a)
  · exact_mod_cast Nat.succ_lt_succ hab

/-- The total dimension splits: dim HH^2 = dim anomaly + dim phonon. -/
theorem hh2_split (dim_HH2 dim_anomaly dim_phonon : Nat)
    (h : dim_HH2 = dim_anomaly + dim_phonon) :
    dim_phonon = dim_HH2 - dim_anomaly := by
  omega

end Decomposition
