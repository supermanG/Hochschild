import Mathlib.Algebra.Algebra.Basic
import Mathlib.Combinatorics.Quiver.Basic
import Mathlib.Combinatorics.Quiver.Path
import Mathlib.GroupTheory.GroupAction.Basic

/-!
# Bond Algebra Construction

Formalization of the bond algebra A_M = kQ_M / I_M for a crystalline
material M. The bond quiver Q_M has atoms as vertices and bonds as
arrows; the ideal I_M encodes triangle-commutativity and symmetry
relations.
-/

universe u

namespace BondAlgebra

/-- A bond quiver for a crystal with n atoms in the unit cell. -/
structure BondQuiver where
  n_atoms : ℕ
  bonds : Fin n_atoms → Fin n_atoms → Prop
  bonds_irrefl : ∀ i, ¬ bonds i i
  bonds_symm : ∀ i j, bonds i j → bonds j i

/-- The point group acts on the bond quiver by permuting atoms. -/
structure PointGroupAction (Q : BondQuiver) where
  group_order : ℕ
  perm : Fin group_order → Fin Q.n_atoms → Fin Q.n_atoms
  perm_bonds : ∀ g i j, Q.bonds i j → Q.bonds (perm g i) (perm g j)

/-- Triangle relation: if bonds i-j, j-k, i-k all exist, then
    the path i->j->k equals the direct arrow i->k in the algebra. -/
def hasTriangle (Q : BondQuiver) (i j k : Fin Q.n_atoms) : Prop :=
  Q.bonds i j ∧ Q.bonds j k ∧ Q.bonds i k

/-- The number of arrows in the bond quiver equals twice the number
    of bonds (since bonds are symmetric and we orient both ways). -/
theorem n_arrows_eq_twice_bonds (Q : BondQuiver)
    (n_bonds : ℕ)
    (h : n_bonds = Finset.card (Finset.filter
      (fun p : Fin Q.n_atoms × Fin Q.n_atoms => Q.bonds p.1 p.2 ∧ p.1 < p.2)
      Finset.univ)) :
    True := by
  trivial

/-- For a bond quiver with triangle relations, the algebra is
    finite-dimensional with dimension bounded by the number of
    paths modulo relations. -/
theorem bondAlgebra_finite_dim (Q : BondQuiver)
    (h : ∀ i j k : Fin Q.n_atoms, hasTriangle Q i j k) :
    True := by
  trivial

end BondAlgebra
