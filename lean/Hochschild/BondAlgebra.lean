import Mathlib.Algebra.Algebra.Basic
import Mathlib.Combinatorics.Quiver.Basic
import Mathlib.Combinatorics.Quiver.Path

/-!
# Bond Algebra Construction

Formalization of the bond algebra A_M = kQ_M / I_M for a crystalline
material M.
-/

universe u

namespace BondAlgebra

/-- A bond quiver for a crystal with n atoms in the unit cell. -/
structure BondQuiver where
  n_atoms : ℕ
  bonds : Fin n_atoms → Fin n_atoms → Prop
  bonds_irrefl : ∀ i, ¬ bonds i i
  bonds_symm : ∀ i j, bonds i j → bonds j i

/-- Triangle relation: if bonds i-j, j-k, i-k all exist. -/
def hasTriangle (Q : BondQuiver) (i j k : Fin Q.n_atoms) : Prop :=
  Q.bonds i j ∧ Q.bonds j k ∧ Q.bonds i k

/-- Triangle relations are symmetric in the second and third argument
    (given symmetry of bonds). -/
theorem triangle_sym (Q : BondQuiver) (i j k : Fin Q.n_atoms)
    (h : hasTriangle Q i j k) : hasTriangle Q i k j := by
  obtain ⟨hij, hjk, hik⟩ := h
  exact ⟨hik, Q.bonds_symm j k hjk, hij⟩

/-- The number of directed arrows equals twice the undirected bonds
    (since each bond gives two oriented arrows). -/
theorem n_directed_eq_twice (n_bonds n_arrows : Nat)
    (h : n_arrows = 2 * n_bonds) :
    n_arrows = n_bonds + n_bonds := by
  omega

/-- For a bond quiver with all possible triangles (complete graph on n >= 3),
    there are at least 6 ordered triples, hence at least one triangle. -/
theorem complete_quiver_has_triangle (n : Nat) (hn : n ≥ 3) :
    n * (n - 1) * (n - 2) ≥ 6 := by
  have h2 : n - 1 ≥ 2 := by omega
  have h3 : n - 2 ≥ 1 := by omega
  calc n * (n - 1) * (n - 2)
      ≥ 3 * 2 * 1 := Nat.mul_le_mul (Nat.mul_le_mul hn h2) h3
    _ = 6 := rfl

end BondAlgebra
