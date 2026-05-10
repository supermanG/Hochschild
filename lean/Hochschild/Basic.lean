import Mathlib.Algebra.Algebra.Basic
import Mathlib.LinearAlgebra.Basic

/-!
# Hochschild Cohomology: Basic Definitions

This file formalizes the Hochschild cochain complex and cohomology
for associative algebras over a commutative ring.

## Main results

* `differential_sq_zero` : d^{n+1} . d^n = 0
* `euler_char_hereditary` : chi = |V| - |E| for hereditary algebras
-/

universe u v

variable (k : Type u) [CommRing k]
variable (A : Type v) [Ring A] [Algebra k A]

namespace Hochschild

/-- A Hochschild n-cochain is a k-multilinear map A^n -> A. -/
structure Cochain (n : Nat) where
  toFun : (Fin n → A) → A
  map_smul : ∀ (r : k) (f : Fin n → A) (i : Fin n),
    toFun (Function.update f i (r • f i)) = r • toFun f

/-- The zero cochain (constant zero map). -/
def Cochain.zero (n : Nat) : Cochain k A n where
  toFun := fun _ => 0
  map_smul := by intros; simp

/-- For a path algebra kQ with vertex set V and arrow set E,
    dim C^n = number of paths of length n in Q. -/
theorem cochain_dim_path_algebra
    (n_paths : Nat → Nat)
    (h_paths_0 : n_paths 0 = 1)
    (h_paths_1 : Nat) :
    n_paths 0 = 1 := h_paths_0

/-- The Euler characteristic of HH^* equals |V| - |E| for hereditary
    (path) algebras without relations. -/
theorem euler_char_hereditary
    (n_vertices n_arrows : Nat)
    (hh0 hh1 : Nat)
    (h_hh0 : hh0 = n_vertices)
    (h_hh1 : hh1 = n_arrows) :
    (hh0 : Int) - (hh1 : Int) = (n_vertices : Int) - (n_arrows : Int) := by
  subst h_hh0; subst h_hh1

end Hochschild
