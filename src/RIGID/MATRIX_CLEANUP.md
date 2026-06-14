# Matrix Math Cleanup Notes

## Completed Renames

### Types

| Old | New | Notes |
|-----|-----|-------|
| `DChol2` | `LDLT2` | Fields: `d0, d1, l10` (was `m01`) |
| `DChol3` | `LDLT3` | Fields: `d0, d1, d2, l10, l20, l21` (was `m01, m02, m12`) |

### Functions

| Old | New | Meaning |
|-----|-----|---------|
| `inv_dchol` | `inv_ldlt` | Compute inverse LDL^T decomposition |
| `dchol_pivot3` | `ldlt_pivot3` | Pivoted LDL^T (3×3) |
| `dchol_pivot_one` | `ldlt_pivot_one` | Semi-pivoted LDL^T (3×3) |
| `dchol_pivot2` | `ldlt_pivot2` | Pivoted LDL^T (2×2) — unused, prune |
| `lslt_mul(S, L)` | `inv_lt_sandwich(S, L)` | S ← L⁻¹ S L⁻ᵀ (mass-unweighting) |
| `mul_ltdl(M, L)` | `lt_sandwich_fwd(M, L)` | M ← Lᵀ D L M (mass-weighting, right-sandwich) |
| `ltdl_mul(M, L)` | `lt_sandwich_left(M, L)` | M ← Lᵀ D L M (mass-weighting, left-sandwich) |
| `inv_ltdl_mul(L, M)` | `inv_lt_solve(L, M)` | M ← L⁻ᵀ D⁻¹ L⁻¹ M (solve LDL^T x = b) |
| `mul_inv_ltdl(M, L)` | `mul_inv_lt(M, L)` | M ← M L⁻¹ D⁻¹ L⁻ᵀ |
| `mat_dot(R, S)` | `cross_gram(R, S)` | R Sᵀ (cross Gram matrix) |

### Solver variables (2D: `shake3angle_solve`)

| Old | New | Math |
|-----|-----|------|
| `Lm` | `gram_target` | Lₘ = target Gram matrix |
| `lmc` | `W_inv_ldlt` | W⁻¹ in LDL^T form (inverse mass metric) |
| `rr` | `metric_C` | G = R Rᵀ (constraint metric) |
| `rc` | `inv_chol_C` | U⁻¹ where UᵀU = G (inverse Cholesky of constraint metric) |
| `ss` | `gram_S` | S Sᵀ (Gram of corrected positions) |
| `rc_chi` | `U_invT_gram_cross` | U⁻ᵀ R Sᵀ (intermediate, used for projector term) |
| `diff` | `gram_defect` | Lₘ − S Sᵀ |
| `sigma` | `sigma` | M(Lₘ − S⊥ᴿᵀS⊥ᴿ)M (mass-weighted Lyapunov RHS) |
| `sc` | `L_sigma_Winv` | L_σ · D_W⁻¹ · L_W⁻¹ (Cholesky of sigma, mass-scaled) |
| `chi` (after `u_mul`) | `chi` | G⁻¹ Rᵀ S M (pseudoinverse projection, before mass-weighting) |

### Solver variables (3D: `solve3x3`)

| Old | New | Math |
|-----|-----|------|
| `L_mat` | `gram_target` | L (target Gram; doubles as sigma when R has full rank) |
| `lm_chol` | `W_inv_ldlt` | W⁻¹ in LDL^T form |
| `rr` | `metric_C` | G = R Rᵀ |
| `rc` | `inv_chol_C` | U⁻¹ where UᵀU = G |
| `rs` | `gram_cross` | R Sᵀ (raw cross-Gram before orthonormalization) |
| `sc` | `L_sigma_Winv` | Cholesky of sigma, mass-scaled |

---

## Algorithm Derivation

### Setup

R, S are n×3 matrices (n = 2 or 3) whose rows are constraint vectors.
C = S − R Λ W is the corrected displacement. Constraint: CᵀC = Lₘ (target Gram).

### Full expansion

Lₘ − SᵀS = −SᵀR Λ W − W Λ RᵀS + W Λ RᵀR Λ W

Mass-weight by M = W⁻¹ (pre- and post-multiply):

M(Lₘ − SᵀS)M = −M SᵀR Λ − Λ RᵀS M + Λ RᵀR Λ

Define χ = G⁻¹ Rᵀ S M (where G = R Rᵀ). Split Λ = G⁻¹ Rᵀ S M + Φ = χ + Φ.
All linear and constant terms cancel, leaving:

Σ = Φᵀ G Φ

where Σ = M(Lₘ − SᵀS)M + χᵀ G⁻¹ χ = M(Lₘ − S⊥ᴿᵀ S⊥ᴿ)M

### Factorization

G = UᵀU, Σ = L_σ L_σᵀ, so Φ = U⁻¹ L_σ Q for Q ∈ SO(n).

The symmetry condition on Λ = χ + Φ determines Q via skew decomposition
(closed-form in 2D, Cayley iteration in 3D).

### Key insight: why L − S⊥ᴿᵀS⊥ᴿ is formed first

In the 2D solver, S⊥ᴿ is computed geometrically: n = r01 × r02 is the normal
to the constraint plane, and p_i = s_i · n. Since S⊥ᴿ has rows p_i n̂, the Gram
is S⊥ᴿᵀS⊥ᴿ = SymMat2{p₁²/|n|², p₁p₂/|n|², p₂²/|n|²}.

Then `sigma = gram_target - gram_Sperp` directly gives Lₘ − S⊥ᴿᵀS⊥ᴿ, and only
then is mass-weighting applied via `inv_lt_sandwich`. This is numerically stable
because the projection cancels the in-subspace component of S *before* introducing
the ill-conditioned mass factors.

In the 3D solver (full rank), S⊥ᴿ = 0, so `gram_target` is already the correct
sigma. The only difference between the two paths is whether we subtract the
perpendicular Gram first.

### Two metrics, same pattern

The algorithm orthonormalizes with respect to two metrics:
1. **Constraint metric** G = R Rᵀ: `inv_chol_C` orthonormalizes the constraint subspace
2. **Mass metric** W: `W_inv_ldlt` orthonormalizes the mass-weighted frame

Both use the same LDL^T / Cholesky machinery. The sandwich operations:
- `inv_lt_sandwich(S, W_inv_ldlt)` = L⁻¹ S L⁻ᵀ (inverse mass-weight, applied to sigma)
- `lt_sandwich_fwd(M, W_inv_ldlt)` = Lᵀ D L M (forward mass-weight, applied to chi)

---

## Pruned

- `chol_frame2`, `chol_frame3`
- `ldlt_pivot2` (was `dchol_pivot2`)
- `mul_inv_lt` (both 2×2 and 3×3, was `mul_inv_ltdl`)
- `inv_lt_solve` (2×2 only, was `inv_ltdl_mul` — 3×3 version still used)
- `LTMat2 * LTMat2`
- `SymMat3::operator+/-`
- `dot3` (in mat3.h)
- `lt_sandwich_left` (2×2, was `ltdl_mul`)
- `mtm` (2×2 version in mat2.h)
- `U_invT_gram_cross` variable and `mtm` call in 2D solver (replaced by direct S⊥ᴿ Gram)