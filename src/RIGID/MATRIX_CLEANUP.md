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
| `lslt_mul(S, L)` | `lt_sandwich(S, L)` | S ← L̃ S L̃ᵀ (congruence by unit lower-triangular; with inv_ldlt output, computes L⁻¹ S L⁻ᵀ) |
| `mul_ltdl(M, L)` | `rmul_ldlt(M, L)` | M ← Lᵀ D L M (mass-weighting, right-multiply by LDL^T) |
| `ltdl_mul(M, L)` | `lt_sandwich_left(M, L)` | M ← Lᵀ D L M (mass-weighting, left-sandwich) |
| `inv_ltdl_mul(L, M)` | `inv_lt_solve(L, M)` | M ← L⁻ᵀ D⁻¹ L⁻¹ M (solve LDL^T x = b) |
| `mul_inv_ltdl(M, L)` | `mul_inv_lt(M, L)` | M ← M L⁻¹ D⁻¹ L⁻ᵀ |
| `mat_dot(R, S)` | `cross_gram(R, S)` | R Sᵀ (cross Gram matrix) |

### Solver variables (2D: `shake3angle_solve`)

| Old | New | Math |
|-----|-----|------|
| `Lm` | `gram_target` | Lₘ = target Gram matrix |
| `lmc` | `reduced_mass_ldlt` | Λ = (B M⁻¹ Bᵀ)⁻¹ in LDL^T form (reduced mass, NOT inverse mass) |
| `rr` | `metric_rr` | G = R Rᵀ (constraint metric) |
| `rc` | `inv_chol_C` → `metric_whitener` → `rr_frame_normalizer` | G⁻¹ᐟ² — frame normalizer for constraint metric (see note below) |
| `ss` | `ss_perp_gram` | S⊥ᴿᵀ S⊥ᴿ (perpendicular Gram) |
| `rc_chi` | `U_invT_gram_cross` | U⁻ᵀ R Sᵀ (intermediate, used for projector term) |
| `diff` | `gram_defect` | Lₘ − S Sᵀ |
| `sigma` | `rr_residual` → `rr_residual_M` | Lₘ − S⊥ᴿᵀS⊥ᴿ (residual Gram), then L_A⁻¹ · residual · L_A⁻ᵀ (L-factors of Delassus removed from both sides) |
| `sc` | `resid_ltchol_M` | L_σ D_A⁻¹ L_A⁻¹ — LᵀL Cholesky (`chol_lower_lt`) of residual (L_Aᵀ removed from right), then right-multiplied by inverse-mass upper factors |
| `chi` (after `u_mul`) | `rs_coproj` → `rs_coproj_M` | G⁻¹ Rᵀ S (contravariant projection), then right-multiplied by L_Aᵀ D_A L_A (full mass on right) |
| `ltmp` | `rr_chi_sym` | Symmetrized coproj (linearization loop iteration variable) |

### Solver variables (3D: `solve3x3`)

| Old | New | Math |
|-----|-----|------|
| `L_mat` | `gram_target` → `rr_target_M` | L (target Gram; doubles as residual when R has full rank), then L_A⁻¹ · L · L_A⁻ᵀ (L-factors of Delassus removed) |
| `lm_chol` | `reduced_mass_ldlt` | Λ = (B M⁻¹ Bᵀ)⁻¹ in LDL^T form |
| `rr` | `metric_rr` | G = R Rᵀ |
| `rc` | `inv_chol_C` → `metric_whitener` → `rr_frame_normalizer` | G⁻¹ᐟ² — frame normalizer for constraint metric (see note below) |
| `rs` | `rs_cross_gram` | R Sᵀ (cross-Gram before orthonormalization) |
| `sc` | `resid_ltchol_M` | L_σ D_A⁻¹ L_A⁻¹ — LᵀL Cholesky (`chol_to_ltl_lower`) of residual (L_Aᵀ removed from right), right-multiplied by inverse-mass upper factors |

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

χ is the **contravariant projection coefficient**: the in-subspace least-squares fit
of S in the R-basis. Its entries are χ_{ij} = (G⁻¹)_{ik} (r_k · s_j), the solution
to R Rᵀ α = R s_j. The actual 3D projection is Rᵀχ = Rᵀ(R Rᵀ)⁻¹ R s_j (Moore-Penrose).

### Factorization

G = UᵀU, Σ = L_σᵀ L_σ (via `chol_to_ltl_lower`), so Φ = U⁻¹ L_σ Q for Q ∈ SO(n).

The correction Φ decomposes as a **polar decomposition in the orthonormalized frame**:
L_σ carries the stretch (residual Gram magnitude), Q carries the rotation (determined
by the symmetry constraint). The frame_normalizer U⁻¹ maps back from orthonormalized
to physical coordinates.

The symmetry condition on Λ = χ + Φ determines Q via skew decomposition
(closed-form in 2D, Cayley iteration in 3D).

### Key insight: why L − S⊥ᴿᵀS⊥ᴿ is formed first

In the 2D solver, S⊥ᴿ is computed geometrically: n = r01 × r02 is the normal
to the constraint plane, and p_i = s_i · n. Since S⊥ᴿ has rows p_i n̂, the Gram
is S⊥ᴿᵀS⊥ᴿ = SymMat2{p₁²/|n|², p₁p₂/|n|², p₂²/|n|²}.

Then `sigma = gram_target - gram_Sperp` directly gives Lₘ − S⊥ᴿᵀS⊥ᴿ, and only
then is lt_sandwich applied. This is numerically stable
because the projection cancels the in-subspace component of S *before* introducing
the ill-conditioned mass factors.

In the 3D solver (full rank), S⊥ᴿ = 0, so `gram_target` is already the correct
sigma. The only difference between the two paths is whether we subtract the
perpendicular Gram first.

### Two metrics, same pattern

The algorithm orthonormalizes with respect to two metrics:
1. **Constraint metric** G = R Rᵀ: `rr_frame_normalizer` orthonormalizes the constraint subspace
2. **Reduced mass** Λ = (B M⁻¹ Bᵀ)⁻¹ = L D Lᵀ: `reduced_mass_ldlt` stores L, D

Both use the same LDL^T / Cholesky machinery. The sandwich operations:
- `lt_sandwich(S, reduced_mass_ldlt)` = L̃ S L̃ᵀ (congruence by unit lower-triangular; with inv_ldlt output, removes L-factors from both sides, preparing for Cholesky)
- `rmul_ldlt(M, reduced_mass_ldlt)` = M Lᵀ D L (applies full Λ on the right)

---

## Pruned

- `chol_frame2`, `chol_frame3`
- `ldlt_pivot2` (was `dchol_pivot2`)
- `mul_inv_lt` (both 2×2 and 3×3, was `mul_inv_ltdl`)
- `inv_lt_solve` (2×2 only, was `inv_ltdl_mul` — 3×3 version still used)
- `LTMat2 * LTMat2`
- `SymMat3::operator+/-`
- `dot3` (in mat3.h)
- `lt_sandwich_fwd` (both 2×2 and 3×3, renamed to `rmul_ldlt`)
- `U_invT_gram_cross` variable and `mtm` call in 2D solver (replaced by direct S⊥ᴿ Gram)

---

## Naming Philosophy: Role Over Mechanism

Names should describe *what a quantity is mathematically*, not *how we computed it*.

### `inv_chol_C` → `metric_whitener` → `rr_frame_normalizer`

Regardless of factorization (Cholesky, eigendecomposition, polar square root), what we need is
G⁻¹ᐟ² — the linear map that normalizes the constraint frame: it pushes tangent vectors from
the oblique frame (where inner products are measured by G) to the orthonormal frame (where
G collapses to I). This is a **frame normalizer** for the RᵀR metric. Cholesky is just the cheap
way to compute a square-root factor. Naming it "inverse Cholesky" confuses mechanism with role;
"metric whitener" borrows from statistics and won't land with MD users.

### `M_ldlt` → `reduced_mass_ldlt`

Stores Λ = D⁻¹ where D = B M⁻¹ Bᵀ is the Delassus matrix. The old name "M_ldlt" was
misleading — M suggests the mass matrix, but this is actually the **inverse operational-space
inertia matrix** (inverse OSIM), or equivalently the factored inverse of the Delassus matrix.
"reduced_mass_ldlt" clarifies that this is the LDL^T factor of the reduced (constraint-space) mass.

### Naming convention: `_M` suffix

Variables with `_M` suffix have been right-multiplied by the reduced mass matrix Λ = L D Lᵀ
(stored as `reduced_mass_ldlt`). The unweighted form is always computed first and kept.
- `rs_coproj_M`: coproj right-multiplied by Λ (via `rmul_ldlt`)
- `rr_residual_M`: residual after `lt_sandwich` with `reduced_mass_ldlt` (L_A⁻¹ · residual · L_A⁻ᵀ)
- `rr_target_M`: target after `lt_sandwich` with `reduced_mass_ldlt` (L_A⁻¹ · target · L_A⁻ᵀ)

Note: `rr_residual_M` and `rr_target_M` are NOT `M·residual·M` — they have only the L-factors
of the Delassus operator removed from both sides, preparing for Cholesky. The full `M·S·M`
structure is split across `lt_sandwich` (removes L from both sides), `chol_lower` (absorbs D),
and `mul_dl` (re-attaches D⁻¹ and L⁻¹ on the right).

### Multibody dynamics terminology mapping

| Current name | Math | Multibody dynamics term |
|---|---|---|
| `metric_rr` | G = R Rᵀ | Task-space metric (constraint metric / Delassus w/o mass-weighting) |
| `M_ldlt` → `reduced_mass_ldlt` | Λ = (B M⁻¹ Bᵀ)⁻¹ | Inverse OSIM / factored inverse Delassus |
| `inv_chol_C` → `rr_frame_normalizer` | G⁻¹ᐟ² | Frame normalizer for constraint metric |
| `gram_target` | Lₘ | Target task-space Gram |
| `rs_cross_gram` | R Sᵀ | Task-space cross Gram |
| `rs_coproj` / `rs_coproj_M` | G⁻¹ Rᵀ S / G⁻¹ Rᵀ S · Λ | Contravariant projection (unweighted / right-multiplied by reduced mass) |

### Demotion: ill-conditioning, not rank deficiency

For n=2 (angle constraint): B has rank 2. S⊥ᴿ lives in the 1D null space (perpendicular to
constraint plane). For n=3 (improper/dihedral): B has full rank, S⊥ᴿ = 0.

Demotion is about **pathological conditioning** — one direction has vanishingly small effective
inertia — not rank loss. M⁻¹ᐟ² preserves rank, so rank(B M⁻¹ Bᵀ) = rank(B) always.

### SHAKE's tangent-space interpretation

SHAKE linearizes constraints at x(t₀) (on the manifold): ∇C|_{x(t₀)} · (x − x(t₀)) = 0.
The rows of R are the constraint gradients. The tangent space is only well-defined at points
on the manifold. x(t₀) was corrected to lie on M, so its tangent space is correct. x_predicted
is off the manifold — projecting from there gives the wrong tangent space. This is the same
pattern as Newton-Raphson on nonlinear constraints: tangent stiffness is always formed at the
last known feasible state.

---

## Decomposition of `lookup_or_compute_matrices`

The monolithic function was decomposed into:

- **`lookup_or_compute_matrices()`** — thin dispatcher calling sub-functions by cluster type, then `propagate_demoted_tags()`
- **`propagate_demoted_tags()`** — extracted demoted-tag propagation loop
- **`lookup_or_compute_angle()`** — angle (flag 1) path, returns int
- **`lookup_or_compute_improper()`** — improper (flag 5) path with demotion logic, returns int
- **`lookup_or_compute_dihedral()`** — dihedral (flag 6) path, returns int
- **`get_inv_mass()`** — inline helper that unifies `rmass` vs `mass[type[]]` distinction

Each sub-function handles both rmass and non-rmass paths internally (only differing in cache key
format and which storage to write to), using `get_inv_mass()` to compute inverse masses uniformly.

### Storage naming

- `gram_target_cached` (was `L_entries`) — cached target Gram matrices, indexed by constraint type
- `reduced_mass_ldlt_cached` (was `lm_entries`) — cached reduced mass LDL^T factors, indexed by type combo (non-rmass path)
- `reduced_rmass_ldlt` (was `rigs_lm_atom`) — per-atom reduced mass LDL^T factors (rmass path)

Note: `reduced_mass_ldlt` is a **mass** matrix (LDL^T of (B M⁻¹ Bᵀ)⁻¹), not an inverse mass matrix.
The `inv_ldlt` function is called on inverse masses but inverts the result, producing a mass.