# Rank Deficiency, Linear Constraints, and Constraint Demotion

## 1. The problem

When the constrained particles are rank-deficient relative to the ambient space (e.g., four particles constrained to a plane), the Gram matrix G₀ = ½(RL)ᵀ(RL) is singular. The SHAKE/RIGS correction R′ = S − RLΛLᵀ can only produce displacements in the column space of RL. Perturbations orthogonal to that space (e.g., out-of-plane displacements) cannot be corrected by any choice of Λ.

Practically, the Cholesky decomposition of G₀ (or MG₀M) fails with a zero (or near-zero) pivot, which is how this issue manifests in the solver.

## 2. Linear frame constraints

A linear frame constraint has the form:

C(R) = nᵀrᵢ − d = 0

or collecting all such constraints into one matrix equation: NR = d, where each row of N picks out a particle and imposes a linear condition on its position.

| | Gram matrix | Linear frame |
|---|---|---|
| Form | ½(RL)ᵀ(RL) = G₀ | NR = d |
| In R | Quadratic | Linear |
| Jacobian (∂C/∂R) | Depends on R | Constant |
| Solution | Iterative (SHAKE) | Direct projection |

Because the Jacobian is constant, the correction is orthogonal projection:

R′ = S − Nᵀ(NNᵀ)⁻¹(NS − d)

No iteration needed. In the Lagrange multiplier language:

ℒ = ½∥R′ − S∥²_F + λᵀ(NR′ − d)

giving R′ = S − Nᵀλ with λ = (NNᵀ)⁻¹(NS − d).

## 3. Composability

Linear and Gram constraints compose cleanly when they act on orthogonal subspaces. For the planar example: in-plane perturbations are corrected by RIGS (preserving distances/angles), and out-of-plane perturbations are corrected by projecting back onto the plane. The two corrections commute because RIGS doesn't generate out-of-plane displacements, and the projection doesn't change in-plane distances to first order.

## 4. Constraint demotion strategy

The solution is to "demote" one particle's positional constraint to a linear (frame) constraint, reducing the Gram system to full rank. The remaining Gram constraints handle internal geometry, and the demoted particle gets a linear constraint (e.g., n · rᵢ = d for the plane normal n).

### Which particle to demote?

G₀ and MG₀M share the same null space, but their null vectors differ: if v is in null(G₀), then M⁻¹v is in null(MG₀M). This means the "worst offender" — the particle most redundant in the constraint system — can differ depending on whether you consider geometry alone (G₀) or geometry weighted by mass (MG₀M).

In the star topology (center + 3 partners):
- G₀'s smallest unpivoted pivot corresponds to the shortest displacement vector
- MG₀M's smallest unpivoted pivot corresponds to the lightest-particle constraint
- These need not agree

### Pivoted Cholesky

Pivoted Cholesky decomposition solves this cleanly. At each step, it selects the largest remaining diagonal element:

```
Pivoted Cholesky of A ∈ PSD^{n×n}:
  Π = identity
  for k = 1, ..., n:
    i = argmax Diag(A_k)[k:]
    swap rows/cols k ↔ i in A_k
    record swap in Π
    L_kk = sqrt(A_k[k,k])
    L_k, = A_k[k+1:,k] / L_kk
    A_{k+1} = A_k[k+1:,k+1:] - L_k, L_k,^T
```

This guarantees d₁ ≥ d₂ ≥ ... ≥ dₙ ≥ 0, so near-zero pivots are always at the end. The accumulated permutation Π maps back to original constraint indices. The last position (or the one where the pivot drops below threshold) identifies the constraint to demote.

Key property: **pivoted Cholesky on MG₀M accounts for both geometry and mass** — exactly what the RIGS solver needs. The permutation Π retrofits to the original constraint ordering to identify which particle to demote.

### Demotion procedure

1. Form MG₀M and attempt pivoted Cholesky decomposition
2. The near-zero (or zero) pivot identifies the redundant constraint
3. Demote the corresponding particle to a linear frame constraint:
   n = r₀₁ × r₀₂ / |r₀₁ × r₀₂| (the out-of-plane direction)
4. Solve the remaining full-rank 2×2 Gram system normally
5. Apply the linear constraint as a direct projection

This reduces the 3×3 singular system to a 2×2 full-rank RIGS solve plus a linear projection — both of which are well-conditioned and non-iterative.

## 5. Force redistribution from the demoted particle

When a particle (e.g., a virtual site) is demoted from the Gram constraint system, its force must be redistributed to the remaining constrained atoms before the RIGS solver runs. This ensures:

1. **Total force is conserved** (Newton's third law)
2. **Torque about the constrained frame is preserved** (rotational dynamics)
3. **The RIGS constraint force** acts on a physically reasonable mass distribution

### The redistribution problem

Given force **F_d** on demoted particle i3, redistribute to {i0, i1, i2} such that:

- Force balance: **F_i0** + **F_i1** + **F_i2** + **F_i3** = **F_d**
- Torque balance about any point: Σ **r_k** × **F_k** = 0

This is underdetermined (9 unknowns, 6 equations). The minimum-norm solution picks the redistribution with smallest total Euclidean norm.

### Implementation in `shake4demoted`

The demoted particle's force is decomposed in the RIGS frame (e1, e2, n) built from the constrained geometry:

```
F_d = f_n · n + f_1 · e1 + f_2 · e2

where:
  f_n = F_d · n    (perpendicular to constraint plane)
  f_1 = F_d · e1   (along bond i0→i1)
  f_2 = F_d · e2   (in-plane, perpendicular to e1)
```

The RIGS frame uses the pre-computed Cholesky values from `rigs_lm[ilist]`:
- `l00 = rigs_lm[ilist][3]` — length of bond i0→i1
- `m01 = rigs_lm[ilist][4]` — Gram-Schmidt coefficient
- `l11 = rigs_lm[ilist][5]` — length of Gram-Schmidt orthogonalized bond i0→i2

```
e1 = r01 / l00
e2 = (r02 - m01 · r01) / l11
n = e1 × e2
```

The fraction of force to redistribute is `fraction = 1 - m_i3 / M_total`, leaving the remaining `m_i3 / M_total` on the demoted particle so the total system mass is conserved.

### Redistributing by bond length weights

The perpendicular (in-plane) force components `(f_1, f_2)` are split between i1 and i2 in proportion to their bond lengths:

```
w1 = l00 / (l00 + l11)
w2 = l11 / (l00 + l11)

F_i1 += (f_1 · e1 + f_2 · e2) · w1 · fraction
F_i2 += (f_1 · e1 + f_2 · e2) · w2 · fraction
```

The perpendicular component `f_n · n` goes entirely to i0 (the central atom), which has no perpendicular lever arm in the frame, so this choice preserves torque with zero net torque about i0.

### Summary of the redistribution

| Component | Destination | Weight |
|-----------|-------------|--------|
| `f_n · n` (perpendicular) | i0 | `fraction` |
| `(f_1·e1 + f_2·e2)` | i1 | `w1 · fraction` |
| `(f_1·e1 + f_2·e2)` | i2 | `w2 · fraction` |
| Removed from | i3 | `fraction` total |

This is a **simplified ansatz** — not the full minimum-norm solution, which would require proper perpendicular distances from the rotation axis. The weights `w1`, `w2` proportional to bond lengths are geometrically motivated and satisfy force + torque balance for the 3-atom constrained subsystem.