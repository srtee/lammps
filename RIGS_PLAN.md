# RIGS Implementation Plan

## Data layout

### shake_flag values

| Value | Meaning | shake_type (3 cols) | rigs_type (3 cols) |
|-------|---------|---------------------|--------------------|
| -1    | Sentinel: claimed by RIGS, not yet assigned | — | — |
| 0     | Not in cluster | — | — |
| 1     | 3-atom angle cluster | [bond1, bond2, angletype] | — |
| 2     | 2-atom bond cluster | [bond1, —, —] | — |
| 3     | 3-atom bond-only cluster | [bond1, bond2, —] | — |
| 4     | 4-atom bond-only cluster (star) | [bond1, bond2, bond3] | — |
| 5     | 4-atom improper cluster (RIGS) | [bond1, bond2, bond3] | [angletype1, angletype2, angletype3] |
| 6     | 4-atom dihedral cluster (RIGS) | [bond1, bond2, bond3] | [angletype1, angletype2, dihedraltype] |

-1 sentinel: set on partner atoms A and D before `find_clusters()`. `find_clusters()`
resets -1 to 0 (with clean shake_atom/shake_type) and skips them. After `find_clusters()`
+ `shake_info()`, A and D get their cluster data from propagation.

### Flag 5 (improper, star topology center A + partners B,C,D)

- Detection: after `find_clusters()`, upgrade flag 4 star clusters where
  all 3 angles through center are in `angle_flag`, or an improper type is in `improper_flag`
- Owner: A (central atom), `shake_atom[A] = {A, B, C, D}`
- 6 distance constraints: d(A,B), d(A,C), d(A,D), d(B,C), d(B,D), d(C,D)
- shake_type = [bondtype(A-B), bondtype(A-C), bondtype(A-D)]
- rigs_type = [angletype(B-A-C), angletype(B-A-D), angletype(C-A-D) or -impropertype]
- rigs_type[2] negative means improper type supplies the 3rd non-bond constraint
  (angle-angle-improper); positive means 3rd angle (angle-angle-angle)

### Flag 6 (dihedral, chain A-B-C-D)

- Detection: before `find_clusters()`, scan atom dihedral topology
- Owner: B (lower interior ID), `shake_atom[B] = {B, A, C, D}`
- C is pre-assigned with same cluster data (implicit B→C hop)
- A and D get -1 sentinels, then 0, then small clusters, then overwritten by
  `shake_info()` propagation from B and C
- 6 distance constraints: d(A,B), d(B,C), d(C,D), d(A,C), d(B,D), d(A,D)
- shake_type = [bondtype(A-B), bondtype(B-C), bondtype(C-D)]
- rigs_type = [angletype(A-B-C), angletype(B-C-D), dihedraltype]

### rigs_type array

- `int **rigs_type` allocated as nmax × 3, owned by FixRigs
- Only populated for shake_flag == 5 or 6
- `rigs_type[0]` and `rigs_type[1]` are always angle types (positive)
- `rigs_type[2]` interpretation depends on shake_flag:
  - shake_flag == 6 (dihedral): dihedral type (positive)
  - shake_flag == 5 (improper): 3rd angle type (positive) or improper type (negative)

### shake_type sizing

- **Unchanged** — remains nmax × 3
- For flags 5 and 6, all 3 bond types fit in the existing 3 columns

## Detection algorithm (FixRigs::post_constructor)

### Flow

1. **Dihedral detection (before find_clusters):** scan atom dihedral topology
   for chains A-B-C-D where all 3 bonds are SHAKE-eligible, both angles are
   in `angle_flag`, and the dihedral type is in `dihedral_flag`. Set flag 6
   on owner B and on C (B→C hop). Set -1 sentinels on A and D.
   Currently limited to `Atom::MOLECULAR` (template support TODO).

2. **find_clusters():** skips pre-assigned atoms (flag != 0). Resets -1 sentinels
   to 0 with clean arrays. Error checks: nshake > 3 (unconditional), connected
   clusters (gated on !rigsflag), RIGS nshake checks (gated on rigsflag).

3. **shake_info():** propagates from B→A and C→D (single pass covers the whole
   chain since both B and C are pre-assigned with full cluster data).

4. **Improper detection (after find_clusters):** upgrade flag 4 star clusters
   to flag 5 where all 3 angles are in `angle_flag`, or an improper type is
   in `improper_flag`. Fill `rigs_type` via `fill_improper_types()`.

5. **Type negation:**
   - Flag 5 (improper): bond types and angle types from `find_clusters()` (flag 5 case)
     plus angle types in `post_constructor()`
   - Flag 6 (dihedral): decomposed into B-half (bonds A-B, B-C; angle A-B-C;
     dihedral type) and C-half (bond C-D; angle B-C-D). B and C each handle
     their own local neighborhood.

### Error checks

1. **nshake > 3** (unconditional in find_clusters): no atom with more than 3
   SHAKE-eligible bonds.
2. **Connected clusters** (gated on !rigsflag): skipped for RIGS since interior
   dihedral atoms have nshake > 1.
3. **RIGS nshake checks** (gated on rigsflag, in find_clusters):
   - flag 5 owner: nshake must equal 3 (star center, 3 bonds)
   - flag 6 owner (B): nshake must equal 2 (chain interior, 2 bonds)
   Catches cases like 1,2-dichloroethene where C-Cl bonds fall outside
   a constrained H-C-C-H dihedral.
4. **Group membership** filters which bonds are SHAKE-eligible, providing
   an escape hatch for users to avoid conflicts.

### shake_atom ordering convention

- **Flag 5 (improper):** shake_atom = [A, B, C, D] — center first (matches SHAKE flag 4)
- **Flag 6 (dihedral):** shake_atom = [B, A, C, D] — B (owner/lowest interior ID) first
  Both B and C have identical shake_atom entries after pre-assignment.

## Constraint solver (in progress)

### What exists

- `FixShake::shake4()` currently handles flag 4, 5, and 6 clusters as a
  placeholder (constrains 3 bonds only, no angle/dihedral constraints)

### What needs to be added

- **FixRigs::shake4rigs()** (or split into shake4improper / shake4dihedral):
  the RIGS constraint solver for flag 5 and flag 6 clusters
- **Central 3×3 matrix solve:** will be implemented by the user
- The solver enforces 6 distance constraints per cluster:
  - Flag 5: 3 bonds + 3 non-bond distances (from angles/improper)
  - Flag 6: 3 bonds + 3 non-bond distances (from 2 angles + 1 dihedral)

### Dispatch

- `post_force()` / `post_force_respa()`: route flag 5 and 6 to the RIGS solver
  instead of `shake4()`
- `min_post_force()`: same routing needed
- The RIGS solver needs atom indices from `closest_list` (4 atoms for flag 5/6)

### Equilibrium distances

- Bond distances: `bond_distance[shake_type[m][0..2]]` (already computed by FixShake::init)
- Angle distances: need `rigs_angle_distance[]` array in FixRigs for the non-bond
  pair distances derived from angle equilibrium values
- Dihedral distance: need `dihedral_distance[]` for the end-to-end distance
  derived from dihedral equilibrium value (for flag 6)

### What the user will implement

The central 3×3 matrix solve algorithm. This is the core numerical method that
solves for the Lagrange multipliers enforcing the 6 distance constraints.
The surrounding infrastructure (atom mapping, distance lookups, force updates)
will be set up to call into this algorithm.

### FLOP cost comparison: SHAKE vs RIGS

#### SHAKE 3×3 (FixShake::shake3angle, per iteration of convergence loop)

The SHAKE algorithm iterates a fixed-point loop solving a 3×3 linear system
with quadratic corrections. Per iteration:

| Component | Multiplies | Adds/Subs |
|---|---|---|
| Quadratic terms (3 × 6-term forms) | 36 | 15 |
| RHS (b1, b2, b3) | 3 | 3 |
| Matrix-vector (3×3 inverse × 3-vec) | 9 | 6 |
| Convergence check | 0 | 3 |
| Overflow guard | 0 | 0 |
| **Per iteration** | **48** | **27** |

~78 FLOPs + 6 `fabs` per iteration. Pre-iteration setup (matrix inverse,
9 dot products, 9 invmass ops) is ~60 FLOPs + 2 sqrts + 1 div.

#### RIGS 2×2 (FixRigs::shake3angle, one-shot — no iteration)

The RIGS algorithm uses a direct matrix decomposition that solves the
2×2 constrained system without iteration. From mass setup through
lamda extraction (lines 757–829 of fix_rigs.cpp):

| Component | Mult | Add/Sub | Div | Sqrt |
|---|---|---|---|---|
| invmass setup | 0 | 2 | 3 | 0 |
| sym_dot ×2 (rr, ss) | 18 | 12 | 0 | 0 |
| L + diff | 2 | 3 | 0 | 0 |
| chi = RS (4 × 3D dot products) | 12 | 8 | 0 | 0 |
| inv_dchol → lm | 1 | 1 | 3 | 0 |
| inv_chol_upper → rc | 2 | 1 | 4 | 2 |
| ut_mul (rc, chi) | 6 | 2 | 0 | 0 |
| mtm(chi) + diff → sigma | 6 | 6 | 0 | 0 |
| u_mul (rc, chi) | 6 | 2 | 0 | 0 |
| lslt_mul (sigma, lm) | 3 | 3 | 0 | 0 |
| chol_lower (sigma) | 1 | 1 | 1 | 2 |
| mul_dl → sc | 4 | 1 | 0 | 0 |
| mul_ltdl (chi, lm) | 8 | 4 | 0 | 0 |
| rc * sc → phiC | 5 | 1 | 0 | 0 |
| rc * J * sc → phiS | 12 | 4 | 0 | 0 |
| skew ×3 | 0 | 3 | 0 | 0 |
| Asq / sinp / sskew / cskew | 7 | 3 | 2 | 1 |
| Lamda extraction | 5 | 5 | 0 | 0 |
| **Total (one-shot)** | **98** | **61** | **13** | **5** |

~159 FLOPs + 5 sqrts + 13 divs. The inv_dchol path for lm saves 2 sqrts
and 4 mults vs the old trans_inv_chol_upper path. Only rc (from rr) and
chol_lower (from sigma) still need sqrts.

#### RIGS 3×3 (FixRigs::shake4improper, one-shot + Cayley iteration)

The 3×3 RIGS solver for 4-atom improper/dihedral clusters. Non-iterative
portion (lines 942–965 of fix_rigs.cpp):

| Component | Mult | Add/Sub | Div | Sqrt |
|---|---|---|---|---|
| invmass setup | 0 | 3 | 4 | 0 |
| sym_dot(R) | 18 | 12 | 0 | 0 |
| sym_dot(S) | 18 | 12 | 0 | 0 |
| L + diff | 6 | 6 | 0 | 0 |
| inv_dchol → lm | 7 | 7 | 6 | 0 |
| mat_dot(R, S) | 18 | 9 | 0 | 0 |
| inv_chol_upper → rc | 12 | 5 | 6 | 3 |
| ut_mul (rc, chi) | 18 | 9 | 0 | 0 |
| sym_dot(chi) + diff → sigma | 18 | 18 | 0 | 0 |
| u_mul (rc, chi) | 18 | 9 | 0 | 0 |
| lslt_mul (sigma, lm) | 12 | 12 | 0 | 0 |
| chol_lower (sigma) | 3 | 3 | 3 | 3 |
| mul_dl → sc | 9 | 4 | 0 | 0 |
| mul_ltdl (chi, lm) | 24 | 18 | 0 | 0 |
| G construction | 20 | 8 | 0 | 0 |
| G.invert() | 8 | 1 | 3 | 0 |
| skew(chi) | 0 | 3 | 0 | 0 |
| **Non-iterative total** | **189** | **131** | **22** | **6** |

Per Cayley iteration (lines 302–310 of mat3.h):

| Component | Mult | Add/Sub | Sqrt |
|---|---|---|---|
| negskew_ut_mul | 12 | 10 | 0 |
| subtract skewChi + normsq | 3 | 5 | 0 |
| G.mat_vec | 6 | 3 | 0 |
| cayley_rotate (w + 3 cols × 2 crosses + update) | 66 | 39 | 1 |
| **Per iteration** | **87** | **57** | **1** |

Post-loop: u_mul(rc, gamma) = 18m + 9a, lamda += chi = 9a.

Total with N Cayley iterations: ~320 + 144N FLOPs + (6+N) sqrts + 22 divs.
Typical N=5: ~1040 FLOPs + 11 sqrts + 22 divs.

#### SHAKE 6×6 extrapolated cost (per iteration)

Extrapolating from the SHAKE 3×3 iteration structure to a 6-lambda system
(for flag 5/6 clusters with 6 distance constraints):

| Component | Mult | Add/Sub | fabs |
|---|---|---|---|
| Quadratic terms (6 × 21-term forms) | 252 | 120 | 0 |
| RHS (b1…b6) | 6 | 6 | 0 |
| Matrix-vector (6×6 inverse × 6-vec) | 36 | 30 | 0 |
| Convergence check | 0 | 6 | 6 |
| Overflow guard | 0 | 0 | 6 |
| **Per iteration** | **294** | **162** | **12** |

~462 FLOPs + 12 fabs per iteration. The 6×6 inverse itself (pre-iteration,
Cholesky or cofactor) adds a further ~100 FLOPs + 3 sqrts.

#### Summary

- **RIGS 2×2** solves 3 constraints in ~159 FLOPs + 5 sqrts + 13 divs (one-shot, no
  iteration). The inv_dchol decomposition eliminates 2 sqrts and 4 mults compared
  to the old Cholesky inverse path for the mass matrix.
- **RIGS 3×3** solves 6 constraints in ~320 + 144N FLOPs + (6+N) sqrts + 22 divs,
  where N is the number of Cayley iterations (typically 3–8). The inv_dchol path
  for lm also eliminates 3 sqrts here.
- **SHAKE 3×3** solves 3 constraints in ~78 FLOPs + 6 fabs per iteration (typically
  10–50 iterations to converge, so 780–3900 FLOPs total).
- **SHAKE 6×6** would solve 6 constraints in ~462 FLOPs + 12 fabs per iteration
  (similarly 10–50 iterations → 4620–23100 FLOPs total).
- The RIGS approach avoids SHAKE's fixed-point iteration at the cost of more
  upfront linear algebra. For 3 constraints, RIGS 2×2 is cheaper after ~3 SHAKE
  iterations. For 6 constraints, RIGS 3×3 at N=5 (~1040 FLOPs) is comparable to
  just 2–3 SHAKE 6×6 iterations (~930–1386 FLOPs), making RIGS the clear winner
  given typical convergence requires 10–50 iterations.

## Completed data integrity updates (fix_shake.cpp)

All dispatch sites in fix_shake.cpp now handle shake_flag 5 and 6:

| Method | Change |
|---|---|
| Destructor | 5 \|\| 6: restores 3 bond types |
| `dof()` | 4 \|\| 5 \|\| 6: +3 DOF removed |
| `pre_neighbor()` list builder | 1 \|\| 3 → 3-atom path; else → 4-atom path (flags 4,5,6) |
| `post_force()` / `post_force_respa()` | 4 \|\| 5 \|\| 6 → `shake4()` (placeholder for RIGS) |
| `min_post_force()` | 4 \|\| 5 \|\| 6 → 4-atom bond force path |
| `copy_arrays()` | 4 \|\| 5 \|\| 6: copies 4 atoms + 3 types |
| `update_arrays()` | 4 \|\| 5 \|\| 6: adjusts 4 atom offsets |
| `set_molecule()` | 4 \|\| 5 \|\| 6: copies 4 atoms + 3 types |
| `pack_exchange()` | 4 \|\| 5 \|\| 6: packs 4 atoms + 3 types |
| `unpack_exchange()` | 4 \|\| 5 \|\| 6: unpacks 4 atoms + 3 types |
| `find_clusters()` type negation | 5: 3 bonds; 6: handled in post_constructor |
| `find_clusters()` stats | count5 and count6 for improper/dihedral clusters |
| `stats()` | Bond loop capped at n=4 to avoid out-of-bounds |
| -1 sentinel | Reset to 0 with clean arrays, skipped by cluster assignment loop |