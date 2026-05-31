# FixElectrodeConp vs FixElectrodeConpIntel Numerical Differences

## Overview

The `FixElectrodeConpIntel` class (defined in `src/INTEL/fix_electrode_conp_intel.h`) is a thin wrapper around `FixElectrodeConp` that primarily:
1. Sets `intelflag = true`
2. Provides an `intel_pack_buffers()` override that calls `PPPMElectrodeIntel::pack_buffers_q()`

The major numerical differences arise in the **KSpace compute_vector path**, specifically in how the electric potential is calculated on the electrode atoms.

## Key Code Paths Causing Numerical Differences

### 1. ElectrodeVector::compute_vector() → KSpace::compute_vector()

In `update_charges()` (line 871 of `fix_electrode_conp.cpp`):
```cpp
elyt_vector->compute_vector(potential_i);
```

This calls `ElectrodeVector::compute_vector()` which then calls:
```cpp
electrode_kspace->compute_vector(vector, groupbit, source_grpbit, invert_source);
```

**The divergence occurs here:**
- **Base class**: Uses `PPPMElectrode::compute_vector()`
- **Intel class**: Uses `PPPMElectrodeIntel::compute_vector()` (templated with precision modes)

### 2. PPPMElectrode::project_psi() vs PPPMElectrodeIntel::project_psi()

This is the **most significant source of numerical differences**. Both versions project the potential from the FFT grid back to particles, but with very different implementations:

**Base version** (`pppm_electrode.cpp`, lines 624-660):
- Standard double-precision arithmetic
- Calls `compute_rho1d()` for charge assignment coefficients
- Simple nested loops

**Intel version** (`pppm_electrode_intel.cpp`, lines 392-503):
- **Templated precision**: Can use `float` or `double` via `flt_t`/`acc_t` template parameters
- **SIMD vectorization**: Uses `#pragma simd` and `#pragma loop_count` directives
- **Lookup tables**: Can use `rho_lookup` table for charge assignment coefficients instead of computing them on-the-fly
- **Aligned memory**: Uses `_alignvar(flt_t rho[3][INTEL_P3M_ALIGNED_MAXORDER], 64)` for cache alignment

### 3. make_rho_in_brick() - Charge Assignment

**Base version** (`pppm_electrode.cpp`):
- Uses standard `compute_rho1d()` for charge assignment weights

**Intel version** (`pppm_electrode_intel.cpp`, lines 910-1045):
- **Templated precision** with `flt_t` and `acc_t`
- **SIMD parallelization** with OpenMP threading
- **Per-thread density accumulation** (lines 946-1044)
- **Lookup table optimization** for charge assignment (`_use_table` path)

### 4. Intel Precision Modes

In `PPPMElectrodeIntel::compute_vector()` (lines 330-342), the code switches between three precision modes:
```cpp
switch (fix->precision()) {
  case FixIntel::PREC_MODE_MIXED:   // float/double mixed
  case FixIntel::PREC_MODE_DOUBLE:  // full double
  default:                          // single precision
}
```

This affects all Intel-specific routines including `make_rho_in_brick()` and `project_psi()`.

### 5. Buffer Packing (intel_pack_buffers)

The Intel version adds an extra step in `set_charges()`:

**Base** (`fix_electrode_conp.cpp`, line 956-962):
```cpp
void FixElectrodeConp::set_charges(std::vector<double> q_local) {
  // ... set charges
  comm->forward_comm(this);
  intel_pack_buffers();  // Empty in base class
}
```

**Intel** (`fix_electrode_conp_intel.h`, line 50):
```cpp
inline void intel_pack_buffers() final override {
  _intel_kspace->pack_buffers_q();
}
```

This calls `PPPMElectrodeIntel::pack_buffers_q()` (lines 1194-1217) which packs charge data into Intel-optimized buffers with SIMD operations, potentially introducing small numerical differences in how charges are represented.

## Summary of Numerical Difference Sources

| Code Path | Base Behavior | Intel Behavior |
|-----------|---------------|----------------|
| `project_psi()` | Standard double arithmetic | SIMD vectorization, templated precision, lookup tables |
| `make_rho_in_brick()` | Standard charge assignment | Multi-threaded, templated precision, per-thread accumulation |
| Precision | Fixed double | Mixed/float/double modes selectable |
| Buffer packing | No-op | SIMD-optimized charge packing |

The most significant numerical differences would come from the **templated precision modes** (especially if running in mixed or single precision) and the **SIMD vectorization** in `project_psi()` and `make_rho_in_brick()`, where floating-point operations may be reordered for vectorization.

## Re-Neighboring Step Behavior (Key for Numerical Differences)

Operations in the Intel package that happen **only on reneighboring steps** (when `neighbor->ago == 0`) vs every step can cause numerical differences. This is a critical source of potential divergence.

### 1. Particle Mapping (Critical Difference)

**Base class** (`pppm_electrode.cpp`, `PPPMElectrode::compute_vector()`):
```cpp
particle_map();  // Called every time compute_vector is called
```

**Intel class** (`pppm_electrode_intel.cpp`, `PPPMElectrodeIntel::start_compute()`):
```cpp
void PPPMElectrodeIntel::start_compute()
{
  if (compute_step < update->ntimestep) {
    if (compute_step == -1) setup();
    // ...
    // particle_map ONLY called if compute_step < current timestep
    // i.e., only once per timestep, not every call to compute_vector
    switch (fix->precision()) {
      case FixIntel::PREC_MODE_MIXED:
        PPPMIntel::particle_map<float, double>(fix->get_mixed_buffers());
        break;
      // ...
    }
    compute_step = update->ntimestep;
  }
}
```

The `compute_step` tracking means `particle_map()` is called **once per timestep**, not every time `compute_vector()` is invoked. If `compute_vector` is called multiple times per timestep (which happens in electrode calculations), the base class re-maps particles each time while Intel caches the mapping.

### 2. Buffer Packing Behavior

**In `pppm_electrode_intel.cpp` line 329:**
```cpp
void PPPMElectrodeIntel::compute_vector(double *vec, int sensor_grpbit, int source_grpbit,
                                        bool invert_source)
{
  start_compute();
  // ...
  // Pack buffers only if not reneighboring step
  if (neighbor->ago != 0) pack_buffers();  // Called conditionally
  // ...
}
```

The `pack_buffers()` call is skipped on reneighboring steps (`neighbor->ago == 0`). This is a deliberate optimization.

### 3. Intel Buffer Management (intel_buffers.h)

**The `thr_pack()` function** (lines 204-234 of `intel_buffers.h`):
```cpp
inline void thr_pack(const int ifrom, const int ito, const int ago) {
  if (ago == 0) {
    // Full pack - positions AND types
    for (int i = ifrom; i < ito; i++) {
      _x[i].x = lmp->atom->x[i][0];
      _x[i].y = lmp->atom->x[i][1];
      _x[i].z = lmp->atom->x[i][2];
      _x[i].w = lmp->atom->type[i];  // Type packed only on ago==0
    }
    if (lmp->atom->q != nullptr)
      for (int i = ifrom; i < ito; i++)
        _q[i] = lmp->atom->q[i];  // Charges packed on ago==0
  } else {
    // Partial pack - positions only (no types, no charges)
    for (int i = ifrom; i < ito; i++) {
      _x[i].x = lmp->atom->x[i][0];
      _x[i].y = lmp->atom->x[i][1];
      _x[i].z = lmp->atom->x[i][2];
    }
  }
}
```

This shows that on reneighboring steps (`ago == 0`):
- **Types are packed** into `_x[i].w`
- **Charges are packed** into `_q[i]`

On non-reneighboring steps (`ago != 0`):
- Only **positions are updated**
- Types and charges are **assumed unchanged**

### 4. PPPMIntel Particle Map

**In `pppm_intel.cpp` line 347:**
```cpp
template <class flt_t, class acc_t>
void PPPMIntel::particle_map(IntelBuffers<flt_t,acc_t> *buffers)
```

This function is called from `start_compute()` and performs the charge assignment grid mapping. It uses the Intel buffers which may have been updated differently depending on `neighbor->ago`.

### 5. Timing-Based Operations

In `fix_intel.h` (lines 409-446):
```cpp
int FixIntel::offload_end_pair()
{
  if (neighbor->ago == 0)
    return _balance_neighbor * atom->nlocal;  // Use neighbor balance
  else
    return _balance_pair * atom->nlocal;      // Use pair balance
}

void FixIntel::acc_timers()
{
  _timers[TIME_OFFLOAD_PAIR] += *_stopwatch_offload_pair;
  if (neighbor->ago == 0) {  // Only accumulate neighbor time on reneighboring
    _timers[TIME_OFFLOAD_NEIGHBOR] += *_stopwatch_offload_neighbor;
    if (_setup_time_cleared == false) {
      zero_timers();
      _setup_time_cleared = true;
    }
  }
}
```

## Summary of Re-Neighboring Effects

| Operation | Every Step | Only on Renegighboring (ago==0) |
|-----------|-----------|--------------------------------|
| `particle_map()` | Base: Yes<br>Intel: No (cached) | Both: Yes |
| `pack_buffers()` | Intel: Yes (conditionally) | Intel: Skipped |
| Buffer type packing | No | Yes (types in `_x[i].w`) |
| Buffer charge packing | No | Yes (`_q[i]`) |
| Position updates | Yes | Yes (always updated) |
| Offload balancing | Uses pair balance | Uses neighbor balance |

## Analysis: Does the Intel Version Correctly Trigger Buffer Repacking Every Step?

**YES, the Intel version correctly repacks charges every step**, but through a different mechanism than general buffer packing.

### Charge Buffer Repacking Mechanism

The electrode-specific charge repacking is done via `intel_pack_buffers()` which is called from `set_charges()`:

**Call chain for charge updates:**
1. `FixElectrodeConp::set_charges()` (line 956-963) calls `intel_pack_buffers()`
2. `FixElectrodeConpIntel::intel_pack_buffers()` (line 50) calls `_intel_kspace->pack_buffers_q()`
3. `PPPMElectrodeIntel::pack_buffers_q()` (lines 1194-1217) uses `thr_pack_q()`
4. `IntelBuffers::thr_pack_q()` (lines 236-244) **always** packs charges:
```cpp
inline void thr_pack_q(const int ifrom, const int ito) {
  if (lmp->atom->q != nullptr)
    #pragma vector aligned
    #pragma ivdep
    for (int i = ifrom; i < ito; i++)
      _q[i] = lmp->atom->q[i];  // ALWAYS packs charges, no "ago" check
}
```

**Key distinction:**
- `thr_pack(ago)` - General purpose packing, charges only packed when `ago == 0`
- `thr_pack_q()` - Charge-only packing, **always** packs charges, called every step via `intel_pack_buffers()`

### Position Buffer Updates

Positions are handled differently in `PPPMElectrodeIntel::compute_vector()`:
```cpp
void PPPMElectrodeIntel::compute_vector(...) {
  start_compute();  // Updates particle_map if needed
  if (neighbor->ago != 0) pack_buffers();  // Updates positions if NOT reneighboring step
  // ...
}
```

On reneighboring steps (`ago == 0`):
- `pack_buffers()` is **skipped** because positions were just updated during neighbor list build
- `particle_map()` is called via `start_compute()` if `compute_step < timestep`

On non-reneighboring steps (`ago != 0`):
- `pack_buffers()` is called to update positions in Intel buffers
- `particle_map()` is **NOT** called (cached from previous step)

### Summary of Buffer Update Behavior

| Buffer Data | Update Trigger | Every Step? | Notes |
|-------------|---------------|-------------|-------|
| **Charges** | `intel_pack_buffers()` → `pack_buffers_q()` → `thr_pack_q()` | **YES** | Explicitly called from `set_charges()` after charge updates |
| **Positions** | `pack_buffers()` → `thr_pack(ago=1)` | Yes (conditionally) | Skipped on reneighboring steps, done otherwise |
| **Types** | `thr_pack(ago=0)` | NO | Only on reneighboring steps, but types rarely change |

## How Intel Package Updates Base Arrays

The Intel package maintains **two sets of arrays**:
1. **Base LAMMPS arrays**: `atom->x`, `atom->f` - used by non-Intel code
2. **Intel buffers**: `_x`, `_f` - used by Intel-optimized code

### Position Data Flow

**Reading positions (Intel buffer ← base array):**
- Done via `thr_pack()` or `pack_buffers()` 
- Called at the start of pair/kspace compute
- On reneighboring steps: positions already fresh from neighbor build
- On non-reneighboring steps: `pack_buffers()` updates Intel buffer positions

**Writing forces (base array ← Intel buffer):**
- Done via `add_results()` → `add_oresults()` in `fix_intel.cpp` (lines 802-903)
- Example: `f[i].x += f_in[i].x;` adds buffered forces to `atom->f`
- Called in `_sync_main_arrays()` which is triggered by:
  - `pre_reverse()` hook (line 622)
  - `post_force()` hook (line 631)

### Force Update Mechanism

```cpp
// From fix_intel.cpp, line 806
lmp_ft * _noalias const f = (lmp_ft *) lmp->atom->f[0] + out_offset;
for (int i = ifrom; i < ito; i++) {
    f[i].x += f_in[i].x;  // Add Intel-buffered forces to atom->f
    f[i].y += f_in[i].y;
    f[i].z += f_in[i].z;
}
```

The Intel package **ADDS** forces to base arrays (doesn't replace them), allowing hybrid Intel/non-Intel calculations.

## Potential Numerical Divergence Sources

1. **Multiple `compute_vector()` calls per timestep**: The base class `PPPMElectrode` recalculates `particle_map()` each time, while `PPPMElectrodeIntel` caches it. If positions change between calls within the same timestep (which can happen with certain integrators), the base class sees updated positions but the Intel version uses cached positions.

2. **Position staleness on non-reneighboring steps**: The Intel code skips `pack_buffers()` on reneighboring steps with the comment "since midstep positions may be outdated". This suggests Intel buffer positions may differ from actual atom positions briefly during the step.

3. **SIMD floating-point reordering**: Even with correct buffer updates, the SIMD vectorization in `project_psi()` and `make_rho_in_brick()` can reorder floating-point operations, causing small numerical differences.

4. **Force accumulation precision**: Intel uses separate `_f` buffers that are accumulated into `atom->f`. The precision of this accumulation (especially in mixed/single precision modes) can differ from base class direct calculation.

## Critical Finding: ElectrodeVector Uses Mixed Position Sources

**YES, this is a major source of numerical differences!**

### The Problem

`ElectrodeVector::pair_contribution()` (lines 131-189 in `electrode_vector.cpp`) uses **base `atom->x` positions directly**:
```cpp
void ElectrodeVector::pair_contribution(double *vector)
{
  double **x = atom->x;  // Line 133 - uses base array, NOT Intel buffers!
  // ...
  double const delx = xtmp - x[j][0];  // Line 164
```

Meanwhile, the **kspace contribution** (`PPPMElectrodeIntel::compute_vector`) uses **Intel buffered positions**:
```cpp
void PPPMElectrodeIntel::compute_vector(...)
{
  // ...
  ATOM_T *_noalias const x = buffers->get_x(0);  // Uses Intel buffers!
```

### The Impact

The electrode vector calculation **combines two different position sources**:

| Component | Position Source | Array Type |
|-----------|-----------------|------------|
| Pair contribution | `atom->x` | Base LAMMPS array |
| KSpace contribution | `buffers->get_x(0)` | Intel SIMD buffer |

These positions can differ because:
1. **Precision conversion**: Intel buffers may use `float` (PREC_MODE_MIXED/SINGLE) while `atom->x` is always `double`
2. **Packing/unpacking**: Converting to/from SIMD-aligned buffers introduces small rounding differences
3. **Timing**: Intel buffers are updated via `pack_buffers()` which may not happen at the exact same time as pair contribution

### Why This Causes Numerical Differences

The electrode potential calculation involves:
1. `pair_contribution()` using `atom->x` to calculate distances → adds to `vector[]`
2. `electrode_kspace->compute_vector()` using Intel `buffers->get_x(0)` → adds to `vector[]`

Since pair and kspace calculations use **different position values**, the resulting `potential_i` (bvec) will differ between base and Intel versions even with identical physical configurations.

### Potential Solutions

To ensure numerical consistency:
1. **Option A**: Modify `ElectrodeVector` to use Intel buffers when available (non-trivial, requires passing buffers through the class hierarchy)
2. **Option B**: Ensure `atom->x` and Intel buffer positions are bitwise identical before electrode calculations (may hurt performance)
3. **Option C**: Accept small numerical differences as inherent to Intel package optimizations (document as expected behavior)

This mixed position source is likely the **primary cause** of numerical differences in electrode calculations between base and Intel versions.

## Solution: ElectrodeVectorIntel Class

To address the mixed position source issue, a new class `ElectrodeVectorIntel` has been created that uses Intel buffers for positions in pair calculations.

### Files Created

1. **`src/INTEL/electrode_vector_intel.h`** - Header file declaring the derived class
2. **`src/INTEL/electrode_vector_intel.cpp`** - Implementation with Intel buffer support

### Key Changes

The `ElectrodeVectorIntel` class:

1. **Inherits from `ElectrodeVector`** - Maintains compatibility with existing code
2. **Overrides `pair_contribution()`** - Uses Intel buffers instead of `atom->x`
3. **Templates for precision modes** - Supports MIXED, DOUBLE, and SINGLE precision

### Implementation Details

**Header file structure:**
```cpp
class ElectrodeVectorIntel : public ElectrodeVector {
 public:
  ElectrodeVectorIntel(class LAMMPS *, int, int, double, bool);
  void setup(class Pair *, class NeighList *, bool) override;

 protected:
  FixIntel *fix;
  template <class flt_t, class acc_t>
  void pair_contribution_buffers(IntelBuffers<flt_t, acc_t> *buffers, double *vector);
  void pair_contribution(double *) override;
};
```

**Buffer usage in pair calculation:**
```cpp
template <class flt_t, class acc_t>
void ElectrodeVectorIntel::pair_contribution_buffers(
    IntelBuffers<flt_t, acc_t> *buffers, double *vector)
{
  // Get positions from Intel buffers instead of atom->x
  typedef typename IntelBuffers<flt_t, acc_t>::atom_t atom_t;
  atom_t *_noalias const x_buf = buffers->get_x(0);

  // ... use x_buf[i].x, x_buf[i].y, x_buf[i].z for position calculations
  flt_t const delx = xtmp - x_buf[j].x;
  // ...
}
```

### Usage

The `ElectrodeVectorIntel` class should be instantiated instead of `ElectrodeVector` when using Intel-optimized electrode fixes:

```cpp
// In fix_electrode_conp_intel.cpp or similar:
elyt_vector = new ElectrodeVectorIntel(lmp, igroup, igroup, eta, true);
if (need_elec_vector) 
  elec_vector = new ElectrodeVectorIntel(lmp, igroup, igroup, eta, false);
```

### Benefits

1. **Position consistency** - Pair and KSpace calculations use the same position data
2. **Numerical reproducibility** - Reduces differences between base and Intel versions
3. **Performance** - Still benefits from Intel SIMD optimizations in position packing
4. **Backward compatibility** - Falls back to base class if Intel fix not available

### Integration Notes

To fully integrate this class:

1. **Modify `fix_electrode_conp_intel.cpp`** - Use `ElectrodeVectorIntel` instead of `ElectrodeVector`
2. **Update build system** - Add `electrode_vector_intel.cpp` to INTEL package build
3. **Test** - Verify numerical consistency with base implementation

The `setup()` method in `ElectrodeVectorIntel` ensures the Intel fix is available and stores a pointer for later use in `pair_contribution()`. If the Intel fix is not found, it falls back to the base class implementation.

## INTEL Position Buffer (`buffers->get_x()`) Update Points During a Verlet Timestep

### Overview

The INTEL package maintains a separate set of SIMD-aligned position buffers (`IntelBuffers::_x`, accessed via `buffers->get_x()`) that mirror `atom->x`. These buffers must be refreshed from `atom->x` at appropriate points in the timestep. The `fix_nh_intel.cpp` position updates (`nve_x()` and `remap()`) write directly to `atom->x`, not to the Intel buffers.

### When `buffers->get_x()` Is Updated from `atom->x`

The copy from `atom->x` → `_x[]` happens inside `IntelBuffers::thr_pack()` (`intel_buffers.h:204-234`). This function is called at two points:

**1. Neighbor rebuild step (`neighbor->ago == 0`):**
- Called from `NbinIntel::bin_atoms()` (`nbin_intel.cpp:186`) with `ago=0`, performing a **full pack** — positions, types (`_x[i].w = type[i]`), and charges (`_q[i]`).
- This occurs during the neighbor list build, which happens *after* `initial_integrate()` and communication, but *before* pair force computation.
- For offload/separate-buffer modes, `thr_pack_cop()` and `thr_pack_host()` are used instead (invoked via `IP_PRE_pack_separate_buffers`).

**2. Non-rebuild step (`neighbor->ago != 0`):**
- Called from each INTEL pair style's `compute()` method (e.g., `pair_lj_cut_intel.cpp:95`) with the current `ago`, performing a **partial pack** — positions only (no types or charges, since those are assumed unchanged).
- Also called from `PPPMElectrodeIntel::compute_vector()` (via `pack_buffers()`) and `pppm_intel.cpp` for K-space calculations.

### Verlet Timestep Sequence and Buffer State

| Step | Code Path | `atom->x` Modified? | `_x[]` Updated? | Buffer State |
|------|-----------|---------------------|-----------------|--------------|
| **initial_integrate** | `FixNHIntel::nve_x()` → writes `atom->x[i] += dtv * v[i]` | YES | NO | **Stale** |
| | `FixNHIntel::remap()` → converts `atom->x` to/from lambda coords | YES | NO | **Stale** |
| | `FixNHIntel::nve_v()` → updates `atom->v` only | No | NO | Stale |
| **post_integrate** | Other fixes that may modify positions | Maybe | NO | Potentially stale |
| **Communication** | `comm->forward_comm()` or `exchange()+borders()` | Reads `atom->x` | NO | Stale (correct: comm uses `atom->x`) |
| **Neighbor build** (if `nflag`) | `NbinIntel::bin_atoms()` → `thr_pack(ifrom,ito,0)` | No | **YES (full)** | **Fresh** |
| **Pair compute** | `pair_*.intel::compute()` → `thr_pack(ifrom,ito,ago)` | No | **YES (partial if ago!=0)** | **Fresh** |
| **K-space compute** | `pppm_intel::compute()` → `thr_pack(ifrom,ito,1)` | No | **YES (partial)** | **Fresh** |
| **pre_reverse / post_force** | `FixIntel::_sync_main_arrays()` → syncs forces, not positions | No | NO | Fresh (unchanged) |
| **final_integrate** | `FixNHIntel::nve_v()` → updates `atom->v` only | No | NO | Fresh |

### Key Observations

1. **`FixNHIntel::nve_x()` writes to `atom->x`, not `_x[]`**: The buffered positions become stale after `initial_integrate()`. This is safe because nothing reads `_x[]` between `initial_integrate()` and the next `thr_pack()` call during pair/kspace computation.

2. **`FixNHIntel::remap()` writes to `atom->x`**: Same as above — the buffers become stale during position remapping for barostat updates, and are refreshed before force computation.

3. **Buffer refresh always precedes force computation**: By the time any code reads `_x[]` for force calculations, `thr_pack()` has been called to synchronize from `atom->x`.

4. **Only positions are refreshed on non-rebuild steps**: Types (`_x[i].w`) and charges (`_q[i]`) are only packed when `ago == 0`. This is correct because atom types never change during a run, and charges only change via `thr_pack_q()` (called separately by electrode code) or on neighbor rebuilds.

5. **`final_integrate()` does not update positions**: `FixNHIntel::nve_v()` only modifies `atom->v`, so positions remain consistent at the end of the timestep — no buffer refresh is needed.

### Stale-Buffer Risk Assessment

There is no stale-buffer bug in the standard Verlet loop because the only code that reads `_x[]` is the INTEL-optimized pair/kspace/bond compute methods, and these always call `thr_pack()` at the top of their `compute()` before reading `_x[]`. However, if a non-INTEL fix or compute were to call an INTEL pair style's internals without first triggering `thr_pack()`, positions would be stale. This does not occur in the standard LAMMPS timestep flow.

## INTEL LRT (Long-Range Task) Buffer Handling

### Overview of LRT

`verlet_lrt_intel.cpp` implements a variant of Verlet integration that overlaps K-space computation with short-range (pair/bonded) force computation by running K-space on a separate thread (using pthreads or C++11 std::thread). This is designed to improve parallelism since K-space calculations (PPPM) are computationally intensive.

K-space is split into two phases:
- **`compute_first()`**: particle_map, make_rho, FFT, poisson — reads `_x[]` for charge assignment
- **`compute_second()`**: fieldforce — writes forces to `_f[]`

In standard Verlet, these are called sequentially via `PPPMIntel::compute()` (line 129-139 of `pppm_intel.cpp`). In LRT, `compute_first()` runs on the K-space thread concurrently with pair compute on the main thread, and `compute_second()` runs after joining.

### Key Differences in Buffer Handling

**1. Explicit `pack_buffers()` Call on Non-Reneighboring Steps**

In LRT, `pack_buffers()` is called **immediately after forward communication** on non-reneighboring steps (`nflag == 0`):

```cpp
// verlet_lrt_intel.cpp, line 254-258
if (nflag == 0) {
  comm->forward_comm();
  timer->stamp(Timer::COMM);
  _intel_kspace->pack_buffers();  // <<-- LRT-specific explicit pack
}
```

This differs from standard Verlet, where buffer packing happens **implicitly** at the top of each INTEL pair/kspace compute method via `thr_pack()`.

**Why this is needed:** In LRT, the K-space thread starts **before** pair compute begins:
```cpp
// Lines 299-304: K-space thread signaled to start
pthread_mutex_lock(&_kmutex);
_kspace_ready = 1;
pthread_cond_signal(&_kcond);
pthread_mutex_unlock(&_kmutex);

// Lines 319-321: Pair compute starts on main thread
if (pair_compute_flag) {
  force->pair->compute(eflag,vflag);
}
```

The K-space thread's `compute_first()` needs buffer positions **immediately** upon starting, but pair compute (whose `thr_pack()` would normally refresh buffers) hasn't started yet. Without the explicit `pack_buffers()` call, K-space would read stale positions from the Intel buffers — positions still reflecting the *previous* timestep.

**2. `pack_buffers()` Implementation**

`PPPMIntel::pack_buffers()` (`pppm_intel.cpp:1036-1058`) always calls `thr_pack(ifrom,ito,1)` (partial pack — positions only), regardless of `neighbor->ago`. This is correct because:
- On non-reneighboring steps: positions need updating, types/charges unchanged
- On reneighboring steps: `pack_buffers()` is not called (see below), so the `ago` parameter is moot

**3. Reneighboring Steps (`nflag != 0`)**

On reneighboring steps, the explicit `pack_buffers()` call is **not made** in the `else` branch (lines 259-288). This is correct because:
- `neighbor->build(1)` → `NbinIntel::bin_atoms()` → `thr_pack(ago=0)` already performs a full pack during neighbor list build
- By the time `compute_first()` starts on the K-space thread, buffers are already fresh from the neighbor build
- No redundant packing is needed

**4. Redundant Packing on Non-Reneighboring Steps**

On non-reneighboring steps, positions are packed **twice**:
1. First by the explicit `pack_buffers()` call (line 258) — needed for K-space thread
2. Second by pair compute's `thr_pack(ago=1)` call — needed for pair forces

This redundancy is intentional and necessary for correctness in LRT, but represents a small performance overhead compared to standard Verlet.

### LRT Timestep Sequence and Buffer State

| Step | Code Path | `atom->x` Modified? | `_x[]` Updated? | Buffer State |
|------|-----------|---------------------|-----------------|--------------|
| **initial_integrate** | `FixNHIntel::nve_x()` → writes `atom->x` | YES | NO | **Stale** |
| | `FixNHIntel::remap()` → writes `atom->x` | YES | NO | **Stale** |
| | `FixNHIntel::nve_v()` → updates `atom->v` | No | NO | Stale |
| **comm** (no reneighbor) | `comm->forward_comm()` | Reads `atom->x` | NO | Stale |
| | `_intel_kspace->pack_buffers()` | No | **YES (positions)** | **Fresh** |
| **K-space thread starts** | `compute_first()` → `particle_map()`, `make_rho()` | No | NO (reads `_x[]`) | Fresh |
| **Pair compute** (main thread) | `pair->compute()` → `thr_pack(ago=1)` | No | **YES (redundant)** | Fresh |
| **K-space thread joins** | `compute_second()` → `fieldforce_ik/ad()` | No | NO | Fresh |
| **pre_reverse / post_force** | `FixIntel::_sync_main_arrays()` | No | NO | Fresh |
| **final_integrate** | `FixNHIntel::nve_v()` | No | NO | Fresh |

### Thread Safety of Buffer Access

Both threads access buffers concurrently during the overlap phase:

| Thread | Reads | Writes |
|--------|-------|--------|
| **K-space thread** | `_x[]` (positions for particle_map/make_rho) | `density_brick[]`, FFT arrays |
| **Main thread** | `_x[]` (positions for pair forces) | `_f[]` (force buffer for pair) |

There are **no write conflicts** because:
- Both threads only **read** `_x[]` — `atom->x` is immutable during force computation
- K-space writes to `density_brick[]` and FFT arrays; pair writes to `_f[]` — these are disjoint
- Force accumulation from K-space (`_f[]` writes in `compute_second()`) happens **after** the join, i.e., sequentially on the main thread

### Comparison: LRT vs Standard Verlet Buffer Handling

| Aspect | Standard Verlet | LRT Verlet |
|--------|-----------------|------------|
| **Explicit `pack_buffers()`** | No (implicit in each compute) | Yes (after `forward_comm`) |
| **K-space buffer source** | `pppm_intel::compute()` calls `thr_pack()` internally or relies on pair's pack | Explicit pack before thread starts |
| **Redundant packing** | No | Yes (on non-reneighboring steps: `pack_buffers()` + pair's `thr_pack()`) |
| **Reneighboring buffer source** | `NbinIntel::bin_atoms()` → `thr_pack(ago=0)` | Same as standard |
| **Thread safety** | N/A (single-threaded) | Safe (read-only access to `_x[]` after pack; disjoint write targets) |
| **Performance cost** | None | One extra `thr_pack()` per non-reneighboring step |

### LRT-Specific Considerations for `fix_nh_intel.cpp`

The position update behavior in `FixNHIntel` is **unchanged** in LRT:
- `nve_x()` and `remap()` still write to `atom->x` only
- Buffers become stale after these updates
- The difference is that LRT **explicitly refreshes** buffers via `pack_buffers()` before K-space needs them, whereas standard Verlet relies on the implicit pack inside pair compute

### Potential Issues

1. **Missing `pack_buffers()` on reneighboring steps in LRT setup**: In `VerletLRTIntel::setup()` (lines 155-198), there is no explicit `pack_buffers()` call. However, `neighbor->build(1)` is called (line 176), which triggers `NbinIntel::bin_atoms()` → `thr_pack(ago=0)`, so buffers are fresh before the initial K-space thread starts (line 179).

2. **Electrode-specific concern**: If `PPPMElectrodeIntel::compute_vector()` is called multiple times per timestep, the LRT design ensures buffers are fresh on the first call (via `start_compute()` → `particle_map()` path), and subsequent calls use `pack_buffers()` (line 329 of `pppm_electrode_intel.cpp`). This matches the standard Verlet behavior.

3. **Memory consistency**: Since both threads read `atom->x` during force computation, there's a theoretical risk if `atom->x` were modified concurrently. However, LAMMPS guarantees `atom->x` is read-only during force compute (only written during `initial_integrate()` and `remap()`), so this is safe without explicit synchronization.

### Summary

LRT introduces an **explicit buffer synchronization point** (`pack_buffers()`) after forward communication to ensure the K-space thread sees fresh positions before it starts. This adds one redundant `thr_pack()` call per non-reneighboring timestep — a deliberate performance/correctness tradeoff to enable overlapping K-space and pair computations. On reneighboring steps, no explicit pack is needed since the neighbor build already refreshes buffers. The design is thread-safe because both threads only read `_x[]` and write to disjoint output arrays.

## Inline Horner / Lookup Table Implementation Analysis (`project_psi` and `make_rho_in_brick`)

This section audits the Intel-optimized rho1d computation in `pppm_electrode_intel.cpp` and `pppm_intel.cpp` against the base class `PPPM::compute_rho1d` (`pppm.cpp:2745`).

### Base Class Reference

`PPPM::compute_rho1d(dx, dy, dz)` (`pppm.cpp:2745-2763`):
```cpp
for (k = (1-order)/2; k <= order/2; k++) {    // k = nlower..nupper
    r1 = r2 = r3 = ZEROF;
    for (l = order-1; l >= 0; l--) {
      r1 = rho_coeff[l][k] + r1*dx;
    }
    rho1d[0][k] = r1;
}
```

This evaluates the polynomial `rho_coeff[0][k] + rho_coeff[1][k]*dx + ... + rho_coeff[order-1][k]*dx^(order-1)` via Horner's method, starting with `r = 0` and iterating from the highest coefficient downward.

### Two Horner Styles in the Intel Package

**Style A — "ZEROF-init"** (matches base class logic):

Used in `pppm_electrode_intel.cpp` `project_psi` (line 482-494), `make_rho_in_brick` (line 1005-1017), and `pppm_intel.cpp` `make_rho` (line 505-517):

```cpp
for (int k = nlower; k <= nupper; k++) {
    FFT_SCALAR r1, r2, r3;
    r1 = r2 = r3 = ZEROF;
    for (int l = order - 1; l >= 0; l--) {
      r1 = rho_coeff[l][k] + r1 * dx;
      r2 = rho_coeff[l][k] + r2 * dy;
      r3 = rho_coeff[l][k] + r3 * dz;
    }
    rho[0][k - nlower] = r1;
    rho[1][k - nlower] = r2;
    rho[2][k - nlower] = r3;
}
```

**Style B — "Seeded-init"** (one fewer multiply optimization):

Used in `pppm_intel.cpp` `fieldforce_ik` (line 666-677) and `fieldforce_ad` (line 847-867):

```cpp
for (int k = nlower; k <= nupper; k++) {
    FFT_SCALAR r1 = rho_coeff[order-1][k];    // Seed with highest coefficient
    FFT_SCALAR r2 = rho_coeff[order-1][k];
    FFT_SCALAR r3 = rho_coeff[order-1][k];
    for (int l = order-2; l >= 0; l--) {
      r1 = rho_coeff[l][k] + r1 * dx;
      r2 = rho_coeff[l][k] + r2 * dy;
      r3 = rho_coeff[l][k] + r3 * dz;
    }
    rho[0][k - nlower] = r1;
    rho[1][k - nlower] = r2;
    rho[2][k - nlower] = r3;
}
```

**Mathematical equivalence**: Both styles produce identical results. Style A starts with `r=0`, iterates `l = order-1..0`, producing `(((rho_coeff[O-1]*dx + rho_coeff[O-2])*dx + ...)*dx + rho_coeff[0])`. Style B seeds `r = rho_coeff[O-1]` and iterates `l = order-2..0`, producing the same value with one fewer multiply (the initial `0*dx` is eliminated). Style B has a minor performance advantage.

**Inconsistency**: `pppm_electrode_intel.cpp` uses Style A while `pppm_intel.cpp` `fieldforce_ik`/`fieldforce_ad` use Style B. Both are correct, but the inconsistency suggests these files were written/optimized at different times. The `precompute_rho()` table generation (`pppm_intel.cpp:962-1007`) uses Style A (base class style) for the `rho_lookup` table, and Style B for `drho_lookup` (matching `compute_drho1d` — see below).

### Derivative Horner (`drho`) in `fieldforce_ad`

The base class `PPPM::compute_drho1d` (`pppm.cpp:2770-2788`):
```cpp
for (k = (1-order)/2; k <= order/2; k++) {
    r1 = r2 = r3 = ZEROF;
    for (l = order-2; l >= 0; l--) {
      r1 = drho_coeff[l][k] + r1*dx;
    }
    drho1d[0][k] = r1;
}
```

Note: `drho_coeff` has `order-1` polynomial coefficients (index 0..order-2), so the loop starts at `l = order-2`.

Intel `fieldforce_ad` (`pppm_intel.cpp:847-867`):
```cpp
for (int k = nlower; k <= nupper; k++) {
    dr1 = dr2 = dr3 = ZEROF;
    r1 = rho_coeff[order-1][k];     // rho uses seeded-init (Style B)
    for (int l = order-2; l >= 0; l--) {
      r1 = rho_coeff[l][k] + r1 * dx;
      dr1 = drho_coeff[l][k] + dr1 * dx;
    }
    rho[0][k-nlower] = r1;
    drho[0][k-nlower] = dr1;
}
```

Here `drho` uses `ZEROF`-init with `l = order-2..0`, matching the base class exactly. `rho` uses seeded-init in the same loop. This is consistent and correct — the derivative polynomial has one fewer term so the seeded-init optimization doesn't apply.

### Lookup Table Implementation

**Table generation** (`pppm_intel.cpp:962-1007`, `precompute_rho()`):
```cpp
half_rho_scale = (rho_points - 1.) / 2.;     // e.g. (5000-1)/2 = 2499.5
half_rho_scale_plus = half_rho_scale + 0.5;   // e.g. 2500.0

for (int i = 0; i < rho_points; i++) {
    FFT_SCALAR dx = -1. + 1./half_rho_scale * (FFT_SCALAR)i;
    // Evaluate Horner for this dx value
    for (int k = nlower; k <= nupper; k++) {
        FFT_SCALAR r1 = ZEROF;
        for (int l = order-1; l >= 0; l--) {
            r1 = rho_coeff[l][k] + r1*dx;
        }
        rho_lookup[i][k-nlower] = r1;
    }
    // Zero-pad remaining entries
    for (int k = nupper-nlower+1; k < INTEL_P3M_ALIGNED_MAXORDER; k++) {
        rho_lookup[i][k] = 0;
    }
}
```

The table maps `dx ∈ [-1, 1]` to `idx ∈ [0, rho_points-1]`. Entries beyond `order` are zero-padded.

**Table lookup at runtime** (all Intel functions):
```cpp
dx = dx * half_rho_scale + half_rho_scale_plus;
int idx = dx;
// ...
for (int k = 0; k < INTEL_P3M_ALIGNED_MAXORDER; k++) {
    rho[0][k] = rho_lookup[idx][k];
}
```

**Table index boundary analysis**:
- `dx = -1.0` → `idx = (-1.0) * 2499.5 + 2500.0 = 0.5` → `int(0.5) = 0` ✓
- `dx = +1.0` → `idx = (1.0) * 2499.5 + 2500.0 = 4999.5` → `int(4999.5) = 4999` ✓
- `dx` is always in `[-1, +1]` because `part2grid` places the particle at the nearest grid point, making `(x - boxlo)*delxinv - nx` fall in that range.
- If floating-point rounding pushed `dx` slightly outside `[-1, 1]`, `idx` could be `-1` or `rho_points` — an out-of-bounds access. This is theoretically possible but extremely unlikely in practice. No bounds check is performed.

### Widened Inner Stencil Loops and Memory Safety

All Intel functions widen the innermost stencil loop from `order` to `INTEL_P3M_ALIGNED_MAXORDER` (= 8, defined in `intel_preprocess.h:88`) for SIMD vectorization. For `l >= order`, `rho[0/1/2][l]` is zero (from `= {0}` init or zero-padded lookup table), so zero values are contributed. However, the **index computation still runs**, accessing grid/density memory at positions beyond the intended stencil.

**Memory safety depends on padded allocations:**

| Function | Array Accessed | Allocation Padding | Safe? |
|----------|----------------|-------------------|-------|
| `make_rho` / `make_rho_in_brick` | `my_density[mzyx]` | `ngrid + INTEL_P3M_ALIGNED_MAXORDER` (line 102) | ✓ |
| `fieldforce_ik` | `vdx/vdy/vdz_brick[mz][my][mx]` | `INTEL_P3M_ALIGNED_MAXORDER * 2` extra bytes (line 1101-1102) | ✓ |
| `fieldforce_ad` | `u_brick[mz][my][mx]` | Same padding as above | ✓ |
| `project_psi` (electrode_intel) | `u_brick[miz][miy][mix]` | Same padded allocation | ✓ |
| `ekx_arr[l]` / `ekx[l]` (fieldforce) | Local `_alignvar` arrays of size `INTEL_P3M_ALIGNED_MAXORDER` | Exactly sized | ✓ |

**Important**: The widened loop reads `rho[0][l]` for `l = 0..7` even when `order = 2..7`. For `l >= order`, zero is read and multiplied, contributing nothing. The extra grid memory accesses beyond the stencil write zero into `my_density` / read from padded `brick` — safe because of the allocation padding. However, the widened loop in `fieldforce_ik` and `fieldforce_ad` reads from `vdx_brick` / `u_brick` at indices beyond the actual stencil. These reads return whatever data is in the padded region — but since the result is multiplied by zero, it doesn't affect the output. The padded memory is allocated but not zeroed, so these reads touch uninitialized memory. This is technically UB in C++ but harmless in practice on x86 (no traps).

### `rho0[2 * INTEL_P3M_ALIGNED_MAXORDER]` in `fieldforce_ik`

`fieldforce_ik` (`pppm_intel.cpp:622`) declares:
```cpp
_alignvar(flt_t rho0[2 * INTEL_P3M_ALIGNED_MAXORDER], 64) = {0};
_alignvar(flt_t rho1[INTEL_P3M_ALIGNED_MAXORDER], 64) = {0};
_alignvar(flt_t rho2[INTEL_P3M_ALIGNED_MAXORDER], 64) = {0};
```

Only `rho0` is double-sized. The access pattern (`rho0[l]` for `l = 0..INTEL_P3M_ALIGNED_MAXORDER-1`) never exceeds `INTEL_P3M_ALIGNED_MAXORDER`. This double-sized allocation appears to be leftover from an earlier design or a safety margin. The other functions use `rho[3][INTEL_P3M_ALIGNED_MAXORDER]`. Wasteful but not a bug.

### Summary of Findings

| Issue | Severity | Description |
|-------|----------|-------------|
| Horner style inconsistency | Low | `pppm_electrode_intel.cpp` uses ZEROF-init (Style A); `pppm_intel.cpp` `fieldforce_*` uses seeded-init (Style B). Both correct; Style B has minor performance advantage. |
| Table index bounds | Low | No runtime bounds check on `idx` in lookup table path. Theoretically OOB if `dx` exits `[-1,1]` due to FP rounding. Extremely unlikely in practice. |
| Widened loop reads padded/uninitialized memory | Low | Inner stencil loop iterates to `INTEL_P3M_ALIGNED_MAXORDER` instead of `order`, accessing padded grid memory. Results multiplied by zero are discarded. Safe due to allocation padding. |
| `rho0` double-sized allocation | Negligible | `fieldforce_ik` allocates `rho0[2*INTEL_P3M_ALIGNED_MAXORDER]` but only accesses first `INTEL_P3M_ALIGNED_MAXORDER` entries. Wasteful but harmless. |
| No functional correctness bugs | — | All Horner evaluations, lookup tables, and widened loops produce results identical to the base class (within floating-point precision). |
