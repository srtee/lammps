/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License as published by the Free Software
   Foundation.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Shern Tee (UQ)
------------------------------------------------------------------------- */

#include "atom_masks.h"
#include "domain.h"
#include "grid3d.h"
#include "pppm_electrode_kokkos.h"

#include "atom_kokkos.h"
#include "boundary_correction.h"
#include "comm.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "grid3d_kokkos.h"
#include "memory_kokkos.h"
#include "slab_dipole.h"
#include "update.h"
#include "wire_dipole.h"

#include <algorithm>
#include <unordered_map>

// mirrors of KSPACE/pppm.cpp file-local constants (not exported in pppm.h)
constexpr int ELEK_OFFSET = 16384;
constexpr FFT_SCALAR ELEK_ZEROF = 0.0;

namespace LAMMPS_NS {




template<class DeviceType>
PPPMElectrodeKokkos<DeviceType>::PPPMElectrodeKokkos(class LAMMPS *lmp) :
    PPPMKokkos<DeviceType>(lmp), ElectrodeKSpace(), elec_vector_setup(false),
    elec_part2grid_max(0)
{
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::init()
{
  Base::init();
}

/* ----------------------------------------------------------------------
   the base compute() only refreshes qsum/qsqsum when the atom count
   changes, but the electrode solvers change charges between calls —
   refresh unconditionally, then run the standard device pipeline
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::compute(int eflag, int vflag)
{
  this->qsum_qsq();
  this->natoms_original = this->atom->natoms;
  Base::compute(eflag, vflag);
}

template<class DeviceType>
PPPMElectrodeKokkos<DeviceType>::~PPPMElectrodeKokkos()
{
  if (this->copymode) return;
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::allocate()
{
  // EW3DC boundary correction, same dispatch as host PPPMElectrode::allocate
  delete boundcorr;
  if (this->slabflag == 1)
    boundcorr = new SlabDipole(this->lmp);
  else if (this->wireflag == 1)
    boundcorr = new WireDipole(this->lmp);
  else
    boundcorr = new BoundaryCorrection(this->lmp);

  Base::allocate();

  // electrode-specific device buffers, sized like the base density plane
  d_elec_density_brick = typename FFT_AT::t_FFT_SCALAR_3d(
      "pppm/electrode/kk:elec_density_brick", this->nzhi_out - this->nzlo_out + 1,
      this->nyhi_out - this->nylo_out + 1, this->nxhi_out - this->nxlo_out + 1);
  d_elec_density_fft = typename FFT_AT::t_FFT_SCALAR_1d(
      "pppm/electrode/kk:elec_density_fft", this->nfft_both);
  d_psi_brick = FFT_DAT::tdual_FFT_SCALAR_3d("pppm/electrode/kk:psi_brick",
      this->nzhi_out - this->nzlo_out + 1, this->nyhi_out - this->nylo_out + 1,
      this->nxhi_out - this->nxlo_out + 1);
  d_psi_fft = typename FFT_AT::t_FFT_SCALAR_1d("pppm/electrode/kk:psi_fft", this->nfft_both);
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::deallocate()
{
  // base deallocation is handled by PPPMKokkos::deallocate (invoked through
  // the PPPM destructor chain); drop the electrode views explicitly so the
  // device memory is returned as soon as the grid shrinks
  d_elec_density_brick = typename FFT_AT::t_FFT_SCALAR_3d();
  d_elec_density_fft = typename FFT_AT::t_FFT_SCALAR_1d();
  d_psi_brick = FFT_DAT::tdual_FFT_SCALAR_3d();
  d_psi_fft = typename FFT_AT::t_FFT_SCALAR_1d();
}
/* ----------------------------------------------------------------------
   per-timestep pre-computation for the electrode vector path: mirror of
   PPPMElectrode::start_compute, on device views
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::start_compute_electrode()
{
  // base setup: bind box, per-atom arrays, allocate grid on first call
  if (!elec_vector_setup) {
    Base::setup();
    elec_vector_setup = true;
  }

  this->boxlo[0] = this->domain->boxlo[0];
  this->boxlo[1] = this->domain->boxlo[1];
  this->boxlo[2] = this->domain->boxlo[2];
  this->boxlo_kk[0] = static_cast<KK_FLOAT>(this->boxlo[0]);
  this->boxlo_kk[1] = static_cast<KK_FLOAT>(this->boxlo[1]);
  this->boxlo_kk[2] = static_cast<KK_FLOAT>(this->boxlo[2]);

  this->atomKK->sync(this->execution_space, X_MASK | Q_MASK | MASK_MASK);

  const int nlocal = this->atomKK->nlocal;
  this->x = this->atomKK->k_x.template view<DeviceType>();
  this->q = this->atomKK->k_q.template view<DeviceType>();
  this->f = this->atomKK->k_f.template view<DeviceType>();
  d_q = this->atomKK->k_q.template view<DeviceType>();
  d_mask = this->atomKK->k_mask.template view<DeviceType>();

  // the base compute() allocates these lazily; the electrode vector path
  // runs before any compute() call, so mirror the allocation here
  if (this->atom->nmax > this->nmax) {
    this->nmax = this->atom->nmax;
    this->d_part2grid =
        typename AT::t_int_1d_3("pppm:part2grid", this->nmax);
    this->d_rho1d = typename FFT_AT::t_FFT_SCALAR_2d_3(
        "pppm:rho1d", this->nmax, this->order / 2 + this->order / 2 + 1);
  }

  // recompute the base per-atom mapping (shared d_part2grid + rho1d tables)
  Base::particle_map();
}

/* ----------------------------------------------------------------------
   ELECTRODE vector: psi_sensor = W_sensor . G . rho_source
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::compute_vector(double *vec, int sensor_grpbit,
                                                     int source_grpbit, bool invert_source)
{
  const int nlocal = this->atomKK->nlocal;

  start_compute_electrode();

  // kernel selection state
  sensor_grpbit_kk = sensor_grpbit;
  source_grpbit_kk = source_grpbit;
  invert_source_kk = invert_source ? 1 : 0;

  // grid sizes for the flat-index kernels
  this->numz_out = this->nzhi_out - this->nzlo_out + 1;
  this->numy_out = this->nyhi_out - this->nylo_out + 1;
  this->numx_out = this->nxhi_out - this->nxlo_out + 1;
  const int inum_out = this->numz_out * this->numy_out * this->numx_out;

  // inverse-volume weight (host mirrors of base members)
  this->delxinv_kk = static_cast<KK_FLOAT>(this->delxinv);
  this->delyinv_kk = static_cast<KK_FLOAT>(this->delyinv);
  this->delzinv_kk = static_cast<KK_FLOAT>(this->delzinv);
  this->delvolinv_kk = static_cast<KK_FLOAT>(this->delvolinv);

  this->copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_make_rho_zero>(0, inum_out), *this);
  this->copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_make_rho_atomic>(0, nlocal), *this);
  this->copymode = 0;

  // all procs communicate density values from their ghost cells to fully
  // sum contribution in their 3d bricks (same as base make_rho); the base
  // reverse hooks read d_density_brick, so swap pointers around the call
  auto d_density_brick_saved = this->d_density_brick;
  this->d_density_brick = d_elec_density_brick;
  this->gc->reverse_comm(Grid3d::KSPACE, this, this->REVERSE_RHO, 1, sizeof(FFT_SCALAR),
                         this->k_gc_buf1, this->k_gc_buf2, MPI_FFT_SCALAR);
  this->d_density_brick = d_density_brick_saved;
  // brick (in) -> fft pencil: copy the inner region and remap
  this->numx_inout = (this->nxhi_in - this->nxlo_out) - (this->nxlo_in - this->nxlo_out) + 1;
  this->numy_inout = (this->nyhi_in - this->nylo_out) - (this->nylo_in - this->nylo_out) + 1;
  this->numz_inout = (this->nzhi_in - this->nzlo_out) - (this->nzlo_in - this->nzlo_out) + 1;
  const int inum_inout = this->numz_inout * this->numy_inout * this->numx_inout;

  this->copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_brick2fft>(0, inum_inout), *this);
  this->copymode = 0;



  this->remap->perform(d_elec_density_fft, d_elec_density_fft, this->d_work1);

  // forward FFT (r -> k)
  this->copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_poisson_pot1>(0, this->nfft), *this);
  this->copymode = 0;

  this->fft1->compute(this->d_work1, this->d_work1, FFT3dKokkos<DeviceType>::FORWARD);

  // multiply by the modified Green's function (scaleinv folded in) in k-space
  this->copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_poisson_pot2>(0, this->nfft), *this);
  this->copymode = 0;

  // k -> r FFT of Green * rho = potential, lands in the in-brick frame
  this->fft2->compute(this->d_work2, this->d_work2, FFT3dKokkos<DeviceType>::BACKWARD);

  // copy the potential into the psi brick for ghost exchange; zero the
  // ghost region first so unwrapped cells can never carry garbage
  Kokkos::deep_copy(d_psi_brick.view<DeviceType>(), 0);
  this->copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_poisson_pot3>(0, inum_inout), *this);
  this->copymode = 0;
  d_psi_brick.template modify<DeviceType>();

  // ghost exchange of u_brick (FORWARD_AD, 1 scalar per point)
  this->gc->forward_comm(Grid3d::KSPACE, this, this->FORWARD_AD, 1, sizeof(FFT_SCALAR),
                         this->k_gc_buf1, this->k_gc_buf2, MPI_FFT_SCALAR);

  // interpolate u_brick back onto the sensor atoms (device)
  this->scaleinv = 1.0 / (static_cast<double>(this->nx_pppm) * this->ny_pppm * this->nz_pppm);
  this->scaleinv_kk = static_cast<KK_FLOAT>(this->scaleinv);
  sensor_grpbit_kk = sensor_grpbit;

  // zero the device-side sensor accumulation buffer, project on device,
  // then copy the contributions back to the host vector
  if (d_u_pot.extent(0) < (std::size_t) nlocal)
    d_u_pot = typename FFT_AT::t_FFT_SCALAR_1d("pppm/electrode/kk:u_pot", nlocal);
  Kokkos::deep_copy(d_u_pot, 0);

  this->copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_project_psi>(0, nlocal), *this);
  this->copymode = 0;
  Kokkos::fence();

  auto h_pot = Kokkos::create_mirror(Kokkos::HostSpace(), d_u_pot);
  Kokkos::deep_copy(h_pot, d_u_pot);
  for (int i = 0; i < nlocal; i++) vec[i] += h_pot(i);
}

/* ----------------------------------------------------------------------
   EW3DC boundary corrections: delegate to the host correction object
   (dipole/wire corrections are O(N) host reductions)
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::compute_vector_corr(double *vec, int sensor_grpbit,
                                                          int source_grpbit, bool invert_source)
{
  this->boundcorr->vector_corr(vec, sensor_grpbit, source_grpbit, invert_source);
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::compute_matrix(bigint *imat, double **matrix,
                                                     bool timer_flag)
{
  // Matrix construction on the host (one-time setup cost): mirror the
  // device Green's function, k->r transform through a temporary host FFT
  // plan, then run the PPPMElectrode contraction algorithms (private
  // copies; same pattern as PPPMElectrodeIntel).

  // matrix setup runs in setup_post_neighbor(), BEFORE force->kspace->setup();
  // unlike the host PPPM (whose init() calls setup() at the end), PPPMKokkos
  // only sets volume/del*inv/greensfn inside setup(), so the matrix path must
  // run it explicitly — start_compute_electrode() does that on first call,
  // then syncs host->device atom data (the run loop has not synced yet, so
  // stale device x would send particle_map() out of range on every rank)
  start_compute_electrode();

  const bigint cache_key = (bigint) this->nz_pppm * this->ny_pppm * this->nx_pppm;
  if (m_greens_real_cache == nullptr || m_greens_cache_key != cache_key) {
    delete[] m_greens_real_cache;
    m_greens_real_cache = new double[(std::size_t) this->nz_pppm * this->ny_pppm *
                                     this->nx_pppm]();

    // k -> r transform on the device, reusing the K3 vector-path machinery:
    // greensfn is already in FFT decomposition (compute_gf_ik in setup());
    // pack it as the FFT input, run the existing backward plan, and let the
    // existing pot3 kernel scatter the inner brick region. A temporary host
    // FFT3d plan corrupts its remap/MPI communicator at multi-rank, so the
    // device plan is the only safe route.
    const int nfft = (this->nxhi_fft - this->nxlo_fft + 1) *
        (this->nyhi_fft - this->nylo_fft + 1) * (this->nzhi_fft - this->nzlo_fft + 1);

    this->copymode = 1;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_greens_pack>(0, nfft), *this);

    this->fft2->compute(this->d_work2, this->d_work2, FFT3dKokkos<DeviceType>::BACKWARD);

    this->numx_inout = this->nxhi_in - this->nxlo_in + 1;
    this->numy_inout = this->nyhi_in - this->nylo_in + 1;
    this->numz_inout = this->nzhi_in - this->nzlo_in + 1;
    const int inum_inout = this->numz_inout * this->numy_inout * this->numx_inout;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_poisson_pot3>(0, inum_inout), *this);
    this->copymode = 0;
    Kokkos::fence();
    d_psi_brick.template modify<DeviceType>();
    d_psi_brick.template sync<Kokkos::HostSpace>();

    auto h_psi = d_psi_brick.view_host();
    for (int k = this->nzlo_in, n = 0; k <= this->nzhi_in; k++)
      for (int j = this->nylo_in; j <= this->nyhi_in; j++)
        for (int i = this->nxlo_in; i <= this->nxhi_in; i++) {
          m_greens_real_cache[(std::size_t) this->ny_pppm * this->nx_pppm * k +
                              this->nx_pppm * j + i] = h_psi(k - this->nzlo_out,
                                                             j - this->nylo_out,
                                                             i - this->nxlo_out);
        }
    MPI_Allreduce(MPI_IN_PLACE, m_greens_real_cache,
                  this->nz_pppm * this->ny_pppm * this->nx_pppm, MPI_DOUBLE, MPI_SUM,
                  this->world);
    m_greens_cache_key = cache_key;
  }

  const int nlocal = this->atom->nlocal;
  int nmat = std::count_if(&imat[0], &imat[nlocal], [](int x) { return x >= 0; });
  MPI_Allreduce(MPI_IN_PLACE, &nmat, 1, MPI_INT, MPI_SUM, this->world);
  // gather electrode coordinates (global order via Allreduce, as host)
  double **x_ele;
  this->memory->create(x_ele, nmat, 3, "pppm/electrode/kk:x_ele");
  memset(&(x_ele[0][0]), 0, (std::size_t) nmat * 3 * sizeof(double));
  auto x = this->atom->x;
  for (int i = 0; i < nlocal; i++) {
    const int ipos = imat[i];
    if (ipos < 0) continue;
    for (int dim = 0; dim < 3; dim++) x_ele[ipos][dim] = x[i][dim];
  }
  MPI_Allreduce(MPI_IN_PLACE, &(x_ele[0][0]), nmat * 3, MPI_DOUBLE, MPI_SUM, this->world);
  // ensure the host weight table exists and matches the current order
  if (m_order != this->order) {
    if (m_order > 0) {
      this->memory->destroy2d_offset(m_rho1d, -m_order / 2);
      this->memory->destroy2d_offset(m_rho_coeff, (1 - m_order) / 2);
      this->memory->destroy2d_offset(m_drho_coeff, (1 - m_order) / 2);
    }
    this->memory->create2d_offset(m_rho1d, 3, -this->order / 2, this->order / 2,
                                  "pppm/electrode/kk:rho1d");
    this->memory->create2d_offset(m_rho_coeff, this->order, (1 - this->order) / 2,
                                  this->order / 2, "pppm/electrode/kk:rho_coeff");
    this->memory->create2d_offset(m_drho_coeff, this->order, (1 - this->order) / 2,
                                  this->order / 2, "pppm/electrode/kk:drho_coeff");
    m_order = this->order;
    matrix_compute_rho_coeff();
  }

  // gather per-atom 1d weights of ALL electrode atoms in imat order
  const int order3 = 3 * this->order;
  std::vector<FFT_SCALAR> rho1d_all(nmat * order3);
  for (int i = 0; i < nlocal; i++) {
    const int ipos = imat[i];
    if (ipos < 0) continue;
    double *xi_ele = x_ele[ipos];
    const int nix = static_cast<int>((xi_ele[0] - this->boxlo[0]) * this->delxinv + this->shift) -
        ELEK_OFFSET;
    const int niy = static_cast<int>((xi_ele[1] - this->boxlo[1]) * this->delyinv + this->shift) -
        ELEK_OFFSET;
    const int niz = static_cast<int>((xi_ele[2] - this->boxlo[2]) * this->delzinv + this->shift) -
        ELEK_OFFSET;
    const FFT_SCALAR dix = nix + this->shiftone - (xi_ele[0] - this->boxlo[0]) * this->delxinv;
    const FFT_SCALAR diy = niy + this->shiftone - (xi_ele[1] - this->boxlo[1]) * this->delyinv;
    const FFT_SCALAR diz = niz + this->shiftone - (xi_ele[2] - this->boxlo[2]) * this->delzinv;
    matrix_compute_rho1d(dix, diy, diz);
    for (int dim = 0; dim < 3; dim++)
      for (int oi = 0; oi < this->order; oi++)
        rho1d_all[(std::size_t) ipos * order3 + dim * this->order + oi] =
            m_rho1d[dim][oi + this->nlower];
  }
  MPI_Allreduce(MPI_IN_PLACE, rho1d_all.data(), nmat * order3, MPI_FFT_SCALAR, MPI_SUM,
                this->world);

  if (this->conp_one_step)
    matrix_one_step(imat, m_greens_real_cache, x_ele, matrix, nmat, rho1d_all.data(), timer_flag);
  else
    matrix_two_step(imat, m_greens_real_cache, x_ele, matrix, nmat, timer_flag);
  this->memory->destroy(x_ele);
}


template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::compute_matrix_corr(bigint *imat, double **matrix)
{
  this->boundcorr->matrix_corr(imat, matrix);
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::compute_group_group(int, int, int)
{
  this->error->all(FLERR, "group group interaction not implemented in pppm/electrode/kk yet");
}

/* ----------------------------------------------------------------------
   device kernels
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PPPMElectrodeKokkos<DeviceType>::operator()(TagPPPMElectrode_make_rho_zero, const int &ii) const
{
  int iz = ii / (this->numy_out * this->numx_out);
  int iy = (ii - iz * this->numy_out * this->numx_out) / this->numx_out;
  int ix = ii - iz * this->numy_out * this->numx_out - iy * this->numx_out;
  d_elec_density_brick(iz, iy, ix) = 0;
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PPPMElectrodeKokkos<DeviceType>::operator()(TagPPPMElectrode_make_rho_atomic, const int &i) const
{
  // only atoms in the (possibly inverted) source group contribute
  const bool i_in_source = !!(d_mask(i) & source_grpbit_kk) != (invert_source_kk != 0);
  if (!i_in_source) return;

  Kokkos::View<FFT_SCALAR ***, Kokkos::LayoutRight, typename KKDevice<DeviceType>::value,
               Kokkos::MemoryTraits<Kokkos::Atomic | Kokkos::Unmanaged>>
      a_density_brick = d_elec_density_brick;

  const int nx = this->d_part2grid(i, 0);
  const int ny = this->d_part2grid(i, 1);
  const int nz = this->d_part2grid(i, 2);
  const FFT_SCALAR dx = static_cast<FFT_SCALAR>(static_cast<KK_FLOAT>(nx) + this->shiftone_kk -
                                                (this->x(i, 0) - this->boxlo_kk[0]) * this->delxinv_kk);
  const FFT_SCALAR dy = static_cast<FFT_SCALAR>(static_cast<KK_FLOAT>(ny) + this->shiftone_kk -
                                                (this->x(i, 1) - this->boxlo_kk[1]) * this->delyinv_kk);
  const FFT_SCALAR dz = static_cast<FFT_SCALAR>(static_cast<KK_FLOAT>(nz) + this->shiftone_kk -
                                                (this->x(i, 2) - this->boxlo_kk[2]) * this->delzinv_kk);

  const int nzl = nz - this->nzlo_out;
  const int nyl = ny - this->nylo_out;
  const int nxl = nx - this->nxlo_out;

  Base::compute_rho1d(i, dx, dy, dz);

  const FFT_SCALAR z0 = static_cast<FFT_SCALAR>(this->delvolinv_kk * d_q(i));
  for (int n = this->nlower; n <= this->nupper; n++) {
    const int mz = n + nzl;
    const FFT_SCALAR y0 = z0 * this->d_rho1d(i, n + Base::order / 2, 2);
    for (int m = this->nlower; m <= this->nupper; m++) {
      const int my = m + nyl;
      const FFT_SCALAR x0 = y0 * this->d_rho1d(i, m + Base::order / 2, 1);
      for (int l = this->nlower; l <= this->nupper; l++) {
        const int mx = l + nxl;
        a_density_brick(mz, my, mx) += x0 * this->d_rho1d(i, l + Base::order / 2, 0);
      }
    }
  }
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PPPMElectrodeKokkos<DeviceType>::operator()(TagPPPMElectrode_brick2fft, const int &ii) const
{
  int k = ii / (this->numy_inout * this->numx_inout);
  int j = (ii - k * this->numy_inout * this->numx_inout) / this->numx_inout;
  int i = ii - k * this->numy_inout * this->numx_inout - j * this->numx_inout;
  k += this->nzlo_in - this->nzlo_out;
  j += this->nylo_in - this->nylo_out;
  i += this->nxlo_in - this->nxlo_out;
  d_elec_density_fft[ii] = d_elec_density_brick(k, j, i);
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PPPMElectrodeKokkos<DeviceType>::operator()(TagPPPMElectrode_poisson_pot1, const int &i) const
{
  this->d_work1[2 * i] = d_elec_density_fft[i];
  this->d_work1[2 * i + 1] = 0;
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PPPMElectrodeKokkos<DeviceType>::operator()(TagPPPMElectrode_greens_pack, const int &i) const
{
  // matrix path: greensfn (real) is the FFT input, written straight into
  // the backward plan's buffer; imaginary part zero
  this->d_work2[2 * i] = static_cast<FFT_SCALAR>(this->d_greensfn[i]);
  this->d_work2[2 * i + 1] = 0;
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PPPMElectrodeKokkos<DeviceType>::operator()(TagPPPMElectrode_poisson_pot2, const int &i) const
{
  this->d_work2[2 * i] = this->d_work1[2 * i] * static_cast<FFT_SCALAR>(this->d_greensfn[i]);
  this->d_work2[2 * i + 1] = this->d_work1[2 * i + 1] * static_cast<FFT_SCALAR>(this->d_greensfn[i]);
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PPPMElectrodeKokkos<DeviceType>::operator()(TagPPPMElectrode_poisson_pot3, const int &ii) const
{
  int k = ii / (this->numy_inout * this->numx_inout);
  int j = (ii - k * this->numy_inout * this->numx_inout) / this->numx_inout;
  int i = ii - k * this->numy_inout * this->numx_inout - j * this->numx_inout;
  k += this->nzlo_in - this->nzlo_out;
  j += this->nylo_in - this->nylo_out;
  i += this->nxlo_in - this->nxlo_out;
  d_psi_brick.view<DeviceType>()(k, j, i) = this->d_work2[2 * ii];
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PPPMElectrodeKokkos<DeviceType>::operator()(TagPPPMElectrode_project_psi, const int &i) const
{
  if (!(d_mask(i) & sensor_grpbit_kk)) return;

  const int nix = this->d_part2grid(i, 0);
  const int niy = this->d_part2grid(i, 1);
  const int niz = this->d_part2grid(i, 2);
  const FFT_SCALAR dix = static_cast<FFT_SCALAR>(static_cast<KK_FLOAT>(nix) + this->shiftone_kk -
                                                 (this->x(i, 0) - this->boxlo_kk[0]) * this->delxinv_kk);
  const FFT_SCALAR diy = static_cast<FFT_SCALAR>(static_cast<KK_FLOAT>(niy) + this->shiftone_kk -
                                                 (this->x(i, 1) - this->boxlo_kk[1]) * this->delyinv_kk);
  const FFT_SCALAR diz = static_cast<FFT_SCALAR>(static_cast<KK_FLOAT>(niz) + this->shiftone_kk -
                                                 (this->x(i, 2) - this->boxlo_kk[2]) * this->delzinv_kk);

  Base::compute_rho1d(i, dix, diy, diz);

  FFT_SCALAR v = 0;
  for (int ni = this->nlower; ni <= this->nupper; ni++) {
    const int miz = ni + niz - this->nzlo_out;
    const FFT_SCALAR iz0 = this->d_rho1d(i, ni + Base::order / 2, 2);
    for (int mi = this->nlower; mi <= this->nupper; mi++) {
      const int miy = mi + niy - this->nylo_out;
      const FFT_SCALAR iy0 = iz0 * this->d_rho1d(i, mi + Base::order / 2, 1);
      for (int li = this->nlower; li <= this->nupper; li++) {
        const int mix = li + nix - this->nxlo_out;
        v += iy0 * this->d_rho1d(i, li + Base::order / 2, 0) * d_psi_brick.view<DeviceType>()(miz, miy, mix);
      }
    }
  }
  d_u_pot[i] += static_cast<KK_FLOAT>(v * this->scaleinv_kk);
}

/* ----------------------------------------------------------------------
   host-side grid ghost exchange for the potential brick (FORWARD_AD):
   the base KK hooks only handle FORWARD_IK (3 scalars); the electrode
   vector path exchanges a single scalar per grid point from d_psi_brick
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::pack_forward_grid(int flag, void *vbuf, int nlist, int *list)
{
  if (flag != this->FORWARD_AD) {
    Base::pack_forward_grid(flag, vbuf, nlist, list);
    return;
  }
  // psi potential brick ghost exchange: single scalar per grid point.
  // Device data is pulled to the host for the (small) ghost swap.
  d_psi_brick.template sync<Kokkos::HostSpace>();
  auto d_psi_brick_h = d_psi_brick.view_host();
  auto *buf = (FFT_SCALAR *) vbuf;
  const int ny = this->nyhi_out - this->nylo_out + 1;
  const int nx = this->nxhi_out - this->nxlo_out + 1;
  for (int i = 0; i < nlist; i++) {
    const int idx = list[i];
    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);
    buf[i] = d_psi_brick_h(iz, iy, ix);
  }
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::unpack_forward_grid(int flag, void *vbuf, int nlist, int *list)
{
  if (flag != this->FORWARD_AD) {
    Base::unpack_forward_grid(flag, vbuf, nlist, list);
    return;
  }
  d_psi_brick.template sync<Kokkos::HostSpace>();
  auto d_psi_brick_h = d_psi_brick.view_host();
  auto *buf = (FFT_SCALAR *) vbuf;
  const int ny = this->nyhi_out - this->nylo_out + 1;
  const int nx = this->nxhi_out - this->nxlo_out + 1;
  for (int i = 0; i < nlist; i++) {
    const int idx = list[i];
    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);
    d_psi_brick_h(iz, iy, ix) = buf[i];
  }
  d_psi_brick.template modify<Kokkos::HostSpace>();
  d_psi_brick.template sync<DeviceType>();
}

/* ----------------------------------------------------------------------
   device grid-comm hooks for the potential brick (FORWARD_AD): the base
   kernels only handle FORWARD_IK (3 scalars from vdx/vdy/vdz); these pack
   and unpack a single scalar per grid point from d_psi_brick
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::pack_forward_grid_kokkos(int flag, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf,
                                                               int nlist, DAT::tdual_int_2d_lr &k_list,
                                                               int index)
{
  typename AT::t_int_2d_lr_um d_list = k_list.view<DeviceType>();
  this->d_list_index = Kokkos::subview(d_list, index, Kokkos::ALL());
  this->d_buf = k_buf.view<DeviceType>();

  this->nx = (this->nxhi_out - this->nxlo_out + 1);
  this->ny = (this->nyhi_out - this->nylo_out + 1);

  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_pack_fwd>(0, nlist), *this);
  this->copymode = 0;
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PPPMElectrodeKokkos<DeviceType>::operator()(TagPPPMElectrode_pack_fwd, const int &i) const
{
  const double dlist = static_cast<double>(this->d_list_index[i]);
  const int iz = static_cast<int>(dlist / (this->nx * this->ny));
  const int iy = static_cast<int>((dlist - iz * this->nx * this->ny) / this->nx);
  const int ix = this->d_list_index[i] - iz * this->nx * this->ny - iy * this->nx;
  this->d_buf[i] = d_psi_brick.view<DeviceType>()(iz, iy, ix);
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::unpack_forward_grid_kokkos(int flag, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf,
                                                                 int offset, int nlist,
                                                                 DAT::tdual_int_2d_lr &k_list, int index)
{
  typename AT::t_int_2d_lr_um d_list = k_list.view<DeviceType>();
  this->d_list_index = Kokkos::subview(d_list, index, Kokkos::ALL());
  this->d_buf = k_buf.view<DeviceType>();
  this->unpack_offset = offset;

  this->nx = (this->nxhi_out - this->nxlo_out + 1);
  this->ny = (this->nyhi_out - this->nylo_out + 1);

  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMElectrode_unpack_fwd>(0, nlist), *this);
  this->copymode = 0;
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PPPMElectrodeKokkos<DeviceType>::operator()(TagPPPMElectrode_unpack_fwd, const int &i) const
{
  const double dlist = static_cast<double>(this->d_list_index[i]);
  const int iz = static_cast<int>(dlist / (this->nx * this->ny));
  const int iy = static_cast<int>((dlist - iz * this->nx * this->ny) / this->nx);
  const int ix = this->d_list_index[i] - iz * this->nx * this->ny - iy * this->nx;
  d_psi_brick.view<DeviceType>()(iz, iy, ix) = this->d_buf[i + this->unpack_offset];
}


/* ----------------------------------------------------------------------
   host matrix weights: verbatim copies of PPPM::compute_rho_coeff and
   PPPM::compute_rho1d (PPPMKokkos::allocate never builds the host tables)
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::matrix_compute_rho_coeff()
{
  int j, k, l, m;
  FFT_SCALAR s;

  FFT_SCALAR **a;
  this->memory->create2d_offset(a, m_order, -m_order, m_order, "pppm/electrode/kk:a");

  for (k = -m_order; k <= m_order; k++)
    for (l = 0; l < m_order; l++)
      a[l][k] = 0.0;

  a[0][0] = 1.0;
  for (j = 1; j < m_order; j++) {
    for (k = -j; k <= j; k += 2) {
      s = 0.0;
      for (l = 0; l < j; l++) {
        a[l + 1][k] = (a[l][k + 1] - a[l][k - 1]) / (l + 1);
#ifdef FFT_SINGLE
        s += powf(0.5, (float) l + 1) *
            (a[l][k - 1] + powf(-1.0, (float) l) * a[l][k + 1]) / (l + 1);
#else
        s += pow(0.5, (double) l + 1) *
            (a[l][k - 1] + pow(-1.0, (double) l) * a[l][k + 1]) / (l + 1);
#endif
      }
      a[0][k] = s;
    }
  }

  m = (1 - m_order) / 2;
  for (k = -(m_order - 1); k < m_order; k += 2) {
    for (l = 0; l < m_order; l++)
      m_rho_coeff[l][m] = a[l][k];
    for (l = 1; l < m_order; l++)
      m_drho_coeff[l - 1][m] = l * a[l][k];
    m++;
  }

  this->memory->destroy2d_offset(a, -m_order);
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::matrix_compute_rho1d(const FFT_SCALAR &dx,
                                                           const FFT_SCALAR &dy,
                                                           const FFT_SCALAR &dz)
{
  int k, l;
  FFT_SCALAR r1, r2, r3;

  for (k = (1 - m_order) / 2; k <= m_order / 2; k++) {
    r1 = r2 = r3 = ELEK_ZEROF;

    for (l = m_order - 1; l >= 0; l--) {
      r1 = m_rho_coeff[l][k] + r1 * dx;
      r2 = m_rho_coeff[l][k] + r2 * dy;
      r3 = m_rho_coeff[l][k] + r3 * dz;
    }
    m_rho1d[0][k] = r1;
    m_rho1d[1][k] = r2;
    m_rho1d[2][k] = r3;
  }
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::matrix_build_amesh(const int dx, const int dy,
                                                         const int dz, double *amesh,
                                                         double *const greens_real)
{
  auto fmod = [](int x, int n) {    // fast unsigned mod
    int r = abs(x);
    while (r >= n) r -= n;
    return r;
  };
  int ind_amesh = 0;

  for (int iz = 0; iz < this->order; iz++)
    for (int jz = 0; jz < this->order; jz++) {
      const int mz = fmod(dz + jz - iz, this->nz_pppm) * this->nx_pppm * this->ny_pppm;
      for (int iy = 0; iy < this->order; iy++)
        for (int jy = 0; jy < this->order; jy++) {
          const int my = fmod(dy + jy - iy, this->ny_pppm) * this->nx_pppm;
          for (int ix = 0; ix < this->order; ix++)
            for (int jx = 0; jx < this->order; jx++) {
              const int mx = fmod(dx + jx - ix, this->nx_pppm);
              amesh[ind_amesh] = greens_real[mz + my + mx];
              ind_amesh++;
            }
        }
    }
}

template<class DeviceType>
template <int AXIS>
void PPPMElectrodeKokkos<DeviceType>::matrix_conv_axis(double *out, const double *in, int origin)
{
  const int nx_ele = this->nxhi_out - this->nxlo_out + 1;
  const int ny_ele = this->nyhi_out - this->nylo_out + 1;
  const int nz_ele = this->nzhi_out - this->nzlo_out + 1;
  const FFT_SCALAR *w = m_rho1d[AXIS] + this->nlower;
  auto fmod = [](int x, int n) {    // fast unsigned mod
    int r = abs(x);
    while (r >= n) r -= n;
    return r;
  };

  if (AXIS == 0) {
    // pass 1: input is greens_real with its native kernel pitch
    const size_t in_pz = (size_t) this->nx_pppm * this->ny_pppm, in_py = (size_t) this->nx_pppm;
    for (int mz = 0; mz < nz_ele; mz++)
      for (int my = 0; my < ny_ele; my++) {
        const double *inl = in + (size_t) mz * in_pz + (size_t) my * in_py;
        double *outl = out + (size_t) mz * ny_ele * nx_ele + (size_t) my * nx_ele;
        for (int mx = 0; mx < nx_ele; mx++) {
          double acc = 0.;
          for (int j = 0; j < this->order; j++)
            acc += w[j] * inl[fmod(mx - origin - j - this->nlower, this->nx_pppm)];
          outl[mx] = acc;
        }
      }
  } else if (AXIS == 1) {
    for (int mz = 0; mz < nz_ele; mz++) {
      const double *inz = in + (size_t) mz * ny_ele * nx_ele;
      double *outz = out + (size_t) mz * ny_ele * nx_ele;
      for (int my = 0; my < ny_ele; my++) {
        double *outl = outz + (size_t) my * nx_ele;
        for (int j = 0; j < this->order; j++) {
          const double *inl = inz + (size_t) fmod(my - origin - j - this->nlower, this->ny_pppm) * nx_ele;
          const double wj = w[j];
          for (int mx = 0; mx < nx_ele; mx++) outl[mx] += wj * inl[mx];
        }
      }
    }
  } else {
    for (int mz = 0; mz < nz_ele; mz++) {
      double *outz = out + (size_t) mz * ny_ele * nx_ele;
      for (int j = 0; j < this->order; j++) {
        const double *inz =
            in + (size_t) fmod(mz - origin - j - this->nlower, this->nz_pppm) * ny_ele * nx_ele;
        const double wj = w[j];
        for (int my = 0; my < ny_ele; my++) {
          const double *inl = inz + (size_t) my * nx_ele;
          double *outl = outz + (size_t) my * nx_ele;
          for (int mx = 0; mx < nx_ele; mx++) outl[mx] += wj * inl[mx];
        }
      }
    }
  }
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::matrix_one_step(bigint *imat, double *greens_real,
                                                      double **x_ele, double **matrix,
                                                      const int nmat,
                                                      const FFT_SCALAR *rho1d_all,
                                                      bool timer_flag)
{
  // verbatim port of PPPMElectrode::one_step_multiplication (full-row
  // scheme; no checkerboard parity skip, no mirror writes)
  const int nlocal = this->atom->nlocal;
  MPI_Barrier(this->world);
  double step1_time = MPI_Wtime();

  std::vector<int> i_list;
  for (int i = 0; i < nlocal; i++) {
    const int ipos = imat[i];
    if (ipos >= 0) i_list.push_back(ipos);
  }
  std::sort(i_list.begin(), i_list.end());

  std::unordered_map<std::int64_t, std::vector<double>> amesh_cache;
  amesh_cache.reserve(2 * nmat);
  auto offset_key = [&](int dx, int dy, int dz) {
    return ((std::int64_t) (dz + 4 * this->nz_pppm) * (8 * (std::int64_t) this->ny_pppm) +
            (std::int64_t) (dy + 4 * this->ny_pppm)) * (8 * (std::int64_t) this->nx_pppm) +
        (dx + 4 * this->nx_pppm);
  };
  const int order2 = this->order * this->order;
  const int order6 = order2 * order2 * order2;
  const int order3 = 3 * this->order;
  double *amesh;
  this->memory->create(amesh, order6, "pppm/electrode/kk:amesh");
  for (const int ipos : i_list) {
    double *_noalias xi_ele = x_ele[ipos];
    int nix = static_cast<int>((xi_ele[0] - this->boxlo[0]) * this->delxinv + this->shift) -
        ELEK_OFFSET;
    int niy = static_cast<int>((xi_ele[1] - this->boxlo[1]) * this->delyinv + this->shift) -
        ELEK_OFFSET;
    int niz = static_cast<int>((xi_ele[2] - this->boxlo[2]) * this->delzinv + this->shift) -
        ELEK_OFFSET;
    FFT_SCALAR const dix = nix + this->shiftone - (xi_ele[0] - this->boxlo[0]) * this->delxinv;
    FFT_SCALAR const diy = niy + this->shiftone - (xi_ele[1] - this->boxlo[1]) * this->delyinv;
    FFT_SCALAR const diz = niz + this->shiftone - (xi_ele[2] - this->boxlo[2]) * this->delzinv;
    matrix_compute_rho1d(dix, diy, diz);
    const FFT_SCALAR *rho_i[3] = {&rho1d_all[ipos * order3],
                                  &rho1d_all[ipos * order3 + this->order],
                                  &rho1d_all[ipos * order3 + 2 * this->order]};
    int njx = -1;
    int njy = -1;
    int njz = -1;    // force initial build_amesh
    for (int jpos = 0; jpos < nmat; jpos++) {
      double *_noalias xj_ele = x_ele[jpos];
      const int njx_new = static_cast<int>((xj_ele[0] - this->boxlo[0]) * this->delxinv + this->shift) -
          ELEK_OFFSET;
      const int njy_new = static_cast<int>((xj_ele[1] - this->boxlo[1]) * this->delyinv + this->shift) -
          ELEK_OFFSET;
      const int njz_new = static_cast<int>((xj_ele[2] - this->boxlo[2]) * this->delzinv + this->shift) -
          ELEK_OFFSET;
      if (njx != njx_new || njy != njy_new || njz != njz_new) {
        njx = njx_new;
        njy = njy_new;
        njz = njz_new;
        const std::int64_t key = offset_key(njx - nix, njy - niy, njz - niz);
        auto it = amesh_cache.find(key);
        if (it == amesh_cache.end()) {
          matrix_build_amesh(njx - nix, njy - niy, njz - niz, amesh, greens_real);
          amesh_cache.emplace(key, std::vector<double>(amesh, amesh + order6));
        } else {
          std::copy(it->second.begin(), it->second.end(), amesh);
        }
      }
      const FFT_SCALAR *rho_j[3] = {&rho1d_all[jpos * order3],
                                    &rho1d_all[jpos * order3 + this->order],
                                    &rho1d_all[jpos * order3 + 2 * this->order]};
      double aij = 0.;
      int ind_amesh = 0;
      for (int ni = 0; ni < this->order; ni++) {
        FFT_SCALAR const iz0 = rho_i[2][ni];
        for (int nj = 0; nj < this->order; nj++) {
          FFT_SCALAR const jz0 = rho_j[2][nj];
          for (int mi = 0; mi < this->order; mi++) {
            FFT_SCALAR const iy0 = iz0 * rho_i[1][mi];
            for (int mj = 0; mj < this->order; mj++) {
              FFT_SCALAR const jy0 = jz0 * rho_j[1][mj];
              for (int li = 0; li < this->order; li++) {
                FFT_SCALAR const ix0 = iy0 * rho_i[0][li];
                double aij_xscan = 0.;
                for (int lj = 0; lj < this->order; lj++) {
                  aij_xscan += (double) amesh[ind_amesh] * rho_j[0][lj];
                  ind_amesh++;
                }
                aij += (double) ix0 * jy0 * aij_xscan;
              }
            }
          }
        }
      }
      matrix[ipos][jpos] += aij / this->volume;
    }
  }
  this->memory->destroy(amesh);
  MPI_Barrier(this->world);
  if (timer_flag && (this->comm->me == 0))
    utils::logmesg(this->lmp, "Single step time: {:.4g} s\n", MPI_Wtime() - step1_time);
}

template<class DeviceType>
void PPPMElectrodeKokkos<DeviceType>::matrix_two_step(bigint *imat, double *greens_real,
                                                      double **x_ele, double **matrix,
                                                      const int nmat, bool timer_flag)
{
  // verbatim port of PPPMElectrode::two_step_multiplication (fused
  // j-outer convolution + contraction; no persistent N x nxyz cache)
  const int nlocal = this->atom->nlocal;
  MPI_Barrier(this->world);
  double start_time = MPI_Wtime();
  int nx_ele = this->nxhi_out - this->nxlo_out + 1;
  int ny_ele = this->nyhi_out - this->nylo_out + 1;
  int nz_ele = this->nzhi_out - this->nzlo_out + 1;
  int nxyz = nx_ele * ny_ele * nz_ele;

  if (m_gw_cache_nxyz < nxyz) {
    this->memory->destroy(m_gw_cache);
    this->memory->create(m_gw_cache, nxyz, "pppm/electrode/kk:gw_cache");
    m_gw_cache_nxyz = nxyz;
  }
  double *gw = m_gw_cache;

  if (m_conv_scratch1 == nullptr) {
    this->memory->create(m_conv_scratch1, nxyz, "pppm/electrode/kk:conv_scratch1");
    this->memory->create(m_conv_scratch2, nxyz, "pppm/electrode/kk:conv_scratch2");
  }

  std::vector<int> i_list;
  std::vector<int> i_atom;
  int nfrag = 0;
  for (int i = 0; i < nlocal; i++) {
    const int ipos = imat[i];
    if (ipos >= 0) {
      i_list.push_back(ipos);
      i_atom.push_back(i);
    }
  }
  std::vector<int> order_(nfrag = (int) i_list.size());
  for (int k = 0; k < nfrag; k++) order_[k] = k;
  std::sort(order_.begin(), order_.end(),
            [&](int a, int b) { return i_list[a] < i_list[b]; });
  std::vector<int> si_list(nfrag), si_atom(nfrag);
  for (int k = 0; k < nfrag; k++) {
    si_list[k] = i_list[order_[k]];
    si_atom[k] = i_atom[order_[k]];
  }
  i_list = si_list;
  i_atom = si_atom;
  const int order3 = 3 * this->order;
  std::vector<FFT_SCALAR> rho_i(nfrag * order3);
  std::vector<int> grid_origin(nfrag * 3);
  for (int ifrag = 0; ifrag < nfrag; ifrag++) {
    const int ipos = i_list[ifrag];
    const int ia = i_atom[ifrag];
    double *_noalias xi_ele = x_ele[ipos];
    const int nix = static_cast<int>((xi_ele[0] - this->boxlo[0]) * this->delxinv + this->shift) -
        ELEK_OFFSET;
    const int niy = static_cast<int>((xi_ele[1] - this->boxlo[1]) * this->delyinv + this->shift) -
        ELEK_OFFSET;
    const int niz = static_cast<int>((xi_ele[2] - this->boxlo[2]) * this->delzinv + this->shift) -
        ELEK_OFFSET;
    matrix_compute_rho1d(nix + this->shiftone - (xi_ele[0] - this->boxlo[0]) * this->delxinv,
                         niy + this->shiftone - (xi_ele[1] - this->boxlo[1]) * this->delyinv,
                         niz + this->shiftone - (xi_ele[2] - this->boxlo[2]) * this->delzinv);
    for (int dim = 0; dim < 3; dim++)
      for (int oi = 0; oi < this->order; oi++)
        rho_i[ifrag * order3 + dim * this->order + oi] = m_rho1d[dim][oi + this->nlower];
    grid_origin[ifrag * 3 + 0] = nix;
    grid_origin[ifrag * 3 + 1] = niy;
    grid_origin[ifrag * 3 + 2] = niz;
  }

  for (int jpos = 0; jpos < nmat; jpos++) {
    double *_noalias xj_ele = x_ele[jpos];
    const int njx = static_cast<int>((xj_ele[0] - this->boxlo[0]) * this->delxinv + this->shift) -
        ELEK_OFFSET;
    const int njy = static_cast<int>((xj_ele[1] - this->boxlo[1]) * this->delyinv + this->shift) -
        ELEK_OFFSET;
    const int njz = static_cast<int>((xj_ele[2] - this->boxlo[2]) * this->delzinv + this->shift) -
        ELEK_OFFSET;
    matrix_compute_rho1d(njx + this->shiftone - (xj_ele[0] - this->boxlo[0]) * this->delxinv,
                         njy + this->shiftone - (xj_ele[1] - this->boxlo[1]) * this->delyinv,
                         njz + this->shiftone - (xj_ele[2] - this->boxlo[2]) * this->delzinv);

    memset(m_conv_scratch2, 0, (std::size_t) nxyz * sizeof(double));
    memset(gw, 0, (std::size_t) nxyz * sizeof(double));
    matrix_conv_axis<0>(m_conv_scratch1, greens_real, njx - this->nxlo_out);
    matrix_conv_axis<1>(m_conv_scratch2, m_conv_scratch1, njy - this->nylo_out);
    matrix_conv_axis<2>(gw, m_conv_scratch2, njz - this->nzlo_out);

    for (int ifrag = 0; ifrag < nfrag; ifrag++) {
      const int ipos = i_list[ifrag];
      const FFT_SCALAR *ri = &rho_i[ifrag * order3];
      double aij = 0.;
      for (int ni = 0; ni < this->order; ni++) {
        double iz0 = ri[2 * this->order + ni];
        int miz = (ni + this->nlower) + grid_origin[ifrag * 3 + 2];
        for (int mi = 0; mi < this->order; mi++) {
          double iy0 = iz0 * ri[1 * this->order + mi];
          int miy = (mi + this->nlower) + grid_origin[ifrag * 3 + 1];
          for (int li = 0; li < this->order; li++) {
            int mix = (li + this->nlower) + grid_origin[ifrag * 3 + 0];
            double ix0 = iy0 * ri[0 * this->order + li];
            int miz0 = miz - this->nzlo_out;
            int miy0 = miy - this->nylo_out;
            int mix0 = mix - this->nxlo_out;
            aij += ix0 * gw[nx_ele * ny_ele * miz0 + nx_ele * miy0 + mix0];
          }
        }
      }
      matrix[ipos][jpos] += aij / this->volume;
    }
  }
  MPI_Barrier(this->world);
  if (timer_flag && (this->comm->me == 0))
    utils::logmesg(this->lmp, "Two step time: {:.4g} s\n", MPI_Wtime() - start_time);
}

/* ----------------------------------------------------------------------
   explicit template instantiations
------------------------------------------------------------------------- */

template class PPPMElectrodeKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PPPMElectrodeKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS