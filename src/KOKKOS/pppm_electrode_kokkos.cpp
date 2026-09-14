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
#include "force.h"
#include "grid3d_kokkos.h"
#include "memory_kokkos.h"
#include "slab_dipole.h"
#include "update.h"
#include "wire_dipole.h"

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
  {
    // TEMP probe: checksum device charges before the scatter
    auto kq = this->atomKK->k_q.template view<DeviceType>();
    auto h_q = Kokkos::create_mirror(Kokkos::HostSpace(), kq);
    Kokkos::deep_copy(h_q, kq);
    double qsum = 0.0, qsum2 = 0.0;
    for (int i = 0; i < nlocal; i++) { qsum += h_q(i); qsum2 += h_q(i) * h_q(i); }
    if (this->comm->me == 0)
      utils::logmesg(this->lmp, fmt::format("DBGQ qsum={:.12g} qsum2={:.12g}\n", qsum, qsum2));
  }
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
  {
    auto h_r = Kokkos::create_mirror(Kokkos::HostSpace(), d_elec_density_brick);
    Kokkos::deep_copy(h_r, d_elec_density_brick);
    double rsum = 0.0;
    for (int iz = 0; iz < h_r.extent(0); iz++)
      for (int iy = 0; iy < h_r.extent(1); iy++)
        for (int ix = 0; ix < h_r.extent(2); ix++) rsum += h_r(iz, iy, ix);
    if (this->comm->me == 0)
      utils::logmesg(this->lmp, fmt::format("DBGRHO elec density brick sum = {:.12g} (nzlo_out={} nzhi_out={})\n", rsum, this->nzlo_out, this->nzhi_out));
  }
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

  {
    // TEMP probe: checksum the full psi brick after ghost exchange
    auto h_psi = Kokkos::create_mirror(Kokkos::HostSpace(), d_psi_brick.view<DeviceType>());
    Kokkos::deep_copy(h_psi, d_psi_brick.view<DeviceType>());
    double psum = 0.0, psum2 = 0.0;
    for (int kz = 0; kz < h_psi.extent(0); kz++)
      for (int ky = 0; ky < h_psi.extent(1); ky++)
        for (int kx = 0; kx < h_psi.extent(2); kx++) {
          double v = h_psi(kz, ky, kx);
          psum += v; psum2 += v * v;
        }
    if (this->comm->me == 0)
      utils::logmesg(this->lmp, fmt::format("DBGPSI psum={:.12g} psum2={:.12g}\n", psum, psum2));
  }
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
  {
    double usum = 0.0, usum2 = 0.0;
    for (int i = 0; i < nlocal; i++) { usum += h_pot(i); usum2 += h_pot(i) * h_pot(i); }
    if (this->comm->me == 0)
      utils::logmesg(this->lmp, fmt::format("DBGU usum={:.12g} usum2={:.12g}\n", usum, usum2));
  }
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
void PPPMElectrodeKokkos<DeviceType>::compute_matrix(bigint *imat, double **matrix, bool)
{
  this->error->all(FLERR, "compute_matrix (algo mat_inv/mat_cg) not implemented in "
                          "pppm/electrode/kk yet (K5 stage); use algo cg with pair mode");
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
   explicit template instantiations
------------------------------------------------------------------------- */

template class PPPMElectrodeKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PPPMElectrodeKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS