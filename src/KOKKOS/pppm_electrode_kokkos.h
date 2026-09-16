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

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(pppm/electrode/kk,PPPMElectrodeKokkos<LMPDeviceType>);
KSpaceStyle(pppm/electrode/kk/device,PPPMElectrodeKokkos<LMPDeviceType>);
KSpaceStyle(pppm/electrode/kk/host,PPPMElectrodeKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PPPM_ELECTRODE_KOKKOS_H
#define LMP_PPPM_ELECTRODE_KOKKOS_H

#include "boundary_correction.h"
#include "electrode_kspace.h"
#include "pppm_kokkos.h"

namespace LAMMPS_NS {

// ELECTRODE kernels on top of the device PPPM machinery: identical
// density/FFT/Green pipeline as the base PPPMKokkos, but the density is
// accumulated from a mask-selected subset of atoms (the electrode charge
// group) into a separate brick, the potential (not the field) is the
// quantity interpolated back to the sensor atoms, and the A-matrix rows
// (twostep path) are contracted from the grid-potential field.
struct TagPPPMElectrode_make_rho_zero{};
struct TagPPPMElectrode_make_rho_atomic{};
struct TagPPPMElectrode_brick2fft{};
struct TagPPPMElectrode_poisson_pot1{};
struct TagPPPMElectrode_poisson_pot2{};
struct TagPPPMElectrode_poisson_pot3{};
struct TagPPPMElectrode_pack_fwd{};
struct TagPPPMElectrode_unpack_fwd{};
struct TagPPPMElectrode_project_psi{};
struct TagPPPMElectrode_project_psi_nele{};
struct TagPPPMElectrode_greens_pack{};

template<class DeviceType>
class PPPMElectrodeKokkos : public PPPMKokkos<DeviceType>, public ElectrodeKSpace {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef FFTArrayTypes<DeviceType> FFT_AT;
  typedef PPPMKokkos<DeviceType> Base;


  PPPMElectrodeKokkos(class LAMMPS *);
  ~PPPMElectrodeKokkos() override;
  void init() override;
  void compute(int, int) override;

  // ElectrodeKSpace interface
  void compute_vector(double *, int, int, bool) override;
  void compute_vector_corr(double *, int, int, bool) override;
  void compute_matrix(bigint *, double **, bool) override;
  void compute_matrix_corr(bigint *, double **) override;

  // Device-resident sensor path (stage K4): same pipeline as compute_vector
  // but the sensor interpolation scatters into d_out (nele entries, iele
  // order) through d_imap instead of a host nlocal buffer.
  void compute_vector_nele(typename AT::t_kkacc_1d &d_out, typename AT::t_int_1d &d_imap,
                           int groupbit, int source_grpbit, bool invert_source);

  // shared density->FFT->Green->psi pipeline; on return d_psi_brick holds
  // the interpolated-ready potential brick (ghost exchange completed)
  void vector_pipeline(int sensor_grpbit, int source_grpbit, bool invert_source);
  void compute_group_group(int, int, int) override;

  // grid ghost exchange for the psi potential brick (FORWARD_AD): the
  // base PPPMKokkos hooks only handle FORWARD_IK/IK_PERATOM, so both the
  // host-side (pack_forward_grid) and device-side (…_kokkos) hooks are
  // overridden below to move a single scalar per grid point
  void pack_forward_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, DAT::tdual_int_2d_lr &, int) override;
  void unpack_forward_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, int, DAT::tdual_int_2d_lr &, int) override;

  void pack_forward_grid(int, void *, int, int *) override;
  void unpack_forward_grid(int, void *, int, int *) override;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_make_rho_zero, const int&) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_make_rho_atomic, const int&) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_brick2fft, const int&) const;
  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_poisson_pot1, const int&) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_poisson_pot2, const int&) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_poisson_pot3, const int&) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_project_psi, const int&) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_project_psi_nele, const int&) const;


  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_pack_fwd, const int&) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_unpack_fwd, const int&) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMElectrode_greens_pack, const int&) const;

 protected:
  void allocate() override;
  void deallocate() override;

  // host matrix construction (mat_inv / mat_cg): one-time cost, runs on
  // the host for both onestep and twostep paths — private copies of the
  // PPPMElectrode algorithms, same pattern as PPPMElectrodeIntel
  void matrix_one_step(bigint *imat, double *greens_real, double **x_ele, double **matrix,
                       const int nmat, const FFT_SCALAR *rho1d_all, bool timer_flag);
  void matrix_two_step(bigint *imat, double *greens_real, double **x_ele, double **matrix,
                       const int nmat, bool timer_flag);
  void matrix_build_amesh(int dx, int dy, int dz, double *amesh, double *greens_real);
  template <int AXIS> void matrix_conv_axis(double *out, const double *in, int origin);


  void matrix_compute_rho_coeff();
  void matrix_compute_rho1d(const FFT_SCALAR &, const FFT_SCALAR &, const FFT_SCALAR &);

  // matrix weights (own copies; PPPMKokkos::allocate never builds the
  // host rho1d/rho_coeff arrays) — allocated on first matrix call
  FFT_SCALAR **m_rho1d = nullptr;      // [3][-nlower..nupper]
  FFT_SCALAR **m_rho_coeff = nullptr;  // [order][(1-order)/2..order/2]
  FFT_SCALAR **m_drho_coeff = nullptr;
  int m_order = -1;                    // allocation bookkeeping
  double *m_greens_real_cache = nullptr;
  bigint m_greens_cache_key = -1;
  double *m_conv_scratch1 = nullptr;
  double *m_conv_scratch2 = nullptr;
  double *m_gw_cache = nullptr;
  int m_gw_cache_nxyz = -1;
  typename FFT_AT::t_FFT_SCALAR_3d d_greens_real;    // device greens brick (brick frame)

  // electrode-specific grid state (device mirrors of the host
  // electrolyte_density_brick / electrolyte_density_fft buffers plus a
  // dedicated potential brick for the FORWARD_AD-style ghost exchange)
  typename FFT_AT::t_FFT_SCALAR_3d d_elec_density_brick;
  typename FFT_AT::t_FFT_SCALAR_1d d_elec_density_fft;
  FFT_DAT::tdual_FFT_SCALAR_3d d_psi_brick;    // DualView: host pack/unpack + device kernels
  typename FFT_AT::t_FFT_SCALAR_1d d_psi_fft;
  typename FFTArrayTypes<LMPHostType>::t_FFT_SCALAR_3d d_psi_brick_h;    // host mirror for pack/unpack

  // EW3DC boundary correction (SlabDipole / WireDipole / dummy), mirroring
  // the host PPPMElectrode::boundcorr — the base PPPMKokkos has no member
  class BoundaryCorrection *boundcorr = nullptr;

  // per-call kernel inputs
  typename AT::t_kkfloat_1d_randomread d_q;      // charges (group filter)
  typename AT::t_int_1d_randomread d_mask;
  typename FFT_AT::t_FFT_SCALAR_1d d_u_pot;      // sensor output (atomic add)

  // device-resident CG path (compute_vector_nele): scatter targets bound per call
  typename AT::t_kkacc_1d d_out_nele;            // nele-ordered accumulation vector
  typename AT::t_int_1d_randomread d_imap_nele;  // local atom -> electrode index (negative = skip)

  // sensor group / source group selection for the current call
  int sensor_grpbit_kk;
  int source_grpbit_kk;
  int invert_source_kk;    // 0 or 1

  // group-scoped part2grid (unused in K3 vector-only scope; kept for K5)
  typename AT::t_int_1d_3 d_ele_part2grid;

  // per-call host state
  bool elec_vector_setup;      // per-run one-time setup done
  bigint elec_part2grid_max;

  void start_compute_electrode();
};

}    // namespace LAMMPS_NS

#endif
#endif