/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef PAIR_CLASS

// clang-format off
PairStyle(lj/cut/coul/long/gauss/kk,PairLJCutCoulLongGaussKokkos<LMPDeviceType>);
PairStyle(lj/cut/coul/long/gauss/kk/device,PairLJCutCoulLongGaussKokkos<LMPDeviceType>);
PairStyle(lj/cut/coul/long/gauss/kk/host,PairLJCutCoulLongGaussKokkos<LMPHostType>);
// clang-format on

#else

// clang-format off
#ifndef LMP_PAIR_LJ_CUT_COUL_LONG_GAUSS_KOKKOS_H
#define LMP_PAIR_LJ_CUT_COUL_LONG_GAUSS_KOKKOS_H

#include "kokkos_type.h"
#include "pair_kokkos.h"    // EV_FLOAT, KK_FLOAT, SBBITS, AtomMask enums, pair_virial_fdotr_compute
#include "pair_lj_cut_coul_long_gauss.h"
#include "electrode_pair_kokkos.h"

namespace LAMMPS_NS {

class NeighList;
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairGaussForce{};
template<int NEWTON_PAIR>
struct TagPairGaussVector{};
template<int NEWTON_PAIR>
struct TagPairGaussMatrix{};
template<int NEWTON_PAIR>
struct TagPairGaussVectorNele{};
template<int NEWTON_PAIR>
struct TagPairGaussSelfNele{};

template<class DeviceType>
class PairLJCutCoulLongGaussKokkos : public PairLJCutCoulLongGauss, public ElectrodePairKokkos<DeviceType> {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairLJCutCoulLongGaussKokkos(class LAMMPS *);
  ~PairLJCutCoulLongGaussKokkos() override;

  void compute(int, int) override;
  void init_style() override;
  double init_one(int, int) override;
  void build_device_tables();

  // ElectrodePair device paths
  void compute_vector(double *, int, int, bool) override;
  void compute_vector_self(double *, int, int, bool) override;
  void compute_matrix(bigint *, double **, int) override;
  void compute_matrix_self(bigint *, double **, int) override;

  // ElectrodePairKokkos device paths (device-resident CG, stage K4)
  void compute_vector_nele(NeighList *, typename AT::t_kkacc_1d &, typename AT::t_int_1d &,
                           int, int, bool) override;
  void compute_vector_self_nele(typename AT::t_kkacc_1d &, typename AT::t_int_1d &, int, int,
                                bool) override;

  // device kernels
  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairGaussForce<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &, EV_FLOAT &) const;
  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairGaussForce<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &) const;

  // ElectrodePair device kernels
  template<int NEWTON_PAIR>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairGaussVector<NEWTON_PAIR>, const int &) const;
  template<int NEWTON_PAIR>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairGaussMatrix<NEWTON_PAIR>, const int &) const;
  template<int NEWTON_PAIR>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairGaussVectorNele<NEWTON_PAIR>, const int &) const;
  template<int NEWTON_PAIR>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairGaussSelfNele<NEWTON_PAIR>, const int &) const;
  friend void pair_virial_fdotr_compute<PairLJCutCoulLongGaussKokkos<DeviceType>>(
      PairLJCutCoulLongGaussKokkos<DeviceType> *);

  int neighflag, newton_pair;
  int nlocal, nall, eflag, vflag;
  KK_FLOAT qqrd2e_kk, g_ewald_kk;

  // execution_space inherited from Pair (assigned in constructor)
  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_kkfloat_1d_randomread q;
  typename AT::t_int_1d_randomread type;
  typename AT::t_int_1d_randomread mask;
  typename AT::t_int_1d d_ispoint;
  typename AT::t_kkfloat_1d d_eta_ii;      // diagonal eta (per type)
  typename AT::t_kkfloat_2d d_eta_ij;      // mixed eta table (ntypes+1)^2
  typename AT::t_kkfloat_2d d_lj1, d_lj2, d_lj3, d_lj4, d_offset;
  typename AT::t_kkfloat_2d d_cut_ljsq;

  typename AT::t_kkfloat_2d d_cutsq;

  KK_FLOAT special_lj_kk[4], special_coul_kk[4];
  KK_FLOAT cut_coulsq_kk;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  // ElectrodePair kernel state (bound per call, not owned)
  typename AT::t_kkacc_1d d_vec;           // sensor potential accumulator (nall)
  typename AT::t_kkfloat_1d d_q;           // gathered electrode charges (iele order)
  typename AT::t_int_1d_randomread d_mpos; // atom index -> iele position (-1 none)
  typename AT::t_kkfloat_2d d_matrix;      // device matrix fragment view
  typename AT::t_int_1d_randomread d_imap; // atom index -> local electrode index (nele, K4)
  int vec_groupbit, vec_source_grpbit;
  bool vec_inv;
};

}    // namespace LAMMPS_NS

#endif
#endif
