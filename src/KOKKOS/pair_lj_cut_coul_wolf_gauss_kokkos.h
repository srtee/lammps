/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/coul/wolf/gauss/kk,PairLJCutCoulWolfGaussKokkos<LMPDeviceType>);
PairStyle(lj/cut/coul/wolf/gauss/kk/device,PairLJCutCoulWolfGaussKokkos<LMPDeviceType>);
PairStyle(lj/cut/coul/wolf/gauss/kk/host,PairLJCutCoulWolfGaussKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_COUL_WOLF_GAUSS_KOKKOS_H
#define LMP_PAIR_LJ_CUT_COUL_WOLF_GAUSS_KOKKOS_H

#include "kokkos_type.h"
#include "pair_kokkos.h"    // EV_FLOAT, KK_FLOAT, SBBITS, atom-array view types
#include "pair_lj_cut_coul_long_gauss_kokkos.h"    // TagPairGauss{Force,Vector,Matrix}
#include "pair_lj_cut_coul_wolf_gauss.h"
namespace LAMMPS_NS {


template<class DeviceType>
class PairLJCutCoulWolfGaussKokkos : public PairLJCutCoulWolfGauss {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  friend void pair_virial_fdotr_compute<PairLJCutCoulWolfGaussKokkos<DeviceType>>(
      PairLJCutCoulWolfGaussKokkos<DeviceType> *);
  PairLJCutCoulWolfGaussKokkos(class LAMMPS *);
  ~PairLJCutCoulWolfGaussKokkos() override;

  void compute(int, int) override;
  void init_style() override;

  // ElectrodePair methods (device kernels; wolf shares the Gauss tag types
  // from pair_lj_cut_coul_long_gauss_kokkos.h)
  void compute_vector(double *, int, int, bool) override;
  void compute_vector_self(double *, int, int, bool) override;
  void compute_matrix(bigint *, double **, int) override;
  void compute_matrix_self(bigint *, double **, int) override;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void operator()(TagPairGaussForce<NEIGHFLAG, NEWTON_PAIR, EVFLAG>,
                                         const int &, EV_FLOAT &) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void operator()(TagPairGaussForce<NEIGHFLAG, NEWTON_PAIR, EVFLAG>,
                                         const int &) const;
  template<int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION void operator()(TagPairGaussVector<NEWTON_PAIR>, const int &) const;
  template<int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION void operator()(TagPairGaussMatrix<NEWTON_PAIR>, const int &) const;

  typename AT::t_kkfloat_2d d_eta_ij;
  typename AT::t_int_1d d_ispoint;
  typename AT::t_kkfloat_2d d_eshift_eta, d_fshift_eta;
  typename AT::t_kkfloat_2d d_lj1, d_lj2, d_lj3, d_lj4, d_offset;
  typename AT::t_kkfloat_2d d_cut_ljsq;

  typename AT::t_kkfloat_2d d_cutsq;
  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_kkfloat_1d_randomread q;
  typename AT::t_int_1d_randomread type;
  typename AT::t_int_1d_randomread mask;

  KK_FLOAT special_lj_kk[4], special_coul_kk[4];
  KK_FLOAT cut_coulsq_kk;
  int neighflag, newton_pair;
  int nlocal, nall, eflag, vflag;

  KK_FLOAT qqrd2e_kk, alpha_kk, e_shift_kk;
  KK_FLOAT selfint_kk, pre_eta_kk, pre_wolf_kk;

  int vec_groupbit, vec_source_grpbit;
  bool vec_inv;

  typename AT::t_kkacc_1d d_vec;
  typename AT::t_kkfloat_2d d_matrix;
  typename AT::t_int_1d d_mpos;

 private:
  void build_device_tables();
};

}    // namespace LAMMPS_NS

#endif
#endif