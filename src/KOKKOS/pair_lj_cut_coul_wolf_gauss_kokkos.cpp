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

/* ----------------------------------------------------------------------
   Kokkos port of lj/cut/coul/wolf/gauss (ELECTRODE pair style).
   Mirrors pair_lj_cut_coul_long_gauss_kokkos.cpp: same kernel structure,
   same list contract (default half list), same member-staleness discipline.
   Wolf-specific physics: erfc(alpha r) with energy/force shifting at
   cut_coul, plus the eta self/shift corrections.
------------------------------------------------------------------------- */

#include "pair_lj_cut_coul_wolf_gauss_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "electrode_math.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "ewald_const.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "utils.h"

#include <cmath>

namespace LAMMPS_NS {

using namespace MathConst;
using namespace EwaldConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCutCoulWolfGaussKokkos<DeviceType>::PairLJCutCoulWolfGaussKokkos(class LAMMPS *lmp) :
    PairLJCutCoulWolfGauss(lmp)
{
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCutCoulWolfGaussKokkos<DeviceType>::~PairLJCutCoulWolfGaussKokkos() {}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutCoulWolfGaussKokkos<DeviceType>::init_style()
{
  PairLJCutCoulWolfGauss::init_style();
  // the ElectrodePair matrix/vector kernels assume the standard half list
  // (host pair semantics); see pair_lj_cut_coul_long_gauss_kokkos.cpp
  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
}

/* ----------------------------------------------------------------------
   device tables: type-pair physics replicated from host arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutCoulWolfGaussKokkos<DeviceType>::build_device_tables()
{
  static thread_local bool built = false;
  if (built) return;
  built = true;
  int n = atom->ntypes;

  d_cutsq = typename AT::t_kkfloat_2d("pair:wolf_gauss:cutsq", n + 1, n + 1);
  d_eta_ij = typename AT::t_kkfloat_2d("pair:wolf_gauss:eta_ij", n + 1, n + 1);
  d_eshift_eta = typename AT::t_kkfloat_2d("pair:wolf_gauss:eshift_eta", n + 1, n + 1);
  d_fshift_eta = typename AT::t_kkfloat_2d("pair:wolf_gauss:fshift_eta", n + 1, n + 1);
  d_lj1 = typename AT::t_kkfloat_2d("pair:wolf_gauss:lj1", n + 1, n + 1);
  d_lj2 = typename AT::t_kkfloat_2d("pair:wolf_gauss:lj2", n + 1, n + 1);
  d_lj3 = typename AT::t_kkfloat_2d("pair:wolf_gauss:lj3", n + 1, n + 1);
  d_lj4 = typename AT::t_kkfloat_2d("pair:wolf_gauss:lj4", n + 1, n + 1);
  d_offset = typename AT::t_kkfloat_2d("pair:wolf_gauss:offset", n + 1, n + 1);
  d_cut_ljsq = typename AT::t_kkfloat_2d("pair:wolf_gauss:cut_ljsq", n + 1, n + 1);
  d_ispoint = typename AT::t_int_1d("pair:wolf_gauss:ispoint", n + 1);

  auto h_cutsq = Kokkos::create_mirror_view(d_cutsq);
  auto h_eta_ij = Kokkos::create_mirror_view(d_eta_ij);
  auto h_eshift = Kokkos::create_mirror_view(d_eshift_eta);
  auto h_fshift = Kokkos::create_mirror_view(d_fshift_eta);
  auto h_lj1 = Kokkos::create_mirror_view(d_lj1);
  auto h_lj2 = Kokkos::create_mirror_view(d_lj2);
  auto h_lj3 = Kokkos::create_mirror_view(d_lj3);
  auto h_lj4 = Kokkos::create_mirror_view(d_lj4);
  auto h_offset = Kokkos::create_mirror_view(d_offset);
  auto h_cut_ljsq = Kokkos::create_mirror_view(d_cut_ljsq);
  auto h_ispoint = Kokkos::create_mirror_view(d_ispoint);

  for (int a = 1; a <= n; a++) {
    h_ispoint(a) = ispoint[a];
    for (int b = 1; b <= n; b++) {
      h_cutsq(a, b) = static_cast<KK_FLOAT>(cutsq[a][b]);
      h_eta_ij(a, b) = static_cast<KK_FLOAT>(eta[a][b]);
      h_eshift(a, b) = static_cast<KK_FLOAT>(eshift_eta[a][b]);
      h_fshift(a, b) = static_cast<KK_FLOAT>(fshift_eta[a][b]);
      h_lj1(a, b) = static_cast<KK_FLOAT>(lj1[a][b]);
      h_lj2(a, b) = static_cast<KK_FLOAT>(lj2[a][b]);
      h_lj3(a, b) = static_cast<KK_FLOAT>(lj3[a][b]);
      h_lj4(a, b) = static_cast<KK_FLOAT>(lj4[a][b]);
      h_offset(a, b) = static_cast<KK_FLOAT>(offset[a][b]);
      h_cut_ljsq(a, b) = static_cast<KK_FLOAT>(cut_ljsq[a][b]);
    }
  }
  Kokkos::deep_copy(d_cutsq, h_cutsq);
  Kokkos::deep_copy(d_eta_ij, h_eta_ij);
  Kokkos::deep_copy(d_eshift_eta, h_eshift);
  Kokkos::deep_copy(d_fshift_eta, h_fshift);
  Kokkos::deep_copy(d_lj1, h_lj1);
  Kokkos::deep_copy(d_lj2, h_lj2);
  Kokkos::deep_copy(d_lj3, h_lj3);
  Kokkos::deep_copy(d_lj4, h_lj4);
  Kokkos::deep_copy(d_offset, h_offset);
  Kokkos::deep_copy(d_cut_ljsq, h_cut_ljsq);
  Kokkos::deep_copy(d_ispoint, h_ispoint);

  for (int m = 0; m < 4; m++) {
    special_lj_kk[m] = static_cast<KK_FLOAT>(force->special_lj[m]);
    special_coul_kk[m] = static_cast<KK_FLOAT>(force->special_coul[m]);
  }
  qqrd2e_kk = static_cast<KK_FLOAT>(force->qqrd2e);
  alpha_kk = static_cast<KK_FLOAT>(alpha);
  cut_coulsq_kk = static_cast<KK_FLOAT>(cut_coulsq);
  const KK_FLOAT alpha_cut = alpha_kk * static_cast<KK_FLOAT>(cut_coul);
  const KK_FLOAT expm2_cut = Kokkos::exp(-alpha_cut * alpha_cut);
  e_shift_kk = ElectrodeMath::safe_erfc(alpha_kk * static_cast<KK_FLOAT>(cut_coul)) /
      static_cast<KK_FLOAT>(cut_coul);
  selfint_kk = static_cast<KK_FLOAT>(2.0) / static_cast<KK_FLOAT>(MY_PIS) * alpha_kk;
  pre_eta_kk = static_cast<KK_FLOAT>(2.0) / static_cast<KK_FLOAT>(MY_PIS);
  pre_wolf_kk = e_shift_kk;
}

/* ----------------------------------------------------------------------
   force/energy compute: mirrors PairLJCutCoulWolfGauss::compute
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutCoulWolfGaussKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  build_device_tables();
  eflag = eflag_in;
  vflag = vflag_in;

  ev_init(eflag, vflag, 0);

  atomKK->sync(execution_space, X_MASK | F_MASK | Q_MASK | TYPE_MASK);
  if (eflag || vflag) atomKK->modified(execution_space, datamask_modify);
  else atomKK->modified(execution_space, F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  q = atomKK->k_q.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  special_lj_kk[0] = static_cast<KK_FLOAT>(force->special_lj[0]);
  special_lj_kk[1] = static_cast<KK_FLOAT>(force->special_lj[1]);
  special_lj_kk[2] = static_cast<KK_FLOAT>(force->special_lj[2]);
  special_lj_kk[3] = static_cast<KK_FLOAT>(force->special_lj[3]);
  special_coul_kk[0] = static_cast<KK_FLOAT>(force->special_coul[0]);
  special_coul_kk[1] = static_cast<KK_FLOAT>(force->special_coul[1]);
  special_coul_kk[2] = static_cast<KK_FLOAT>(force->special_coul[2]);
  special_coul_kk[3] = static_cast<KK_FLOAT>(force->special_coul[3]);

  newton_pair = force->newton_pair;

  const int inum = list->inum;
  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  // dispatch on the list geometry (always half here), not lmp->kokkos->neighflag
  EV_FLOAT ev;
  copymode = 1;
  if (newton_pair) {
    if (evflag)
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairGaussForce<HALF, 1, 1>>(0, inum), *this, ev);
    else
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType, TagPairGaussForce<HALF, 1, 0>>(0, inum), *this);
  } else {
    if (evflag)
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairGaussForce<HALF, 0, 1>>(0, inum), *this, ev);
    else
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType, TagPairGaussForce<HALF, 0, 0>>(0, inum), *this);
  }

  copymode = 0;
  if (eflag_global) eng_vdwl += static_cast<double>(ev.evdwl);
  if (eflag_global) eng_coul += static_cast<double>(ev.ecoul);
  if (vflag_global) {
    for (int n = 0; n < 6; n++) virial[n] += static_cast<double>(ev.v[n]);
  }
  if (vflag_fdotr) pair_virial_fdotr_compute(this);
}

/* ----------------------------------------------------------------------
   vector kernel: accumulate a_ij q_j into the sensor potential
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutCoulWolfGaussKokkos<DeviceType>::compute_vector(double *vec, int groupbit,
                                                              int source_grpbit, bool inv)
{
  build_device_tables();
  // the kernels read the nlocal/newton_pair members; keep them current
  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;
  const int inum = list->inum;

  atomKK->sync(execution_space, X_MASK | Q_MASK | TYPE_MASK | MASK_MASK);

  x = atomKK->k_x.view<DeviceType>();
  q = atomKK->k_q.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  const int nall = nlocal + atom->nghost;
  d_vec = typename AT::t_kkacc_1d("pair:wolf_gauss_vec", nall);

  vec_groupbit = groupbit;
  vec_source_grpbit = source_grpbit;
  vec_inv = inv;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGaussVector<0>>(0, inum), *this);

  auto h_vec = Kokkos::create_mirror_view(d_vec);
  Kokkos::deep_copy(h_vec, d_vec);
  for (int i = 0; i < nall; i++) vec[i] += static_cast<double>(h_vec(i));

  copymode = 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutCoulWolfGaussKokkos<DeviceType>::compute_vector_self(double *vec, int groupbit,
                                                                   int source_grpbit, bool inv)
{
  build_device_tables();
  atomKK->sync(Host, Q_MASK | TYPE_MASK | MASK_MASK);
  int *h_mask = atom->mask;
  int *h_type = atom->type;
  double *h_q = atom->q;
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (!(h_mask[i] & groupbit)) continue;
    const int itype = h_type[i];
    const bool i_in_source = !!(h_mask[i] & source_grpbit) != inv;
    if (i_in_source) {
      vec[i] -= (selfint_kk + pre_wolf_kk) * h_q[i];
      if (!ispoint[itype])
        vec[i] += (pre_eta_kk * eta[itype][itype] + eshift_eta[itype][itype]) * h_q[i];
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION void PairLJCutCoulWolfGaussKokkos<DeviceType>::operator()(
    TagPairGaussVector<NEWTON_PAIR>, const int &ii) const
{
  const int i = d_ilist[ii];
  const bool i_in_sensor = (mask(i) & vec_groupbit);
  const bool i_in_source = !!(mask(i) & vec_source_grpbit) != vec_inv;
  if (!(i_in_sensor || i_in_source)) return;
  const KK_FLOAT xtmp = x(i, 0);
  const KK_FLOAT ytmp = x(i, 1);
  const KK_FLOAT ztmp = x(i, 2);
  const KK_FLOAT qtmp = q[i];
  const int itype = type[i];
  const bool ipoint = d_ispoint(itype) != 0;

  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i, jj);
    j &= NEIGHMASK;
    const bool j_in_sensor = (mask(j) & vec_groupbit);
    const bool j_in_source = !!(mask(j) & vec_source_grpbit) != vec_inv;
    const bool compute_ij = i_in_sensor && j_in_source;
    const bool compute_ji = (newton_pair || j < nlocal) && (j_in_sensor && i_in_source);
    if (!(compute_ij || compute_ji)) continue;

    const KK_FLOAT delx = xtmp - x(j, 0);
    const KK_FLOAT dely = ytmp - x(j, 1);
    const KK_FLOAT delz = ztmp - x(j, 2);
    const KK_FLOAT rsq = delx * delx + dely * dely + delz * delz;
    const int jtype = type[j];
    if (rsq >= d_cutsq(itype, jtype)) continue;

    const KK_FLOAT factor_coul = special_coul_kk[j >> SBBITS & 3];
    const KK_FLOAT r = Kokkos::sqrt(rsq);
    const KK_FLOAT rinv = static_cast<KK_FLOAT>(1.0) / r;
    KK_FLOAT aij = rinv * ElectrodeMath::safe_erfc(alpha_kk * r) - e_shift_kk;
    KK_FLOAT erfc_eta = 0.0;
    if (!(ipoint && d_ispoint(jtype) != 0)) {
      erfc_eta = ElectrodeMath::safe_erfc(d_eta_ij(itype, jtype) * r);
      aij -= rinv * erfc_eta - d_eshift_eta(itype, jtype);
    }
    if (factor_coul < static_cast<KK_FLOAT>(1.0))
      aij -= (static_cast<KK_FLOAT>(1.0) - factor_coul) * rinv *
          (static_cast<KK_FLOAT>(1.0) - erfc_eta);
    if (i_in_sensor) d_vec(i) += static_cast<KK_ACC_FLOAT>(aij * q(j));
    if (j_in_sensor && (!vec_inv || !i_in_sensor)) d_vec(j) += static_cast<KK_ACC_FLOAT>(aij * qtmp);
  }
}

/* ----------------------------------------------------------------------
   matrix kernel: accumulate a_ij into [ipos][jpos]
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutCoulWolfGaussKokkos<DeviceType>::compute_matrix(bigint *mpos, double **array,
                                                              int groupbit)
{
  build_device_tables();
  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;

  int ngroup = -1;
  for (int i = 0; i < atom->nlocal; i++)
    if (mpos[i] + 1 > ngroup) ngroup = static_cast<int>(mpos[i]) + 1;
  MPI_Allreduce(MPI_IN_PLACE, &ngroup, 1, MPI_INT, MPI_MAX, world);

  atomKK->sync(execution_space, X_MASK | TYPE_MASK | MASK_MASK);
  x = atomKK->k_x.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  const int nall = nlocal + atom->nghost;
  DAT::tdual_int_1d k_mpos("pair:wolf_gauss:mpos", nall);
  auto h_mpos = k_mpos.view_host();
  for (int i = 0; i < nall; i++) h_mpos(i) = static_cast<int>(mpos[i]);
  k_mpos.modify_host();
  k_mpos.sync<DeviceType>();
  d_mpos = k_mpos.view<DeviceType>();

  typename AT::t_kkfloat_2d k_mat("pair:wolf_gauss:matrix", ngroup, ngroup);
  Kokkos::deep_copy(k_mat, 0.0);
  d_matrix = k_mat;
  vec_groupbit = groupbit;
  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGaussMatrix<0>>(0, list->inum),
                       *this);

  auto h_mat = Kokkos::create_mirror_view(d_matrix);
  Kokkos::deep_copy(h_mat, d_matrix);
  for (int a = 0; a < ngroup; a++)
    for (int b = 0; b < ngroup; b++) array[a][b] += static_cast<double>(h_mat(a, b));

  copymode = 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutCoulWolfGaussKokkos<DeviceType>::compute_matrix_self(bigint *mpos, double **array,
                                                                   int groupbit)
{
  build_device_tables();    // selfint_kk/pre_wolf_kk members are set here
  atomKK->sync(Host, Q_MASK | TYPE_MASK | MASK_MASK);
  int *h_mask = atom->mask;
  int *h_type = atom->type;
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (h_mask[i] & groupbit) {
      const int itype = h_type[i];
      array[mpos[i]][mpos[i]] -= selfint_kk + pre_wolf_kk;
      if (!ispoint[itype])
        array[mpos[i]][mpos[i]] += pre_eta_kk * eta[itype][itype] + eshift_eta[itype][itype];
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION void PairLJCutCoulWolfGaussKokkos<DeviceType>::operator()(
    TagPairGaussMatrix<NEWTON_PAIR>, const int &ii) const
{
  const int i = d_ilist[ii];
  if (i >= nlocal) return;
  if (!(mask(i) & vec_groupbit)) return;
  const int itype = type[i];
  const bool ipoint = d_ispoint(itype) != 0;
  const bigint ipos = d_mpos(i);
  if (ipos < 0) return;
  const KK_FLOAT xtmp = x(i, 0);
  const KK_FLOAT ytmp = x(i, 1);
  const KK_FLOAT ztmp = x(i, 2);

  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i, jj);
    j &= NEIGHMASK;
    if (!(mask(j) & vec_groupbit)) continue;
    const bigint jpos = d_mpos(j);
    if (jpos < 0) continue;

    const KK_FLOAT delx = xtmp - x(j, 0);
    const KK_FLOAT dely = ytmp - x(j, 1);
    const KK_FLOAT delz = ztmp - x(j, 2);
    const KK_FLOAT rsq = delx * delx + dely * dely + delz * delz;
    const int jtype = type[j];

    if (rsq < d_cutsq(itype, jtype)) {
      const KK_FLOAT factor_coul = special_coul_kk[j >> SBBITS & 3];
      const KK_FLOAT r = Kokkos::sqrt(rsq);
      const KK_FLOAT rinv = static_cast<KK_FLOAT>(1.0) / r;
      KK_FLOAT aij = rinv * ElectrodeMath::safe_erfc(alpha_kk * r) - e_shift_kk;
      KK_FLOAT erfc_eta = 0.0;
      if (!(ipoint && d_ispoint(jtype) != 0)) {
        erfc_eta = ElectrodeMath::safe_erfc(d_eta_ij(itype, jtype) * r);
        aij -= rinv * erfc_eta - d_eshift_eta(itype, jtype);
      }
      if (factor_coul < static_cast<KK_FLOAT>(1.0))
        aij -= (static_cast<KK_FLOAT>(1.0) - factor_coul) * rinv *
            (static_cast<KK_FLOAT>(1.0) - erfc_eta);
      if (!newton_pair && j >= nlocal) aij *= static_cast<KK_FLOAT>(0.5);
      d_matrix(ipos, jpos) += static_cast<KK_ACC_FLOAT>(aij);
    }
  }
}

/* ----------------------------------------------------------------------
   force/energy kernel
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   force/energy kernel: mirrors PairLJCutCoulWolfGauss::compute
------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION void PairLJCutCoulWolfGaussKokkos<DeviceType>::operator()(
    TagPairGaussForce<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii, EV_FLOAT &ev) const
{
  Kokkos::View<KK_ACC_FLOAT *[3], typename DAT::t_kkacc_1d_3::array_layout,
               typename KKDevice<DeviceType>::value, Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value>>
      a_f = f;

  const int i = d_ilist[ii];
  const KK_FLOAT xtmp = x(i, 0);
  const KK_FLOAT ytmp = x(i, 1);
  const KK_FLOAT ztmp = x(i, 2);
  const KK_FLOAT qtmp = q[i];
  const int itype = type[i];
  const bool ipoint = d_ispoint(itype) != 0;
  const KK_FLOAT half_factor = ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) &&
                                (NEWTON_PAIR || i < nlocal))
      ? static_cast<KK_FLOAT>(1.0)
      : static_cast<KK_FLOAT>(0.5);

  // wolf self + shifted Coulomb self energy
  if (EVFLAG && eflag) {
    const KK_FLOAT q2 = qtmp * qtmp;
    KK_FLOAT e = -((e_shift_kk / static_cast<KK_FLOAT>(2.0) + alpha_kk /
                     static_cast<KK_FLOAT>(MY_PIS)) * qqrd2e_kk) * q2;
    if (!ipoint)
      e += (d_eshift_eta(itype, itype) / static_cast<KK_FLOAT>(2.0) +
            d_eta_ij(itype, itype) / static_cast<KK_FLOAT>(MY_PIS)) * qqrd2e_kk * q2;
    ev.ecoul += static_cast<KK_ACC_FLOAT>(e) * half_factor;
  }

  KK_ACC_FLOAT fxtmp = 0.0;
  KK_ACC_FLOAT fytmp = 0.0;
  KK_ACC_FLOAT fztmp = 0.0;

  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i, jj);
    const KK_FLOAT factor_lj = special_lj_kk[j >> SBBITS & 3];
    const KK_FLOAT factor_coul = special_coul_kk[j >> SBBITS & 3];
    j &= NEIGHMASK;

    const KK_FLOAT delx = xtmp - x(j, 0);
    const KK_FLOAT dely = ytmp - x(j, 1);
    const KK_FLOAT delz = ztmp - x(j, 2);
    const KK_FLOAT rsq = delx * delx + dely * dely + delz * delz;
    const int jtype = type[j];
    const bool gausscorr = !ipoint || d_ispoint(jtype) == 0;
    const KK_FLOAT jhalf = ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) &&
                            (NEWTON_PAIR || j < nlocal))
        ? static_cast<KK_FLOAT>(1.0)
        : static_cast<KK_FLOAT>(0.5);

    if (rsq < d_cutsq(itype, jtype)) {
      const KK_FLOAT r2inv = static_cast<KK_FLOAT>(1.0) / rsq;
      KK_FLOAT forcecoul = 0.0;
      KK_FLOAT ecoul = 0.0;
      KK_FLOAT erfc_eta = 0.0;
      if (rsq < cut_coulsq_kk) {
        const KK_FLOAT r = Kokkos::sqrt(rsq);
        const KK_FLOAT grij = alpha_kk * r;
        const KK_FLOAT expm2 = Kokkos::exp(-grij * grij);
        const KK_FLOAT t = static_cast<KK_FLOAT>(1.0) /
            (static_cast<KK_FLOAT>(1.0) + static_cast<KK_FLOAT>(EWALD_P) * grij);
        const KK_FLOAT erfc = t * (static_cast<KK_FLOAT>(A1) +
                              t * (static_cast<KK_FLOAT>(A2) +
                              t * (static_cast<KK_FLOAT>(A3) +
                              t * (static_cast<KK_FLOAT>(A4) +
                              t * static_cast<KK_FLOAT>(A5))))) * expm2;
        const KK_FLOAT prefactor = qqrd2e_kk * qtmp * q(j) / r;
        const KK_FLOAT f_shift = -(e_shift_kk + static_cast<KK_FLOAT>(2.0) * alpha_kk /
                                   static_cast<KK_FLOAT>(MY_PIS) * expm2) /
            Kokkos::sqrt(cut_coulsq_kk);
        forcecoul = prefactor * (erfc + static_cast<KK_FLOAT>(EWALD_F) * grij * expm2 +
                                 f_shift * rsq);
        KK_FLOAT forcecorr = 0.0;
        if (gausscorr) {
          const KK_FLOAT etarij = d_eta_ij(itype, jtype) * r;
          const KK_FLOAT expm2_eta = Kokkos::exp(-etarij * etarij);
          const KK_FLOAT te = static_cast<KK_FLOAT>(1.0) /
              (static_cast<KK_FLOAT>(1.0) + static_cast<KK_FLOAT>(EWALD_P) * etarij);
          erfc_eta = te * (static_cast<KK_FLOAT>(A1) +
                      te * (static_cast<KK_FLOAT>(A2) +
                      te * (static_cast<KK_FLOAT>(A3) +
                      te * (static_cast<KK_FLOAT>(A4) +
                      te * static_cast<KK_FLOAT>(A5))))) * expm2_eta;
          forcecorr = erfc_eta + static_cast<KK_FLOAT>(EWALD_F) * etarij * expm2_eta;
          forcecoul -= prefactor * (forcecorr + d_fshift_eta(itype, jtype) * rsq);
        }
        if (factor_coul < static_cast<KK_FLOAT>(1.0))
          forcecoul -= (static_cast<KK_FLOAT>(1.0) - factor_coul) * prefactor *
              (static_cast<KK_FLOAT>(1.0) - forcecorr);
        if (EVFLAG && eflag) {
          ecoul = prefactor * (erfc - e_shift_kk * r);
          if (gausscorr) ecoul -= prefactor * (erfc_eta - d_eshift_eta(itype, jtype) * r);
          if (factor_coul < static_cast<KK_FLOAT>(1.0))
            ecoul -= (static_cast<KK_FLOAT>(1.0) - factor_coul) * prefactor *
                (static_cast<KK_FLOAT>(1.0) - erfc_eta);
        }
      }

      KK_FLOAT forcelj = 0.0;
      KK_FLOAT evdwl = 0.0;
      if (rsq < d_cut_ljsq(itype, jtype)) {
        const KK_FLOAT r6inv = r2inv * r2inv * r2inv;
        forcelj = r6inv * (d_lj1(itype, jtype) * r6inv - d_lj2(itype, jtype));
        if (EVFLAG && eflag)
          evdwl = r6inv * (d_lj3(itype, jtype) * r6inv - d_lj4(itype, jtype)) -
              d_offset(itype, jtype);
      }

      const KK_FLOAT fpair = (forcecoul + factor_lj * forcelj) * r2inv;
      const KK_FLOAT fdx = delx * fpair;
      const KK_FLOAT fdy = dely * fpair;
      const KK_FLOAT fdz = delz * fpair;

      fxtmp += static_cast<KK_ACC_FLOAT>(fdx);
      fytmp += static_cast<KK_ACC_FLOAT>(fdy);
      fztmp += static_cast<KK_ACC_FLOAT>(fdz);
      if ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) && (NEWTON_PAIR || j < nlocal)) {
        a_f(j, 0) -= static_cast<KK_ACC_FLOAT>(fdx);
        a_f(j, 1) -= static_cast<KK_ACC_FLOAT>(fdy);
        a_f(j, 2) -= static_cast<KK_ACC_FLOAT>(fdz);
      }

      if (EVFLAG) {
        if (eflag) {
          ev.ecoul += ecoul * jhalf;
          ev.evdwl += evdwl * factor_lj * jhalf;
        }
        if (vflag_global) {
          const KK_FLOAT vfx = delx * fpair;
          const KK_FLOAT vfy = dely * fpair;
          const KK_FLOAT vfz = delz * fpair;
          ev.v[0] += jhalf * vfx * delx;
          ev.v[1] += jhalf * vfy * dely;
          ev.v[2] += jhalf * vfz * delz;
          ev.v[3] += jhalf * vfx * dely;
          ev.v[4] += jhalf * vfx * delz;
          ev.v[5] += jhalf * vfy * delz;
        }
      }
    }
  }

  a_f(i, 0) += fxtmp;
  a_f(i, 1) += fytmp;
  a_f(i, 2) += fztmp;
}

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION void PairLJCutCoulWolfGaussKokkos<DeviceType>::operator()(
    TagPairGaussForce<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii) const
{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG, NEWTON_PAIR, EVFLAG>(
      TagPairGaussForce<NEIGHFLAG, NEWTON_PAIR, EVFLAG>(), ii, ev);
}

template class PairLJCutCoulWolfGaussKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLJCutCoulWolfGaussKokkos<LMPHostType>;
#endif

}    // namespace LAMMPS_NS