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

#include "pair_lj_cut_coul_long_gauss_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "electrode_math.h"
#include "atom_vec_kokkos.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "math_const.h"
#include "ewald_const.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"
#include "neigh_request.h"
#include "respa.h"
#include "update.h"
#include "utils.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>


namespace LAMMPS_NS {

using namespace MathConst;
using namespace EwaldConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCutCoulLongGaussKokkos<DeviceType>::PairLJCutCoulLongGaussKokkos(class LAMMPS *lmp) :
    PairLJCutCoulLongGauss(lmp)
{
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  // stock pair masks: never claim X/Q/TYPE as device-modified (only what
  // the kernels write); the ALL_MASK defaults from the host base Pair ctor
  // break the verlet/dump force sync pipeline
  atomKK = (AtomKokkos *) atom;
  datamask_read = X_MASK | F_MASK | Q_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}
/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCutCoulLongGaussKokkos<DeviceType>::~PairLJCutCoulLongGaussKokkos()
{
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutCoulLongGaussKokkos<DeviceType>::init_style()
{
  PairLJCutCoulLongGauss::init_style();

  // error if rRESPA with inner levels

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style, "^respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
    if (respa) error->all(FLERR, "Cannot use Kokkos pair style with rRESPA inner/middle");
  }

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
  // the ElectrodePair matrix/vector kernels assume the standard half list
  // (host pair semantics; the host pair style never enables full). Full lists
  // double-iterate every pair and the newton-off convention differs, so the
  // default half request is used unconditionally.
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double PairLJCutCoulLongGaussKokkos<DeviceType>::init_one(int i, int j)
{
  return PairLJCutCoulLongGauss::init_one(i, j);
}

/* ----------------------------------------------------------------------
   device tables built after all init_one calls (init_style runs last)
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutCoulLongGaussKokkos<DeviceType>::build_device_tables()
{
  static thread_local bool built = false;
  if (built) return;
  built = true;
  int n = atom->ntypes;

  d_cutsq = typename AT::t_kkfloat_2d("pair:cutsq", n + 1, n + 1);
  auto h_cutsq = Kokkos::create_mirror_view(d_cutsq);
  for (int a = 1; a <= n; a++)
    for (int b = 1; b <= n; b++) h_cutsq(a, b) = static_cast<KK_FLOAT>(cutsq[a][b]);
  Kokkos::deep_copy(d_cutsq, h_cutsq);

  d_eta_ij = typename AT::t_kkfloat_2d("pair:eta_ij", n + 1, n + 1);
  d_eta_ii = typename AT::t_kkfloat_1d("pair:eta_ii", n + 1);
  d_ispoint = typename AT::t_int_1d("pair:ispoint", n + 1);
  d_lj1 = typename AT::t_kkfloat_2d("pair:lj1", n + 1, n + 1);
  d_lj2 = typename AT::t_kkfloat_2d("pair:lj2", n + 1, n + 1);
  d_lj3 = typename AT::t_kkfloat_2d("pair:lj3", n + 1, n + 1);
  d_lj4 = typename AT::t_kkfloat_2d("pair:lj4", n + 1, n + 1);
  d_offset = typename AT::t_kkfloat_2d("pair:offset", n + 1, n + 1);
  d_cut_ljsq = typename AT::t_kkfloat_2d("pair:cut_ljsq", n + 1, n + 1);

  auto h_eta_ij = Kokkos::create_mirror_view(d_eta_ij);
  auto h_eta_ii = Kokkos::create_mirror_view(d_eta_ii);
  auto h_ispoint = Kokkos::create_mirror_view(d_ispoint);
  auto h_lj1 = Kokkos::create_mirror_view(d_lj1);
  auto h_lj2 = Kokkos::create_mirror_view(d_lj2);
  auto h_lj3 = Kokkos::create_mirror_view(d_lj3);
  auto h_lj4 = Kokkos::create_mirror_view(d_lj4);
  auto h_offset = Kokkos::create_mirror_view(d_offset);
  auto h_cut_ljsq = Kokkos::create_mirror_view(d_cut_ljsq);

  for (int a = 1; a <= n; a++) {
    h_eta_ii(a) = static_cast<KK_FLOAT>(eta[a][a]);
    h_ispoint(a) = ispoint[a];
    for (int b = 1; b <= n; b++) {
      h_eta_ij(a, b) = static_cast<KK_FLOAT>(eta[a][b]);
      h_lj1(a, b) = static_cast<KK_FLOAT>(lj1[a][b]);
      h_lj2(a, b) = static_cast<KK_FLOAT>(lj2[a][b]);
      h_lj3(a, b) = static_cast<KK_FLOAT>(lj3[a][b]);
      h_lj4(a, b) = static_cast<KK_FLOAT>(lj4[a][b]);
      h_offset(a, b) = static_cast<KK_FLOAT>(offset[a][b]);
      h_cut_ljsq(a, b) = static_cast<KK_FLOAT>(cut_ljsq[a][b]);
    }
  }
  Kokkos::deep_copy(d_eta_ij, h_eta_ij);
  Kokkos::deep_copy(d_eta_ii, h_eta_ii);
  Kokkos::deep_copy(d_ispoint, h_ispoint);
  Kokkos::deep_copy(d_lj1, h_lj1);
  Kokkos::deep_copy(d_lj2, h_lj2);
  Kokkos::deep_copy(d_lj3, h_lj3);
  Kokkos::deep_copy(d_lj4, h_lj4);
  Kokkos::deep_copy(d_offset, h_offset);
  Kokkos::deep_copy(d_cut_ljsq, h_cut_ljsq);

  g_ewald_kk = static_cast<KK_FLOAT>(g_ewald);
  qqrd2e_kk = static_cast<KK_FLOAT>(force->qqrd2e);
  cut_coulsq_kk = static_cast<KK_FLOAT>(cut_coulsq);
  for (int m = 0; m < 4; m++) {
    special_lj_kk[m] = static_cast<KK_FLOAT>(force->special_lj[m]);
    special_coul_kk[m] = static_cast<KK_FLOAT>(force->special_coul[m]);
  }
}

/* ----------------------------------------------------------------------
   force/energy compute: mirrors PairLJCutCoulLongGauss::compute (no coulomb
   tables; the gauss erfc terms are cheap direct evaluations)
------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION void PairLJCutCoulLongGaussKokkos<DeviceType>::operator()(
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

  if (!ipoint && EVFLAG && eflag) {
    const KK_FLOAT e = qqrd2e_kk * d_eta_ii(itype) / static_cast<KK_FLOAT>(MY_PIS) * qtmp * qtmp;
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
    const bool needcorr = !ipoint || !d_ispoint(jtype);
    const KK_FLOAT jhalf = ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) &&
                            (NEWTON_PAIR || j < nlocal))
        ? static_cast<KK_FLOAT>(1.0)
        : static_cast<KK_FLOAT>(0.5);

    if (rsq < d_cutsq(itype, jtype)) {
      const KK_FLOAT r2inv = static_cast<KK_FLOAT>(1.0) / rsq;
      KK_FLOAT forcecoul = 0.0;
      KK_FLOAT ecoul = 0.0;
      KK_FLOAT erfc_eta = 0.0;
      KK_FLOAT forcecorr = 0.0;
      if (rsq < cut_coulsq) {
        const KK_FLOAT r = Kokkos::sqrt(rsq);
        const KK_FLOAT grij = g_ewald_kk * r;
        const KK_FLOAT expm2 = Kokkos::exp(-grij * grij);
        const KK_FLOAT t = static_cast<KK_FLOAT>(1.0) /
            (static_cast<KK_FLOAT>(1.0) + static_cast<KK_FLOAT>(EWALD_P) * grij);
        const KK_FLOAT erfc = t * (static_cast<KK_FLOAT>(A1) +
                              t * (static_cast<KK_FLOAT>(A2) +
                              t * (static_cast<KK_FLOAT>(A3) +
                              t * (static_cast<KK_FLOAT>(A4) +
                              t * static_cast<KK_FLOAT>(A5))))) * expm2;
        const KK_FLOAT prefactor = qqrd2e_kk * qtmp * q[j] / r;
        forcecoul = prefactor * (erfc + static_cast<KK_FLOAT>(EWALD_F) * grij * expm2);
        if (needcorr) {
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
          forcecoul -= prefactor * forcecorr;
        }
        if (factor_coul < static_cast<KK_FLOAT>(1.0))
          forcecoul -= (static_cast<KK_FLOAT>(1.0) - factor_coul) * prefactor *
              (static_cast<KK_FLOAT>(1.0) - forcecorr);
        if (EVFLAG && eflag) {
          ecoul = prefactor * erfc;
          if (needcorr) ecoul -= prefactor * erfc_eta;
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
        if (eflag) ev.ecoul += ecoul * jhalf;
        ev.evdwl += evdwl * factor_lj * jhalf;
        if (vflag_either) {
          const KK_FLOAT vfx = delx * fpair;
          const KK_FLOAT vfy = dely * fpair;
          const KK_FLOAT vfz = delz * fpair;
          if (vflag_global) {
            ev.v[0] += jhalf * vfx * delx;
            ev.v[1] += jhalf * vfy * dely;
            ev.v[2] += jhalf * vfz * delz;
            ev.v[3] += jhalf * vfx * dely;
            ev.v[4] += jhalf * vfx * delz;
            ev.v[5] += jhalf * vfy * delz;
          }
          if (vflag_atom) {
            const int voff = EVFLAG == 0 ? 0 : 0;    // placeholder, per-atom via d_vatom below
            (void) voff;
            if (i < nlocal) {
              d_vatom(i, 0) += static_cast<KK_ACC_FLOAT>(jhalf * vfx * delx);
              d_vatom(i, 1) += static_cast<KK_ACC_FLOAT>(jhalf * vfy * dely);
              d_vatom(i, 2) += static_cast<KK_ACC_FLOAT>(jhalf * vfz * delz);
              d_vatom(i, 3) += static_cast<KK_ACC_FLOAT>(jhalf * vfx * dely);
              d_vatom(i, 4) += static_cast<KK_ACC_FLOAT>(jhalf * vfx * delz);
              d_vatom(i, 5) += static_cast<KK_ACC_FLOAT>(jhalf * vfy * delz);
            }
          }
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
KOKKOS_INLINE_FUNCTION void PairLJCutCoulLongGaussKokkos<DeviceType>::operator()(
    TagPairGaussForce<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii) const
{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG, NEWTON_PAIR, EVFLAG>(
      TagPairGaussForce<NEIGHFLAG, NEWTON_PAIR, EVFLAG>(), ii, ev);
}

template<class DeviceType>
void PairLJCutCoulLongGaussKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  build_device_tables();
  eflag = eflag_in;
  vflag = vflag_in;

  ev_init(eflag, vflag, 0);

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space, datamask_read);
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

  copymode = 1;

  EV_FLOAT ev;
  // the pair's neighbor list is always half (see init_style); the force
  // kernel must run HALFTHREAD regardless of lmp->kokkos->neighflag
  // (FULL on GPU): a half list means several rows write the same twin's
  // force slot, which needs atomic accumulation -- HALF's non-atomic
  // unmanaged alias silently drops updates on concurrent backends
  if (newton_pair) {
    if (evflag)
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairGaussForce<HALFTHREAD, 1, 1>>(0, inum), *this, ev);
    else
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType, TagPairGaussForce<HALFTHREAD, 1, 0>>(0, inum), *this);
  } else {
    if (evflag)
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairGaussForce<HALFTHREAD, 0, 1>>(0, inum), *this, ev);
    else
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType, TagPairGaussForce<HALFTHREAD, 0, 0>>(0, inum), *this);
  }

  if (eflag_global) {
    eng_vdwl += static_cast<double>(ev.evdwl);
    eng_coul += static_cast<double>(ev.ecoul);
  }
  if (vflag_global) {
    virial[0] += static_cast<double>(ev.v[0]);
    virial[1] += static_cast<double>(ev.v[1]);
    virial[2] += static_cast<double>(ev.v[2]);
    virial[3] += static_cast<double>(ev.v[3]);
    virial[4] += static_cast<double>(ev.v[4]);
    virial[5] += static_cast<double>(ev.v[5]);
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

/* ----------------------------------------------------------------------
   ElectrodePair device paths. The pair's own neighbor list is a
   NeighListKokkos under a KOKKOS build (its host arrays are empty), so the
   vector and matrix contributions run as device kernels over the device
   lists -- they cannot delegate to the base-class host loops.
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutCoulLongGaussKokkos<DeviceType>::compute_vector(double *vec, int groupbit,
                                                              int source_grpbit, bool inv)
{
  build_device_tables();
  // the kernels read the nlocal/newton_pair members; keep them current
  // (neither compute() nor compute_matrix() may have run yet)
  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;
  const int inum = list->inum;

  // charges are the atom array (sensor/source read through group masks)
  atomKK->sync(execution_space, X_MASK | Q_MASK | TYPE_MASK | MASK_MASK);

  x = atomKK->k_x.view<DeviceType>();
  q = atomKK->k_q.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  // vec lives on the host (the fix gathers it); mirror to device, zero, accumulate
  const int nall = nlocal + atom->nghost;
  d_vec = typename AT::t_kkacc_1d("pair:gauss_vec", nall);
  Kokkos::deep_copy(d_vec, 0);    // kernel accumulates: fresh view is NOT guaranteed zero on GPU
  vec_groupbit = groupbit;
  vec_source_grpbit = source_grpbit;
  vec_inv = inv;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGaussVector<0>>(0, inum), *this);
  auto h_vec = Kokkos::create_mirror_view(d_vec);
  Kokkos::deep_copy(h_vec, d_vec);
  for (int i = 0; i < nlocal + atom->nghost; i++) vec[i] += static_cast<double>(h_vec(i));

  copymode = 0;
}

template<class DeviceType>
void PairLJCutCoulLongGaussKokkos<DeviceType>::compute_vector_self(double *vec, int groupbit,
                                                                   int source_grpbit, bool inv)
{
  atomKK->sync(Host, Q_MASK | TYPE_MASK | MASK_MASK);

  const KK_FLOAT selfint = static_cast<KK_FLOAT>(2.0) / static_cast<KK_FLOAT>(MY_PIS) * g_ewald_kk;
  const KK_FLOAT preta = static_cast<KK_FLOAT>(2.0) / static_cast<KK_FLOAT>(MY_PIS);

  // host atom arrays are valid under a KOKKOS build after sync; the pair's own
  // list->ilist is not (KK lists keep host pointers empty)
  atomKK->sync(Host, Q_MASK | TYPE_MASK | MASK_MASK);
  int *h_mask = atom->mask;
  int *h_type = atom->type;
  double *h_q = atom->q;
  for (int i = 0; i < nlocal; i++) {
    if (!(h_mask[i] & groupbit)) continue;
    const int itype = h_type[i];
    const bool i_in_source = !!(h_mask[i] & source_grpbit) != inv;
    if (i_in_source) {
      vec[i] -= selfint * h_q[i];
      if (!ispoint[itype]) vec[i] += preta * eta[itype][itype] * h_q[i];
    }
  }
}

/* ----------------------------------------------------------------------
   vector kernel: accumulate a_ij q_j into the sensor potential
------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION void PairLJCutCoulLongGaussKokkos<DeviceType>::operator()(
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

  KK_ACC_FLOAT vtmp = 0.0;
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
    KK_FLOAT aij = rinv * ElectrodeMath::safe_erfc(g_ewald_kk * r);
    KK_FLOAT erfc_eta = 0.0;
    if (!(ipoint && d_ispoint(jtype) != 0)) {
      erfc_eta = ElectrodeMath::safe_erfc(d_eta_ij(itype, jtype) * r);
      aij -= rinv * erfc_eta;
    }
    if (factor_coul < static_cast<KK_FLOAT>(1.0))
      aij -= (static_cast<KK_FLOAT>(1.0) - factor_coul) * rinv *
          (static_cast<KK_FLOAT>(1.0) - erfc_eta);
    if (i_in_sensor) Kokkos::atomic_add(&d_vec(i), static_cast<KK_ACC_FLOAT>(aij * q(j)));
    if (j_in_sensor && (!vec_inv || !i_in_sensor))
      Kokkos::atomic_add(&d_vec(j), static_cast<KK_ACC_FLOAT>(aij * qtmp));
  }
}

template<class DeviceType>
void PairLJCutCoulLongGaussKokkos<DeviceType>::compute_matrix(bigint *mpos, double **array,
                                                              int groupbit)
{
  build_device_tables();    // after all init_one calls; cheap enough at setup
  newton_pair = force->newton_pair;

  int ngroup = -1;
  for (int i = 0; i < atom->nlocal; i++)
    if (mpos[i] + 1 > ngroup) ngroup = static_cast<int>(mpos[i]) + 1;
  MPI_Allreduce(MPI_IN_PLACE, &ngroup, 1, MPI_INT, MPI_MAX, world);

  atomKK->sync(execution_space, X_MASK | TYPE_MASK | MASK_MASK);
  x = atomKK->k_x.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  // the matrix kernels read the nlocal member; keep it current (compute()
  // may not have run yet at matrix-build time)
  nlocal = atom->nlocal;

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  // host mpos -> device; host matrix rows are the fix's double** (the
  // ElectrodeMatrix full-array contract), so build on device then copy back
  const int nall = nlocal + atom->nghost;
  DAT::tdual_int_1d k_mpos("pair:gauss:mpos", nall);
  auto h_mpos = k_mpos.view_host();
  for (int i = 0; i < nall; i++) h_mpos(i) = static_cast<int>(mpos[i]);
  k_mpos.modify_host();
  k_mpos.sync<DeviceType>();
  d_mpos = k_mpos.view<DeviceType>();

  // the full matrix is small only in the sharded world; here we mirror the
  // host contract: allocate ngroup x ngroup, accumulate, copy back
  typename AT::t_kkfloat_2d k_mat("pair:gauss:matrix", ngroup, ngroup);
  d_matrix = k_mat;
  Kokkos::deep_copy(d_matrix, 0);    // kernel accumulates: fresh view is NOT guaranteed zero on GPU
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

template<class DeviceType>
void PairLJCutCoulLongGaussKokkos<DeviceType>::compute_matrix_self(bigint *mpos, double **array,
                                                                   int groupbit)
{
  const int nlocal = atom->nlocal;
  atomKK->sync(Host, Q_MASK | TYPE_MASK | MASK_MASK);

  const KK_FLOAT selfint = static_cast<KK_FLOAT>(2.0) / static_cast<KK_FLOAT>(MY_PIS) * g_ewald_kk;
  const KK_FLOAT preta = static_cast<KK_FLOAT>(2.0) / static_cast<KK_FLOAT>(MY_PIS);

  int *h_mask = atom->mask;
  int *h_type = atom->type;
  for (int i = 0; i < nlocal; i++) {
    if (h_mask[i] & groupbit) {
      const int itype = h_type[i];
      array[mpos[i]][mpos[i]] -= selfint;
      if (!ispoint[itype]) array[mpos[i]][mpos[i]] += preta * eta[itype][itype];
    }
  }
}

template<class DeviceType>
template<int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION void PairLJCutCoulLongGaussKokkos<DeviceType>::operator()(
    TagPairGaussMatrix<NEWTON_PAIR>, const int &ii) const
{
  const int i = d_ilist[ii];
  // ghost rows own no matrix entry of their own; their contribution is
  // folded back through their local twin's row (the host convention relies
  // on the local row's ghost-j visits). Skip only the row-write, not the
  // atom: ghost i has d_mpos(i) < 0 by tag_to_iele construction? No: ghosts
  // carry their twin's tag, so d_mpos(ghost) >= 0 and iterating a ghost row
  // would double-count every pair already visited from the local side.
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
      KK_FLOAT aij = rinv * ElectrodeMath::safe_erfc(g_ewald_kk * r);
      KK_FLOAT erfc_eta = 0.0;
      if (!(ipoint && d_ispoint(jtype) != 0)) {
        erfc_eta = ElectrodeMath::safe_erfc(d_eta_ij(itype, jtype) * r);
        aij -= rinv * erfc_eta;
      }
      if (factor_coul < static_cast<KK_FLOAT>(1.0))
        aij -= (static_cast<KK_FLOAT>(1.0) - factor_coul) * rinv *
            (static_cast<KK_FLOAT>(1.0) - erfc_eta);
      // full-list write rule (mirrors the host kernels): each local row
      // writes only its own [ipos][jpos] at full weight; the owner of the
      // mirror entry fills it from its row, and compute_array symmetrizes
      // after the MPI row exchange (a no-op by construction)
      d_matrix(ipos, jpos) += static_cast<KK_ACC_FLOAT>(aij);
    }
  }
}

template class PairLJCutCoulLongGaussKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLJCutCoulLongGaussKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS