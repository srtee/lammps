/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Device-resident CG recurrence shared by ElectrodeCGKokkos (algo cg) and
   ElectrodeMatCGKokkos (algo mat_cg).  Every routine mirrors the host
   ElectrodeCG numerics step for step: cg_solve (projected CG with the
   20-step residual reset), predict_q/push_history (ASPC), and
   constraint_projection.  See electrode_cg.cpp for the derivations.
------------------------------------------------------------------------- */

#include "electrode_cg_device_kokkos.h"

#include "electrode_pair_kokkos.h"

#include "atom.h"
#include "atom_masks.h"
#include "electrode_mat_cg.h"
#include "error.h"
#include "fix_electrode_conp.h"
#include "fix_electrode_conp_kokkos.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "atom_kokkos.h"
#include "neigh_list_kokkos.h"
#include "pair.h"
#include "pppm_electrode_kokkos.h"
#include "update.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <typeinfo>

namespace LAMMPS_NS {

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_setup()
{
  fix_kk = dynamic_cast<FixElectrodeConpKokkos<DeviceType> *>(this->fix);
  if (fix_kk == nullptr) this->error->all(FLERR, "Device electrode CG requires fix electrode/conp/kk");
  if (fix_kk->get_cg_neighlist() == nullptr)
    this->error->all(FLERR, "Device electrode CG requires the device CG neighbor list");

  dev_elyt = fix_kk->device_elyt_vector();
  if (dev_elyt == nullptr) this->error->all(FLERR, "Device electrode CG without an electrolyte vector");
  if (fix_kk->device_elyt_group() != dev_elyt->get_groupbit())
    this->error->all(FLERR, "Device electrode CG sensor group does not match the electrolyte vector");

  pair_kk = dynamic_cast<ElectrodePairKokkos<DeviceType> *>(this->force->pair);
  if (pair_kk == nullptr)
    this->error->all(FLERR, "Pair style does not implement the ELECTRODE device interface");
  kspace_kk = nullptr;
  if (dev_elyt->get_kspaceflag()) {
    kspace_kk = dynamic_cast<PPPMElectrodeKokkos<DeviceType> *>(dev_elyt->get_electrode_kspace());
    if (kspace_kk == nullptr)
      this->error->all(FLERR, "KSpace style does not implement the ELECTRODE device vector path");
  }

  dev_mode = fix_kk->device_resident();
}

/* ----------------------------------------------------------------------
   refresh all solver-sized device state after update_solver(); the base
   call has already rebuilt taglist/iele_to_group/nele/nele_world/ngroups
------------------------------------------------------------------------- */

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_refresh_maps()
{
  const int nele_local = this->nele;
  if ((int) d_bvec.extent(0) < nele_local) {
    d_bvec = dview("electrode/cgdev:d_bvec", nele_local);
    d_b = dview("electrode/cgdev:d_b", nele_local);
    d_x = dview("electrode/cgdev:d_x", nele_local);
    d_q = dview("electrode/cgdev:d_q", nele_local);
    d_r = dview("electrode/cgdev:d_r", nele_local);
    d_d = dview("electrode/cgdev:d_d", nele_local);
    d_p = dview("electrode/cgdev:d_p", nele_local);
    d_y = dview("electrode/cgdev:d_y", nele_local);
    d_hist = Kokkos::View<double **, Kokkos::LayoutRight, DeviceType>(
        "electrode/cgdev:d_hist", this->predictor_cols > 0 ? this->predictor_cols : 1, nele_local);
    d_hist_tmp = Kokkos::View<double **, Kokkos::LayoutRight, DeviceType>(
        "electrode/cgdev:d_hist_tmp", this->predictor_cols > 0 ? this->predictor_cols : 1,
        nele_local);
    d_cb = dview("electrode/cgdev:d_cb", nele_local);
    d_cx = dview("electrode/cgdev:d_cx", nele_local);
    d_iele2grp = iview("electrode/cgdev:d_iele2grp", nele_local);
  }
  if ((int) d_vg.extent(0) < this->ngroups) {
    d_vg = dview("electrode/cgdev:d_vg", this->ngroups);
    d_sums = dview("electrode/cgdev:d_sums", this->ngroups);
    d_gcounts = iview("electrode/cgdev:d_gcounts", this->ngroups);
  }

  dev_refresh_imap();

  {
    auto h_iele2grp = Kokkos::create_mirror_view(d_iele2grp);
    for (int i = 0; i < nele_local; i++) h_iele2grp(i) = this->iele_to_group[i];
    Kokkos::deep_copy(d_iele2grp, h_iele2grp);
  }
  {
    // GLOBAL per-group atom counts (host update_solver makes ngroups global;
    // the projection normalizes by the world count)
    std::vector<int> counts(this->ngroups, 0);
    for (int i = 0; i < nele_local; i++) counts[this->iele_to_group[i]]++;
    MPI_Allreduce(MPI_IN_PLACE, counts.data(), this->ngroups, MPI_INT, MPI_SUM, this->world);
    auto m_counts = Kokkos::create_mirror_view(d_gcounts);
    for (int g = 0; g < this->ngroups; g++) m_counts(g) = counts[g];
    Kokkos::deep_copy(d_gcounts, m_counts);
  }

  // ASPC weights, padded to cols+2 like the host rows (Kolafa 2003)
  const int cols = this->predictor_cols;
  if (cols > 0) {
    if ((int) d_w.extent(0) < cols) {
      d_w = Kokkos::View<double **, Kokkos::LayoutRight, DeviceType>(
          "electrode/cgdev:d_w", cols, cols + 2);
    }
    auto h_w = Kokkos::create_mirror_view(d_w);
    for (int k = 0; k < cols; k++)
      for (int j = 0; j < cols + 2; j++) h_w(k, j) = 0.0;
    for (int k = 0; k < cols; k++)
      for (size_t j = 0; j < this->predictor_weights[k].size() && (int) j < cols + 2; j++)
        h_w(k, j) = this->predictor_weights[k][j];
    Kokkos::deep_copy(d_w, h_w);
  }

  dev_refresh_solver_maps();
}

/* ----------------------------------------------------------------------
   atom -> iele map on the device; rebuilt only when the atom/tag layout
   changes (same fingerprint scheme as ElectrodeInvKokkos)
------------------------------------------------------------------------- */

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_refresh_imap()
{
  const int nele_local = this->nele;
  const int nlocal = this->atom->nlocal;
  if ((int) d_imap.extent(0) < nlocal) {
    d_imap = iview("electrode/cgdev:d_imap", nlocal);
    imap_fingerprint = 0;
  }

  static_cast<AtomKokkos *>(this->atom)->sync(Host, TAG_MASK);
  uint64_t new_fingerprint = 0x9e3779b97f4a7c15ULL;
  new_fingerprint ^= (uint64_t) nele_local * 0x9e3779b1ULL;
  for (int i = 0; i < nlocal; i++) new_fingerprint ^= (uint64_t) this->atom->tag[i] + i;
  if (new_fingerprint != imap_fingerprint) {
    imap_fingerprint = new_fingerprint;
    // ordering changed (device sort or migration): refresh the host atom map
    // before translating tags, then fill EVERY atom's slot (-1 = not
    // electrode) -- mirror entries left uninitialized read garbage on device
    if (this->atom->map_style != Atom::MAP_NONE) this->atom->map_set();
    h_imap = Kokkos::create_mirror_view(d_imap);
    for (int i = 0; i < nlocal; i++) {
      int ipos = -1;
      for (int k = 0; k < nele_local; k++)
        if (this->taglist[k] == this->atom->tag[i]) {
          ipos = k;
          break;
        }
      h_imap(i) = ipos;
    }
    h_rowpos.assign(nele_local, -1);
    for (int i = 0; i < nlocal; i++)
      if (h_imap(i) >= 0) h_rowpos[h_imap(i)] = i;
    if ((int) d_rowpos.extent(0) < nele_local)
      d_rowpos = iview("electrode/cgdev:d_rowpos", nele_local);
    auto h_rp = Kokkos::create_mirror_view(d_rowpos);
    for (int k = 0; k < nele_local; k++) h_rp(k) = h_rowpos[k];
    Kokkos::deep_copy(d_imap, h_imap);
    Kokkos::deep_copy(d_rowpos, h_rp);
  }
}

/* ----------------------------------------------------------------------
   device b-assembly, mirror of ElectrodeInvKokkos::assemble_b_device +
   set_elyt_pot: pair + self (device), kspace vector (device) + corr
   (host, atom-order staging), electronegativity (host, device add)
------------------------------------------------------------------------- */

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_set_elyt_pot()
{
  this->elyt_step = this->update->ntimestep;
  dev_refresh_imap();
  Kokkos::deep_copy(d_bvec, 0.0);

  const int sensor_grpbit = dev_elyt->get_groupbit();
  const int source_grpbit = dev_elyt->get_source_grpbit();
  const bool invert_source = dev_elyt->get_invert_source();

  class NeighList *neigh = fix_kk->get_cg_neighlist();
  pair_kk->compute_vector_nele(neigh, d_bvec, d_imap, sensor_grpbit, source_grpbit, invert_source);
  pair_kk->compute_vector_self_nele(d_bvec, d_imap, sensor_grpbit, source_grpbit, invert_source);

  if (dev_elyt->get_kspaceflag()) {
    // re-cast in case the kspace object was swapped (fix tune/kspace)
    kspace_kk = dynamic_cast<PPPMElectrodeKokkos<DeviceType> *>(dev_elyt->get_electrode_kspace());
    if (kspace_kk == nullptr)
      this->error->all(FLERR, "KSpace style does not implement the ELECTRODE device vector path");
    kspace_kk->compute_vector_nele(d_bvec, d_imap, sensor_grpbit, source_grpbit, invert_source);
    double *corr;
    this->memory->create(corr, this->atom->nmax, "electrode/cgdev:corr");
    memset(corr, 0, this->atom->nmax * sizeof(double));
    kspace_kk->compute_vector_corr(corr, sensor_grpbit, source_grpbit, invert_source);
    auto h_bv = Kokkos::create_mirror_view(d_bvec);
    Kokkos::deep_copy(h_bv, d_bvec);
    for (int i = 0; i < this->nele; i++)
      h_bv(i) += static_cast<KK_ACC_FLOAT>(corr[h_rowpos[i]]);
    Kokkos::deep_copy(d_bvec, h_bv);
    this->memory->destroy(corr);
  }

  const int en_idx = this->fix->electronegativity_index();
  if (en_idx >= 0) {
    if ((int) d_en.extent(0) < this->nele) d_en = dview("electrode/cgdev:d_en", this->nele);
    auto h_en = Kokkos::create_mirror_view(d_en);
    for (int i = 0; i < this->nele; i++)
      h_en(i) = this->atom->dvector[en_idx][h_rowpos[i]] / this->force->qqrd2e;
    Kokkos::deep_copy(d_en, h_en);
    auto lbv = d_bvec;
    auto len = d_en;
    const int nele_local = this->nele;
    Kokkos::parallel_for(
        "electrode/cgdev:en", Kokkos::RangePolicy<DeviceType>(0, nele_local),
        KOKKOS_LAMBDA(const int i) { lbv(i) += len(i); });
  }
}

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_sync_bvec_host()
{
  auto h_bv = Kokkos::create_mirror_view(d_bvec);
  Kokkos::deep_copy(h_bv, d_bvec);
  if ((int) this->bvec.size() != this->nele) this->bvec.resize(this->nele);
  for (int i = 0; i < this->nele; i++) this->bvec[i] = h_bv(i);
}

/* ----------------------------------------------------------------------
   the solve: mirror of ElectrodeCG::solve -- rhs b = evscale*v[g] - bvec,
   predictor, projected CG, history push.  d_q holds the solution in
   taglist order on the device.
------------------------------------------------------------------------- */

template<class DeviceType, class HostSolver>
std::vector<double> ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_solve(std::vector<double> v)
{
  if (this->constraint == ChargeSolver::ChargeConstraint::GROUP) v.assign(v.size(), 0.);

  {
    auto h_vg = Kokkos::create_mirror_view(d_vg);
    for (int g = 0; g < this->ngroups; g++) h_vg(g) = v[g];
    Kokkos::deep_copy(d_vg, h_vg);
  }

  {
    auto lb = d_b;
    auto lbv = d_bvec;
    auto lvg = d_vg;
    auto l2g = d_iele2grp;
    auto ev = this->evscale;
    const int nele_local = this->nele;
    Kokkos::parallel_for(
        "electrode/cgdev:rhs", Kokkos::RangePolicy<DeviceType>(0, nele_local),
        KOKKOS_LAMBDA(const int i) { lb(i) = ev * lvg(l2g(i)) - lbv(i); });
  }
  this->applied_psi = v;
  this->sb_stale = true;
  this->a_cached_flag = false;
  this->ncall++;

  dev_predict_q();
  dev_cg_solve(d_b, d_x, true, this->nele_world);
  Kokkos::deep_copy(d_q, d_x);
  // host mirror for set_charges/thermo consumers (one D2H per step)
  auto h_q = Kokkos::create_mirror_view(d_q);
  Kokkos::deep_copy(h_q, d_q);
  if ((int) this->q_ele.size() != this->nele) this->q_ele.resize(this->nele);
  for (int i = 0; i < this->nele; i++) this->q_ele[i] = h_q(i);

  return this->q_ele;
}

/* ----------------------------------------------------------------------
   mirror of ElectrodeCG::predict_q: first solve seeds x from the current
   device charges; afterwards ASPC over q + history.  The history shift
   (host lines 263-273) runs with the PRE-predict count and stores the
   CURRENT charges in hist[0].
------------------------------------------------------------------------- */

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_predict_q()
{
  const int cols = this->predictor_cols;
  if (!this->predictor_count || cols <= 0) {
    dev_seed_from_atoms(d_x);
  } else {
    const int k = this->predictor_count - 1;
    auto lw = d_w;
    auto lhist = d_hist;
    auto lq = d_q;
    auto lx = d_x;
    const int nele_local = this->nele;
    Kokkos::parallel_for(
        "electrode/cgdev:predict", Kokkos::RangePolicy<DeviceType>(0, nele_local),
        KOKKOS_LAMBDA(const int i) {
          double acc = lw(k, 0) * lq(i);
          for (int j = 1; j <= k + 1; j++) acc += lw(k, j) * lhist(j - 1, i);
          lx(i) = acc;
        });
  }

  if (cols > 0) {
    const int count = this->predictor_count;
    auto lhist = d_hist;
    auto ltmp = d_hist_tmp;
    auto lq = d_q;
    const int nele_local = this->nele;
    Kokkos::parallel_for(
        "electrode/cgdev:push", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {cols, nele_local}),
        KOKKOS_LAMBDA(const int m, const int i) {
          if (m == 0) ltmp(m, i) = lq(i);
          else if (m <= count) ltmp(m, i) = lhist(m - 1, i);
          else ltmp(m, i) = lhist(m, i);
        });
    std::swap(this->d_hist_tmp, this->d_hist);
    if (this->predictor_count < cols) this->predictor_count++;
  }
}

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_seed_from_atoms(dview &d_x)
{
  auto *atomKK = static_cast<AtomKokkos *>(this->atom);
  atomKK->sync(Device, Q_MASK);
  auto dq = atomKK->k_q.template view<DeviceType>();
  auto lmap = d_imap;
  auto lx = d_x;
  const int nele_local = this->nele;
  Kokkos::parallel_for(
      "electrode/cgdev:seed", Kokkos::RangePolicy<DeviceType>(0, nele_local),
      KOKKOS_LAMBDA(const int i) { lx(i) = dq(lmap(i)); });
}

/* ----------------------------------------------------------------------
   projected CG, mirror of ElectrodeCG::cg_solve.  evscale lives inside
   the matvec, projections run on x/r as the host does, and every 20
   iterations the residual is recomputed from scratch.
------------------------------------------------------------------------- */

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_cg_solve(
    const dview &d_b, dview &d_x, bool constrain, int max_iter)
{
  const int nele_local = this->nele;

  dev_project(d_x, d_x, constrain && (this->constraint != ChargeSolver::ChargeConstraint::NONE));

  dev_matvec(d_x, d_y);    // y = (evscale*A) x
  {
    auto lr = d_r;
    auto lb = d_b;
    auto ly = d_y;
    Kokkos::parallel_for(
        "electrode/cgdev:res", Kokkos::RangePolicy<DeviceType>(0, nele_local),
        KOKKOS_LAMBDA(const int i) { lr(i) = lb(i) - ly(i); });
  }
  // projected CG, faithful mirror of ElectrodeCG::cg_solve: conjugate
  // search direction with the residual recurrence, exact refresh of both
  // every 20 iterations. Plain steepest descent (fresh d = P r each step)
  // diverges on the projected operator -- it is indefinite in the
  // sum-charge-free subspace.
  dev_project(d_r, d_d, false);    // d = P r
  double dot_old = dev_dot(d_r, d_d);
  double delta = dot_old;
  const bool do_constrain =
      constrain && (this->constraint != ChargeSolver::ChargeConstraint::NONE);
  for (int k = 0; k < max_iter; k++, this->nstep++) {
    if (delta <= this->threshold) break;
    dev_matvec(d_d, d_y);    // y = (evscale*A) d
    const double den = dev_dot(d_d, d_y);
    if (den == 0.0) break;
    const double alpha = delta / den;
    {
      auto lx = d_x;
      auto ld = d_d;
      auto la = alpha;
      Kokkos::parallel_for(
          "electrode/cgdev:axpy", Kokkos::RangePolicy<DeviceType>(0, nele_local),
          KOKKOS_LAMBDA(const int i) { lx(i) += la * ld(i); });
    }
    if (do_constrain && ((k + 1) % 20 == 0)) {
      // avoid shifting the residual: exact reprojection + refresh
      dev_project(d_x, d_x, constrain);
      dev_matvec(d_x, d_y);
      auto lr = d_r;
      auto lb = d_b;
      auto ly = d_y;
      Kokkos::parallel_for(
          "electrode/cgdev:res", Kokkos::RangePolicy<DeviceType>(0, nele_local),
          KOKKOS_LAMBDA(const int i) { lr(i) = lb(i) - ly(i); });
    } else {
      auto lr = d_r;
      auto ly = d_y;
      auto la = alpha;
      Kokkos::parallel_for(
          "electrode/cgdev:raxpy", Kokkos::RangePolicy<DeviceType>(0, nele_local),
          KOKKOS_LAMBDA(const int i) { lr(i) -= la * ly(i); });
    }
    dev_project(d_r, d_p, false);    // p = P r
    const double dot_new = dev_dot(d_r, d_p);
    {
      auto ld = d_d;
      auto lp = d_p;
      auto beta = dot_new / dot_old;
      Kokkos::parallel_for(
          "electrode/cgdev:daxpy", Kokkos::RangePolicy<DeviceType>(0, nele_local),
          KOKKOS_LAMBDA(const int i) { ld(i) = lp(i) + beta * ld(i); });
    }
    delta = dev_dot(d_r, d_d);
    dot_old = dot_new;
  }
  if ((delta > this->threshold) && (this->comm->me == 0))
    this->error->warning(FLERR, "CG threshold not reached");
}

template<class DeviceType, class HostSolver>
double ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_dot(const dview &a, const dview &b)
{
  double local = 0.0;
  const int nele_local = this->nele;
  Kokkos::parallel_reduce(
      "electrode/cgdev:dot", Kokkos::RangePolicy<DeviceType>(0, nele_local),
      KOKKOS_LAMBDA(const int i, double &acc) { acc += a(i) * b(i); }, local);
  MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_DOUBLE, MPI_SUM, this->world);
  return local;
}

template<class DeviceType, class HostSolver>
double ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_sum(const dview &a)
{
  double local = 0.0;
  const int nele_local = this->nele;
  Kokkos::parallel_reduce(
      "electrode/cgdev:sum", Kokkos::RangePolicy<DeviceType>(0, nele_local),
      KOKKOS_LAMBDA(const int i, double &acc) { acc += a(i); }, local);
  MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_DOUBLE, MPI_SUM, this->world);
  return local;
}

/* ----------------------------------------------------------------------
   constraint projection on the device, mirror of
   ElectrodeCG::constraint_projection
------------------------------------------------------------------------- */

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_project(
    const dview &src, dview &dst, bool correction)
{
  const int nele_local = this->nele;
  switch (this->constraint) {
    case ChargeSolver::ChargeConstraint::NONE:
      if (&src != &dst) Kokkos::deep_copy(dst, src);
      return;
    case ChargeSolver::ChargeConstraint::SINGLE: {
      double sum = dev_sum(src);
      if (correction) sum -= this->qtotal;
      const double c = sum / this->nele_world;
      auto ls = src;
      auto ld = dst;
      Kokkos::parallel_for(
          "electrode/cgdev:proj1", Kokkos::RangePolicy<DeviceType>(0, nele_local),
          KOKKOS_LAMBDA(const int i) { ld(i) = ls(i) - c; });
      return;
    }
    case ChargeSolver::ChargeConstraint::GROUP: {
      Kokkos::deep_copy(d_sums, 0.0);
      auto ls = src;
      auto lsums = d_sums;
      auto l2g = d_iele2grp;
      const int ngroups_local = this->ngroups;
      Kokkos::parallel_for(
          "electrode/cgdev:gsum", Kokkos::RangePolicy<DeviceType>(0, nele_local),
          KOKKOS_LAMBDA(const int i) { Kokkos::atomic_add(&lsums(l2g(i)), ls(i)); });
      auto h_sums = Kokkos::create_mirror_view(d_sums);
      Kokkos::deep_copy(h_sums, d_sums);
      MPI_Allreduce(MPI_IN_PLACE, h_sums.data(), this->ngroups, MPI_DOUBLE, MPI_SUM, this->world);
      auto h_counts = Kokkos::create_mirror_view(d_gcounts);
      Kokkos::deep_copy(h_counts, d_gcounts);
      for (int g = 0; g < this->ngroups; g++) {
        if (correction) h_sums(g) -= this->qtotal_group[g];
        h_sums(g) /= h_counts(g);
      }
      Kokkos::deep_copy(d_sums, h_sums);
      auto lc = d_sums;
      auto ld = dst;
      Kokkos::parallel_for(
          "electrode/cgdev:projg", Kokkos::RangePolicy<DeviceType>(0, nele_local),
          KOKKOS_LAMBDA(const int i) { ld(i) = ls(i) - lc(l2g(i)); });
      return;
    }
    default:
      this->error->all(FLERR, "Constraint not implemented");
  }
}

/* ----------------------------------------------------------------------
   scatter a taglist-order device charge vector into atom->q (device) and
   the ghosts via forward_comm: the matvec's device-side charge staging.
   The production solve scatters d_q each step via scatter_device().
------------------------------------------------------------------------- */

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_scatter_into(const dview &src)
{
  auto *atomKK = static_cast<AtomKokkos *>(this->atom);
  auto dq = atomKK->k_q.template view<DeviceType>();
  auto lmap = d_rowpos;    // row k -> atom position (d_imap is the inverse!)
  auto ls = src;
  const int nele_local = this->nele;
  Kokkos::parallel_for(
      "electrode/cgdev:scatter", Kokkos::RangePolicy<DeviceType>(0, nele_local),
      KOKKOS_LAMBDA(const int i) { dq(lmap(i)) = static_cast<double>(ls(i)); });
  atomKK->modified(Device, Q_MASK);
  this->comm->forward_comm(this->fix);
}

template<class DeviceType, class HostSolver>
void ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_scatter_q()
{
  dev_scatter_into(d_q);
}

/* ----------------------------------------------------------------------
   mirror of ElectrodeCG::compute_potentials (electrode/thermo support):
   a_cached = (evscale*A) q with q read straight off the device
------------------------------------------------------------------------- */

template<class DeviceType, class HostSolver>
std::vector<double> ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::dev_compute_potentials()
{
  if (!this->a_cached_flag) {
    this->a_cached_flag = true;
    dev_matvec(d_q, d_y);
    auto h_y = Kokkos::create_mirror_view(d_y);
    Kokkos::deep_copy(h_y, d_y);
    if ((int) this->a_cached.size() != this->nele) this->a_cached.resize(this->nele);
    for (int i = 0; i < this->nele; i++) this->a_cached[i] = h_y(i);
  }
  dev_sync_bvec_host();

  const int ngroups_local = this->ngroups;
  std::vector<double> pots(ngroups_local, 0.);
  std::vector<int> counts(ngroups_local, 0);
  for (int i = 0; i < this->nele; i++) {
    const int g = this->iele_to_group[i];
    pots[g] += (this->a_cached[i] + this->bvec[i]) / this->evscale;
    counts[g]++;
  }
  MPI_Allreduce(MPI_IN_PLACE, pots.data(), ngroups_local, MPI_DOUBLE, MPI_SUM, this->world);
  MPI_Allreduce(MPI_IN_PLACE, counts.data(), ngroups_local, MPI_INT, MPI_SUM, this->world);
  for (int g = 0; g < ngroups_local; g++) pots[g] /= counts[g];
  return pots;
}

/* ----------------------------------------------------------------------
   cg_solve override: compute_macro_calibration (vacuum_capacitance for
   the CG thermo) drives the host virtual with arbitrary rhs vectors.
   Device lane: scratch views + dev_cg_solve, then restore the production
   charges (host calibration restores via fix->set_charges(q_ele)).
------------------------------------------------------------------------- */

template<class DeviceType, class HostSolver>
std::vector<double> ElectrodeCGDeviceKokkos<DeviceType, HostSolver>::cg_solve(
    std::vector<double> b, const std::vector<double> &x_init, bool constrain, int max_iter)
{
  if (!dev_mode) return HostSolver::cg_solve(std::move(b), x_init, constrain, max_iter);

  const int nele_local = this->nele;
  if ((int) d_cb.extent(0) < nele_local) {
    d_cb = dview("electrode/cgdev:d_cb", nele_local);
    d_cx = dview("electrode/cgdev:d_cx", nele_local);
  }
  {
    auto h_b = Kokkos::create_mirror_view(d_cb);
    for (int i = 0; i < nele_local; i++) h_b(i) = b[i];
    Kokkos::deep_copy(d_cb, h_b);
    auto h_x = Kokkos::create_mirror_view(d_cx);
    for (int i = 0; i < nele_local; i++) h_x(i) = x_init[i];
    Kokkos::deep_copy(d_cx, h_x);
  }
  dev_cg_solve(d_cb, d_cx, constrain, max_iter);
  std::vector<double> out(nele_local);
  auto h_x = Kokkos::create_mirror_view(d_cx);
  Kokkos::deep_copy(h_x, d_cx);
  for (int i = 0; i < nele_local; i++) out[i] = h_x(i);
  dev_scatter_into(this->d_q);    // restore electrode charges on the atoms
  return out;
}

/* explicit instantiation: vector-CG pair + matrix-CG pair */
template class ElectrodeCGDeviceKokkos<LMPDeviceType, ElectrodeCG>;
#ifdef LMP_KOKKOS_GPU
template class ElectrodeCGDeviceKokkos<LMPHostType, ElectrodeCG>;
#endif
template class ElectrodeCGDeviceKokkos<LMPDeviceType, ElectrodeMatCG>;

}    // namespace LAMMPS_NS
