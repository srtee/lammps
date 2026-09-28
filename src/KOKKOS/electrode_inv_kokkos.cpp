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
   Contributing author: Shern Tee (UQ)
------------------------------------------------------------------------- */

#include "electrode_inv_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "electrode_vector.h"
#include "error.h"
#include "fix_electrode_conp.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include "pair.h"
#include "update.h"

#include <algorithm>

namespace LAMMPS_NS {

namespace {

// fold the corr staging into b; KOKKOS_LAMBDA needs a namespace-scope
// enclosing function (nvcc rejects extended lambdas in private members)
template<class V>
void corr_fold_dev(V v_b, V v_c, int n)
{
  Kokkos::parallel_for("electrode/inv/kk:corr_fold", n,
                       KOKKOS_LAMBDA(const int i) { v_b(i) += v_c(i); });
}

}    // namespace


template<class DeviceType>
ElectrodeInvKokkos<DeviceType>::ElectrodeInvKokkos(class LAMMPS *lmp_in, class FixElectrodeConp *fix_in) :
    ElectrodeInv(lmp_in), fix(fix_in)
{
}

template<class DeviceType>
ElectrodeInvKokkos<DeviceType>::~ElectrodeInvKokkos() noexcept
{
  if (corr_scratch != nullptr) memory->destroy(corr_scratch);
  if (timer_print && comm->me == 0)
    utils::logmesg(lmp,
                   "InvKK legs: bar={:.4g} pull={:.4g} pair={:.4g} ks={:.4g} corr={:.4g} "
                   "b2h={:.4g} gth={:.4g} p2d={:.4g} mv={:.4g} q2h={:.4g} nsolve={}\n",
                   t_bar, t_pull, t_pair, t_ks, t_corr, t_b2h, t_gth, t_p2d, t_mv, t_q2h,
                   nsolve);
}

/* ----------------------------------------------------------------------
   setup_device(): locate the device pair/kspace interfaces and mirror
   the fragment matrix (nele_local x nele_world) at setup_solver() /
   update_solver() time.
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeInvKokkos<DeviceType>::setup_solver(int groupbit_in, std::unordered_map<tagint, int> tag_to_iele,
                                                  std::vector<int> group_bits_in, bool ffield,
                                                  bool timer_flag)
{
  ElectrodeInv::setup_solver(groupbit_in, std::move(tag_to_iele), std::move(group_bits_in), ffield,
                             timer_flag);
  timer_print = timer_flag;
  setup_device();
  refresh_device_matrix();
}
template<class DeviceType>
void ElectrodeInvKokkos<DeviceType>::setup_device()
{
  dev_vec = fix->device_elyt_vector();
  if (dev_vec == nullptr) error->all(FLERR, "ElectrodeInvKokkos without an electrolyte vector");
  if (fix->device_elyt_group() != dev_vec->get_groupbit())
    error->all(FLERR, "ElectrodeInvKokkos sensor group does not match the electrolyte vector");

  fix_kk = dynamic_cast<FixElectrodeConpKokkos<DeviceType> *>(fix);
  if (fix_kk == nullptr) error->all(FLERR, "ElectrodeInvKokkos requires fix electrode/conp/kk");
  if (fix_kk->get_cg_neighlist() == nullptr)
    error->all(FLERR, "ElectrodeInvKokkos requires the device vector neighbor list");

  pair_kk = dynamic_cast<ElectrodePairKokkos<DeviceType> *>(force->pair);
  if (pair_kk == nullptr)
    error->all(FLERR, "Pair style does not implement the ELECTRODE device interface");
  if (dev_vec->get_kspaceflag()) {
    kspace_kk = dynamic_cast<PPPMElectrodeKokkos<DeviceType> *>(dev_vec->get_electrode_kspace());
    if (kspace_kk == nullptr)
      error->all(FLERR, "KSpace style does not implement the ELECTRODE device vector path");
  }
}

/* ----------------------------------------------------------------------
   called at construction and after every atom migration: base re-compacts
   cap_frag so fragment r is the row of iele_local[r]; mirror it to device
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeInvKokkos<DeviceType>::update_solver(std::vector<tagint> taglist,
                                                   std::vector<int> iele_to_group_local)
{
  ElectrodeInv::update_solver(std::move(taglist), std::move(iele_to_group_local));
  if (fix_kk != nullptr) refresh_device_matrix();
}

template<class DeviceType>
void ElectrodeInvKokkos<DeviceType>::refresh_device_matrix()
{
  const int nrow = nlocalele;
  const int ncol = nele_world;
  if ((int) d_capfrag.extent(0) != nrow || (int) d_capfrag.extent(1) != ncol)
    d_capfrag = decltype(d_capfrag)("electrode/inv/kk:d_capfrag", nrow, ncol);
  if (nrow == 0 || ncol == 0) return;
  auto h_cap = Kokkos::create_mirror_view(d_capfrag);
  for (int r = 0; r < nrow; r++) {
    const auto &row = cap_frag[r];
    if ((int) row.size() != ncol)
      error->all(FLERR, "ElectrodeInvKokkos: fragment row {} has wrong width", r);
    for (int j = 0; j < ncol; j++) h_cap(r, j) = static_cast<KK_ACC_FLOAT>(row[j]);
  }
  Kokkos::deep_copy(d_capfrag, h_cap);
}

/* ----------------------------------------------------------------------
   b vector assembly + matvec on device.  b_nall is ignored (the host
   potential array is never built); the pipeline is:

   1. pair+self+kspace kernels assemble d_b in taglist order (corr folded
      on device via d_corr staging); the atom->taglist imap is cached on
      device behind a tag-array fingerprint
   2. gather to world-iele order (same convention as buffer_and_gather)
   3. one H2D (potentials) and one D2H (charges) per solve, both through
      persistent host mirrors (no per-solve mirror allocations);
      sb_charges accumulate host-side, exactly as the base class does
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeInvKokkos<DeviceType>::assemble_b_device()
{
  const int nele_local = nlocalele;
  const int nlocal = atom->nlocal;
  if ((int) d_b.extent(0) < nele_local) {
    d_b = typename ArrayTypes<DeviceType>::t_kkacc_1d("electrode/inv/kk:d_b", nele_local);
    h_b = Kokkos::create_mirror_view(d_b);
  }
  Kokkos::deep_copy(d_b, 0.0);
  // the device solve bypasses base buffer_and_gather, which is where the
  // world-iele gather staging is normally allocated
  if (buf_gathered == nullptr)
    memory->create(buf_gathered, nele_world, "ElectrodeInv:buf_gathered");
  // Under the device exchange/sort pipeline the device views are the
  // authoritative copies: exchange_device() lands migrated atoms there and
  // sort_device() permutes them in place, so the host arrays -- and the
  // tag->local map, a host hash -- are stale until pulled.  Everything the
  // host path reads below (tags for the imap fingerprint, kspace boundary
  // corr via atom->x/q/mask, the electronegativity term) pulls current
  // here; the flag-gated sync is a no-op while the device is quiet, and
  // AtomKokkos::sort()'s legacy branch does exactly this pull.
  double t0 = MPI_Wtime();
  static_cast<AtomKokkos *>(atom)->sync(Host, TAG_MASK | MASK_MASK | X_MASK | Q_MASK);
  t_pull += MPI_Wtime() - t0;
  if ((int) d_imap.extent(0) < nlocal) {
    d_imap = typename ArrayTypes<DeviceType>::t_int_1d("electrode/inv/kk:d_imap", nlocal);
    h_imap = Kokkos::create_mirror_view(d_imap);
  }
  uint64_t fp = 1469598103934665603ull;
  for (int i = 0; i < nlocal; i++) fp = (fp ^ (uint64_t) atom->tag[i]) * 1099511628211ull;
  for (int i = 0; i < nele_local; i++) fp = (fp ^ (uint64_t) taglist_local[i]) * 1099511628211ull;
  fp = (fp ^ (uint64_t) nlocal) * 1099511628211ull;
  if (fp != imap_fingerprint) {
    // ordering changed (device sort or migration): rebuild the map so
    // atom->map(tag) addresses the pulled layout
    if (atom->map_style != Atom::MAP_NONE) atom->map_set();
    for (int i = 0; i < nlocal; i++) {
      int ipos = -1;
      for (int k = 0; k < nele_local; k++)
        if (taglist_local[k] == atom->tag[i]) {
          ipos = k;
          break;
        }
      h_imap(i) = ipos;
    }
    Kokkos::deep_copy(d_imap, h_imap);
    imap_fingerprint = fp;
  }

  // electrolyte charges (incl. ghosts) must be current on the device
  fix_kk->device_charge_sync();

  const int sensor_grpbit = dev_vec->get_groupbit();
  const int source_grpbit = dev_vec->get_source_grpbit();
  const bool invert_source = dev_vec->get_invert_source();
  class NeighList *neigh = fix_kk->get_cg_neighlist();

  double t1 = MPI_Wtime();
  pair_kk->compute_vector_nele(neigh, d_b, d_imap, sensor_grpbit, source_grpbit, invert_source);
  pair_kk->compute_vector_self_nele(d_b, d_imap, sensor_grpbit, source_grpbit, invert_source);
  t_pair += MPI_Wtime() - t1;

  if (dev_vec->get_kspaceflag()) {
    double t2 = MPI_Wtime();
    kspace_kk->compute_vector_nele(d_b, d_imap, sensor_grpbit, source_grpbit, invert_source);
    t_ks += MPI_Wtime() - t2;
    double t3 = MPI_Wtime();
    // boundary corrections stay host (O(N) sums), then fold into d_b
    double *corr;
    memory->create(corr, atom->nmax, "electrode/inv/kk:corr_scratch");
    memset(corr, 0, atom->nmax * sizeof(double));
    kspace_kk->compute_vector_corr(corr, sensor_grpbit, source_grpbit, invert_source);
    auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_b);
    for (int i = 0; i < nele_local; i++)
      h_b(i) += static_cast<KK_ACC_FLOAT>(corr[atom->map(taglist_local[i])]);
    Kokkos::deep_copy(d_b, h_b);
    memory->destroy(corr);
    t_corr += MPI_Wtime() - t3;
  }
}

template<class DeviceType>
void ElectrodeInvKokkos<DeviceType>::set_elyt_pot(double *b_nall)
{
  (void) b_nall;    // ignored: the b vector is assembled on the device
  elyt_step = update->ntimestep;
  if (fix_kk == nullptr) error->all(FLERR, "ElectrodeInvKokkos used before setup_solver");

  ++nsolve;
  double t0 = MPI_Wtime();
  MPI_Barrier(world);
  t_bar += MPI_Wtime() - t0;
  double mult_start = MPI_Wtime();

  assemble_b_device();

  // D2H b, add electronegativity correction (per-atom property, host)
  const int nele_local = nlocalele;
  double t1 = MPI_Wtime();
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_b);
  const int en_idx = fix->electronegativity_index();
  buf_iele.resize(nele_local);
  for (int i = 0; i < nele_local; i++) {
    buf_iele[i] = h_b(i);
    if (en_idx >= 0)
      buf_iele[i] += atom->dvector[en_idx][atom->map(taglist_local[i])] / force->qqrd2e;
  }
  t_b2h += MPI_Wtime() - t1;

  // gather to world-iele order, same convention as buffer_and_gather
  double t2 = MPI_Wtime();
  MPI_Allgatherv(buf_iele.data(), nele_local, MPI_DOUBLE, buf_gathered, recvcounts, displs,
                 MPI_DOUBLE, world);
  for (int i = 0; i < nele_world; i++) potential_iele[iele_gathered[i]] = buf_gathered[i];
  t_gth += MPI_Wtime() - t2;

  // H2D potentials, matvec q(r) = -(cap_frag . pot)
  double t3 = MPI_Wtime();
  if ((int) d_pot.extent(0) != nele_world)
    d_pot = typename ArrayTypes<DeviceType>::t_kkacc_1d("electrode/inv/kk:d_pot", nele_world);
  {
    auto h_pot = Kokkos::create_mirror_view(d_pot);
    for (int j = 0; j < nele_world; j++) h_pot(j) = static_cast<KK_ACC_FLOAT>(potential_iele[j]);
    Kokkos::deep_copy(d_pot, h_pot);
  }
  t_p2d += MPI_Wtime() - t3;

  double t4 = MPI_Wtime();
  if ((int) d_qvec.extent(0) != nele_local)
    d_qvec = typename ArrayTypes<DeviceType>::t_kkacc_1d("electrode/inv/kk:d_qvec", nele_local);
  if (nele_local > 0) {
    auto v_cap = d_capfrag;
    auto v_pot = d_pot;
    auto v_q = d_qvec;
    const int ncol = nele_world;
    Kokkos::parallel_for("electrode/inv/kk:matvec", nele_local, KOKKOS_LAMBDA(const int r) {
      double acc = 0.0;
      for (int j = 0; j < ncol; j++) acc -= v_cap(r, j) * v_pot(j);
      v_q(r) = acc;
    });
  }
  t_mv += MPI_Wtime() - t4;
  // D2H charges; sb_charges accumulate host-side like the base class
  std::fill(sb_charges.begin(), sb_charges.end(), 0.);
  double t5 = MPI_Wtime();
  if (nele_local > 0) {
    auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_qvec);
    for (int i = 0; i < nele_local; i++) {
      qvec[i] = h_q(i);
      sb_charges[iele_to_group[iele_local[i]]] += qvec[i];
    }
  }
  t_q2h += MPI_Wtime() - t5;
  double t6 = MPI_Wtime();
  MPI_Allreduce(MPI_IN_PLACE, sb_charges.data(), ngroups, MPI_DOUBLE, MPI_SUM, world);
  MPI_Barrier(world);
  t_bar += MPI_Wtime() - t6;
  mult_time += MPI_Wtime() - mult_start;
}

template<class DeviceType>
double ElectrodeInvKokkos<DeviceType>::memory_use()
{
  double bytes = ElectrodeInv::memory_use();
  bytes += (double) d_capfrag.size() * sizeof(KK_ACC_FLOAT);
  bytes += (double) (d_pot.size() + d_b.size() + d_corr.size() + d_qvec.size()) * sizeof(KK_ACC_FLOAT);
  bytes += (double) d_imap.size() * sizeof(int);
  if (corr_scratch_nmax > 0) bytes += (double) corr_scratch_nmax * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

template class ElectrodeInvKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ElectrodeInvKokkos<LMPHostType>;
#endif

}    // namespace LAMMPS_NS
