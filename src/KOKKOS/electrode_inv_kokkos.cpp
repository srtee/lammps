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
#include "electrode_vector.h"
#include "error.h"
#include "fix_electrode_conp.h"
#include "force.h"
#include "comm.h"
#include "memory.h"

#include <cstdio>
#include <cstdlib>
#include "pair.h"
#include "update.h"

#include <algorithm>

namespace LAMMPS_NS {


template<class DeviceType>
ElectrodeInvKokkos<DeviceType>::ElectrodeInvKokkos(class LAMMPS *lmp_in, class FixElectrodeConp *fix_in) :
    ElectrodeInv(lmp_in), fix(fix_in)
{
}

template<class DeviceType>
ElectrodeInvKokkos<DeviceType>::~ElectrodeInvKokkos() noexcept = default;


/* ----------------------------------------------------------------------
   setup: base builds iele_local / cap_frag bookkeeping, then bind the
   device pair/kspace interfaces and mirror the fragment matrix
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeInvKokkos<DeviceType>::setup_solver(int groupbit_in, std::unordered_map<tagint, int> tag_to_iele,
                                                  std::vector<int> group_bits_in, bool ffield,
                                                  bool timer_flag)
{
  ElectrodeInv::setup_solver(groupbit_in, std::move(tag_to_iele), std::move(group_bits_in), ffield,
                             timer_flag);
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

   1. pair+self+kspace kernels assemble d_b in taglist order
   2. one D2H; electronegativity correction stays host (per-atom property)
   3. gather to world-iele order (same convention as buffer_and_gather)
   4. one H2D (potentials), device matvec, one D2H (charges); sb_charges
      accumulate host-side, exactly as ElectrodeInv::set_elyt_pot does
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeInvKokkos<DeviceType>::assemble_b_device()
{
  const int nele_local = nlocalele;
  if ((int) d_b.extent(0) < nele_local)
    d_b = typename ArrayTypes<DeviceType>::t_kkacc_1d("electrode/inv/kk:d_b", nele_local);
  Kokkos::deep_copy(d_b, 0.0);
  // the device solve bypasses base buffer_and_gather, which is where the
  // world-iele gather staging is normally allocated
  if (buf_gathered == nullptr)
    memory->create(buf_gathered, nele_world, "ElectrodeInv:buf_gathered");
  // atom -> taglist position map (-1 for non-electrode atoms)
  const int nlocal = atom->nlocal;
  if ((int) d_imap.extent(0) < nlocal)
    d_imap = typename ArrayTypes<DeviceType>::t_int_1d("electrode/inv/kk:d_imap", nlocal);
  {
    auto h_imap = Kokkos::create_mirror_view(d_imap);
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
  }

  // electrolyte charges (incl. ghosts) must be current on the device
  fix_kk->device_charge_sync();

  const int sensor_grpbit = dev_vec->get_groupbit();
  const int source_grpbit = dev_vec->get_source_grpbit();
  const bool invert_source = dev_vec->get_invert_source();
  class NeighList *neigh = fix_kk->get_cg_neighlist();

  pair_kk->compute_vector_nele(neigh, d_b, d_imap, sensor_grpbit, source_grpbit, invert_source);
  pair_kk->compute_vector_self_nele(d_b, d_imap, sensor_grpbit, source_grpbit, invert_source);

  if (dev_vec->get_kspaceflag()) {
    kspace_kk->compute_vector_nele(d_b, d_imap, sensor_grpbit, source_grpbit, invert_source);
    // boundary corrections stay host (O(N) sums), then fold into d_b
    double *corr_scratch;
    memory->create(corr_scratch, atom->nmax, "electrode/inv/kk:corr_scratch");
    memset(corr_scratch, 0, atom->nmax * sizeof(double));
    kspace_kk->compute_vector_corr(corr_scratch, sensor_grpbit, source_grpbit, invert_source);
    auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_b);
    for (int i = 0; i < nele_local; i++)
      h_b(i) += static_cast<KK_ACC_FLOAT>(corr_scratch[atom->map(taglist_local[i])]);
    Kokkos::deep_copy(d_b, h_b);
    memory->destroy(corr_scratch);
  }
}

template<class DeviceType>
void ElectrodeInvKokkos<DeviceType>::set_elyt_pot(double *b_nall)
{
  (void) b_nall;    // ignored: the b vector is assembled on the device
  elyt_step = update->ntimestep;
  if (fix_kk == nullptr) error->all(FLERR, "ElectrodeInvKokkos used before setup_solver");

  MPI_Barrier(world);
  double mult_start = MPI_Wtime();

  assemble_b_device();

  // D2H b, add electronegativity correction (per-atom property, host)
  const int nele_local = nlocalele;
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_b);
  const int en_idx = fix->electronegativity_index();
  buf_iele.resize(nele_local);
  for (int i = 0; i < nele_local; i++) {
    buf_iele[i] = h_b(i);
    if (en_idx >= 0)
      buf_iele[i] += atom->dvector[en_idx][atom->map(taglist_local[i])] / force->qqrd2e;
  }

  // gather to world-iele order, same convention as buffer_and_gather
  MPI_Allgatherv(buf_iele.data(), nele_local, MPI_DOUBLE, buf_gathered, recvcounts, displs, MPI_DOUBLE,
                 world);
  for (int i = 0; i < nele_world; i++) potential_iele[iele_gathered[i]] = buf_gathered[i];

  // H2D potentials, matvec q(r) = -(cap_frag . pot)
  if ((int) d_pot.extent(0) != nele_world)
    d_pot = typename ArrayTypes<DeviceType>::t_kkacc_1d("electrode/inv/kk:d_pot", nele_world);
  {
    auto h_pot = Kokkos::create_mirror_view(d_pot);
    for (int j = 0; j < nele_world; j++) h_pot(j) = static_cast<KK_ACC_FLOAT>(potential_iele[j]);
    Kokkos::deep_copy(d_pot, h_pot);
  }

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

  // D2H charges; sb_charges accumulate host-side like the base class
  std::fill(sb_charges.begin(), sb_charges.end(), 0.);
  if (nele_local > 0) {
    auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_qvec);
    for (int i = 0; i < nele_local; i++) {
      qvec[i] = h_q(i);
      sb_charges[iele_to_group[iele_local[i]]] += qvec[i];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, sb_charges.data(), ngroups, MPI_DOUBLE, MPI_SUM, world);
  MPI_Barrier(world);
  mult_time += MPI_Wtime() - mult_start;
}

template<class DeviceType>
double ElectrodeInvKokkos<DeviceType>::memory_use()
{
  double bytes = ElectrodeInv::memory_use();
  bytes += (double) d_capfrag.size() * sizeof(KK_ACC_FLOAT);
  bytes += (double) (d_pot.size() + d_b.size() + d_qvec.size()) * sizeof(KK_ACC_FLOAT);
  bytes += (double) d_imap.size() * sizeof(int);
  return bytes;
}

/* ---------------------------------------------------------------------- */

template class ElectrodeInvKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ElectrodeInvKokkos<LMPHostType>;
#endif

}    // namespace LAMMPS_NS
