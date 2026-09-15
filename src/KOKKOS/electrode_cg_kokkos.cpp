/* ----------------------------------------------------------------------
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
   Contributing authors: Shern Tee (GU)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Device matvec for the ELECTRODE CG solver (stage K4c). Replaces the
   base class's host nall round-trip: pair + self + kspace contributions
   accumulate into a nele-ordered device view addressed through d_imap,
   so no nall-sized buffer and no reverse communication are needed (the
   matvec list is full + newton-off). Charges still round-trip per
   iteration through fix->set_charges (K4 contract).
------------------------------------------------------------------------- */

#include "electrode_cg_kokkos.h"

#include "atom.h"
#include "electrode_pair_kokkos.h"
#include "electrode_vector.h"
#include "error.h"
#include "fix_electrode_conp.h"
#include "fix_electrode_conp_kokkos.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "pair.h"
#include "pppm_electrode_kokkos.h"

#include <unordered_map>

namespace LAMMPS_NS {

template<class DeviceType>
ElectrodeCGKokkos<DeviceType>::ElectrodeCGKokkos(class LAMMPS *lmp, class FixElectrodeConp *fix_in) :
    ElectrodeCG(lmp, fix_in)
{
}

template<class DeviceType>
ElectrodeCGKokkos<DeviceType>::~ElectrodeCGKokkos() noexcept = default;

/* ----------------------------------------------------------------------
   locate the device interfaces; must run after FixElectrodeConp::init()
   has bound elec_vec to the device-CG neighbor list (id 4)
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeCGKokkos<DeviceType>::setup_device()
{
  dev_vec = elec_vec;
  if (dev_vec == nullptr) error->all(FLERR, "ElectrodeCGKokkos without electrode vector");

  auto *fix_kk_local = dynamic_cast<FixElectrodeConpKokkos<DeviceType> *>(fix);
  if (fix_kk_local == nullptr) error->all(FLERR, "ElectrodeCGKokkos requires fix electrode/conp/kk");
  if (fix_kk_local->get_cg_neighlist() == nullptr)
    error->all(FLERR, "ElectrodeCGKokkos requires the device CG neighbor list (id 4)");
  fix_kk = fix_kk_local;

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
   device matvec: out = (evscale * A) . q, mirroring
   ElectrodeVector::compute_pot: pair + self (device), tf + hardness
   (host), kspace vector (device) + corr (host). One nele-sized D2H.
------------------------------------------------------------------------- */

template<class DeviceType>
std::vector<double> ElectrodeCGKokkos<DeviceType>::ele_ele_interaction(
    const std::vector<double> &q_ele_in)
{
  double mult_start = MPI_Wtime();

  // 1. charges -> atoms (host) -> device; forward_comm fills ghosts
  fix->set_charges(q_ele_in);

  const int nele_local = nele;
  if ((int) d_out.extent(0) < nele_local)
    d_out = typename ArrayTypes<DeviceType>::t_kkacc_1d("electrode/cg/kk:d_out", nele_local);
  Kokkos::deep_copy(d_out, 0);

  // 2. atom -> iele map on device (nlocal-sized; -1 = not electrode)
  const int nlocal = atom->nlocal;
  if ((int) d_imap.extent(0) < nlocal)
    d_imap = typename ArrayTypes<DeviceType>::t_int_1d("electrode/cg/kk:d_imap", nlocal);
  {
    auto h_imap = Kokkos::create_mirror_view(d_imap);
    for (int i = 0; i < nlocal; i++) {
      const tagint tag = atom->tag[i];
      int ipos = -1;
      for (int k = 0; k < nele_local; k++)
        if (taglist[k] == tag) { ipos = k; break; }
      h_imap(i) = ipos;
    }
    Kokkos::deep_copy(d_imap, h_imap);
  }

  const int sensor_grpbit = dev_vec->get_groupbit();
  const int source_grpbit = dev_vec->get_source_grpbit();
  const bool invert_source = dev_vec->get_invert_source();
  // 3. pair + self on device over the full newton-off list
  class NeighList *neigh = fix_kk->get_cg_neighlist();
  pair_kk->compute_vector_nele(neigh, d_out, d_imap, sensor_grpbit, source_grpbit, invert_source);
  pair_kk->compute_vector_self_nele(d_out, d_imap, sensor_grpbit, source_grpbit, invert_source);

  if (dev_vec->get_tfflag() || dev_vec->get_hardnessflag()) {
    double *potential_host;
    memory->create(potential_host, atom->nmax, "electrode/cg/kk:tf_scratch");
    memset(potential_host, 0, atom->nmax * sizeof(double));
    dev_vec->nonpair_contribution(potential_host);
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    for (int i = 0; i < nele_local; i++)
      h_out(i) += static_cast<KK_ACC_FLOAT>(
          potential_host[atom->map(taglist[i])] * evscale);
    Kokkos::deep_copy(d_out, h_out);
    memory->destroy(potential_host);
  }

  if (dev_vec->get_kspaceflag()) {
    kspace_kk->compute_vector_nele(d_out, d_imap, sensor_grpbit, source_grpbit, invert_source);
    double *corr_scratch;
    memory->create(corr_scratch, atom->nmax, "electrode/cg/kk:corr_scratch");
    memset(corr_scratch, 0, atom->nmax * sizeof(double));
    kspace_kk->compute_vector_corr(corr_scratch, sensor_grpbit, source_grpbit, invert_source);
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    for (int i = 0; i < nele_local; i++)
      h_out(i) += static_cast<KK_ACC_FLOAT>(corr_scratch[atom->map(taglist[i])]);
    Kokkos::deep_copy(d_out, h_out);
    memory->destroy(corr_scratch);
  }

  // 6. gather to host in iele order (the CG recurrence's only D2H)
  auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
  std::vector<double> out(nele_local);
  for (int i = 0; i < nele_local; i++) out[i] = h_out(i);

  mult_time += MPI_Wtime() - mult_start;
  return out;
}

/* ---------------------------------------------------------------------- */

template class ElectrodeCGKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ElectrodeCGKokkos<LMPHostType>;
#endif

}    // namespace LAMMPS_NS