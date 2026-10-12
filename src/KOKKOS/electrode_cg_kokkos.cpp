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
   Contributing authors: Shern Tee (GU)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Device matvec for the ELECTRODE CG solver, "algo cg":
   - device lane (device on): out = (evscale * A) . in entirely on the
     device; pair + self + kspace accumulate into a nele-ordered view
     addressed through d_imap, charges are staged through the device q
     array + forward_comm, and only the nonpair (tf/hardness) and
     boundary-correction terms touch the host.
   - host lane (device no): K4 behavior -- host charges round trip, device
     kernels over the full newton-off list, one nele-sized D2H.
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
    ElectrodeCGDeviceKokkos<DeviceType, ElectrodeCG>(lmp, fix_in)
{
}

template<class DeviceType>
ElectrodeCGKokkos<DeviceType>::~ElectrodeCGKokkos() noexcept = default;

/* ----------------------------------------------------------------------
   solver setup: run the base binding (elec_vec -> device-CG list), then
   bind the device interfaces. Runs from setup_post_neighbor() where the
   fix constructs the solver -- never from fix init(), where charge_solver
   does not exist yet.
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeCGKokkos<DeviceType>::setup_solver(double cg_threshold, ElectrodeVector *vec,
                                                 int predictor_cols)
{
  ElectrodeCG::setup_solver(cg_threshold, vec, predictor_cols);
  dev_vec = this->elec_vec;
  if (dev_vec == nullptr) this->error->all(FLERR, "ElectrodeCGKokkos without electrode vector");
  this->dev_setup();
}

/* ----------------------------------------------------------------------
   shared matvec core: zero d_out, accumulate pair/self/kspace on the
   device plus tf/hardness/corr on the host.  Charges must already sit on
   the atoms (host or device lane).
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeCGKokkos<DeviceType>::dev_matvec(const typename Base::dview &in,
                                               typename Base::dview &out)
{
  const double mult_start = MPI_Wtime();

  this->check_kspace_refresh();    // fix tune/kspace may have re-created the kspace
  this->dev_scatter_into(in);      // charges -> atoms (device) -> ghosts
  this->dev_refresh_imap();

  Kokkos::deep_copy(out, 0);

  const int sensor_grpbit = dev_vec->get_groupbit();
  const int source_grpbit = dev_vec->get_source_grpbit();
  const bool invert_source = dev_vec->get_invert_source();

  class NeighList *neigh = this->fix_kk->get_cg_neighlist();
  this->pair_kk->compute_vector_nele(neigh, out, this->d_imap, sensor_grpbit, source_grpbit,
                                     invert_source);
  this->pair_kk->compute_vector_self_nele(out, this->d_imap, sensor_grpbit, source_grpbit,
                                          invert_source);

  if (dev_vec->get_tfflag() || dev_vec->get_hardnessflag()) {
    double *potential_host;
    this->memory->create(potential_host, this->atom->nmax, "electrode/cg/kk:tf_scratch");
    memset(potential_host, 0, this->atom->nmax * sizeof(double));
    dev_vec->nonpair_contribution(potential_host);
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    for (int i = 0; i < this->nele; i++)
      h_out(i) += static_cast<KK_ACC_FLOAT>(
          potential_host[this->atom->map(this->taglist[i])] * this->evscale);
    Kokkos::deep_copy(out, h_out);
    this->memory->destroy(potential_host);
  }

  if (dev_vec->get_kspaceflag()) {
    // re-cast in case the kspace object was swapped (fix tune/kspace)
    this->kspace_kk =
        dynamic_cast<PPPMElectrodeKokkos<DeviceType> *>(dev_vec->get_electrode_kspace());
    if (this->kspace_kk == nullptr)
      this->error->all(FLERR, "KSpace style does not implement the ELECTRODE device vector path");
    this->kspace_kk->compute_vector_nele(out, this->d_imap, sensor_grpbit, source_grpbit,
                                         invert_source);
    double *corr_scratch;
    this->memory->create(corr_scratch, this->atom->nmax, "electrode/cg/kk:corr_scratch");
    memset(corr_scratch, 0, this->atom->nmax * sizeof(double));
    this->kspace_kk->compute_vector_corr(corr_scratch, sensor_grpbit, source_grpbit, invert_source);
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    for (int i = 0; i < this->nele; i++)
      h_out(i) += static_cast<KK_ACC_FLOAT>(corr_scratch[this->atom->map(this->taglist[i])]);
    Kokkos::deep_copy(out, h_out);
    this->memory->destroy(corr_scratch);
  }

  this->mult_time += MPI_Wtime() - mult_start;
}

/* ----------------------------------------------------------------------
   host-lane matvec (device no): K4 behavior, charges round trip through
   the host set_charges and the result comes back in one nele-sized D2H.
------------------------------------------------------------------------- */

template<class DeviceType>
std::vector<double> ElectrodeCGKokkos<DeviceType>::ele_ele_interaction(
    const std::vector<double> &q_ele_in)
{
  double mult_start = MPI_Wtime();

  // 1. charges -> atoms (host) -> device; forward_comm fills ghosts
  this->fix->set_charges(q_ele_in);
  this->check_kspace_refresh();    // fix tune/kspace may have re-created the kspace

  const int nele_local = this->nele;
  if ((int) d_out.extent(0) < nele_local)
    d_out = typename ArrayTypes<DeviceType>::t_kkacc_1d("electrode/cg/kk:d_out", nele_local);
  Kokkos::deep_copy(d_out, 0);

  // 2. atom -> iele map on device (nlocal-sized; -1 = not electrode)
  const int nlocal = this->atom->nlocal;
  if ((int) this->d_imap.extent(0) < nlocal)
    this->d_imap = typename ArrayTypes<DeviceType>::t_int_1d("electrode/cg/kk:d_imap", nlocal);
  {
    auto h_imap = Kokkos::create_mirror_view(this->d_imap);
    for (int i = 0; i < nlocal; i++) {
      const tagint tag = this->atom->tag[i];
      int ipos = -1;
      for (int k = 0; k < nele_local; k++)
        if (this->taglist[k] == tag) { ipos = k; break; }
      h_imap(i) = ipos;
    }
    Kokkos::deep_copy(this->d_imap, h_imap);
  }

  const int sensor_grpbit = dev_vec->get_groupbit();
  const int source_grpbit = dev_vec->get_source_grpbit();
  const bool invert_source = dev_vec->get_invert_source();
  // 3. pair + self on device over the full newton-off list
  class NeighList *neigh = this->fix_kk->get_cg_neighlist();
  this->pair_kk->compute_vector_nele(neigh, d_out, this->d_imap, sensor_grpbit, source_grpbit,
                                     invert_source);
  this->pair_kk->compute_vector_self_nele(d_out, this->d_imap, sensor_grpbit, source_grpbit,
                                          invert_source);

  if (dev_vec->get_tfflag() || dev_vec->get_hardnessflag()) {
    double *potential_host;
    this->memory->create(potential_host, this->atom->nmax, "electrode/cg/kk:tf_scratch");
    memset(potential_host, 0, this->atom->nmax * sizeof(double));
    dev_vec->nonpair_contribution(potential_host);
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    for (int i = 0; i < nele_local; i++)
      h_out(i) += static_cast<KK_ACC_FLOAT>(
          potential_host[this->atom->map(this->taglist[i])] * this->evscale);
    Kokkos::deep_copy(d_out, h_out);
    this->memory->destroy(potential_host);
  }

  if (dev_vec->get_kspaceflag()) {
    // re-cast in case the kspace object was swapped (fix tune/kspace)
    this->kspace_kk =
        dynamic_cast<PPPMElectrodeKokkos<DeviceType> *>(dev_vec->get_electrode_kspace());
    if (this->kspace_kk == nullptr)
      this->error->all(FLERR, "KSpace style does not implement the ELECTRODE device vector path");
    this->kspace_kk->compute_vector_nele(d_out, this->d_imap, sensor_grpbit, source_grpbit,
                                         invert_source);
    double *corr_scratch;
    this->memory->create(corr_scratch, this->atom->nmax, "electrode/cg/kk:corr_scratch");
    memset(corr_scratch, 0, this->atom->nmax * sizeof(double));
    this->kspace_kk->compute_vector_corr(corr_scratch, sensor_grpbit, source_grpbit, invert_source);
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    for (int i = 0; i < nele_local; i++)
      h_out(i) += static_cast<KK_ACC_FLOAT>(corr_scratch[this->atom->map(this->taglist[i])]);
    Kokkos::deep_copy(d_out, h_out);
    this->memory->destroy(corr_scratch);
  }

  // 6. gather to host in iele order (the CG recurrence's only D2H)
  auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
  std::vector<double> out(nele_local);
  for (int i = 0; i < nele_local; i++) out[i] = h_out(i);

  this->mult_time += MPI_Wtime() - mult_start;
  return out;
}

/* ---------------------------------------------------------------------- */

template class ElectrodeCGKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ElectrodeCGKokkos<LMPHostType>;
#endif

}    // namespace LAMMPS_NS
