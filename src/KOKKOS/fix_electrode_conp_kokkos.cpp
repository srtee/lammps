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

#include "fix_electrode_conp_kokkos.h"



#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "electrode_cg_kokkos.h"
#include "electrode_inv_kokkos.h"

#include <algorithm>

namespace LAMMPS_NS {

template<class DeviceType>
FixElectrodeConpKokkos<DeviceType>::FixElectrodeConpKokkos(class LAMMPS *lmp, int narg,
                                                           char **arg) :
    FixElectrodeConp(lmp, narg, arg), atomKK(static_cast<AtomKokkos *>(atom))
{
  // the base constructor already created host ElectrodeVector objects unless
  // intelflag matched "/intel"; mark ourselves so a future device vector
  // implementation can swap in here (K2/K3)
  if constexpr (!std::is_same_v<DeviceType, LMPHostType>) {
    if (device_solve) {
      // keep ghost charges, atom migration, and sorting on the device: the
      // host fallbacks force a full atom-array host<->device sync every
      // reneighboring step. The host-solver lane (device off) keeps the
      // legacy host exchange: its per-step host q writes require the host
      // to stay the q authority (same discipline as fix qeq/reaxff/kk)
      forward_comm_device = 1;
      exchange_comm_device = 1;
      sort_device = 1;
    }
  }
}

template<class DeviceType>
void FixElectrodeConpKokkos<DeviceType>::set_charges(std::vector<double> q_local)
{
  // the device exchange pipeline may have left k_q device-authoritative
  // (migrating atoms arrive with their charges device-side); pull before
  // the host write so it cannot clobber arrived data. The base write ends
  // with the virtual device_charge_sync() republish.
  if (atomKK == nullptr) atomKK = static_cast<AtomKokkos *>(atom);
  double ts = MPI_Wtime();
  atomKK->sync(Host, Q_MASK);
  t_setq_pre += MPI_Wtime() - ts;
  sync_site = 1;
  FixElectrodeConp::set_charges(std::move(q_local));
  sync_site = 0;
}

template<class DeviceType>
void FixElectrodeConpKokkos<DeviceType>::device_charge_sync()
{
  // the CG matvec changes electrode charges between device kspace calls;
  // mark the host q write so the device view is refreshed
  if (atomKK == nullptr) atomKK = static_cast<AtomKokkos *>(atom);
  double ts = MPI_Wtime();
  atomKK->modified(Host, Q_MASK);
  atomKK->sync(this->execution_space, Q_MASK);
  t_push[sync_site] += MPI_Wtime() - ts;
}

template<class DeviceType>
void FixElectrodeConpKokkos<DeviceType>::host_data_sync(uint64_t mask)
{
  // under the device exchange/sort pipeline the device views are
  // authoritative between force computations (exchange_device() and
  // sort_device() mutate them in place without a hostward pull); callers
  // request only the arrays they read on the host, so steady-state runs
  // hit the modified-flag check and skip untouched arrays entirely
  if (atomKK == nullptr) atomKK = static_cast<AtomKokkos *>(atom);
  double ts = MPI_Wtime();
  atomKK->sync(Host, mask);
  t_hsync += MPI_Wtime() - ts;
}
/* ----------------------------------------------------------------------
   device ghost-charge exchange: q travels device-side through the Kokkos
   comm pipeline, removing the per-step host round trip of the legacy path
------------------------------------------------------------------------- */

template<class DeviceType>
int FixElectrodeConpKokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                                                 DAT::tdual_double_1d &k_buf,
                                                                 int /*pbc_flag*/, int * /*pbc*/)
{
  // both solve paths keep q current on the device (device_charge_sync);
  // pack straight from the device view
  atomKK->sync(execution_space, Q_MASK);
  auto d_sendlist = k_sendlist.view<DeviceType>();
  auto d_buf = k_buf.view<DeviceType>();
  auto d_q = atomKK->k_q.template view<DeviceType>();
  Kokkos::parallel_for(
      "electrode/conp/kk:pack_fwd", Kokkos::RangePolicy<DeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) { d_buf(i) = static_cast<double>(d_q(d_sendlist(i))); });
  return n;
}

template<class DeviceType>
void FixElectrodeConpKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first,
                                                                    DAT::tdual_double_1d &k_buf)
{
  auto d_buf = k_buf.view<DeviceType>();
  auto d_q = atomKK->k_q.template view<DeviceType>();
  Kokkos::parallel_for(
      "electrode/conp/kk:unpack_fwd", Kokkos::RangePolicy<DeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) { d_q(first + i) = d_buf(i); });
  atomKK->k_q.template modify<DeviceType>();
}

/* ----------------------------------------------------------------------
   atom migration: each electrode atom carries its capacitance-matrix row
   through the exchange buffers. The rows live in host-side ElectrodeInv
   fragments, so pack/unpack run on the host through the dual views --
   migration is rare (reneighboring steps with actual crossings), so this
   costs nothing in steady state and replaces the legacy host-exchange
   fallback that synced every atom array every reneighboring step
------------------------------------------------------------------------- */

template<class DeviceType>
int FixElectrodeConpKokkos<DeviceType>::pack_exchange_kokkos(
    const int &nsend, DAT::tdual_double_2d_lr &k_buf, DAT::tdual_int_1d k_exchange_sendlist,
    DAT::tdual_int_1d k_copylist, ExecutionSpace /*space*/)
{
  // the device exchange pipeline keeps atom arrays device-authoritative;
  // pull tags/masks host-side before the host-side group check
  atomKK->sync(Host, MASK_MASK | TAG_MASK);
  if (charge_solver == nullptr) return 0;    // pre-setup exchange: setup rebuilds all rows

  const int row_size = charge_solver->pack_row_size();
  if (nsend == 0 || row_size == 0) return 0;

  // layout mirrors the device packer of fix shake/kk: one header double per
  // outgoing atom (absolute payload offset), then a dense payload per atom:
  // [iele, row...] for electrode atoms, [-1] for the rest
  k_exchange_sendlist.sync_host();
  const auto &h_sendlist = k_exchange_sendlist.view_host();
  std::vector<double> header(nsend, 0.);
  std::vector<double> payload;
  payload.reserve((size_t) nsend * (row_size + 1));
  std::vector<double> rowbuf(row_size);
  for (int i = 0; i < nsend; i++) {
    header[i] = (double) payload.size();
    const int j = h_sendlist(i);
    if (!(atom->mask[j] & groupbit)) {
      payload.push_back(-1.0);
      continue;
    }
    nlocalele_outdated = 1;
    nlocalele--;    // mirror FixElectrodeConp::pack_exchange bookkeeping
    const int m = charge_solver->pack_row(j, rowbuf.data());
    payload.push_back(rowbuf[0]);
    payload.insert(payload.end(), rowbuf.begin() + 1, rowbuf.begin() + m);
  }

  const int total = nsend + (int) payload.size();
  double *flat = k_buf.view_host().data();
  for (int i = 0; i < nsend; i++) flat[i] = header[i];
  for (int i = 0; i < (int) payload.size(); i++) flat[nsend + i] = payload[i];
  k_buf.modify_host();
  k_buf.sync<DeviceType>();
  return total;
}

template<class DeviceType>
void FixElectrodeConpKokkos<DeviceType>::unpack_exchange_kokkos(
    DAT::tdual_double_2d_lr &k_buf, DAT::tdual_int_1d &k_indices, int nrecv, int nrecv1,
    int nextrarecv1, ExecutionSpace /*space*/)
{
  // refresh host tags/masks so the base-class gather_list_iele() that runs
  // on the next solve sees the post-exchange ordering
  atomKK->sync(Host, MASK_MASK | TAG_MASK);
  if (charge_solver == nullptr || nrecv == 0) return;

  // arrivals are rare: pull the received block host-side and update the
  // ElectrodeInv fragments there; the next solve re-uploads the fragments
  auto h_buf =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k_buf.view<DeviceType>());
  auto h_indices =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k_indices.view<DeviceType>());
  const int row_size = charge_solver->pack_row_size();
  std::vector<double> rowbuf(row_size);
  const double *flat = h_buf.data();
  for (int i = 0; i < nrecv; i++) {
    int m = (int) flat[i];
    if (i >= nrecv1) m = nextrarecv1 + (int) flat[nextrarecv1 + i - nrecv1];
    const double iele = flat[m];
    if (iele < 0) continue;    // non-electrode atom
    // payload layout matches ElectrodeInv::unpack_row: [iele, row...]
    rowbuf[0] = iele;
    for (int k = 1; k < row_size; k++) rowbuf[k] = flat[m + k];
    nlocalele_outdated = 1;
    nlocalele++;    // mirror FixElectrodeConp::unpack_exchange bookkeeping
    charge_solver->unpack_row(h_indices(i), rowbuf.data());
  }
}

template<class DeviceType>
void FixElectrodeConpKokkos<DeviceType>::init()
{
  // gauss-pair mode is the device path; eta mode falls back to the host
  // kernels (K2 scope decision) -- everything else runs host-side for now
  if (!pairflag)
    error->all(FLERR, "fix electrode/conp/kk requires the pair keyword with a "
                      "pair style implementing ElectrodePair (eta mode is host-only)");

  FixElectrodeConp::init();
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    if (device_solve)
      error->all(FLERR, "Fix {} device on requires the Kokkos device lane; /kk/host uses the host solve",
                 style);
  }
  mark_kokkos_lists();
  // device-CG binding happens in ElectrodeCGKokkos::setup_solver(), which
  // runs from setup_post_neighbor() once the solver exists (never here:
  // Modify::init() precedes solver construction)
}

/* ----------------------------------------------------------------------
   device CG: the device-matvec solver over the full newton-off list
------------------------------------------------------------------------- */

template<class DeviceType>
bool FixElectrodeConpKokkos<DeviceType>::cg_device_needs_full_list() const
{
  return true;
}

template<class DeviceType>
ElectrodeCG *FixElectrodeConpKokkos<DeviceType>::new_cg_solver()
{
  return new ElectrodeCGKokkos<DeviceType>(lmp, this);
}

template<class DeviceType>
ElectrodeInv *FixElectrodeConpKokkos<DeviceType>::new_inv_solver()
{
  // the device-resident solve (device on) needs the full newton-off device
  // neighbor list, which only the device lane requests; the host lane and
  // the default (device off) keep the portable host ElectrodeInv
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) return new ElectrodeInv(lmp);
  else if (device_solve) return new ElectrodeInvKokkos<DeviceType>(lmp, this);
  else return new ElectrodeInv(lmp);
}

template<class DeviceType>
int FixElectrodeConpKokkos<DeviceType>::device_elyt_group() const
{
  return elyt_vector->get_groupbit();
}
template<class DeviceType>
void FixElectrodeConpKokkos<DeviceType>::mark_kokkos_lists()
{
  // request device-resident neighbor lists on the device backend
  const bool host_dispatch = std::is_same_v<DeviceType, LMPHostType> &&
      !std::is_same_v<DeviceType, LMPDeviceType>;
  for (int ireq = 0; ireq < neighbor->nrequest; ireq++) {
    auto *req = neighbor->requests[ireq];
    if (req->get_requestor() != this) continue;
    req->set_kokkos_host(host_dispatch);
    req->set_kokkos_device(!host_dispatch);
  }
}

/* ----------------------------------------------------------------------
   the header-only /thermo/kk and /conq/kk styles derive from the host base
   (no MI); they reuse the conp dispatch through this shared free function
------------------------------------------------------------------------- */

template<class DeviceType>
void electrode_kk_sync_q(class FixElectrodeConp *fix)
{
  // mark host-side charge writes so the device q view is refreshed; used
  // by the CG matvec between device kspace vector calls
  auto *atomKK = static_cast<AtomKokkos *>(fix->lmp->atom);
  atomKK->modified(Host, Q_MASK);
  atomKK->sync(fix->execution_space, Q_MASK);
}


template<class DeviceType>
void electrode_kk_mark_lists(class LAMMPS *lmp, FixElectrodeConp *fix)
{
  const bool host_dispatch = std::is_same_v<DeviceType, LMPHostType> &&
      !std::is_same_v<DeviceType, LMPDeviceType>;
  for (int ireq = 0; ireq < lmp->neighbor->nrequest; ireq++) {
    auto *req = lmp->neighbor->requests[ireq];
    if (req->get_requestor() != fix) continue;
    req->set_kokkos_host(host_dispatch);
    req->set_kokkos_device(!host_dispatch);
  }
}

template void electrode_kk_mark_lists<LMPDeviceType>(LAMMPS *, FixElectrodeConp *);
#ifdef LMP_KOKKOS_GPU
template void electrode_kk_mark_lists<LMPHostType>(LAMMPS *, FixElectrodeConp *);
#endif

template class FixElectrodeConpKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixElectrodeConpKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS