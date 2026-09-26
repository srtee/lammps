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
    FixElectrodeConp(lmp, narg, arg)
{
  // the base constructor already created host ElectrodeVector objects unless
  // intelflag matched "/intel"; mark ourselves so a future device vector
  // implementation can swap in here (K2/K3)
}

template<class DeviceType>
void FixElectrodeConpKokkos<DeviceType>::set_charges(std::vector<double> q_local)
{
  FixElectrodeConp::set_charges(std::move(q_local));
}

template<class DeviceType>
void FixElectrodeConpKokkos<DeviceType>::device_charge_sync()
{
  // the CG matvec changes electrode charges between device kspace calls;
  // mark the host q write so the device view is refreshed
  if (atomKK == nullptr) atomKK = static_cast<AtomKokkos *>(atom);
  atomKK->modified(Host, Q_MASK);
  atomKK->sync(this->execution_space, Q_MASK);
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