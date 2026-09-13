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



#include "error.h"
#include "fix_electrode_thermo.h"
#include "kokkos.h"
#include "neigh_request.h"

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
void FixElectrodeConpKokkos<DeviceType>::init()
{
  // gauss-pair mode is the device path; eta mode falls back to the host
  // kernels (K2 scope decision) -- everything else runs host-side for now
  if (!pairflag)
    error->all(FLERR, "fix electrode/conp/kk requires the pair keyword with a "
                      "pair style implementing ElectrodePair (eta mode is host-only)");

  FixElectrodeConp::init();
  mark_kokkos_lists();
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