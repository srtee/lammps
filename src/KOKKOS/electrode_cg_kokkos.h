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
   ELECTRODE CG solver, "algo cg": the matvec runs pair + kspace through
   the Kokkos device kernels (ElectrodePairKokkos / PPPMElectrodeKokkos)
   over the full newton-off matvec list.  With "device on" on a device
   instantiation the CG recurrence is device-resident too (the
   ElectrodeCGDeviceKokkos mixin); with "device no" only the matvec is
   device-side and the recurrence stays in ElectrodeCG (K4 behavior).
   TF / hardness / boundary corrections stay host.
------------------------------------------------------------------------- */

#ifndef LMP_ELECTRODE_CG_KOKKOS_H
#define LMP_ELECTRODE_CG_KOKKOS_H

#include "electrode_cg.h"
#include "electrode_cg_device_kokkos.h"
#include "fix_electrode_conp_kokkos.h"

#include "electrode_pair_kokkos.h"
#include "pppm_electrode_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class ElectrodeCGKokkos : public ElectrodeCGDeviceKokkos<DeviceType, ElectrodeCG> {
  using Base = ElectrodeCGDeviceKokkos<DeviceType, ElectrodeCG>;

 public:
  ElectrodeCGKokkos(class LAMMPS *, class FixElectrodeConp *);
  ~ElectrodeCGKokkos() noexcept override;

  // setup_solver override calls setup_device() after the base binds
  // elec_vec to the device-CG list; locates the device pair/kspace interfaces
  void setup_solver(double, ElectrodeVector *, int) override;
  // host-lane matvec (device no): overrides the base's ElectrodeVector call
  std::vector<double> ele_ele_interaction(const std::vector<double> &) override;

 protected:
  void dev_matvec(const typename Base::dview &in, typename Base::dview &out) override;

 private:
  class ElectrodeVector *dev_vec = nullptr;    // elec_vec bound to the device list
  typename Base::dview d_out;                  // nele accumulator for the host lane
};

}    // namespace LAMMPS_NS

#endif
