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
   ELECTRODE CG solver with a device-resident matvec (stage K4c): the CG
   recurrence stays on the host in ElectrodeCG; only ele_ele_interaction is
   overridden to run pair + kspace contributions through the Kokkos device
   kernels (ElectrodePairKokkos / PPPMElectrodeKokkos) over the full
   newton-off matvec list. TF / hardness / boundary corrections stay host.
------------------------------------------------------------------------- */

#ifndef LMP_ELECTRODE_CG_KOKKOS_H
#define LMP_ELECTRODE_CG_KOKKOS_H

#include "electrode_cg.h"
#include "fix_electrode_conp_kokkos.h"

#include "electrode_pair_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class ElectrodeCGKokkos : public ElectrodeCG {
 public:
  ElectrodeCGKokkos(class LAMMPS *, class FixElectrodeConp *);
  ~ElectrodeCGKokkos() noexcept override;

  // called by FixElectrodeConpKokkos::init() after setup_general bound
  // elec_vec to the device-CG list; locates the device pair/kspace interfaces
  void setup_device();

 protected:
  std::vector<double> ele_ele_interaction(const std::vector<double> &) override;

 private:
  class FixElectrodeConpKokkos<DeviceType> *fix_kk = nullptr;
  typename ArrayTypes<DeviceType>::t_kkacc_1d d_out;    // nele accumulator
  typename ArrayTypes<DeviceType>::t_int_1d d_imap;     // atom -> iele (-1 none)
  class ElectrodeVector *dev_vec = nullptr;    // elec_vec bound to the device list
  ElectrodePairKokkos<DeviceType> *pair_kk = nullptr;
  class ElectrodeKSpace *kspace_kk = nullptr;  // device vector path (PPPMElectrodeKokkos)
};

}    // namespace LAMMPS_NS

#endif