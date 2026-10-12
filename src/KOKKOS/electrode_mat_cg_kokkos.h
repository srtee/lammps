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
   ELECTRODE matrix-CG solver ("algo mat_cg") with a device-resident
   recurrence: the replicated elastance lives in device memory and the
   matvec is a device GEMV over the rank's local rows.  The only per-step
   host traffic is the charge world-scatter (an MPI_Allreduce, as in the
   host solver); the CG recurrence itself never leaves the device.
------------------------------------------------------------------------- */

#ifndef LMP_ELECTRODE_MAT_CG_KOKKOS_H
#define LMP_ELECTRODE_MAT_CG_KOKKOS_H

#include "electrode_cg_device_kokkos.h"
#include "electrode_mat_cg.h"

namespace LAMMPS_NS {

template<class DeviceType>
class ElectrodeMatCGKokkos : public ElectrodeCGDeviceKokkos<DeviceType, ElectrodeMatCG> {
  using Base = ElectrodeCGDeviceKokkos<DeviceType, ElectrodeMatCG>;

 public:
  ElectrodeMatCGKokkos(class LAMMPS *, class FixElectrodeConp *);
  ~ElectrodeMatCGKokkos() noexcept override;

  void setup_solver(double, std::unordered_map<tagint, int>, int) override;
  double memory_use() override;

 public:
  // device overrides public: nvcc forbids extended device lambdas in
  // member functions with protected/private access
  void dev_matvec(const typename Base::dview &in, typename Base::dview &out) override;
  void dev_refresh_solver_maps() override;

 private:
  // replicated elastance on the device (n_mat x n_mat, already evscaled)
  Kokkos::View<double **, Kokkos::LayoutRight, DeviceType> d_elastance;
  typename Base::dview d_qworld;    // nele_world charge gather target
  typename Base::iview d_row;       // nele: local row index -> matrix row
  bool matrix_uploaded = false;
};

}    // namespace LAMMPS_NS

#endif
