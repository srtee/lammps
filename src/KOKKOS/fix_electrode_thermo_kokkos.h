/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef FIX_CLASS

// clang-format off
FixStyle(electrode/thermo/kk,FixElectrodeThermoKokkos<LMPDeviceType>);
FixStyle(electrode/thermo/kk/device,FixElectrodeThermoKokkos<LMPDeviceType>);
FixStyle(electrode/thermo/kk/host,FixElectrodeThermoKokkos<LMPHostType>);
// clang-format on

#else

#ifndef LMP_FIX_ELECTRODE_THERMO_KOKKOS_H
#define LMP_FIX_ELECTRODE_THERMO_KOKKOS_H

#include "fix_electrode_conp_kokkos.h"
#include "fix_electrode_thermo.h"
#include "fix_electrode_conp_kokkos.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType>
class FixElectrodeThermoKokkos : public FixElectrodeThermo {
 public:
  FixElectrodeThermoKokkos(class LAMMPS *lmp, int narg, char **arg) :
    FixElectrodeThermo(lmp, narg, arg)
  {
  }
  void init() override
  {
    if (!pairflag)
      error->all(FLERR, "fix electrode/thermo/kk requires the pair keyword with a "
                        "pair style implementing ElectrodePair (eta mode is host-only)");
    FixElectrodeThermo::init();
    mark_kokkos_lists();
  }

 private:
  void mark_kokkos_lists()
  {
    electrode_kk_mark_lists<DeviceType>(lmp, this);
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
