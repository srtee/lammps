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
FixStyle(electrode/conq/kk,FixElectrodeConqKokkos<LMPDeviceType>);
FixStyle(electrode/conq/kk/device,FixElectrodeConqKokkos<LMPDeviceType>);
FixStyle(electrode/conq/kk/host,FixElectrodeConqKokkos<LMPHostType>);
// clang-format on

#else

#ifndef LMP_FIX_ELECTRODE_CONQ_KOKKOS_H
#define LMP_FIX_ELECTRODE_CONQ_KOKKOS_H

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "fix_electrode_conp_kokkos.h"
#include "fix_electrode_conq.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType>
class FixElectrodeConqKokkos : public FixElectrodeConq {
 public:
  FixElectrodeConqKokkos(class LAMMPS *lmp, int narg, char **arg) :
    FixElectrodeConq(lmp, narg, arg)
  {
  }
  void init() override
  {
    if (!pairflag)
      error->all(FLERR, "fix electrode/conq/kk requires the pair keyword with a "
                        "pair style implementing ElectrodePair (eta mode is host-only)");
    FixElectrodeConq::init();
    mark_kokkos_lists();
  }
  void set_charges(std::vector<double> q_local) override
  {
    FixElectrodeConq::set_charges(std::move(q_local));
  }
  void device_charge_sync() override
  {
    if (atomKK == nullptr) atomKK = static_cast<AtomKokkos *>(atom);
    atomKK->modified(Host, Q_MASK);
    atomKK->sync(this->execution_space, Q_MASK);
  }

 private:
  // shares the neighbor-list dispatch logic with conp/kk via a free function
  void mark_kokkos_lists() { electrode_kk_mark_lists<DeviceType>(lmp, this); }
  class AtomKokkos *atomKK = nullptr;
};

}    // namespace LAMMPS_NS

#endif
#endif