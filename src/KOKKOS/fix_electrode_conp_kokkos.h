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
FixStyle(electrode/conp/kk,FixElectrodeConpKokkos<LMPDeviceType>);
FixStyle(electrode/conp/kk/device,FixElectrodeConpKokkos<LMPDeviceType>);
FixStyle(electrode/conp/kk/host,FixElectrodeConpKokkos<LMPHostType>);
// clang-format on

#else

#ifndef LMP_FIX_ELECTRODE_CONP_KOKKOS_H
#define LMP_FIX_ELECTRODE_CONP_KOKKOS_H

#include "fix_electrode_conp.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class ElectrodeCG;
class FixElectrodeConp;

template<class DeviceType>
void electrode_kk_mark_lists(class LAMMPS *, class FixElectrodeConp *);

template<class DeviceType>
class FixElectrodeConpKokkos : public FixElectrodeConp {
 public:
  FixElectrodeConpKokkos(class LAMMPS *, int, char **);
  void init() override;
  bool cg_device_needs_full_list() const override;
  ElectrodeCG *new_cg_solver() override;
  class NeighList *get_cg_neighlist() const { return cg_kk_neighlist; }
  void set_charges(std::vector<double>) override;
  void device_charge_sync() override;

 protected:
  void mark_kokkos_lists();
  class AtomKokkos *atomKK = nullptr;
 };

}    // namespace LAMMPS_NS

#endif
#endif