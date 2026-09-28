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
#include "kokkos_base.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class ElectrodeInv;
class ElectrodeCG;
class FixElectrodeConp;

template<class DeviceType>
void electrode_kk_mark_lists(class LAMMPS *, class FixElectrodeConp *);

template<class DeviceType>
class FixElectrodeConpKokkos : public FixElectrodeConp, public KokkosBase {
 public:
  FixElectrodeConpKokkos(class LAMMPS *, int, char **);
  void init() override;
  bool cg_device_needs_full_list() const override;
  ElectrodeCG *new_cg_solver() override;
  ElectrodeInv *new_inv_solver() override;
  bool device_mat_inv() const override { return device_solve && !std::is_same_v<DeviceType, LMPHostType>; }
  ElectrodeVector *device_elyt_vector() const override { return elyt_vector; }
  int device_elyt_group() const override;
  class NeighList *get_cg_neighlist() const { return cg_kk_neighlist; }
  void set_charges(std::vector<double>) override;
  void device_charge_sync() override;
  void host_data_sync(uint64_t) override;
  // device exchange/border pipeline: ghost charges and matrix rows travel
  // in the Kokkos comm machinery instead of forcing legacy host exchange
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &, int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
  int pack_exchange_kokkos(const int &, DAT::tdual_double_2d_lr &, DAT::tdual_int_1d,
                           DAT::tdual_int_1d, ExecutionSpace) override;
  void unpack_exchange_kokkos(DAT::tdual_double_2d_lr &, DAT::tdual_int_1d &, int, int, int,
                              ExecutionSpace) override;

 protected:
  void mark_kokkos_lists();
  class AtomKokkos *atomKK = nullptr;
 };

}    // namespace LAMMPS_NS

#endif
#endif