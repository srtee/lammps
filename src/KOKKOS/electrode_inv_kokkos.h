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
   Contributing author: Shern Tee (UQ)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   ElectrodeInv with a device-resident b-assembly and matvec (mat_inv/kk
   stage).  The host ElectrodeInv::solve() (constraint + sd vectors) stays
   host: only the per-step set_elyt_pot pipeline moves to the device.

   - the fragment matrix (cap_frag rows) is mirrored to d_capfrag at
     update_solver() time (construction and every migration)
   - the b vector is assembled through the same device kernels the CG
     matvec uses (ElectrodePairKokkos / PPPMElectrodeKokkos over the full
     newton-off list, id 4), then gathered to world-iele order exactly as
     ElectrodeInv::buffer_and_gather does
   - one nele_world-sized H2D and one nele_local-sized D2H carry the
     potentials in and the charges out; sb_charges stay host
------------------------------------------------------------------------- */

#ifndef LMP_ELECTRODE_INV_KOKKOS_H
#define LMP_ELECTRODE_INV_KOKKOS_H

#include "electrode_inv.h"
#include "fix_electrode_conp_kokkos.h"

#include "electrode_pair_kokkos.h"
#include "pppm_electrode_kokkos.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

template<class DeviceType>
class ElectrodeInvKokkos : public ElectrodeInv {
 public:
  ElectrodeInvKokkos(class LAMMPS *, class FixElectrodeConp *);
  ~ElectrodeInvKokkos() noexcept override;

  // setup_solver override calls setup_device() + refresh_device_matrix()
  // after the base builds iele_local / cap_frag
  void setup_solver(int, std::unordered_map<tagint, int>, std::vector<int>, bool, bool) override;
  void update_solver(std::vector<tagint>, std::vector<int>) override;
  void set_elyt_pot(double *) override;
  double memory_use() override;

  // fine-grained per-solve staging timers: accumulated unconditionally,
  // printed from the destructor when the probe flag is set
  double t_bar = 0., t_pull = 0., t_pair = 0., t_ks = 0., t_corr = 0., t_b2h = 0.,
         t_gth = 0., t_p2d = 0., t_mv = 0., t_q2h = 0.;
  int nsolve = 0;

 private:
  // locate device pair/kspace interfaces; called from setup_solver()
  void setup_device();
  // refill d_capfrag from cap_frag rows (nlocalele x nele_world)
  void refresh_device_matrix();
  // device b-assembly: pair + self + kspace into d_b, taglist order
  void assemble_b_device();
  // destructor leg-timer print gating (mirrors the fix's timer_flag)
  bool timer_print = false;

  class FixElectrodeConp *fix;
  class FixElectrodeConpKokkos<DeviceType> *fix_kk = nullptr;
  class ElectrodeVector *dev_vec = nullptr;       // elyt vector carrying sensor/source bits
  ElectrodePairKokkos<DeviceType> *pair_kk = nullptr;
  PPPMElectrodeKokkos<DeviceType> *kspace_kk = nullptr;

  Kokkos::View<KK_ACC_FLOAT **, Kokkos::LayoutRight, DeviceType> d_capfrag;    // nele_local x nele_world
  typename ArrayTypes<DeviceType>::t_kkacc_1d d_pot;        // nele_world potentials
  typename ArrayTypes<DeviceType>::t_kkacc_1d d_b;          // nele_local b vector
  typename ArrayTypes<DeviceType>::t_kkacc_1d d_qvec;       // nele_local charges out
  typename ArrayTypes<DeviceType>::t_int_1d d_imap;         // atom -> taglist pos (-1 none)
  typename ArrayTypes<DeviceType>::t_kkacc_1d::host_mirror_type h_pot;    // persistent host mirrors
  typename ArrayTypes<DeviceType>::t_kkacc_1d::host_mirror_type h_b;
  typename ArrayTypes<DeviceType>::t_kkacc_1d::host_mirror_type h_q;
  typename ArrayTypes<DeviceType>::t_kkacc_1d::host_mirror_type h_corr;
  typename ArrayTypes<DeviceType>::t_int_1d::host_mirror_type h_imap;
  typename ArrayTypes<DeviceType>::t_kkacc_1d d_corr;       // nele_local corr staging
  double *corr_scratch = nullptr;                            // persistent corr scratch (nmax)
  int corr_scratch_nmax = 0;
  uint64_t imap_fingerprint = 0;                             // tag/taglist stamp; 0 = invalid
};

}    // namespace LAMMPS_NS

#endif
