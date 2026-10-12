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
   Device-resident CG recurrence shared by the two CG-based ELECTRODE
   solvers.  Template parameter HostSolver selects the host base class:

     ElectrodeCGKokkos    = ElectrodeCGDeviceKokkos<DT, ElectrodeCG>
                            (pair/kspace matvec, "algo cg")
     ElectrodeMatCGKokkos = ElectrodeCGDeviceKokkos<DT, ElectrodeMatCG>
                            (replicated-matrix matvec, "algo mat_cg")

   When the device lane is active (fix electrode/conp/kk, "device on",
   non-host instantiation) the WHOLE solve runs on the device: ASPC
   predictor history, rhs assembly, constraint projections, dots, axpys.
   The only host traffic per iteration is the MPI reduction behind each
   CG dot (unavoidable: the convergence test is a global scalar); the
   charge vector itself never crosses the PCIe bus inside the solve.

   Host fallback: with dev_mode false (host instantiation, or "device
   no") every override passes straight through to HostSolver, reproducing
   the existing lanes exactly.
------------------------------------------------------------------------- */

#ifndef LMP_ELECTRODE_CG_DEVICE_KOKKOS_H
#define LMP_ELECTRODE_CG_DEVICE_KOKKOS_H

#include "electrode_cg.h"
#include "electrode_pair_kokkos.h"
#include "fix_electrode_conp_kokkos.h"
#include "pppm_electrode_kokkos.h"

#include "atom_kokkos.h"
#include "comm.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

template<class DeviceType, class HostSolver>
class ElectrodeCGDeviceKokkos : public HostSolver {
 public:
  ElectrodeCGDeviceKokkos(class LAMMPS *lmp, class FixElectrodeConp *fix_in) : HostSolver(lmp, fix_in) {}

  ~ElectrodeCGDeviceKokkos() noexcept override = default;

  // ChargeSolver overrides: device-resident when the device lane is on,
  // verbatim HostSolver behavior otherwise
  void update_solver(std::vector<tagint> taglist_local, std::vector<int> iele_to_group_local) override
  {
    HostSolver::update_solver(std::move(taglist_local), std::move(iele_to_group_local));
    if (dev_mode) dev_refresh_maps();
  }
  void set_elyt_pot(double *b_nall) override
  {
    if (dev_mode) dev_set_elyt_pot();
    else
      HostSolver::set_elyt_pot(b_nall);
  }
  std::vector<double> solve(std::vector<double> v) override
  {
    return dev_mode ? dev_solve(std::move(v)) : HostSolver::solve(std::move(v));
  }
  void scatter_device() override
  {
    if (dev_mode) dev_scatter_q();
  }
  std::vector<double> compute_potentials() override
  {
    if (dev_mode) return dev_compute_potentials();
    return HostSolver::compute_potentials();
  }
  double get_potential(int igroup) override
  {
    if (dev_mode) dev_sync_bvec_host();
    return HostSolver::get_potential(igroup);
  }
  std::vector<double> cg_solve(
      std::vector<double> b, const std::vector<double> &x_init, bool constrain, int max_iter) override;

 public:
  // device-method block is public: nvcc forbids extended device lambdas in
  // member functions with protected/private access
  using dview = typename ArrayTypes<DeviceType>::t_kkacc_1d;
  using iview = typename ArrayTypes<DeviceType>::t_int_1d;
  // bind the device interfaces (fix, pair, kspace, elyt vector) and set
  // dev_mode; call from the final class's setup_solver after the base bind
  void dev_setup();

  // the matvec: out = (evscale * A) . in, entirely on the device
  virtual void dev_matvec(const dview &in, dview &out) = 0;

  // solver-specific device refresh at update_solver() time (matrix
  // branch: the local-row map); default no-op
  virtual void dev_refresh_solver_maps() {}

  // full device map refresh (bvec/vg/weights/counters + refresh_imap +
  // solver-specific maps); runs from update_solver() when dev_mode
  void dev_refresh_maps();
  // atom -> iele map only (cheap); runs at each b-assembly
  void dev_refresh_imap();
  void dev_set_elyt_pot();
  std::vector<double> dev_solve(std::vector<double> v);
  void dev_cg_solve(const dview &d_b, dview &d_x, bool constrain, int max_iter);
  void dev_project(const dview &src, dview &dst, bool correction);
  double dev_dot(const dview &a, const dview &b);
  double dev_sum(const dview &a);
  std::vector<double> dev_cg_solve_override(std::vector<double> b, const std::vector<double> &x_init,
                                            bool constrain, int max_iter);
  void dev_scatter_into(const dview &src);
  void dev_scatter_q();
  std::vector<double> dev_compute_potentials();
  void dev_sync_bvec_host();
  void dev_predict_q();
  void dev_seed_from_atoms(dview &x);

 protected:
  bool dev_mode = false;
  class FixElectrodeConpKokkos<DeviceType> *fix_kk = nullptr;
  ElectrodePairKokkos<DeviceType> *pair_kk = nullptr;
  PPPMElectrodeKokkos<DeviceType> *kspace_kk = nullptr;
  ElectrodeVector *dev_elyt = nullptr;    // elyt vector carrying the b-assembly bits

  // atom -> iele map, shared with the K4 host-recurrence lane
  iview d_imap;
  typename iview::host_mirror_type h_imap;
  uint64_t imap_fingerprint = 0;
  dview d_bvec, d_b, d_x, d_q, d_r, d_d, d_p, d_y, d_vg, d_sums, d_en, d_cb, d_cx;
  iview d_iele2grp, d_gcounts;
  Kokkos::View<double **, Kokkos::LayoutRight, DeviceType> d_hist, d_hist_tmp, d_w;
  std::vector<int> h_rowpos;    // row k -> atom position (inverse of h_imap);
                                // the system may have no atom map (MAP_NONE)
  iview d_rowpos;               // device copy of h_rowpos: the charge scatter
                                // writes row k's charge to atom d_rowpos(k)
  bool solved_once = false;

};

}    // namespace LAMMPS_NS

#endif
