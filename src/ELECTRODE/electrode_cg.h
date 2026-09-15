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
   Contributing authors: Ludwig Ahrens-Iwers (TUHH), Shern Tee (GU), Robert Meissner (Hereon, TUHH)
------------------------------------------------------------------------- */

#ifndef LMP_ELECTRODE_CG_H
#define LMP_ELECTRODE_CG_H

#include "charge_solver.h"
#include "electrode_vector.h"
#include "fix.h"
#include <vector>

namespace LAMMPS_NS {

class FixElectrodeConp;    // forward decl

class ElectrodeCG : public Pointers, public ChargeSolver {
 public:
  // ChargeSolver methods
  ElectrodeCG(class LAMMPS *, class FixElectrodeConp * = nullptr);
  ~ElectrodeCG() noexcept;
  void update_solver(std::vector<tagint>, std::vector<int>) override;
  void set_elyt_pot(double *) override;
  std::vector<double> solve(std::vector<double>) override;
  std::vector<double> compute_potentials() override;
  double get_potential(int) override;
  double get_sb_charges(int) override;
  double get_macro_capacitance(int, int) override;
  double get_macro_elastance(int, int) override;
  void buffer_and_gather(double const *, double *) override;
  double memory_use() override;

  // for electrode/thermo
  double vacuum_capacitance() override;

  //setup
  void setup_solver(double, ElectrodeVector *, int);

 protected:
  int nele, nele_world, ngroups;
  virtual void setup_cg(double, int);
  virtual std::vector<double> ele_ele_interaction(const std::vector<double> &);
  FixElectrodeConp *fix;

  std::vector<tagint> taglist;
  std::vector<int> iele_to_group;
  std::vector<double> q_ele;
  ElectrodeVector *elec_vec;    // bound by setup_solver; device matvec reuses it
  double *potential_i;    // potentials, i-indexed (0 for non-electrode atoms)
  int nmax;
  double evscale, threshold;
  virtual std::vector<double> cg_solve(std::vector<double> b, const std::vector<double> &x_init,
                                       bool constrain, int max_iter);
  virtual std::vector<double> pot_to_vector(double *);
  virtual double dot_product(const std::vector<double> &, const std::vector<double> &);
  void predict_q();

private:
  long nstep, ncall;
  bigint elyt_step;
  bool setup, a_cached_flag;
  int predictor_index, predictor_cols, predictor_count;
  std::vector<std::vector<double>> predictor_weights;
  std::vector<double> bvec, a_cached;

  // macro quantities for electrode/thermo + array output
 protected:
  bool macro_computed, vac_cap_computed, sb_stale;
  double vac_cap;
  std::vector<std::vector<double>> macro_capacitance, macro_elastance;
  std::vector<std::vector<double>> sd_vectors;    // evscale * x_g, x_g = unconstrained solve of M x = evscale e_g
  std::vector<double> sb_charges, applied_psi;

 private:
  void compute_macro_calibration();
  void compute_sb();
  std::vector<double> constraint_projection(std::vector<double>, bool);
};

}    // namespace LAMMPS_NS

#endif
