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
   Contributing authors: Ludwig Ahrens-Iwers (MPSD, TUHH), Shern Tee (GU), Robert Meissner (Hereon, TUHH)
------------------------------------------------------------------------- */

#ifndef LMP_ELECTRODE_MAT_CG_H
#define LMP_ELECTRODE_MAT_CG_H

#include "electrode_cg.h"
#include <unordered_map>

namespace LAMMPS_NS {

class ElectrodeMatCG : public ElectrodeCG {
 public:
  // ChargeSolver methods
  ElectrodeMatCG(class LAMMPS *);
  ElectrodeMatCG(class LAMMPS *, class FixElectrodeConp *);
  ~ElectrodeMatCG() noexcept;
  void update_solver(std::vector<tagint>, std::vector<int>) override;
  void set_elastance(int nele_world, double **elastance);
  double memory_use() override;

  // setup -- virtual so the Kokkos device variant can bind after the base
  virtual void setup_solver(double, std::unordered_map<tagint, int>, int);

 protected:
  int n_mat;
  bool matrix_set;
  double **elastance;
  std::vector<double> qele_world;
  std::unordered_map<tagint, int> tag_to_iele;    // inverse of global taglist:
  std::vector<int> iele_local;                    // electrode IDs owned by me

  std::vector<double> ele_ele_interaction(const std::vector<double> &) override;
};

}    // namespace LAMMPS_NS

#endif

