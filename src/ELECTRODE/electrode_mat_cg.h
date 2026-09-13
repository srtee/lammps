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

#ifndef LMP_ELECTRODE_MAT_CG_H
#define LMP_ELECTRODE_MAT_CG_H

#include "electrode_cg.h"
#include <unordered_map>

namespace LAMMPS_NS {

class ElectrodeMatCG : public ElectrodeCG {
 public:
  // ChargeSolver methods
  ElectrodeMatCG(class LAMMPS *);
  ~ElectrodeMatCG() noexcept;
  void update_solver(std::vector<tagint>, std::vector<int>) override;
  double memory_use() override;

  //setup
  void setup_solver(double, std::unordered_map<tagint, int>, int);
  void set_elastance(int, double **);
  // fragment access for write_inv: fragment r is the row of iele_local[r]
  const std::vector<std::vector<double>> &get_fragments() const { return el_frag; }
  std::vector<int> get_fragment_iele() const { return iele_local; }
  // after setup, the full matrix is scattered into per-rank row fragments;
  // fragment r holds the row of electrode index iele_local[r]
  void fragmentize();

 private:
  int n_mat;
  bool matrix_set, fragmented;
  double **elastance;                        // full matrix during setup only
  std::vector<std::vector<double>> el_frag;  // rows owned by this rank, nele x nele_world
  std::vector<double> qele_world;
  std::unordered_map<tagint, int> tag_to_iele;    // inverse of global taglist:
  std::vector<int> iele_local;                    // electrode IDs owned by me

  std::vector<double> ele_ele_interaction(const std::vector<double> &) override;
};

}    // namespace LAMMPS_NS

#endif

