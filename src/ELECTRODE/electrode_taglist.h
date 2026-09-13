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

#ifndef LMP_ELECTRODE_TAGLIST_H
#define LMP_ELECTRODE_TAGLIST_H

#include "pointers.h"
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

class ElectrodeTaglist : public Pointers {
 public:
  ElectrodeTaglist(class LAMMPS *, std::vector<int>);
  ~ElectrodeTaglist();
  std::unordered_map<tagint, int> get_tag_to_iele();
  void write_to_file(const std::string, double **);
  void write_to_file(const std::string, double *);
  void read_from_file(const std::string &, double **, const std::string &);
  // reassemble a full n x n matrix (taglist order) on rank 0 and broadcast;
  // frag[r] is the row of electrode index frag_iele[r] as held by this rank
  void gather_full_matrix_to_zero(double **dst, const std::vector<std::vector<double>> &frag,
                                  const std::vector<int> &frag_iele, std::size_t n);
  double memory_usage();

 private:
  std::vector<std::vector<double>> sort_by_group(double **);
  std::vector<std::vector<double>> sort_by_group(double *);
  void write_to_file(const std::string, std::vector<std::vector<double>>);
  std::vector<tagint> taglist, taglist_bygroup;
  std::vector<int> group_idx;
  std::unordered_map<tagint, int> tag_to_iele;    // inverse of taglist:
  std::size_t n;
};

}    // namespace LAMMPS_NS

#endif

