/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Authors: Shern Tee (GU), Stephen Sanderson (UQ)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(rigs,FixRigs);
// clang-format on
#else

#ifndef LMP_FIX_RIGS_H
#define LMP_FIX_RIGS_H

#include "fix_shake.h"

namespace LAMMPS_NS {

class FixRigs : public FixShake {
  public:
  FixRigs(class LAMMPS *, int, char **);
  ~FixRigs() override;
  void post_constructor() override;
  void init() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  int size_restart(int) override;
  int maxsize_restart() override;
 protected:
  int **rigs_type;
  double *rigs_angle;
  double *rigs_angle_distance;
  double *rigs_improper_distance;
  double *rigs_dihedral_distance;
  void shake3angle(int) override;
  void shake4(int ilist) override;
  void shake4improper(int ilist);
  void shake4dihedral(int ilist);
  void fill_improper_types(int i);
  int improper_check(int i);
  int bondtype_find(int i, tagint partner, int setflag);
  int impropertype_findset(int i, tagint n0, tagint n1, tagint n2, tagint n3, int setflag);
  int dihedraltype_findset(int i, tagint n1, tagint n2, tagint n3, tagint n4, int setflag);
};

}    // namespace LAMMPS_NS

#endif
#endif

