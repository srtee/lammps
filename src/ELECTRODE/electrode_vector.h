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

#ifndef LMP_ELECTRODE_VECTOR_H
#define LMP_ELECTRODE_VECTOR_H

//#include "pointers.h"
#include "fix.h"
#include <map>

namespace LAMMPS_NS {

class ElectrodeVector : public Fix {
 public:
  ElectrodeVector(class LAMMPS *, int, char **, int, int, double, bool);
  ~ElectrodeVector() override;
  int setmask() override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  void setup_general(class Pair *, class NeighList *, bool, bool);
  void setup_tf(const std::map<int, double> &);
  void setup_hardness(int index);
  void setup_eta(int);
  void compute_pot(double *);
  // host TF/hardness contributions (used by the KOKKOS device matvec, which
  // computes pair+kspace on device and these terms on host)
  void nonpair_contribution(double *);
  // accessors for the KOKKOS device matvec (ElectrodeCGKokkos)
  class NeighList *get_list() const { return list; }
  int get_groupbit() const { return groupbit; }
  int get_source_grpbit() const { return source_grpbit; }
  bool get_invert_source() const { return invert_source; }
  bool get_tfflag() const { return tfflag; }
  bool get_hardnessflag() const { return hardnessflag; }
  bool get_kspaceflag() const { return kspaceflag; }
  class ElectrodeKSpace *get_electrode_kspace() const { return electrode_kspace; }
  int igroup, source_group;
  bool buffers_stale; // for ElVecIntel

 protected: // for ElectrodeVectorIntel
  virtual void get_fix_intel() {}
  virtual void pair_contribution(double *);
  bool invert_source;
  int groupbit, source_grpbit;
  double **cutsq;
  double g_ewald, eta;
  bool etaflag;
  class NeighList *list;
  int eta_index;

 protected:    // device-matvec staging (ElectrodeCGKokkos): same flags,
               // pair/kspace on device, tf/hardness on host
  bool tfflag;
  bool pairflag;
  bool hardnessflag;
  bool kspaceflag;
  class ElectrodeKSpace *electrode_kspace;

 private:
  bigint ngroup;
  std::map<int, double> tf_types;
  int hardness_index;
  class ElectrodePair *electrode_pair;

  void self_contribution(double *);
  void tf_contribution(double *);
  void hardness_contribution(double *);

  double kspace_time_total;
  double pair_time_total;
  double boundary_time_total;
  double b_time_total;

  double *pot;             // potentials, i-indexed (0 for non-electrode atoms)

  bool timer_flag;
};

}    // namespace LAMMPS_NS

#endif
