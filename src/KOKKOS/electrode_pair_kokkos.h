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

/* Interface for pair styles that provide ElectrodePair contributions with
   device-resident output (stage K4, device CG). Templated on DeviceType so
   the view argument types match each PairKokkos instantiation exactly; the
   caller (ElectrodeCGKokkos, templated on the same DeviceType) dynamic_casts
   to ElectrodePairKokkos<DeviceType>.
   Lives in src/KOKKOS so the ELECTRODE package stays Kokkos-free. */

#ifndef LMP_ELECTRODE_PAIR_KOKKOS_H
#define LMP_ELECTRODE_PAIR_KOKKOS_H

#include "kokkos_type.h"

namespace LAMMPS_NS {

class NeighList;

template<class DeviceType>
class ElectrodePairKokkos {
 public:
  virtual ~ElectrodePairKokkos() noexcept(false) {}    // NOLINT: GCC 8 RHEL8

  // vector contribution, device output:
  //   neigh:  caller's neighbor list (full + newton-off for the CG path);
  //           rows of ghost atoms duplicate their local twin's row and are
  //           skipped by the kernel
  //   d_out:  pre-zeroed accumulator, one entry per local electrode atom
  //           (nele, iele order)
  //   d_imap: atom index -> local electrode index, -1 = not an electrode
  //           atom; sensor side written through d_imap, no atomics
  //           (each ordered (sensor, source) pair appears exactly once)
  // groupbit/source_grpbit/inv: same semantics as ElectrodePair::compute_vector
  virtual void compute_vector_nele(NeighList *neigh,
                                   typename ArrayTypes<DeviceType>::t_kkacc_1d &d_out,
                                   typename ArrayTypes<DeviceType>::t_int_1d &d_imap,
                                   int groupbit, int source_grpbit, bool inv) = 0;

  // self contribution (diagonal terms, same semantics as
  // ElectrodePair::compute_vector_self): nele-scattered device output.
  // Requires the view bindings and nlocal from a preceding
  // compute_vector_nele call (always co-issued by ElectrodeVector).
  virtual void compute_vector_self_nele(typename ArrayTypes<DeviceType>::t_kkacc_1d &d_out,
                                        typename ArrayTypes<DeviceType>::t_int_1d &d_imap,
                                        int groupbit, int source_grpbit, bool inv) = 0;
};

}    // namespace LAMMPS_NS

#endif