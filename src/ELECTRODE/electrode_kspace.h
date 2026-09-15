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

#ifndef LMP_ELECTRODE_KSPACE_H
#define LMP_ELECTRODE_KSPACE_H

#ifdef LMP_KOKKOS
#include "kokkos_type.h"
#endif
#include <stdexcept>

#include "lmptype.h"

namespace LAMMPS_NS {
class ElectrodeKSpace {
 public:
  // clang-format off
  // Cannot use =default here because of broken GCC 8 on RHEL 8
  virtual ~ElectrodeKSpace() noexcept(false) {} // NOLINT
  // clang-format on
  virtual void compute_vector(double *, int, int, bool) = 0;
  virtual void compute_vector_corr(double *, int, int, bool) = 0;
  virtual void compute_matrix(bigint *, double **, bool) = 0;
  virtual void compute_matrix_corr(bigint *, double **) = 0;

#ifdef LMP_KOKKOS
  // Device-resident variant (stage K4, Kokkos device CG): same semantics as
  // compute_vector but the sensor output is scattered on the device into
  // d_out (pre-zeroed, nele entries, iele order) through d_imap
  // (atom index -> local electrode index, -1 = not an electrode atom).
  // Only meaningful on styles that support it (KOKKOS-instantiated PPPM);
  // default fails loudly so host CG callers never hit it silently.
  virtual void compute_vector_nele(typename ArrayTypes<LMPDeviceType>::t_kkacc_1d &,
                                   typename ArrayTypes<LMPDeviceType>::t_int_1d &, int, int,
                                   bool)
  {
    throw std::runtime_error("KSpace style does not implement compute_vector_nele");
  }
#endif
};
}    // namespace LAMMPS_NS

#endif
