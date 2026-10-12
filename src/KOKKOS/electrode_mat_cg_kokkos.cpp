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
   Contributing authors: Ludwig Ahrens-Iwers (TUHH), Shern Tee (GU), Robert Meissner (Hereon, TUHH)
------------------------------------------------------------------------- */

#include "electrode_mat_cg_kokkos.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix_electrode_conp.h"
#include "memory.h"

#include <cassert>

namespace LAMMPS_NS {

template<class DeviceType>
ElectrodeMatCGKokkos<DeviceType>::ElectrodeMatCGKokkos(class LAMMPS *lmp,
                                                       class FixElectrodeConp *fix_in) :
    ElectrodeCGDeviceKokkos<DeviceType, ElectrodeMatCG>(lmp, fix_in)
{
}

template<class DeviceType>
ElectrodeMatCGKokkos<DeviceType>::~ElectrodeMatCGKokkos() noexcept = default;

/* ----------------------------------------------------------------------
   solver setup: run the base binding, then bind the device interfaces and
   upload the replicated elastance (which set_elastance() must have set).
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeMatCGKokkos<DeviceType>::setup_solver(double cg_threshold,
                                                    std::unordered_map<tagint, int> tag_to_iele,
                                                    int predictor_cols)
{
  // base: setup_cg + the initial local-row map; then bind the device
  // interfaces (sets dev_mode). The device maps themselves refresh in
  // update_solver(), which the fix calls right after construction
  // (fix_electrode_conp.cpp update_solver(taglist_local, iele_to_group_local)).
  ElectrodeMatCG::setup_solver(cg_threshold, std::move(tag_to_iele), predictor_cols);
  this->dev_setup();
}

template<class DeviceType>
double ElectrodeMatCGKokkos<DeviceType>::memory_use()
{
  double bytes = ElectrodeMatCG::memory_use();
  bytes += (double) d_elastance.extent(0) * d_elastance.extent(1) * sizeof(double);
  bytes += (double) (d_qworld.size() + d_row.size()) * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   local row map: iele_local[i] (the matrix row owned by local electrode i)
   mirrored to the device; rebuilt whenever the taglist changes.
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeMatCGKokkos<DeviceType>::dev_refresh_solver_maps()
{
  const int nele_local = this->nele;
  if ((int) d_row.extent(0) < nele_local)
    d_row = typename Base::iview("electrode/matcg/kk:d_row", nele_local);
  auto h_row = Kokkos::create_mirror_view(d_row);
  for (int i = 0; i < nele_local; i++) {
    const int row = this->iele_local[i];
    if (row < 0 || row >= this->n_mat)
      this->error->all(FLERR, "ElectrodeMatCGKokkos: local electrode row out of range");
    h_row(i) = row;
  }
  Kokkos::deep_copy(d_row, h_row);

  // upload the replicated elastance once per matrix lifetime; the base
  // update_solver errors out if n_mat ever changes
  if (!matrix_uploaded) {
    assert(this->matrix_set);
    const int n = this->n_mat;
    d_elastance = Kokkos::View<double **, Kokkos::LayoutRight, DeviceType>(
        "electrode/matcg/kk:d_elastance", n, n);
    auto h_elastance = Kokkos::create_mirror_view(d_elastance);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) h_elastance(i, j) = this->elastance[i][j];
    matrix_uploaded = true;
    Kokkos::deep_copy(d_elastance, h_elastance);
  }

  if ((int) d_qworld.extent(0) < this->n_mat)
    d_qworld = typename Base::dview("electrode/matcg/kk:d_qworld", this->n_mat);
}

/* ----------------------------------------------------------------------
   device matvec: out(i) = sum_j elastance[row(i), j] * qworld(j).
   qworld = the world-scattered charges: an MPI_Allreduce of the rank's
   local slice, exactly as in ElectrodeMatCG::ele_ele_interaction.
------------------------------------------------------------------------- */

template<class DeviceType>
void ElectrodeMatCGKokkos<DeviceType>::dev_matvec(const typename Base::dview &in,
                                                  typename Base::dview &out)
{
  const double mult_start = MPI_Wtime();
  const int nele_local = this->nele;

  // world-scatter: local charges -> host world vector -> device
  std::vector<double> h_qworld(this->n_mat, 0.);
  {
    auto h_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in);
    for (int i = 0; i < nele_local; i++) h_qworld[this->iele_local[i]] = h_in(i);
  }
  MPI_Allreduce(MPI_IN_PLACE, h_qworld.data(), this->n_mat, MPI_DOUBLE, MPI_SUM, this->world);
  Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_view(
      h_qworld.data(), this->n_mat);
  Kokkos::deep_copy(d_qworld, h_view);

  // GEMV over the rank's local rows; out stays on the device
  const auto elast = d_elastance;
  const auto qworld = d_qworld;
  const auto row = d_row;
  const int n = this->n_mat;
  Kokkos::parallel_for(
      "electrode/matcg/kk:gemv", Kokkos::RangePolicy<DeviceType>(0, nele_local),
      KOKKOS_LAMBDA(const int i) {
        double acc = 0.0;
        const double *erow = &elast(row(i), 0);
        for (int j = 0; j < n; j++) acc += erow[j] * qworld(j);
        out(i) = static_cast<KK_ACC_FLOAT>(acc);
      });

  this->mult_time += MPI_Wtime() - mult_start;
}

/* ---------------------------------------------------------------------- */

template class ElectrodeMatCGKokkos<LMPDeviceType>;

}    // namespace LAMMPS_NS
