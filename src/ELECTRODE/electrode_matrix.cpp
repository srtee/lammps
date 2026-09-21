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

#include "electrode_matrix.h"

#include "atom.h"
#include "comm.h"
#include "electrode_kspace.h"
#include "electrode_math.h"
#include "electrode_pair.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "math_const.h"
#include "neigh_list.h"
#include "pair.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

ElectrodeMatrix::ElectrodeMatrix(LAMMPS *lmp, int electrode_group, double eta) :
    Pointers(lmp), cutsq(nullptr), pair(nullptr), list(nullptr), electrode_kspace(nullptr)
{
  igroup = electrode_group;    // group of all electrode atoms
  groupbit = group->bitmask[igroup];
  ngroup = group->count(igroup);
  this->eta = eta;
  etaflag = false;
  tfflag = false;
  hardnessflag = false;
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatrix::setup(const std::unordered_map<tagint, int> &tag_ids, class Pair *fix_pair,
                            class NeighList *fix_neighlist, bool pairflag)
{
  pair = fix_pair;
  cutsq = pair->cutsq;
  list = fix_neighlist;
  this->pairflag = pairflag;

  if (pairflag) {
    electrode_pair = dynamic_cast<ElectrodePair *>(pair);
    if (electrode_pair == nullptr) error->all(FLERR, "Pair style does not implement ElectrodePair");
  }
  kspaceflag = (force->kspace != nullptr);
  if (kspaceflag) {
    electrode_kspace = dynamic_cast<ElectrodeKSpace *>(force->kspace);
    if (electrode_kspace == nullptr) error->all(FLERR, "KSpace does not implement ElectrodeKSpace");
    g_ewald = force->kspace->g_ewald;
    if (comm->me == 0)
      utils::logmesg(lmp, "ELECTRODE matrix setup with KSpace {}\n", force->kspace_style);
  } else if (comm->me == 0)
    utils::logmesg(lmp, "ELECTRODE matrix setup without KSpace\n");

  tag_to_iele = tag_ids;
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatrix::setup_tf(const std::map<int, double> &tf_types)
{
  tfflag = true;
  this->tf_types = tf_types;
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatrix::setup_hardness(int index)
{
  hardnessflag = true;
  hardness_index = index;
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatrix::setup_eta(int index)
{
  etaflag = true;
  eta_index = index;
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatrix::compute_array(double **array, bool timer_flag)
{
  // setting all entries of coulomb matrix to zero
  size_t nbytes = sizeof(double) * ngroup * ngroup;
  if (nbytes) memset(&array[0][0], 0, nbytes);

  update_mpos();
  if (pairflag) {
    // compute_matrix must traverse the fix's FULL newton-off list (the
    // pairflag kernels write only [ipos][jpos] per row). The pair's own
    // perpetual list is a HALF list: over it each unordered pair is seen
    // once, so only one triangle gets filled and the symmetrize below
    // halves every pair term; kokkos newton-off half lists additionally
    // store periodic-image pairs as two ordered entries, doubling them.
    // Swap the list in for the matrix call and restore afterward.
    NeighList *pair_own_list = pair->list;
    pair->init_list(1, list);
    electrode_pair->compute_matrix(mpos.data(), array, groupbit);
    pair->init_list(1, pair_own_list);
    electrode_pair->compute_matrix_self(mpos.data(), array, groupbit);
  } else {
    pair_contribution(array);
    self_contribution(array);
  }
  if (tfflag) tf_contribution(array);
  if (hardnessflag) hardness_contribution(array);
  if (kspaceflag) {
    MPI_Barrier(world);
    double kspace_time = MPI_Wtime();
    electrode_kspace->compute_matrix(&mpos[0], array, timer_flag);
    electrode_kspace->compute_matrix_corr(&mpos[0], array);
    MPI_Barrier(world);
    if (timer_flag && (comm->me == 0)) {
      utils::logmesg(lmp, "KSpace time: {:.4g} s\n", MPI_Wtime() - kspace_time);
    }
  }

  // every (i,j) entry is written exactly once by i's owner; exchange
  // fragment rows so every rank holds the full matrix for inversion
  const int nprocs = comm->nprocs;
  const int nlocal = atom->nlocal;
  std::vector<int> my_iele;
  for (int i = 0; i < nlocal; i++)
    if (mpos[i] >= 0) my_iele.push_back((int) mpos[i]);
  const int myrows = (int) my_iele.size();

  std::vector<int> rowcnt(nprocs), rowdis(nprocs);
  MPI_Allgather(&myrows, 1, MPI_INT, rowcnt.data(), 1, MPI_INT, world);
  rowdis[0] = 0;
  for (int p = 1; p < nprocs; p++) rowdis[p] = rowdis[p - 1] + rowcnt[p - 1];
  const int total_rows = rowdis[nprocs - 1] + rowcnt[nprocs - 1];

  // gather iele indices (gather order = rank p's local atoms in tag order)
  std::vector<int> iele_gathered(total_rows);
  MPI_Allgatherv(my_iele.data(), myrows, MPI_INT, iele_gathered.data(), rowcnt.data(),
                 rowdis.data(), MPI_INT, world);

  // gather the owned rows in the same order
  std::vector<int> vcnt(nprocs), vdis(nprocs);
  for (int p = 0; p < nprocs; p++) {
    vcnt[p] = rowcnt[p] * (int) ngroup;
    vdis[p] = rowdis[p] * (int) ngroup;
  }
  std::vector<double> sendbuf((std::size_t) myrows * ngroup);
  for (int r = 0; r < myrows; r++)
    std::copy(&array[my_iele[r]][0], &array[my_iele[r]][0] + ngroup,
              sendbuf.begin() + (std::size_t) r * ngroup);
  std::vector<double> allrows((std::size_t) total_rows * ngroup);
  MPI_Allgatherv(sendbuf.data(), myrows * (int) ngroup, MPI_DOUBLE, allrows.data(), vcnt.data(),
                 vdis.data(), MPI_DOUBLE, world);

  for (int idx = 0; idx < total_rows; idx++) {
    const int iele = iele_gathered[idx];
    std::copy(allrows.begin() + (std::size_t) idx * ngroup,
              allrows.begin() + (std::size_t) (idx + 1) * ngroup, &array[iele][0]);
  }

  // the real-space pair stage on a half neighbor list distributes each
  // unordered pair's terms between [i][j] and [j][i] according to the
  // list build's arbitrary pair ownership (backend- and newton-dependent);
  // the physical matrix is symmetric, so enforce symmetry before the
  // matrix is written out or handed to any solver
  for (int i = 0; i < ngroup; i++)
    for (int j = i + 1; j < ngroup; j++) {
      const double avg = 0.5 * (array[i][j] + array[j][i]);
      array[i][j] = avg;
      array[j][i] = avg;
    }
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatrix::pair_contribution(double **array)
{
  int inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz;
  double r, rinv, rsq, aij;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;


  // neighbor list will be ready because called from post_neighbor
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  // skip if I,J are not in 2 groups

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    // skip if atom I is not in either group
    if (!(mask[i] & groupbit)) continue;

    const bigint ipos = mpos[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    double const eta_i = etaflag ? atom->dvector[eta_index][i] : eta;
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // real-space part of matrix is symmetric
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;

      delx = xtmp - x[j][0];    // neighlists take care of pbc
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        double const eta_j = etaflag ? atom->dvector[eta_index][j] : eta;
        double const etaij = eta_i * eta_j / sqrt(eta_i * eta_i + eta_j * eta_j);

        r = sqrt(rsq);
        rinv = 1.0 / r;
        aij = rinv;
        aij *= ElectrodeMath::safe_erfc(g_ewald * r);
        aij -= ElectrodeMath::safe_erfc(etaij * r) * rinv;
        bigint jpos = mpos[j];
        // full newton-off list: each local row writes only its own
        // [ipos][jpos]; a ghost partner is the ONLY visit filling this
        // entry (its owner rank fills the mirror), so never halve
        array[ipos][jpos] += aij;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatrix::self_contribution(double **array)
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  const double selfint = 2.0 / MY_PIS * g_ewald;
  const double preta = MY_SQRT2 / MY_PIS;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double const eta_i = etaflag ? atom->dvector[eta_index][i] : eta;
      array[mpos[i]][mpos[i]] += preta * eta_i - selfint;
    }
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatrix::tf_contribution(double **array)
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  int *mask = atom->mask;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) array[mpos[i]][mpos[i]] += tf_types[type[i]];
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatrix::hardness_contribution(double **array)
{
  double *d_hardness = atom->dvector[hardness_index];
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  bool warn = false;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double hardness = d_hardness[i] / force->qqrd2e;
      array[mpos[i]][mpos[i]] += hardness;
      if (hardness < 0) warn = true;
    }
  }
  if (warn && comm->me == 0)
    error->warning(FLERR, "Hardness smaller than zero. Qeq might not converge.");
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatrix::update_mpos()
{
  const int nall = atom->nlocal + atom->nghost;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  mpos = std::vector<bigint>(nall, -1);

  for (int i = 0; i < nall; i++) {
    if (mask[i] & groupbit)
      mpos[i] = tag_to_iele[tag[i]];
    else
      mpos[i] = -1;
  }
}
