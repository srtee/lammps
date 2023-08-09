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
   Contributing authors: Ludwig Ahrens-Iwers (TUHH), Shern Tee (UQ), Robert Meißner (TUHH)
------------------------------------------------------------------------- */

#include "electrode_vector.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "domain.h"
#include "electrode_math.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "pair.h"

#include <cmath>
#include <exception>

using namespace LAMMPS_NS;
using namespace MathConst;

ElectrodeVector::ElectrodeVector(LAMMPS *lmp, int sensor_group, int source_group, double eta,
                                 bool invert_source) :
    Pointers(lmp)
{
  igroup = sensor_group;                // group of all atoms at which we calculate potential
  this->source_group = source_group;    // group of all atoms influencing potential
  this->invert_source = invert_source;
  groupbit = group->bitmask[igroup];
  ngroup = group->count(igroup);
  source_grpbit = group->bitmask[source_group];
  this->eta = eta;
  tfflag = false;

  kspace_time_total = 0;
  pair_time_total = 0;
  boundary_time_total = 0;
  b_time_total = 0;

  tip4p_flag = false;
  nmax_tip4p = 0;
}

/* ---------------------------------------------------------------------- */

ElectrodeVector::~ElectrodeVector()
{
  if (timer_flag && (comm->me == 0)) {
    try {
      utils::logmesg(lmp, fmt::format("B time: {:.4g} s\n", b_time_total));
      utils::logmesg(lmp, fmt::format("B kspace time: {:.4g} s\n", kspace_time_total));
      utils::logmesg(lmp, fmt::format("B pair time: {:.4g} s\n", pair_time_total));
      utils::logmesg(lmp, fmt::format("B boundary time: {:.4g} s\n", boundary_time_total));
    } catch (std::exception &) {
    }
  }
}

/* ---------------------------------------------------------------------- */

void ElectrodeVector::setup(class Pair *fix_pair, class NeighList *fix_neighlist,
                            bool timer_flag)
{
  pair = fix_pair;
  cutsq = pair->cutsq;
  list = fix_neighlist;
  this->timer_flag = timer_flag;

  electrode_kspace = force->kspace;
  if (!(electrode_kspace)) error->all(FLERR, "ELECTRODE requires KSpace");
  if (!(electrode_kspace->electrodeflag)) error->all(FLERR, "KSpace does not implement ElectrodeKSpace");
  g_ewald = electrode_kspace->g_ewald;
  if (pair->tip4pflag) {
    tip4p_flag = true;
    int itmp = 0;
    auto p_qdist = (double *) force->pair->extract("qdist",itmp);
    auto p_cut_coul = (double *) force->pair->extract("cut_coul",itmp);
    int *p_typeO = (int *) force->pair->extract("typeO",itmp);
    int *p_typeH = (int *) force->pair->extract("typeH",itmp);
    int *p_typeA = (int *) force->pair->extract("typeA",itmp);
    int *p_typeB = (int *) force->pair->extract("typeB",itmp);
    if (!p_qdist || !p_typeO || !p_typeH || !p_typeA || !p_typeB)
      error->all(FLERR,"Pair style is incompatible with fix electrode TIP4P mode");
    qdist = *p_qdist;
    typeO = *p_typeO;
    typeH = *p_typeH;
    int typeA = *p_typeA;
    int typeB = *p_typeB;

    if (force->angle == nullptr || force->bond == nullptr ||
        force->angle->setflag == nullptr || force->bond->setflag == nullptr)
      error->all(FLERR,"Bond and angle potentials must be defined for TIP4P");
    if (typeA < 1 || typeA > atom->nangletypes ||
        force->angle->setflag[typeA] == 0)
      error->all(FLERR,"Bad TIP4P angle type for PPPM/TIP4P");
    if (typeB < 1 || typeB > atom->nbondtypes ||
        force->bond->setflag[typeB] == 0)
      error->all(FLERR,"Bad TIP4P bond type for PPPM/TIP4P");
    double theta = force->angle->equilibrium_angle(typeA);
    double blen = force->bond->equilibrium_distance(typeB);
    alpha = qdist / (cos(0.5*theta) * blen);
  }
}

/* ---------------------------------------------------------------------- */

void ElectrodeVector::setup_tf(const std::map<int, double> &tf_types)
{
  tfflag = true;
  this->tf_types = tf_types;
}

/* ---------------------------------------------------------------------- */

void ElectrodeVector::compute_vector(double *vector)
{
  MPI_Barrier(world);
  double start_time = MPI_Wtime();
  // pair
  double pair_start_time = MPI_Wtime();
  if (tip4p_flag) pair_contribution_tip4p(vector);
  else pair_contribution(vector);
  self_contribution(vector);
  if (tfflag) tf_contribution(vector);
  MPI_Barrier(world);
  pair_time_total += MPI_Wtime() - pair_start_time;
  // kspace
  double kspace_start_time = MPI_Wtime();
  electrode_kspace->potential_group_group(vector, groupbit, source_grpbit, invert_source);
  MPI_Barrier(world);
  kspace_time_total += MPI_Wtime() - kspace_start_time;
  // boundary
  double boundary_start_time = MPI_Wtime();
  electrode_kspace->potential_group_group_corr(vector, groupbit, source_grpbit, invert_source);
  MPI_Barrier(world);
  boundary_time_total += MPI_Wtime() - boundary_start_time;
  b_time_total += MPI_Wtime() - start_time;
}

/* ---------------------------------------------------------------------- */

void ElectrodeVector::pair_contribution(double *vector)
{
  double const etaij = eta * MY_ISQRT2;
  double **x = atom->x;
  double *q = atom->q;
  int *type = atom->type;
  int *mask = atom->mask;
  // neighbor list will be ready because called from post_neighbor
  int const nlocal = atom->nlocal;
  int const inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int newton_pair = force->newton_pair;

  for (int ii = 0; ii < inum; ii++) {
    int const i = ilist[ii];
    bool const i_in_sensor = (mask[i] & groupbit);
    bool const i_in_source = !!(mask[i] & source_grpbit) != invert_source;
    if (!(i_in_sensor || i_in_source)) continue;
    double const xtmp = x[i][0];
    double const ytmp = x[i][1];
    double const ztmp = x[i][2];
    int itype = type[i];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int const j = jlist[jj] & NEIGHMASK;
      bool const j_in_sensor = (mask[j] & groupbit);
      bool const j_in_source = !!(mask[j] & source_grpbit) != invert_source;
      bool const compute_ij = i_in_sensor && j_in_source;
      bool const compute_ji = (newton_pair || j < nlocal) && (j_in_sensor && i_in_source);
      if (!(compute_ij || compute_ji)) continue;
      double const delx = xtmp - x[j][0];    // neighlists take care of pbc
      double const dely = ytmp - x[j][1];
      double const delz = ztmp - x[j][2];
      double const rsq = delx * delx + dely * dely + delz * delz;
      int jtype = type[j];
      if (rsq >= cutsq[itype][jtype]) continue;
      double const r = sqrt(rsq);
      double const rinv = 1.0 / r;
      double aij = rinv;
      aij *= ElectrodeMath::safe_erfc(g_ewald * r);
      if (invert_source) // TODO: safer check for different types' eta
        aij -= ElectrodeMath::safe_erfc(eta * r) * rinv;
      else
        aij -= ElectrodeMath::safe_erfc(etaij * r) * rinv;
      if (i_in_sensor) vector[i] += aij * q[j];
      if (j_in_sensor) vector[j] += aij * q[i];
    }
  }
}

/* ---------------------------------------------------------------------- */

void ElectrodeVector::self_contribution(double *vector)
{
  int const inum = list->inum;
  int *mask = atom->mask;
  int *ilist = list->ilist;
  double *q = atom->q;

  const double selfint = 2.0 / MY_PIS * g_ewald;
  const double preta = MY_SQRT2 / MY_PIS;

  for (int ii = 0; ii < inum; ii++) {
    int const i = ilist[ii];
    bool const i_in_sensor = (mask[i] & groupbit);
    bool const i_in_source = !!(mask[i] & source_grpbit) != invert_source;
    if (i_in_sensor && i_in_source) vector[i] += (preta * eta - selfint) * q[i];
  }
}

/* ---------------------------------------------------------------------- */

void ElectrodeVector::tf_contribution(double *vector)
{
  int const inum = list->inum;
  int *mask = atom->mask;
  int *type = atom->type;
  int *ilist = list->ilist;
  double *q = atom->q;

  for (int ii = 0; ii < inum; ii++) {
    int const i = ilist[ii];
    bool const i_in_sensor = (mask[i] & groupbit);
    bool const i_in_source = !!(mask[i] & source_grpbit) != invert_source;
    if (i_in_sensor && i_in_source) vector[i] += tf_types[type[i]] * q[i];
  }
}

/* ---------------------------------------------------------------------- */

void ElectrodeVector::pair_contribution_tip4p(double *vector)
{
  double const etaij = eta * MY_ISQRT2;
  double **x = atom->x;
  double *q = atom->q;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  // neighbor list will be ready because called from post_neighbor
  int const nlocal = atom->nlocal;
  int const inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int newton_pair = force->newton_pair;

  double const cut_coulsqplus = (cut_coul+2.0*qdist) * (cut_coul+2.0*qdist);

  double *xi, *xj, *xH1, *xH2;
  int iH1, iH2, jH1, jH2;

  int const nall = nlocal + atom->nghost;

  if (atom->nmax > nmax_tip4p) {
    nmax_tip4p = atom->nmax;
    memory->destroy(hneigh);
    memory->create(hneigh,nmax_tip4p,3,"pair:hneigh");
    memory->destroy(newsite);
    memory->create(newsite,nmax_tip4p,3,"pair:newsite");
  }

  if (neighbor->ago == 0)
    for (int i = 0; i < nall; i++) hneigh[i][0] = -1;
  for (int i = 0; i < nall; i++) hneigh[i][2] = 0;

  for (int ii = 0; ii < inum; ii++) {
    int const i = ilist[ii];
    bool const i_in_sensor = (mask[i] & groupbit);
    bool const i_in_source = !!(mask[i] & source_grpbit) != invert_source;
    if (!(i_in_sensor || i_in_source)) continue;
    int itype = type[i];
    if (itype == typeO) {
      if (hneigh[i][0] < 0) {
        iH1 = atom->map(tag[i] + 1);
        iH2 = atom->map(tag[i] + 2);
        if (iH1 == -1 || iH2 == -1)
          error->one(FLERR,"TIP4P hydrogen is missing");
        if (atom->type[iH1] != typeH || atom->type[iH2] != typeH)
          error->one(FLERR,"TIP4P hydrogen has incorrect atom type");
        // set iH1,iH2 to closest image to O
        iH1 = domain->closest_image(i,iH1);
        iH2 = domain->closest_image(i,iH2);
        compute_newsite(x[i],x[iH1],x[iH2],newsite[i]);
        hneigh[i][0] = iH1;
        hneigh[i][1] = iH2;
        hneigh[i][2] = 1;
      } else {
        iH1 = hneigh[i][0];
        iH2 = hneigh[i][1];
        if (hneigh[i][2] == 0) {
          hneigh[i][2] = 1;
          compute_newsite(x[i],x[iH1],x[iH2],newsite[i]);
        }
      }
      xi = newsite[i];
    } else xi = x[i];
    double const xtmp = xi[0];
    double const ytmp = xi[1];
    double const ztmp = xi[2];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int const j = jlist[jj] & NEIGHMASK;
      bool const j_in_sensor = (mask[j] & groupbit);
      bool const j_in_source = !!(mask[j] & source_grpbit) != invert_source;
      bool const compute_ij = i_in_sensor && j_in_source;
      bool const compute_ji = (newton_pair || j < nlocal) && (j_in_sensor && i_in_source);
      if (!(compute_ij || compute_ji)) continue;
      int jtype = type[j];
      if (jtype == typeO) {
        if (hneigh[j][0] < 0) {
          jH1 = atom->map(tag[j] + 1);
          jH2 = atom->map(tag[j] + 2);
          if (jH1 == -1 || jH2 == -1)
            error->one(FLERR,"TIP4P hydrogen is missing");
          if (atom->type[jH1] != typeH || atom->type[jH2] != typeH)
            error->one(FLERR,"TIP4P hydrogen has incorrect atom type");
          // set iH1,iH2 to closest image to O
          jH1 = domain->closest_image(j,jH1);
          jH2 = domain->closest_image(j,jH2);
          compute_newsite(x[j],x[jH1],x[jH2],newsite[j]);
          hneigh[j][0] = jH1;
          hneigh[j][1] = jH2;
          hneigh[j][2] = 1;

        } else {
          iH1 = hneigh[i][0];
          iH2 = hneigh[j][1];
          if (hneigh[j][2] == 0) {
            hneigh[j][2] = 1;
            compute_newsite(x[j],x[jH1],x[jH2],newsite[j]);
          }
        }
        xj = newsite[j];
      } else xj = x[j];
      double const delx = xtmp - xj[0];    // neighlists take care of pbc
      double const dely = ytmp - xj[1];
      double const delz = ztmp - xj[2];
      double const rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cut_coulsqplus) continue;
      double const r = sqrt(rsq);
      double const rinv = 1.0 / r;
      double aij = rinv;
      aij *= ElectrodeMath::safe_erfc(g_ewald * r);
      if (invert_source) // TODO: safer check for different types' eta
        aij -= ElectrodeMath::safe_erfc(eta * r) * rinv;
      else
        aij -= ElectrodeMath::safe_erfc(etaij * r) * rinv;
      if (i_in_sensor) vector[i] += aij * q[j];
      if (j_in_sensor) vector[j] += aij * q[i];
    }
  }
}

void ElectrodeVector::compute_newsite(double *xO, double *xH1,
                                         double *xH2, double *xM)
{
  double delx1 = xH1[0] - xO[0];
  double dely1 = xH1[1] - xO[1];
  double delz1 = xH1[2] - xO[2];

  double delx2 = xH2[0] - xO[0];
  double dely2 = xH2[1] - xO[1];
  double delz2 = xH2[2] - xO[2];

  xM[0] = xO[0] + alpha * 0.5 * (delx1 + delx2);
  xM[1] = xO[1] + alpha * 0.5 * (dely1 + dely2);
  xM[2] = xO[2] + alpha * 0.5 * (delz1 + delz2);
}


