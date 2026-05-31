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

#include "fix_rigs.h"

#include "angle.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;

FixRigs::FixRigs(LAMMPS *lmp, int narg, char **arg) :
    FixShake(lmp, narg, arg), rigs_type(nullptr), demoted_tag(nullptr),
    rigs_angle(nullptr),
    rigs_angle_distance(nullptr), rigs_improper_distance(nullptr),
    rigs_dihedral_distance(nullptr),
    rigs_L(nullptr), rigs_lm(nullptr), rigs_maxlist(0),
    store_lamda_corrections(false), propagate_demoted_clusters(true)
{
  restart_peratom = 1;
  atom->add_callback(Atom::RESTART);
}

FixRigs::~FixRigs()
{
  if (modify->get_fix_by_id(id)) atom->delete_callback(id, Atom::RESTART);
  memory->destroy(rigs_type);
  memory->destroy(demoted_tag);
  delete[] rigs_angle;
  delete[] rigs_angle_distance;
  delete[] rigs_improper_distance;
  delete[] rigs_dihedral_distance;
  memory->destroy(rigs_L);
  memory->destroy(rigs_lm);
}

void FixRigs::grow_arrays(int nmax)
{
  FixShake::grow_arrays(nmax);
  memory->grow(rigs_type, nmax, 3, "rigs:rigs_type");
  memory->grow(demoted_tag, nmax, "rigs:demoted_tag");
}

void FixRigs::copy_arrays(int i, int j, int delflag)
{
  FixShake::copy_arrays(i, j, delflag);
  demoted_tag[j] = demoted_tag[i];
  if (shake_flag[j] == 5 || shake_flag[j] == -5 ||
      shake_flag[j] == 6) {
    rigs_type[j][0] = rigs_type[i][0];
    rigs_type[j][1] = rigs_type[i][1];
    rigs_type[j][2] = rigs_type[i][2];
  }
}

int FixRigs::pack_exchange(int i, double *buf)
{
  int m = FixShake::pack_exchange(i, buf);
  buf[m++] = ubuf(demoted_tag[i]).d;
  if (shake_flag[i] == 5 || shake_flag[i] == -5 || 
      shake_flag[i] == 6) {
    buf[m++] = rigs_type[i][0];
    buf[m++] = rigs_type[i][1];
    buf[m++] = rigs_type[i][2];
  }
  return m;
}

int FixRigs::unpack_exchange(int nlocal, double *buf)
{
  int m = FixShake::unpack_exchange(nlocal, buf);
  demoted_tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  if (shake_flag[nlocal] == 5 || shake_flag[nlocal] == -5 || 
      shake_flag[nlocal] == 6) {
    rigs_type[nlocal][0] = static_cast<int>(buf[m++]);
    rigs_type[nlocal][1] = static_cast<int>(buf[m++]);
    rigs_type[nlocal][2] = static_cast<int>(buf[m++]);
  }
  return m;
}

int FixRigs::pack_restart(int i, double *buf)
{
  int m = 0;
  if (shake_flag[i] == 5 || shake_flag[i] == -5 || 
      shake_flag[i] == 6) {
    buf[m++] = 5;
    buf[m++] = ubuf(demoted_tag[i]).d;
    buf[m++] = rigs_type[i][0];
    buf[m++] = rigs_type[i][1];
    buf[m++] = rigs_type[i][2];
  } else {
    buf[m++] = 2;
    buf[m++] = ubuf(demoted_tag[i]).d;
  }
  return m;
}

void FixRigs::unpack_restart(int i, int nth)
{
  double **extra = atom->extra;

  int m = 0;
  for (int j = 0; j < nth; j++) m += static_cast<int>(extra[i][m]);
  m++;

  int count = static_cast<int>(extra[i][m++]);
  demoted_tag[i] = (tagint) ubuf(extra[i][m++]).i;
  if (count == 5) {
    rigs_type[i][0] = static_cast<int>(extra[i][m++]);
    rigs_type[i][1] = static_cast<int>(extra[i][m++]);
    rigs_type[i][2] = static_cast<int>(extra[i][m++]);
  }
}

int FixRigs::size_restart(int i)
{
  if (shake_flag[i] == 5 || shake_flag[i] == -5 ||
      shake_flag[i] == 6) return 5;
  return 2;
}

int FixRigs::maxsize_restart()
{
  return 5;
}

void FixRigs::init()
{
  FixShake::init();

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  delete[] rigs_angle;
  rigs_angle = new double[atom->nangletypes + 1];

  int nlocal = atom->nlocal;
  for (int i = 1; i <= atom->nangletypes; i++) {
    if (angle_flag[i] == 0) continue;
    if (force->angle == nullptr) continue;

    int bond1_type = 0, bond2_type = 0;
    for (int m = 0; m < nlocal; m++) {
      if (shake_flag[m] != 1) continue;
      if (shake_type[m][2] != i) continue;
      int type1 = MIN(shake_type[m][0], shake_type[m][1]);
      int type2 = MAX(shake_type[m][0], shake_type[m][1]);
      bond1_type = type1;
      bond2_type = type2;
      break;
    }

    int flag_all;
    MPI_Allreduce(&bond1_type, &flag_all, 1, MPI_INT, MPI_MAX, world);
    bond1_type = flag_all;
    MPI_Allreduce(&bond2_type, &flag_all, 1, MPI_INT, MPI_MAX, world);
    bond2_type = flag_all;

    if (bond1_type == 0) {
      rigs_angle[i] = 0.0;
      continue;
    }

    double b1 = bond_distance[bond1_type];
    double b2 = bond_distance[bond2_type];
    double angle = force->angle->equilibrium_angle(i);
    rigs_angle[i] = b1 * b2 * cos(angle);
  }

  delete[] rigs_angle_distance;
  rigs_angle_distance = new double[atom->nangletypes + 1];
  for (int i = 1; i <= atom->nangletypes; i++) {
    if (angle_flag[i] == 0) {
      rigs_angle_distance[i] = 0.0;
      continue;
    }
    if (force->angle == nullptr) {
      rigs_angle_distance[i] = 0.0;
      continue;
    }

    int bond1_type = 0, bond2_type = 0;
    for (int m = 0; m < nlocal; m++) {
      if (shake_flag[m] == 1 && shake_type[m][2] == i) {
        bond1_type = MIN(shake_type[m][0], shake_type[m][1]);
        bond2_type = MAX(shake_type[m][0], shake_type[m][1]);
        break;
      }
      if (shake_flag[m] == 5 && rigs_type[m][0] == i) {
        bond1_type = MIN(shake_type[m][0], shake_type[m][1]);
        bond2_type = MAX(shake_type[m][0], shake_type[m][1]);
        break;
      }
      if (shake_flag[m] == 5 && rigs_type[m][1] == i) {
        bond1_type = MIN(shake_type[m][0], shake_type[m][2]);
        bond2_type = MAX(shake_type[m][0], shake_type[m][2]);
        break;
      }
      if (shake_flag[m] == 5 && rigs_type[m][2] == i) {
        bond1_type = MIN(shake_type[m][1], shake_type[m][2]);
        bond2_type = MAX(shake_type[m][1], shake_type[m][2]);
        break;
      }
      if (shake_flag[m] == 6 && rigs_type[m][0] == i) {
        bond1_type = MIN(shake_type[m][0], shake_type[m][1]);
        bond2_type = MAX(shake_type[m][0], shake_type[m][1]);
        break;
      }
      if (shake_flag[m] == 6 && rigs_type[m][1] == i) {
        bond1_type = MIN(shake_type[m][1], shake_type[m][2]);
        bond2_type = MAX(shake_type[m][1], shake_type[m][2]);
        break;
      }
    }

    int flag_all;
    MPI_Allreduce(&bond1_type, &flag_all, 1, MPI_INT, MPI_MAX, world);
    bond1_type = flag_all;
    MPI_Allreduce(&bond2_type, &flag_all, 1, MPI_INT, MPI_MAX, world);
    bond2_type = flag_all;

    if (bond1_type == 0) {
      rigs_angle_distance[i] = 0.0;
      continue;
    }

    double b1 = bond_distance[bond1_type];
    double b2 = bond_distance[bond2_type];
    double ang = force->angle->equilibrium_angle(i);
    double rsq = b1*b1 + b2*b2 - 2.0*b1*b2*cos(ang);
    rigs_angle_distance[i] = sqrt(rsq);
    rigs_angle[i] = b1 * b2 * cos(ang);
  }

  delete[] rigs_improper_distance;
  rigs_improper_distance = new double[atom->nimpropertypes + 1];

  delete[] rigs_dihedral_distance;
  rigs_dihedral_distance = new double[atom->ndihedraltypes + 1];
}

void FixRigs::pre_neighbor()
{
  FixShake::pre_neighbor();
  prebuild_matrices();
}
