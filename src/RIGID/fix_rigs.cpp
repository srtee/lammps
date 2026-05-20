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
#include "error.h"
#include "force.h"
#include "mat2.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace RigsMath;

FixRigs::FixRigs(LAMMPS *lmp, int narg, char **arg) :
    FixShake(lmp, narg, arg), rigs_type(nullptr), rigs_angle(nullptr) {}

FixRigs::~FixRigs()
{
  memory->destroy(rigs_type);
  delete[] rigs_angle;
}

void FixRigs::post_constructor()
{
  grow_arrays(atom->nmax);

  int i;
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  int *mask_atom = atom->mask;

  atommols = atom->avec->onemols;

  int impropers_allow = atom->avec->impropers_allow;

  int nimproper = 0;

  for (i = 0; i < nlocal; i++) {
    rigs_type[i][0] = 0;
    rigs_type[i][1] = 0;
    rigs_type[i][2] = 0;

    if (shake_flag[i] == 0) continue;
    if (!(mask_atom[i] & groupbit)) continue;
    if (shake_atom[i][0] != tag[i]) continue;
    if (shake_flag[i] == 4) {
      int angles_allow_flag = atom->avec->angles_allow;
      int nangle_found = 0;
      if (angles_allow_flag) {
        tagint a1 = shake_atom[i][1];
        tagint a2 = shake_atom[i][2];
        tagint a3_atom = shake_atom[i][3];
        int n = angletype_findset(i, a1, a2, 0);
        if (n > 0 && angle_flag[n]) nangle_found++;
        n = angletype_findset(i, a1, a3_atom, 0);
        if (n > 0 && angle_flag[n]) nangle_found++;
        n = angletype_findset(i, a2, a3_atom, 0);
        if (n > 0 && angle_flag[n]) nangle_found++;
      }
      if (nangle_found == 3) {
        shake_flag[i] = 5;
        nimproper++;
      } else if (impropers_allow) {
        int impflag = improper_check(i);
        if (impflag) {
          shake_flag[i] = 5;
          nimproper++;
        }
      }
    }
  }

  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] == 5) {
      fill_improper_types(i);
    }
  }

  find_clusters();

  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] == 5) {
      if (rigs_type[i][0] > 0)
        angletype_findset(i, shake_atom[i][1], shake_atom[i][2], -1);
      if (rigs_type[i][1] > 0)
        angletype_findset(i, shake_atom[i][1], shake_atom[i][3], -1);
      if (rigs_type[i][2] > 0)
        angletype_findset(i, shake_atom[i][2], shake_atom[i][3], -1);
    }
  }
}

int FixRigs::improper_check(int i)
{
  tagint a1 = shake_atom[i][1];
  tagint a2 = shake_atom[i][2];
  tagint a3_atom = shake_atom[i][3];

  if (molecular == Atom::MOLECULAR) {
    int nimp = atom->num_improper[i];
    for (int m = 0; m < nimp; m++) {
      if (atom->improper_type[i][m] <= 0) continue;
      if (!improper_flag[atom->improper_type[i][m]]) continue;
      tagint b1 = atom->improper_atom1[i][m];
      tagint b2 = atom->improper_atom2[i][m];
      tagint b3 = atom->improper_atom3[i][m];
      tagint b4 = atom->improper_atom4[i][m];

      if (b1 != atom->tag[i]) continue;
      if ((b2 == a1 && b3 == a2 && b4 == a3_atom) ||
          (b2 == a1 && b3 == a3_atom && b4 == a2) ||
          (b2 == a2 && b3 == a1 && b4 == a3_atom) ||
          (b2 == a2 && b3 == a3_atom && b4 == a1) ||
          (b2 == a3_atom && b3 == a1 && b4 == a2) ||
          (b2 == a3_atom && b3 == a2 && b4 == a1))
        return 1;
    }
  } else {
    int imol = atom->molindex[i];
    int iatom = atom->molatom[i];
    tagint tagprev = atom->tag[i] - iatom - 1;
    int nimp = atommols[imol]->num_improper[iatom];
    for (int m = 0; m < nimp; m++) {
      if (atommols[imol]->improper_type[iatom][m] <= 0) continue;
      if (!improper_flag[atommols[imol]->improper_type[iatom][m]]) continue;
      tagint b1 = atommols[imol]->improper_atom1[iatom][m] + tagprev;
      tagint b2 = atommols[imol]->improper_atom2[iatom][m] + tagprev;
      tagint b3 = atommols[imol]->improper_atom3[iatom][m] + tagprev;
      tagint b4 = atommols[imol]->improper_atom4[iatom][m] + tagprev;
      if (b1 != atom->tag[i]) continue;
      if ((b2 == a1 && b3 == a2 && b4 == a3_atom) ||
          (b2 == a1 && b3 == a3_atom && b4 == a2) ||
          (b2 == a2 && b3 == a1 && b4 == a3_atom) ||
          (b2 == a2 && b3 == a3_atom && b4 == a1) ||
          (b2 == a3_atom && b3 == a1 && b4 == a2) ||
          (b2 == a3_atom && b3 == a2 && b4 == a1))
        return 1;
    }
  }
  return 0;
}

void FixRigs::fill_improper_types(int i)
{
  tagint a1 = shake_atom[i][1];
  tagint a2 = shake_atom[i][2];
  tagint a3_atom = shake_atom[i][3];

  rigs_type[i][0] = 0;
  rigs_type[i][1] = 0;
  rigs_type[i][2] = 0;

  int n = angletype_findset(i, a1, a2, 0);
  if (n > 0 && angle_flag[n]) rigs_type[i][0] = n;

  n = angletype_findset(i, a1, a3_atom, 0);
  if (n > 0 && angle_flag[n]) rigs_type[i][1] = n;

  n = angletype_findset(i, a2, a3_atom, 0);
  if (n > 0 && angle_flag[n]) rigs_type[i][2] = n;

  int nimp = impropertype_findset(i, shake_atom[i][0], a1, a2, a3_atom, 0);
  if (nimp > 0 && improper_flag[nimp]) {
    if (rigs_type[i][2] == 0) rigs_type[i][2] = -nimp;
  }
}

int FixRigs::impropertype_findset(int i, tagint n0, tagint n1, tagint n2, tagint n3, int setflag)
{
  int m, nimp;

  if (molecular == Atom::MOLECULAR) {
    nimp = atom->num_improper[i];
    for (m = 0; m < nimp; m++) {
      tagint b0 = atom->improper_atom1[i][m];
      tagint b1 = atom->improper_atom2[i][m];
      tagint b2 = atom->improper_atom3[i][m];
      tagint b3 = atom->improper_atom4[i][m];
      if (b0 != n0) continue;
      if (b1 == n1 && b2 == n2 && b3 == n3) break;
      if (b1 == n1 && b2 == n3 && b3 == n2) break;
      if (b1 == n2 && b2 == n1 && b3 == n3) break;
    }
  } else {
    int imol = atom->molindex[i];
    int iatom = atom->molatom[i];
    tagint tagprev = atom->tag[i] - iatom - 1;
    nimp = atommols[imol]->num_improper[iatom];
    int *itype = atommols[imol]->improper_type[iatom];
    for (m = 0; m < nimp; m++) {
      tagint b0 = atommols[imol]->improper_atom1[iatom][m] + tagprev;
      tagint b1 = atommols[imol]->improper_atom2[iatom][m] + tagprev;
      tagint b2 = atommols[imol]->improper_atom3[iatom][m] + tagprev;
      tagint b3 = atommols[imol]->improper_atom4[iatom][m] + tagprev;
      if (b0 != n0) continue;
      if (b1 == n1 && b2 == n2 && b3 == n3) break;
      if (b1 == n1 && b2 == n3 && b3 == n2) break;
      if (b1 == n2 && b2 == n1 && b3 == n3) break;
    }
  }

  if (m < nimp) {
    if (setflag == 0) {
      if (molecular == Atom::MOLECULAR) return atom->improper_type[i][m];
      else {
        int *itype = atommols[atom->molindex[i]]->improper_type[atom->molatom[i]];
        return itype[m];
      }
    }
    if (molecular == Atom::MOLECULAR) {
      if ((setflag < 0 && atom->improper_type[i][m] > 0) ||
          (setflag > 0 && atom->improper_type[i][m] < 0))
        atom->improper_type[i][m] = -atom->improper_type[i][m];
    } else {
      int *itype = atommols[atom->molindex[i]]->improper_type[atom->molatom[i]];
      if ((setflag < 0 && itype[m] > 0) ||
          (setflag > 0 && itype[m] < 0))
        itype[m] = -itype[m];
    }
  }

  return 0;
}

void FixRigs::grow_arrays(int nmax)
{
  FixShake::grow_arrays(nmax);
  memory->grow(rigs_type, nmax, 3, "rigs:rigs_type");
}

void FixRigs::copy_arrays(int i, int j, int delflag)
{
  FixShake::copy_arrays(i, j, delflag);
  if (shake_flag[j] == 5 || shake_flag[j] == 6) {
    rigs_type[j][0] = rigs_type[i][0];
    rigs_type[j][1] = rigs_type[i][1];
    rigs_type[j][2] = rigs_type[i][2];
  }
}

int FixRigs::pack_exchange(int i, double *buf)
{
  int m = FixShake::pack_exchange(i, buf);
  if (shake_flag[i] == 5 || shake_flag[i] == 6) {
    buf[m++] = rigs_type[i][0];
    buf[m++] = rigs_type[i][1];
    buf[m++] = rigs_type[i][2];
  }
  return m;
}

int FixRigs::unpack_exchange(int nlocal, double *buf)
{
  int m = FixShake::unpack_exchange(nlocal, buf);
  if (shake_flag[nlocal] == 5 || shake_flag[nlocal] == 6) {
    rigs_type[nlocal][0] = static_cast<int>(buf[m++]);
    rigs_type[nlocal][1] = static_cast<int>(buf[m++]);
    rigs_type[nlocal][2] = static_cast<int>(buf[m++]);
  }
  return m;
}

int FixRigs::pack_restart(int i, double *buf)
{
  int m = FixShake::pack_restart(i, buf);
  if (shake_flag[i] == 5 || shake_flag[i] == 6) {
    buf[m++] = rigs_type[i][0];
    buf[m++] = rigs_type[i][1];
    buf[m++] = rigs_type[i][2];
  }
  return m;
}

void FixRigs::unpack_restart(int i, int ncol, double *buf)
{
  FixShake::unpack_restart(i, ncol, buf);
  // TODO: restore rigs_type from buf for restart
}

int FixRigs::size_restart(int i)
{
  int n = FixShake::size_restart(i);
  if (shake_flag[i] == 5 || shake_flag[i] == 6) n += 3;
  return n;
}

int FixRigs::maxsize_restart()
{
  return FixShake::maxsize_restart() + 3;
}

void FixRigs::init()
{
  FixShake::init();

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
}

/* ----------------------------------------------------------------------
   calculate RIGS constraint forces for size 3 cluster = two bonds + angle
------------------------------------------------------------------------- */

void FixRigs::shake3angle(int ilist)
{
  int atomlist[3];
  double v[6];
  double invmass0,invmass01,invmass02;

  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  double bond1 = bond_distance[shake_type[m][0]];
  double bond2 = bond_distance[shake_type[m][1]];
  double bond12 = rigs_angle[shake_type[m][2]];

  double r01[3];
  r01[0] = x[i0][0] - x[i1][0];
  r01[1] = x[i0][1] - x[i1][1];
  r01[2] = x[i0][2] - x[i1][2];

  double r02[3];
  r02[0] = x[i0][0] - x[i2][0];
  r02[1] = x[i0][1] - x[i2][1];
  r02[2] = x[i0][2] - x[i2][2];

  double s01[3];
  s01[0] = xshake[i0][0] - xshake[i1][0];
  s01[1] = xshake[i0][1] - xshake[i1][1];
  s01[2] = xshake[i0][2] - xshake[i1][2];

  double s02[3];
  s02[0] = xshake[i0][0] - xshake[i2][0];
  s02[1] = xshake[i0][1] - xshake[i2][1];
  s02[2] = xshake[i0][2] - xshake[i2][2];

  SymMat2 rr = sym_dot(r01, r02);
  SymMat2 ss = sym_dot(s01, s02);

  SymMat2 L = {bond1 * bond1, bond12, bond2 * bond2};
  SymMat2 diff = L - ss;

  Mat2 SR;
  SR(0,0) = s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2];
  SR(0,1) = s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2];
  SR(1,0) = s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2];
  SR(1,1) = s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2];

  if (rmass) {
    invmass0 = dtfsq / rmass[i0];
    invmass01 = invmass0 + dtfsq / rmass[i1];
    invmass02 = invmass0 + dtfsq / rmass[i2];
  } else {
    invmass0 = dtfsq / mass[type[i0]];
    invmass01 = invmass0 + dtfsq / mass[type[i1]];
    invmass02 = invmass0 + dtfsq / mass[type[i2]];
  }

  SymMat2 M = inv_sym({invmass01, invmass0, invmass02});

  SymMat2 D = sandwich(M, diff);

  Mat2 K = M * SR;

  SymMat2 rh = inv_sym(rr);

  Mat2 chi = K * rh;
  SymMat2 chiKT = mat_mul_tosym(chi, transpose(K));
  SymMat2 sigma = chiKT + D;

  Mat2 sc = chol_upper(sigma);

  Mat2 rc = inv_chol_lower(rr);

  Mat2 phiC = sc * rc;

  double phiS11 = sc(0,1)*rc(0,0) - sc(0,0)*rc(1,0);
  double phiS12 = -sc(0,0)*rc(1,1);
  double phiS21 = sc(1,1)*rc(0,0);

  double skewC = skew(phiC);
  double skewChi = skew(chi);
  double skewS = phiS12 - phiS21;

  double Asq = skewC * skewC + skewS * skewS;
  double sinp = sqrt(Asq - skewChi * skewChi);
  double sskew = -(skewS * skewChi + skewC * sinp) / Asq;
  double cskew = (skewS * sinp - skewChi * skewC) / Asq;

  double lamda01 = chi(0,0) + cskew*phiC(0,0) + sskew*phiS11;
  double lamda02 = chi(1,1) + cskew*phiC(1,1);
  double lamda12 = chi(0,1) + cskew*phiC(0,1) + sskew*phiS12;

  if (i0 < nlocal) {
    f[i0][0] -= (lamda01+lamda12)*r01[0] + (lamda02+lamda12)*r02[0];
    f[i0][1] -= (lamda01+lamda12)*r01[1] + (lamda02+lamda12)*r02[1];
    f[i0][2] -= (lamda01+lamda12)*r01[2] + (lamda02+lamda12)*r02[2];
  }

  if (i1 < nlocal) {
    f[i1][0] += lamda01*r01[0] + lamda12*r02[0];
    f[i1][1] += lamda01*r01[1] + lamda12*r02[1];
    f[i1][2] += lamda01*r01[2] + lamda12*r02[2];
  }

  if (i2 < nlocal) {
    f[i2][0] += lamda02*r02[0] + lamda12*r01[0];
    f[i2][1] += lamda02*r02[1] + lamda12*r01[1];
    f[i2][2] += lamda02*r02[2] + lamda12*r01[2];
  }

  if (evflag) {
    int count = 0;
    if (i0 < nlocal) atomlist[count++] = i0;
    if (i1 < nlocal) atomlist[count++] = i1;
    if (i2 < nlocal) atomlist[count++] = i2;

    double r12[3];
    r12[0] = r02[0] - r01[0];
    r12[1] = r02[1] - r01[1];
    r12[2] = r02[2] - r01[2];

    double lamda01_shake = - lamda01 - lamda12;
    double lamda02_shake = - lamda02 - lamda12;
    double lamda12_shake = lamda12;

    v[0] = lamda01_shake*r01[0]*r01[0] + lamda02_shake*r02[0]*r02[0] + lamda12_shake*r12[0]*r12[0];
    v[1] = lamda01_shake*r01[1]*r01[1] + lamda02_shake*r02[1]*r02[1] + lamda12_shake*r12[1]*r12[1];
    v[2] = lamda01_shake*r01[2]*r01[2] + lamda02_shake*r02[2]*r02[2] + lamda12_shake*r12[2]*r12[2];
    v[3] = lamda01_shake*r01[0]*r01[1] + lamda02_shake*r02[0]*r02[1] + lamda12_shake*r12[0]*r12[1];
    v[4] = lamda01_shake*r01[0]*r01[2] + lamda02_shake*r02[0]*r02[2] + lamda12_shake*r12[0]*r12[2];
    v[5] = lamda01_shake*r01[1]*r01[2] + lamda02_shake*r02[1]*r02[2] + lamda12_shake*r12[1]*r12[2];

    double fpairlist[] = {lamda01_shake, lamda02_shake, lamda12_shake};
    double dellist[][3]  = {{r01[0], r01[1], r01[2]},
                            {r02[0], r02[1], r02[2]},
                            {r12[0], r12[1], r12[2]}};
    int pairlist[][2] = {{i0,i1}, {i0,i2}, {i1,i2}};
    v_tally(count,atomlist,3.0,v,nlocal,3,pairlist,fpairlist,dellist);
  }
}