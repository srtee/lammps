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

#include "atom.h"
#include "error.h"
#include "mat2.h"
#include "mat3.h"
#include "memory.h"

using namespace LAMMPS_NS;
using namespace RigsMath;

void FixRigs::prebuild_matrices()
{
  if (nlist > rigs_maxlist) {
    memory->destroy(rigs_L);
    memory->destroy(rigs_lm);
    memory->destroy(rigs_R);
    rigs_maxlist = nlist;
    memory->create(rigs_L, rigs_maxlist, 6, "rigs:rigs_L");
    memory->create(rigs_lm, rigs_maxlist, 6, "rigs:rigs_lm");
    memory->create(rigs_R, rigs_maxlist, 9, "rigs:rigs_R");
  }

  for (int ilist = 0; ilist < nlist; ilist++) {
    int m = list[ilist];
    double *L = rigs_L[ilist];
    double *lm = rigs_lm[ilist];

    if (shake_flag[m] == 1) {
      double bond1 = bond_distance[shake_type[m][0]];
      double bond2 = bond_distance[shake_type[m][1]];
      double bond12 = rigs_angle[shake_type[m][2]];

      L[0] = bond1 * bond1;
      L[1] = bond12;
      L[2] = bond2 * bond2;

      int i0 = closest_list[ilist][0];
      int i1 = closest_list[ilist][1];
      int i2 = closest_list[ilist][2];
      double invmass0, invmass01, invmass02;
      if (rmass) {
        invmass0 = 1.0 / rmass[i0];
        invmass01 = invmass0 + 1.0 / rmass[i1];
        invmass02 = invmass0 + 1.0 / rmass[i2];
      } else {
        invmass0 = 1.0 / mass[type[i0]];
        invmass01 = invmass0 + 1.0 / mass[type[i1]];
        invmass02 = invmass0 + 1.0 / mass[type[i2]];
      }
      DChol2 dc = inv_dchol(SymMat2{invmass01, invmass0, invmass02});
      lm[0] = dc.d0;
      lm[1] = dc.d1;
      lm[2] = dc.m01;

      chol_frame2(L, rigs_R[ilist]);

    } else if (shake_flag[m] == 5) {
      double bond1 = bond_distance[shake_type[m][0]];
      double bond2 = bond_distance[shake_type[m][1]];
      double bond3 = bond_distance[shake_type[m][2]];

      L[0] = bond1 * bond1;
      L[1] = rigs_angle[rigs_type[m][0]];
      L[2] = rigs_angle[rigs_type[m][1]];
      L[3] = bond2 * bond2;
      L[4] = rigs_angle[rigs_type[m][2]];
      L[5] = bond3 * bond3;

      int i0 = closest_list[ilist][0];
      int i1 = closest_list[ilist][1];
      int i2 = closest_list[ilist][2];
      int i3 = closest_list[ilist][3];
      double mu0, mu01, mu02, mu03;
      if (rmass) {
        mu0 = 1.0 / rmass[i0];
        mu01 = mu0 + 1.0 / rmass[i1];
        mu02 = mu0 + 1.0 / rmass[i2];
        mu03 = mu0 + 1.0 / rmass[i3];
      } else {
        mu0 = 1.0 / mass[type[i0]];
        mu01 = mu0 + 1.0 / mass[type[i1]];
        mu02 = mu0 + 1.0 / mass[type[i2]];
        mu03 = mu0 + 1.0 / mass[type[i3]];
      }
      DChol3 dc = inv_dchol(SymMat3{mu01, mu0, mu0, mu02, mu0, mu03});
      lm[0] = dc.d0;
      lm[1] = dc.d1;
      lm[2] = dc.d2;
      lm[3] = dc.m01;
      lm[4] = dc.m02;
      lm[5] = dc.m12;

      chol_frame3(L, rigs_R[ilist]);

    } else if (shake_flag[m] == 6) {
      double bond1 = bond_distance[shake_type[m][0]];
      double bond2 = bond_distance[shake_type[m][1]];
      double bond3 = bond_distance[shake_type[m][2]];

      L[0] = bond1 * bond1;
      L[1] = bond1 * bond2;
      L[2] = bond1 * bond3;
      L[3] = bond2 * bond2;
      L[4] = bond2 * bond3;
      L[5] = bond3 * bond3;

      int i0 = closest_list[ilist][0];
      int i1 = closest_list[ilist][1];
      int i2 = closest_list[ilist][2];
      int i3 = closest_list[ilist][3];
      double mu0, mu2, mu10, mu02, mu23;
      if (rmass) {
        mu0 = 1.0 / rmass[i0];
        mu2 = 1.0 / rmass[i2];
        mu10 = 1.0 / rmass[i1] + mu0;
        mu02 = mu0 + mu2;
        mu23 = mu2 + 1.0 / rmass[i3];
      } else {
        mu0 = 1.0 / mass[type[i0]];
        mu2 = 1.0 / mass[type[i2]];
        mu10 = 1.0 / mass[type[i1]] + mu0;
        mu02 = mu0 + mu2;
        mu23 = mu2 + 1.0 / mass[type[i3]];
      }
      DChol3 dc = inv_dchol(SymMat3{mu10, mu0, 0, mu02, mu2, mu23});
      lm[0] = dc.d0;
      lm[1] = dc.d1;
      lm[2] = dc.d2;
      lm[3] = dc.m01;
      lm[4] = dc.m02;
      lm[5] = dc.m12;

      chol_frame3(L, rigs_R[ilist]);
    }
  }
}

void FixRigs::shake4(int ilist)
{
  int m = list[ilist];
  if (shake_flag[m] == 5) {
    shake4improper(ilist);
  } else if (shake_flag[m] == 6) {
    error->one(FLERR,"RIGS dihedral constraint solver not yet implemented");
  } else {
    FixShake::shake4(ilist);
  }
}

/* ----------------------------------------------------------------------
   calculate RIGS constraint forces for size 3 cluster = two bonds + angle
   ------------------------------------------------------------------------- */

void FixRigs::shake3angle(int ilist)
{
  int atomlist[3];
  double v[6];

  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];

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

  SymMat2 L = {rigs_L[ilist][0], rigs_L[ilist][1], rigs_L[ilist][2]};
  SymMat2 diff = L - ss;

  Mat2 chi;
  chi(0, 0) = s01[0] * r01[0] + s01[1] * r01[1] + s01[2] * r01[2];
  chi(1, 0) = s01[0] * r02[0] + s01[1] * r02[1] + s01[2] * r02[2];
  chi(0, 1) = s02[0] * r01[0] + s02[1] * r01[1] + s02[2] * r01[2];
  chi(1, 1) = s02[0] * r02[0] + s02[1] * r02[1] + s02[2] * r02[2];

  DChol2 lm = {rigs_lm[ilist][0], rigs_lm[ilist][1], rigs_lm[ilist][2]};

  UTMat2 rc = inv_chol_upper(rr);
  ut_mul(rc, chi);
  SymMat2 sigma = diff + mtm(chi);
  u_mul(rc, chi);

  lslt_mul(sigma, lm);
  LTMat2 sc = mul_dl(chol_lower(sigma),lm);
  mul_ltdl(chi, lm);

  Mat2 phiC = rc * sc;

  Mat2 J;
  J(0, 0) = 0.0;  J(0, 1) = -1.0;
  J(1, 0) = 1.0;  J(1, 1) = 0.0;
  Mat2 phiS = rc * (J * sc);

  double skewC = skew(phiC);
  double skewChi = skew(chi);
  double skewS = skew(phiS);

  double Asq = skewC * skewC + skewS * skewS;
  double sinp = sqrt(Asq - skewChi * skewChi);
  double sskew = -(skewS * skewChi + skewC * sinp) / Asq;
  double cskew = (skewS * sinp - skewChi * skewC) / Asq;

  double lamda01 = chi(0, 0) + cskew * phiC(0, 0) + sskew * phiS(0, 0);
  double lamda02 = chi(1, 1) + cskew * phiC(1, 1);
  double lamda12 = chi(0, 1) + cskew * phiC(0, 1) + sskew * phiS(0, 1);

  lamda01 /= dtfsq;
  lamda02 /= dtfsq;
  lamda12 /= dtfsq;

  if (i0 < nlocal) {
    f[i0][0] -= (lamda01 + lamda12) * r01[0] + (lamda02 + lamda12) * r02[0];
    f[i0][1] -= (lamda01 + lamda12) * r01[1] + (lamda02 + lamda12) * r02[1];
    f[i0][2] -= (lamda01 + lamda12) * r01[2] + (lamda02 + lamda12) * r02[2];
  }

  if (i1 < nlocal) {
    f[i1][0] += lamda01 * r01[0] + lamda12 * r02[0];
    f[i1][1] += lamda01 * r01[1] + lamda12 * r02[1];
    f[i1][2] += lamda01 * r01[2] + lamda12 * r02[2];
  }

  if (i2 < nlocal) {
    f[i2][0] += lamda02 * r02[0] + lamda12 * r01[0];
    f[i2][1] += lamda02 * r02[1] + lamda12 * r01[1];
    f[i2][2] += lamda02 * r02[2] + lamda12 * r01[2];
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

    double lamda01_shake = -lamda01 - lamda12;
    double lamda02_shake = -lamda02 - lamda12;
    double lamda12_shake = lamda12;

    v[0] = lamda01_shake * r01[0] * r01[0] + lamda02_shake * r02[0] * r02[0] + lamda12_shake * r12[0] * r12[0];
    v[1] = lamda01_shake * r01[1] * r01[1] + lamda02_shake * r02[1] * r02[1] + lamda12_shake * r12[1] * r12[1];
    v[2] = lamda01_shake * r01[2] * r01[2] + lamda02_shake * r02[2] * r02[2] + lamda12_shake * r12[2] * r12[2];
    v[3] = lamda01_shake * r01[0] * r01[1] + lamda02_shake * r02[0] * r02[1] + lamda12_shake * r12[0] * r12[1];
    v[4] = lamda01_shake * r01[0] * r01[2] + lamda02_shake * r02[0] * r02[2] + lamda12_shake * r12[0] * r12[2];
    v[5] = lamda01_shake * r01[1] * r01[2] + lamda02_shake * r02[1] * r02[2] + lamda12_shake * r12[1] * r12[2];

    double fpairlist[] = {lamda01_shake, lamda02_shake, lamda12_shake};
    double dellist[][3] = {{r01[0], r01[1], r01[2]},
                           {r02[0], r02[1], r02[2]},
                           {r12[0], r12[1], r12[2]}};
    int pairlist[][2] = {{i0, i1}, {i0, i2}, {i1, i2}};
    v_tally(count, atomlist, 3.0, v, nlocal, 3, pairlist, fpairlist, dellist);
  }
}

/* ----------------------------------------------------------------------
   calculate RIGS constraint forces for flag 5 = improper cluster
   star topology: center atom 0 + partners 1,2,3
   3x3 Gram matrices with r01, r02, r03 on diagonal
   ------------------------------------------------------------------------- */

void FixRigs::shake4improper(int ilist)
{
  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  int i3 = closest_list[ilist][3];

  double dist12 = rigs_angle_distance[rigs_type[m][0]];
  double dist13 = rigs_angle_distance[rigs_type[m][1]];
  double dist23;
  if (rigs_type[m][2] > 0)
    dist23 = rigs_angle_distance[rigs_type[m][2]];
  else
    dist23 = rigs_improper_distance[-rigs_type[m][2]];

  Mat3 R, S;
  R(0, 0) = x[i0][0] - x[i1][0];
  R(0, 1) = x[i0][1] - x[i1][1];
  R(0, 2) = x[i0][2] - x[i1][2];
  R(1, 0) = x[i0][0] - x[i2][0];
  R(1, 1) = x[i0][1] - x[i2][1];
  R(1, 2) = x[i0][2] - x[i2][2];
  R(2, 0) = x[i0][0] - x[i3][0];
  R(2, 1) = x[i0][1] - x[i3][1];
  R(2, 2) = x[i0][2] - x[i3][2];

  S(0, 0) = xshake[i0][0] - xshake[i1][0];
  S(0, 1) = xshake[i0][1] - xshake[i1][1];
  S(0, 2) = xshake[i0][2] - xshake[i1][2];
  S(1, 0) = xshake[i0][0] - xshake[i2][0];
  S(1, 1) = xshake[i0][1] - xshake[i2][1];
  S(1, 2) = xshake[i0][2] - xshake[i2][2];
  S(2, 0) = xshake[i0][0] - xshake[i3][0];
  S(2, 1) = xshake[i0][1] - xshake[i3][1];
  S(2, 2) = xshake[i0][2] - xshake[i3][2];

  SymMat3 rr = mmt(R);
  SymMat3 ss = mmt(S);

  SymMat3 L = {rigs_L[ilist][0], rigs_L[ilist][1], rigs_L[ilist][2],
               rigs_L[ilist][3], rigs_L[ilist][4], rigs_L[ilist][5]};
  SymMat3 diff = L - ss;

  DChol3 lm = {rigs_lm[ilist][0], rigs_lm[ilist][1], rigs_lm[ilist][2],
               rigs_lm[ilist][3], rigs_lm[ilist][4], rigs_lm[ilist][5]};

  Mat3 chi = mat_dot(R, S);
  UTMat3 rc = inv_chol_upper(rr);
  ut_mul(rc, chi);
  SymMat3 sigma = diff + mtm(chi);
  u_mul(rc, chi);
  lslt_mul(sigma, lm);
  LTMat3 sc = mul_dl(chol_lower(sigma),lm);
  mul_ltdl(chi, lm);

  Mat3 lamda = cayley_converge(rc, sc, chi, max_iter, tolerance);
  lamda += chi;

  Mat43 L_lam = improper_L_lambda(lamda);
  L_lam *= R;

  if (i0 < nlocal)
    for (int i = 0; i < 3; i++) f[i0][i] -= L_lam(0, i) / dtfsq;
  if (i1 < nlocal)
    for (int i = 0; i < 3; i++) f[i1][i] -= L_lam(1, i) / dtfsq;
  if (i2 < nlocal)
    for (int i = 0; i < 3; i++) f[i2][i] -= L_lam(2, i) / dtfsq;
  if (i3 < nlocal)
    for (int i = 0; i < 3; i++) f[i3][i] -= L_lam(3, i) / dtfsq;
}

/* ----------------------------------------------------------------------
   calculate RIGS constraint forces for flag 6 = dihedral cluster
   chain topology: atoms A-B-C-D stored as 1-0-2-3
   3x3 Gram matrices with r10, r02, r23 on diagonal
   ------------------------------------------------------------------------- */

void FixRigs::shake4dihedral(int ilist)
{
  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  int i3 = closest_list[ilist][3];

  double dist12 = rigs_angle_distance[rigs_type[m][0]];
  double dist23 = rigs_angle_distance[rigs_type[m][1]];
  double dist13 = rigs_dihedral_distance[rigs_type[m][2]];

  Mat3 R;
  R(0,0) = x[i1][0] - x[i0][0]; R(0,1) = x[i1][1] - x[i0][1]; R(0,2) = x[i1][2] - x[i0][2];
  R(1,0) = x[i0][0] - x[i2][0]; R(1,1) = x[i0][1] - x[i2][1]; R(1,2) = x[i0][2] - x[i2][2];
  R(2,0) = x[i2][0] - x[i3][0]; R(2,1) = x[i2][1] - x[i3][1]; R(2,2) = x[i2][2] - x[i3][2];

  Mat3 S;
  S(0,0) = xshake[i1][0] - xshake[i0][0]; S(0,1) = xshake[i1][1] - xshake[i0][1]; S(0,2) = xshake[i1][2] - xshake[i0][2];
  S(1,0) = xshake[i0][0] - xshake[i2][0]; S(1,1) = xshake[i0][1] - xshake[i2][1]; S(1,2) = xshake[i0][2] - xshake[i2][2];
  S(2,0) = xshake[i2][0] - xshake[i3][0]; S(2,1) = xshake[i2][1] - xshake[i3][1]; S(2,2) = xshake[i2][2] - xshake[i3][2];

  SymMat3 rr = mmt(R);
  SymMat3 ss = mmt(S);

  SymMat3 L = {rigs_L[ilist][0], rigs_L[ilist][1], rigs_L[ilist][2],
               rigs_L[ilist][3], rigs_L[ilist][4], rigs_L[ilist][5]};
  SymMat3 diff = L - ss;

  DChol3 lm = {rigs_lm[ilist][0], rigs_lm[ilist][1], rigs_lm[ilist][2],
               rigs_lm[ilist][3], rigs_lm[ilist][4], rigs_lm[ilist][5]};

  Mat3 chi = mat_dot(R, S);
  UTMat3 rc = inv_chol_upper(rr);
  ut_mul(rc, chi);
  SymMat3 sigma = diff + mtm(chi);
  u_mul(rc, chi);

  lslt_mul(sigma, lm);
  LTMat3 sc = mul_dl(chol_lower(sigma),lm);
  mul_ltdl(chi, lm);

  Mat3 lamda = cayley_converge(rc, sc, chi, max_iter, tolerance);
  lamda += chi;

  Mat43 L_lam = dihedral_L_lambda(lamda);
  L_lam *= R;

  if (i0 < nlocal)
    for (int i = 0; i < 3; i++) f[i0][i] += dtfsq * L_lam(0, i);
  if (i1 < nlocal)
    for (int i = 0; i < 3; i++) f[i1][i] += dtfsq * L_lam(1, i);
  if (i2 < nlocal)
    for (int i = 0; i < 3; i++) f[i2][i] += dtfsq * L_lam(2, i);
  if (i3 < nlocal)
    for (int i = 0; i < 3; i++) f[i3][i] += dtfsq * L_lam(3, i);
}
