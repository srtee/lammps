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
#include "mat2.h"

#include <cmath>

using namespace LAMMPS_NS;

FixRigs::FixRigs(LAMMPS *lmp, int narg, char **arg) :
    FixShake(lmp, narg, arg), rigs_angle(nullptr) {}

FixRigs::~FixRigs() { delete[] rigs_angle; }

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

  // local atom IDs and constraint distances

  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  double bond1 = bond_distance[shake_type[m][0]];
  double bond2 = bond_distance[shake_type[m][1]];
  double bond12 = rigs_angle[shake_type[m][2]];
  // r01,r02 = distance vec between atoms

  double r01[3];
  r01[0] = x[i0][0] - x[i1][0];
  r01[1] = x[i0][1] - x[i1][1];
  r01[2] = x[i0][2] - x[i1][2];

  double r02[3];
  r02[0] = x[i0][0] - x[i2][0];
  r02[1] = x[i0][1] - x[i2][1];
  r02[2] = x[i0][2] - x[i2][2];

  // s01,s02 = distance vec after unconstrained update

  double s01[3];
  s01[0] = xshake[i0][0] - xshake[i1][0];
  s01[1] = xshake[i0][1] - xshake[i1][1];
  s01[2] = xshake[i0][2] - xshake[i1][2];

  double s02[3];
  s02[0] = xshake[i0][0] - xshake[i2][0];
  s02[1] = xshake[i0][1] - xshake[i2][1];
  s02[2] = xshake[i0][2] - xshake[i2][2];

  // scalar distances between atoms

  SymMat2 rr = sym_dot(r01, r02);
  SymMat2 ss = sym_dot(s01, s02);
  
  // D = M (L - S^T S) M (symm)
  SymMat2 L = {bond1*bond1, bond12, bond2*bond2};
  SymMat2 diff = sym_outer_diff(L, ss);

  // avoid dumb stuff if Gamma is too small
  if (diff.d00*diff.d00 < tolerance &&
      diff.d01*diff.d01 < tolerance &&
      diff.d11*diff.d11 < tolerance) {
    FixShake::shake3angle(ilist);
    return;
  }

  Mat2 SR;
  SR(0,0) = s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2];
  SR(0,1) = s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2];
  SR(1,0) = s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2];
  SR(1,1) = s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2];

  // matrix coeffs and rhs for lamda equations

  if (rmass) {
    invmass0 = 1.0 / rmass[i0];
    invmass01 = invmass0 + 1.0 / rmass[i1];
    invmass02 = invmass0 + 1.0 / rmass[i2];
  } else {
    invmass0 = 1.0 / mass[type[i0]];
    invmass01 = invmass0 + 1.0 / mass[type[i1]];
    invmass02 = invmass0 + 1.0 / mass[type[i2]];
  }

  // M = (mu01 mu0; mu0 mu02)^(-1) (symm)
  SymMat2 M = inv_sym({invmass01, invmass0, invmass02});
  
  // D = M (L - S^T S) M (symm)
  SymMat2 D = sandwich(M, diff);

  // K = M (S^T R)
  Mat2 K = sym_mul(M, SR);

  // rh = (R^T R)^(-1) (symm)
  SymMat2 rh = inv_sym(rr);

  // chi = K * rh
  Mat2 chi = mul_sym(K, rh);

  // sigma = chi K^T - D (symm)
  SymMat2 sigma = chi_KT_minus_D(chi, K, D);
  
  // sc = upper Chol of sigma
  Mat2 sc = chol_upper(sigma);
  
  // rc = inverse of lower Chol of R^T R
  Mat2 rc = inv_chol_lower(rr);

  // phiC = sc x rc
  Mat2 phiC = mat_mul(sc, rc);

  // phiS = sc x [0, -1; 1, 0] x rc
  // J = [0,-1;1,0], so sc*J*rc has entries:
  //   row0 = sc_row0 * J * rc_col = sc00*[0,-1;1,0]_row + sc01*[0,-1;1,0]_row
  // Explicitly:
  double phiS11 = sc(0,1)*rc(0,0) - sc(0,0)*rc(1,0);
  double phiS12 = -sc(0,0)*rc(1,1);
  double phiS21 = sc(1,1)*rc(0,0);

  // cosine solve
  double skewC = skew(phiC);
  double skewChi = skew(chi);
  double skewS = phiS12 - phiS21;
  
  // skewChi - cos skewC - sin skewS = 0
  // cos th sin p + sin th cos p = skewChi/A, A = sqrt(skewC*2 + skewS*2)
  // sin (th + p) = skCh/A, sin p = skewC/A, cos p = skewS/A
  // sin (th) = skCh/A cos p - sqrt(A*A - skCh*skCh)/A sin p
  // = (skewS skCh - skewC sqrt(A*A - skCh*skCh)) / A*A
  double Asq = skewC*skewC + skewS*skewS;
  double sinp = sqrt(Asq - skewChi*skewChi);
  double sskew = (skewS*skewChi + skewC*sinp)/Asq;
  double cskew = sqrt(1-sskew*sskew);

  // and finally!!
  double lamda01 = chi(0,0) - cskew*phiC(0,0) - sskew*phiS11;
  double lamda02 = chi(1,1) - cskew*phiC(1,1);
  double lamda12 = chi(0,1) - cskew*phiC(0,1) - sskew*phiS12; 
  double lamda21 = chi(1,0) - cskew*phiC(1,0) - sskew*phiS21;
  // update forces if atom is owned by this processor

  lamda01 = lamda01/dtfsq;
  lamda02 = lamda02/dtfsq;
  lamda12 = lamda12/dtfsq;

  if (i0 < nlocal) {
    f[i0][0] += (lamda01+lamda12)*r01[0] + (lamda02+lamda12)*r02[0];
    f[i0][1] += (lamda01+lamda12)*r01[1] + (lamda02+lamda12)*r02[1];
    f[i0][2] += (lamda01+lamda12)*r01[2] + (lamda02+lamda12)*r02[2];
  }

  if (i1 < nlocal) {
    f[i1][0] -= lamda01*r01[0] + lamda12*r02[0];
    f[i1][1] -= lamda01*r01[1] + lamda12*r02[1];
    f[i1][2] -= lamda01*r01[2] + lamda12*r02[2];
  }

  if (i2 < nlocal) {
    f[i2][0] -= lamda02*r02[0] + lamda12*r01[0];
    f[i2][1] -= lamda02*r02[1] + lamda12*r01[1];
    f[i2][2] -= lamda02*r02[2] + lamda12*r01[2];
  }
  
  if (evflag) { // taken from shake3 -- TO FIX
    int count = 0;
    if (i0 < nlocal) atomlist[count++] = i0;
    if (i1 < nlocal) atomlist[count++] = i1;
    if (i2 < nlocal) atomlist[count++] = i2;

    v[0] = lamda01*r01[0]*r01[0] + lamda02*r02[0]*r02[0];
    v[1] = lamda01*r01[1]*r01[1] + lamda02*r02[1]*r02[1];
    v[2] = lamda01*r01[2]*r01[2] + lamda02*r02[2]*r02[2];
    v[3] = lamda01*r01[0]*r01[1] + lamda02*r02[0]*r02[1];
    v[4] = lamda01*r01[0]*r01[2] + lamda02*r02[0]*r02[2];
    v[5] = lamda01*r01[1]*r01[2] + lamda02*r02[1]*r02[2];

    double fpairlist[] = {lamda01, lamda02};
    double dellist[][3]  = {{r01[0], r01[1], r01[2]},
                            {r02[0], r02[1], r02[2]}};
    int pairlist[][2] = {{i0,i1}, {i0,i2}};
    v_tally(count,atomlist,3.0,v,nlocal,2,pairlist,fpairlist,dellist);
  }
}
