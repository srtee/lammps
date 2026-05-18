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

#include "comm.h"

using namespace LAMMPS_NS;

FixRigs::FixRigs(LAMMPS *lmp, int narg, char **arg):
    FixShake(lmp, narg, arg) { rigsflag = 1; }

FixRigs::~FixRigs() {}

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
  double bond12 = angle_distance[shake_type[m][2]]; // b1b2cos

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

  double r11 = r01[0]*r01[0] + r01[1]*r01[1] + r01[2]*r01[2];
  double r22 = r02[0]*r02[0] + r02[1]*r02[1] + r02[2]*r02[2];
  double r12 = r01[0]*r02[0] + r01[1]*r02[1] + r01[2]*r02[2];
  double s11 = s01[0]*s01[0] + s01[1]*s01[1] + s01[2]*s01[2];
  double s22 = s02[0]*s02[0] + s02[1]*s02[1] + s02[2]*s02[2];
  double s12 = s01[0]*s02[0] + s01[1]*s02[1] + s01[2]*s02[2];

  double sr11 = s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2];
  double sr12 = s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2];
  double sr21 = s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2];
  double sr22 = s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2];

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

  // M = (mu01 mu0; mu0 mu02)^(-1)

  double detM = invmass01 * invmass02 - (invmass0 * invmass0);
  double M11 = invmass02 / detM;
  double M12 = -invmass0 / detM;
  double M22 = invmass01 / detM;

  // D = M (L - S^T S) M (symm)
  double diff11 = bond1*bond1 - s11;
  double diff22 = bond2*bond2 - s22;
  double diff12 = bond12 - s12;
  if (comm->me == 0) {
    printf("diff11 = %20.14f \n", diff11); 
    printf("diff12 = %20.14f \n", diff12); 
    printf("diff22 = %20.14f \n", diff22); 
  }

  double mD11 = diff11*M11 + diff12*M12;
  double mD12 = diff11*M12 + diff12*M22;
  double mD21 = diff12*M11 + diff22*M12;
  double mD22 = diff12*M12 + diff22*M12;

  double D11 = M11*mD11 + M12*mD12;
  double D12 = M11*mD12 + M12*mD22;
  double D22 = M12*mD12 + M22*mD22;
/*
  double D11 = M11*(diff11*M11 + 2*diff12*M12) + M12*diff22*M12;
  double D22 = M22*(diff22*M22 + 2*diff12*M12) + M12*diff11*M12;
  double D12 = M11*(diff11*M12 + diff12*M22) + M12*(diff12*M12 + diff22*M22);
*/
  // K = M (S^T R)
  double K11 = M11*sr11 + M12*sr21;
  double K12 = M11*sr12 + M12*sr22;
  double K21 = M12*sr11 + M22*sr21;
  double K22 = M12*sr12 + M22*sr22;

  // rh = (R^T R)^(-1) (symm)
  double detR = r11*r22 - (r12*r12);
  double rh11 = r22/detR;
  double rh12 = -r12/detR;
  double rh22 = r11/detR;

  // chi = K rh
  double chi11 = K11*rh11 + K12*rh12;
  double chi12 = K11*rh12 + K12*rh22;
  double chi21 = K21*rh11 + K22*rh12;
  double chi22 = K21*rh12 + K22*rh22;
  if (comm->me == 0) {
    printf("chi11 = %8.4f \n", chi11); 
    printf("chi12 = %8.4f \n", chi12); 
    printf("chi21 = %8.4f \n", chi21); 
    printf("chi22 = %8.4f \n", chi22); 
  }
  
  // sigma = chi K^T - D (symm)
  double sig11 = chi11*K11 + chi12*K12 - D11;
  double sig12 = chi11*K21 + chi12*K22 - D12;
  double sig21 = chi21*K11 + chi22*K12 - D12;
  double sig22 = chi21*K21 + chi22*K22 - D22;

  // sc = upper Chol of sigma
  double sc22 = sqrt(sig22);
  double sc12 = sig12/sc22;
  double sc11 = sqrt(sig11 - sc12*sc12);
  
  // rc = inverse of lower Chol of R^T R
  double rc11 = 1/sqrt(r11);
  double rc21 = r12/sqrt(r11);
  double rc22 = 1/sqrt(r22 - rc21*rc21);
  rc21 *= -1/(rc11*rc22);
  
  // phiC = sc x rc
  double phiC11 = sc11*rc11 + sc12*rc21;
  double phiC12 = sc12*rc22;
  double phiC21 = sc22*rc21;
  double phiC22 = sc22*rc22;
  if (comm->me == 0) {
  printf("phiC11 = %8.4f \n", phiC11); 
  printf("phiC12 = %8.4f \n", phiC12); 
  printf("phiC21 = %8.4f \n", phiC21); 
  printf("phiC22 = %8.4f \n", phiC22); 
  }

  // phiS = sc x [0, -1; 1, 0] x rc
  double phiS11 = sc12*rc11 - sc11*rc21;
  double phiS12 = -sc11*rc22;
  double phiS21 = sc12*rc21;
  if (comm->me == 0) {
  printf("phiS11 = %8.4f \n", phiS11); 
  printf("phiS12 = %8.4f \n", phiS12); 
  printf("phiS21 = %8.4f \n", phiS21); 
  }

  // versine solve
  double skewC = phiC12 - phiC21;
  double skewChi = (chi12 - chi21);
  double skewS = phiS12 - phiS21;
  
  // skewChi - cos skewC - sin skewS = 0
  // cos th sin p + sin th cos p = skewChi/A, A = sqrt(skewC*2 + skewS*2)
  // sin (th + p) = skCh/A, sin p = skewC/A, cos p = skewS/A
  // sin (th) = skCh/A cos p - sqrt(A*A - skCh*skCh)/A sin p
  // = (skewS skCh - skewC sqrt(A*A - skCh*skCh)) / A*A
  double Asq = skewC*skewC + skewS*skewS;
  double sinp = sqrt(Asq - skewChi*skewChi);
  double sskew = -(skewS*skewChi - skewC*sinp)/Asq;
  double cskew = sqrt(1-sskew*sskew);
  if (comm->me == 0) {
  printf("skewChi = %8.4f \n", skewChi); 
  printf("skewC = %8.4f \n", skewC); 
  printf("skewS = %8.4f \n", skewS); 
  printf("cskew = %8.4f \n", cskew); 
  printf("sskew = %8.4f \n", sskew); 
  }

  // and finally!!
  double lamda01 = chi11 - (cskew)*phiC11 - sskew*phiS11;
  double lamda02 = chi22 - (cskew)*phiC22;
  double lamda12 = chi12 - (cskew)*phiC12 - sskew*phiS12; 
  if (comm->me == 0) {
  printf("lamda01 = %8.4f \n", lamda01); 
  printf("lamda02 = %8.4f \n", lamda02); 
  printf("lamda12 = %8.4f \n\n", lamda12); 
  }

  // avoid dumb stuff if Gamma is too small
  if (fabs(diff11) < tolerance &&
      fabs(diff22) < tolerance &&
      fabs(diff12) < tolerance) return;

  // update forces if atom is owned by this processor

  lamda01 = 0.5*lamda01/dtfsq;
  lamda02 = 0.5*lamda02/dtfsq;
  lamda12 = 0.5*lamda12/dtfsq;

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
/* TO FIX
  if (evflag) {
    int count = 0;
    if (i0 < nlocal) atomlist[count++] = i0;
    if (i1 < nlocal) atomlist[count++] = i1;
    if (i2 < nlocal) atomlist[count++] = i2;

    v[0] = lamda01*r01[0]*r01[0]+lamda02*r02[0]*r02[0]+lamda12*r12[0]*r12[0];
    v[1] = lamda01*r01[1]*r01[1]+lamda02*r02[1]*r02[1]+lamda12*r12[1]*r12[1];
    v[2] = lamda01*r01[2]*r01[2]+lamda02*r02[2]*r02[2]+lamda12*r12[2]*r12[2];
    v[3] = lamda01*r01[0]*r01[1]+lamda02*r02[0]*r02[1]+lamda12*r12[0]*r12[1];
    v[4] = lamda01*r01[0]*r01[2]+lamda02*r02[0]*r02[2]+lamda12*r12[0]*r12[2];
    v[5] = lamda01*r01[1]*r01[2]+lamda02*r02[1]*r02[2]+lamda12*r12[1]*r12[2];

    double fpairlist[] = {lamda01, lamda02, lamda12};
    double dellist[][3]  = {{r01[0], r01[1], r01[2]},
                            {r02[0], r02[1], r02[2]},
                            {r12[0], r12[1], r12[2]}};
    int pairlist[][2] = {{i0,i1}, {i0,i2}, {i1,i2}};
    v_tally(count,atomlist,3.0,v,nlocal,3,pairlist,fpairlist,dellist);
  }
*/
}
