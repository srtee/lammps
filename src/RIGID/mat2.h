/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Contributing authors: Shern Tee (GU), Stephen Sanderson (UQ)
 ------------------------------------------------------------------------- */

#ifndef LMP_MAT2_H
#define LMP_MAT2_H

#include <cmath>

namespace RigsMath {

struct LTMat2 {
  double l00, l10, l11;
};

struct UTMat2 {
  double u00, u01, u11;
};

struct SymMat2 {
  double d00, d01, d11;

  SymMat2 operator+(const SymMat2 &B) const { // used, make in-place?
    return {d00 + B.d00, d01 + B.d01, d11 + B.d11};
  }
  SymMat2 operator-(const SymMat2 &B) const { // used
    return {d00 - B.d00, d01 - B.d01, d11 - B.d11};
  }
};

//inline void lslt_mul(SymMat2 &S, const LTMat2 &L)
//{
//  // S <- S * L^T
//  double t01 = S.d01;
//  S.d01 = S.d00 * L.l10 + S.d01 * L.l11;
//  S.d00 *= L.l00;
//  S.d11 = t01 * L.l10 + S.d11 * L.l11;
//  // <- L * S * L^T
//  S.d11 = S.d01 * L.l10 + S.d11 * L.l11;
//  S.d00 *= L.l00;
//  S.d01 *= L.l00;
//}

struct Mat2 {
  double d[2][2];

  double &operator()(int i, int j) { return d[i][j]; }
  double operator()(int i, int j) const { return d[i][j]; }

  Mat2 operator*(const Mat2 &B) const {
    Mat2 R;
    R(0, 0) = d[0][0] * B(0, 0) + d[0][1] * B(1, 0);
    R(0, 1) = d[0][0] * B(0, 1) + d[0][1] * B(1, 1);
    R(1, 0) = d[1][0] * B(0, 0) + d[1][1] * B(1, 0);
    R(1, 1) = d[1][0] * B(0, 1) + d[1][1] * B(1, 1);
    return R;
  }
  
  Mat2 operator*(const LTMat2 &L) const {
    Mat2 R;
    R(0, 0) = d[0][0] * L.l00 + d[0][1] * L.l10;
    R(0, 1) = d[0][1] * L.l11;
    R(1, 0) = d[1][0] * L.l00 + d[1][1] * L.l10;
    R(1, 1) = d[1][1] * L.l11;
    return R;
  }
};

inline void ut_mul(const UTMat2 &U, Mat2 &M) // used
{
  M(1, 0) = U.u11 * M(1, 0) + U.u01 * M(0, 0);
  M(1, 1) = U.u11 * M(1, 1) + U.u01 * M(0, 1);
  M(0, 0) *= U.u00;
  M(0, 1) *= U.u00;
}

inline void u_mul(const UTMat2 &U, Mat2 &M) // used
{
  M(0, 0) = U.u00 * M(0, 0) + U.u01 * M(1, 0);
  M(0, 1) = U.u00 * M(0, 1) + U.u01 * M(1, 1);
  M(1, 0) *= U.u11;
  M(1, 1) *= U.u11;
}

// r1·r1, r1·r2, r2·r2
inline SymMat2 sym_dot(const double r1[3], const double r2[3]) // used
{
  return {r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2],
          r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2],
          r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]};
}

inline SymMat2 mtm(const Mat2 &M) // used
{
  return {M(0, 0) * M(0, 0) + M(1, 0) * M(1, 0),
	  M(0, 0) * M(0, 1) + M(1, 0) * M(1, 1),
	  M(0, 1) * M(0, 1) + M(1, 1) * M(1, 1)};
}

inline double skew(const Mat2 &A) { return A(0, 1) - A(1, 0); }

inline Mat2 operator*(const UTMat2 &U, const Mat2 &B)
{
  Mat2 R;
  R(0, 0) = U.u00 * B(0, 0) + U.u01 * B(1, 0);
  R(0, 1) = U.u00 * B(0, 1) + U.u01 * B(1, 1);
  R(1, 0) = U.u11 * B(1, 0);
  R(1, 1) = U.u11 * B(1, 1);
  return R;
}

inline Mat2 operator*(const UTMat2 &U, const LTMat2 &L)
{
  Mat2 R;
  R(0, 0) = U.u00 * L.l00 + U.u01 * L.l10;
  R(0, 1) = U.u01 * L.l11;
  R(1, 0) = U.u11 * L.l10;
  R(1, 1) = U.u11 * L.l11;
  return R;
}

inline UTMat2 inv_chol_upper(const SymMat2 &A) // used
{
  double u00 = sqrt(A.d00);
  double u01 = A.d01 / u00;
  double u11 = sqrt(A.d11 - u01 * u01);
  return {1.0 / u00, -u01 / (u00 * u11), 1.0 / u11};
}

inline LTMat2 chol_lower(const SymMat2 &A) // used
{
  double l11 = sqrt(A.d11);
  double l10 = A.d01 / l11;
  double l00 = sqrt(A.d00 - l10 * l10);
  return {l00, l10, l11};
}

struct DChol2 {
  double d0, d1, m01;
};

inline DChol2 inv_dchol(const SymMat2 &A)
{
  double m01 = A.d01 / A.d00;
  double d1 = A.d11 - m01 * A.d01;
  return {1/A.d00, 1/d1, -m01};
}

inline void lslt_mul(SymMat2 &S, const DChol2 &L)
{
  // S <- S * L^T
  S.d11 += S.d01 * L.m01;
  S.d01 += S.d00 * L.m01;
  // <- L * S * L^T
  S.d11 += S.d01 * L.m01;
}

inline LTMat2 mul_dl(const LTMat2 &A, const DChol2 &L)
{
  double l00 = A.l00 * L.d0;
  double l11 = A.l11 * L.d1;
  double l10 = A.l10 * L.d0 + l11 * L.m01;
  return {l00, l10, l11};
}


inline LTMat2 operator*(const LTMat2 &A, const LTMat2 &B)
{
  return {A.l00 * B.l00,
          A.l10 * B.l00 + A.l11 * B.l10,
          A.l11 * B.l11};
}

inline void mul_ltdl(Mat2 &M, const DChol2 &L) // used
{
  M(0, 1) += M(0, 0) * L.m01;
  M(1, 1) += M(1, 0) * L.m01;
  M(0, 0) *= L.d0;
  M(1, 0) *= L.d0;
  M(0, 1) *= L.d1;
  M(1, 1) *= L.d1;
  M(0, 0) += M(0, 1) * L.m01;
  M(1, 0) += M(1, 1) * L.m01;
}

}

#endif
