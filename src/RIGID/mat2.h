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

#include "vec3.h"

namespace RigsMath {

struct LTMat2 {
  double l00, l10, l11;
};

struct UTMat2 {
  double u00, u01, u11;
};

struct SymMat2 {
  double d00, d01, d11;

  SymMat2 operator-(const SymMat2 &B) const {
    return {d00 - B.d00, d01 - B.d01, d11 - B.d11};
  }
};

struct Mat2 {
  double d[2][2];

  Mat2() { d[0][0] = d[0][1] = d[1][0] = d[1][1] = 0.0; }
  Mat2(double a00, double a01, double a10, double a11)
    : d{{a00, a01}, {a10, a11}} {}

  double &operator()(int i, int j) { return d[i][j]; }
  double operator()(int i, int j) const { return d[i][j]; }

  Mat2 operator*(const LTMat2 &L) const {
    return Mat2(d[0][0] * L.l00 + d[0][1] * L.l10,
                d[0][1] * L.l11,
                d[1][0] * L.l00 + d[1][1] * L.l10,
                d[1][1] * L.l11);
  }
};

inline Mat2 operator*(const SymMat2 &A, const Mat2 &B)
{
  return Mat2(A.d00 * B(0, 0) + A.d01 * B(1, 0),
              A.d00 * B(0, 1) + A.d01 * B(1, 1),
              A.d01 * B(0, 0) + A.d11 * B(1, 0),
              A.d01 * B(0, 1) + A.d11 * B(1, 1));
}

inline Mat2 operator*(const LTMat2 &L, const UTMat2 &U)
{
  return Mat2(L.l00 * U.u00,
              L.l00 * U.u01,
              L.l10 * U.u00,
              L.l10 * U.u01 + L.l11 * U.u11);
}

inline Mat2 operator*(const LTMat2 &L, const Mat2 &B)
{
  return Mat2(L.l00 * B(0, 0),
              L.l00 * B(0, 1),
              L.l10 * B(0, 0) + L.l11 * B(1, 0),
              L.l10 * B(0, 1) + L.l11 * B(1, 1));
}

inline Mat2 operator*(const Mat2 &A, const UTMat2 &B)
{
  return Mat2(A(0, 0) * B.u00 + A(0, 1) * B.u01,
              A(0, 1) * B.u11,
              A(1, 0) * B.u00 + A(1, 1) * B.u01,
              A(1, 1) * B.u11);
}

inline void ut_mul(const UTMat2 &U, Mat2 &M)
{
  M(1, 0) = U.u11 * M(1, 0) + U.u01 * M(0, 0);
  M(1, 1) = U.u11 * M(1, 1) + U.u01 * M(0, 1);
  M(0, 0) *= U.u00;
  M(0, 1) *= U.u00;
}

inline void u_mul(const UTMat2 &U, Mat2 &M)
{
  M(0, 0) = U.u00 * M(0, 0) + U.u01 * M(1, 0);
  M(0, 1) = U.u00 * M(0, 1) + U.u01 * M(1, 1);
  M(1, 0) *= U.u11;
  M(1, 1) *= U.u11;
}

inline SymMat2 sym_dot(const Vec3 &r1, const Vec3 &r2)
{
  return {dot(r1, r1), dot(r1, r2), dot(r2, r2)};
}

inline double skew(const Mat2 &A) { return A(0, 1) - A(1, 0); }

inline double trace_of_product(const Mat2 &A, const Mat2 &B)
{
  return A(0, 0) * B(0, 0) + A(1, 1) * B(1, 1)
       + A(0, 1) * B(0, 1) + A(1, 0) * B(1, 0);
}

inline LTMat2 chol_lower(const SymMat2 &A)
{
  double l00 = sqrt(A.d00);
  double l10 = A.d01 / l00;
  double l11 = sqrt(A.d11 - l10 * l10);
  return {l00, l10, l11};
}

inline UTMat2 chol_upper(const SymMat2 &A)
{
  LTMat2 L = chol_lower(A);
  return {L.l00, L.l10, L.l11};
}

struct LDU2 {
  double d0, d1, u01;
};

inline LDU2 ldu2(const SymMat2 &A)
{
  double u01 = A.d01 / A.d00;
  double d1 = A.d11 - u01 * A.d01;
  return {A.d00, d1, u01};
}

inline void UsL(SymMat2 &S, const LDU2 &U)
{
  S.d00 += S.d01 * U.u01;
  S.d01 += S.d11 * U.u01;
  S.d00 += S.d01 * U.u01;
}

inline UTMat2 mul_du(const UTMat2 &A, const LDU2 &U)
{
  double u00 = A.u00 * U.d0;
  double u11 = A.u11 * U.d1;
  double u01 = A.u01 * U.d1 + u00 * U.u01;
  return {u00, u01, u11};
}

inline Mat2 operator*(const Mat2 &M, const LDU2 &ldu)
{
  Mat2 R = M;
  R(0, 0) += R(0, 1) * ldu.u01;
  R(1, 0) += R(1, 1) * ldu.u01;
  R(0, 0) *= ldu.d0; R(0, 1) *= ldu.d1;
  R(1, 0) *= ldu.d0; R(1, 1) *= ldu.d1;
  R(0, 1) += R(0, 0) * ldu.u01;
  R(1, 1) += R(1, 0) * ldu.u01;
  return R;
}

inline SymMat2 mass_matrix3(double *m)
{
  double mutot = 1. / (m[0] + m[1] + m[2]);
  double u1 = m[1] * mutot;
  double u2 = m[2] * mutot;
  return SymMat2 { m[1] * (1. - u1), -m[1] * u2,
                   m[2] * (1. - u2)};
}

}
#endif