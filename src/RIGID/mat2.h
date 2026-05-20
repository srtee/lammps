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

struct SymMat2 {
  double d00, d01, d11;
};

struct Mat2 {
  double d[2][2];

  double &operator()(int i, int j) { return d[i][j]; }
  double operator()(int i, int j) const { return d[i][j]; }
};

inline SymMat2 sym_dot(const double r1[3], const double r2[3])
{
  return {r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2],
          r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2],
          r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]};
}

inline SymMat2 sym_outer_diff(const SymMat2 &L, const SymMat2 &S)
{
  return {L.d00 - S.d00, L.d01 - S.d01, L.d11 - S.d11};
}

inline SymMat2 inv_sym(const SymMat2 &A)
{
  double det = A.d00 * A.d11 - A.d01 * A.d01;
  return {A.d11 / det, -A.d01 / det, A.d00 / det};
}

inline SymMat2 sandwich(const SymMat2 &M, const SymMat2 &A)
{
  double MA00 = M.d00 * A.d00 + M.d01 * A.d01;
  double MA01 = M.d00 * A.d01 + M.d01 * A.d11;
  double MA10 = M.d01 * A.d00 + M.d11 * A.d01;
  double MA11 = M.d01 * A.d01 + M.d11 * A.d11;
  return {MA00 * M.d00 + MA01 * M.d01,
          MA00 * M.d01 + MA01 * M.d11,
          MA10 * M.d01 + MA11 * M.d11 };
}

inline Mat2 sym_mul(const SymMat2 &A, const Mat2 &B)
{
  Mat2 R;
  R(0, 0) = A.d00 * B(0, 0) + A.d01 * B(1, 0);
  R(0, 1) = A.d00 * B(0, 1) + A.d01 * B(1, 1);
  R(1, 0) = A.d01 * B(0, 0) + A.d11 * B(1, 0);
  R(1, 1) = A.d01 * B(0, 1) + A.d11 * B(1, 1);
  return R;
}

inline Mat2 mul_sym(const Mat2 &A, const SymMat2 &B)
{
  Mat2 R;
  R(0, 0) = A(0, 0) * B.d00 + A(0, 1) * B.d01;
  R(0, 1) = A(0, 0) * B.d01 + A(0, 1) * B.d11;
  R(1, 0) = A(1, 0) * B.d00 + A(1, 1) * B.d01;
  R(1, 1) = A(1, 0) * B.d01 + A(1, 1) * B.d11;
  return R;
}

inline SymMat2 sym_minus(const SymMat2 &A, const SymMat2 &B)
{
  return {A.d00 - B.d00, A.d01 - B.d01, A.d11 - B.d11};
}

inline SymMat2 sym_plus(const SymMat2 &A, const SymMat2 &B)
{
  return {A.d00 + B.d00, A.d01 + B.d01, A.d11 + B.d11};
}

inline SymMat2 chi_KT_minus_D(const Mat2 &chi, const Mat2 &K, const SymMat2 &D)
{
  return {chi(0, 0) * K(0, 0) + chi(0, 1) * K(0, 1) - D.d00,
          chi(0, 0) * K(1, 0) + chi(0, 1) * K(1, 1) - D.d01,
          chi(1, 0) * K(1, 0) + chi(1, 1) * K(1, 1) - D.d11};
}

inline Mat2 mat_mul(const Mat2 &A, const Mat2 &B)
{
  Mat2 R;
  R(0, 0) = A(0, 0) * B(0, 0) + A(0, 1) * B(1, 0);
  R(0, 1) = A(0, 0) * B(0, 1) + A(0, 1) * B(1, 1);
  R(1, 0) = A(1, 0) * B(0, 0) + A(1, 1) * B(1, 0);
  R(1, 1) = A(1, 0) * B(0, 1) + A(1, 1) * B(1, 1);
  return R;
}

inline SymMat2 mat_mul_tosym(const Mat2 &A, const Mat2 &B)
{
  SymMat2 R;
  R.d00 = A(0, 0) * B(0, 0) + A(0, 1) * B(1, 0);
  R.d01 = A(0, 0) * B(0, 1) + A(0, 1) * B(1, 1);
  R.d11 = A(1, 0) * B(0, 1) + A(1, 1) * B(1, 1);
  return R;
}

inline Mat2 chol_upper(const SymMat2 &A)
{
  double s22 = sqrt(A.d11);
  double s12 = A.d01 / s22;
  double sq00 = sqrt(A.d00);
  double s11 = sqrt((sq00 - s12) * (sq00 + s12));
  Mat2 R;
  R(0, 0) = s11; R(0, 1) = s12;
  R(1, 0) = 0.0; R(1, 1) = s22;
  return R;
}

inline Mat2 inv_chol_lower(const SymMat2 &A)
{
  double l11 = sqrt(A.d00);
  double l21 = A.d01 / l11;
  double sq11 = sqrt(A.d11);
  double l22 = sqrt((sq11 + l21)*(sq11 - l21));
  double inv11 = 1.0 / l11;
  double inv22 = 1.0 / l22;
  double inv21 = -l21 * inv11 * inv22;
  Mat2 R;
  R(0, 0) = inv11; R(0, 1) = 0.0;
  R(1, 0) = inv21; R(1, 1) = inv22;
  return R;
}

inline Mat2 transpose(const Mat2 &A)
{
  Mat2 R;
  R(0, 0) = A(0, 0); R(0, 1) = A(1, 0);
  R(1, 0) = A(0, 1); R(1, 1) = A(1, 1);
  return R;
}

inline double skew(const Mat2 &A) { return A(0, 1) - A(1, 0); }

#endif
