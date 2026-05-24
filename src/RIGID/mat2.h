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

struct SymMat2 {
  double d00, d01, d11;

  SymMat2 operator+(const SymMat2 &B) const {
    return {d00 + B.d00, d01 + B.d01, d11 + B.d11};
  }
  SymMat2 operator-(const SymMat2 &B) const {
    return {d00 - B.d00, d01 - B.d01, d11 - B.d11};
  }
};

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

  Mat2 operator*(const SymMat2 &B) const {
    Mat2 R;
    R(0, 0) = d[0][0] * B.d00 + d[0][1] * B.d01;
    R(0, 1) = d[0][0] * B.d01 + d[0][1] * B.d11;
    R(1, 0) = d[1][0] * B.d00 + d[1][1] * B.d01;
    R(1, 1) = d[1][0] * B.d01 + d[1][1] * B.d11;
    return R;
  }
};

inline Mat2 operator*(const SymMat2 &A, const Mat2 &B)
{
  Mat2 R;
  R(0, 0) = A.d00 * B(0, 0) + A.d01 * B(1, 0);
  R(0, 1) = A.d00 * B(0, 1) + A.d01 * B(1, 1);
  R(1, 0) = A.d01 * B(0, 0) + A.d11 * B(1, 0);
  R(1, 1) = A.d01 * B(0, 1) + A.d11 * B(1, 1);
  return R;
}

// r1·r1, r1·r2, r2·r2
inline SymMat2 sym_dot(const double r1[3], const double r2[3])
{
  return {r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2],
          r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2],
          r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]};
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

inline SymMat2 mat_mul_tosym(const Mat2 &A, const Mat2 &B)
{
  return {A(0, 0) * B(0, 0) + A(0, 1) * B(1, 0),
          A(0, 0) * B(0, 1) + A(0, 1) * B(1, 1),
          A(1, 0) * B(0, 1) + A(1, 1) * B(1, 1)};
}

inline Mat2 transpose(const Mat2 &A)
{
  Mat2 R;
  R(0, 0) = A(0, 0); R(0, 1) = A(1, 0);
  R(1, 0) = A(0, 1); R(1, 1) = A(1, 1);
  return R;
}

inline double skew(const Mat2 &A) { return A(0, 1) - A(1, 0); }

inline Mat2 chol_lower(const SymMat2 &A) //nonstandard!!
{
  double l11 = sqrt(A.d11);
  double l10 = A.d01 / l11;
  double l00 = sqrt(A.d00 - l10 * l10);
  Mat2 R;
  R(0, 0) = l00; R(0, 1) = 0.0;
  R(1, 0) = l10; R(1, 1) = l11;
  return R;
}

inline Mat2 inv_chol_upper(const SymMat2 &A)
{
  double u00 = sqrt(A.d00);
  double u01 = A.d01 / u00;
  double u11 = sqrt(A.d11 - u01*u01);
  double inv00 = 1.0 / u00;
  double inv11 = 1.0 / u11;
  double inv01 = -u01 * inv00 * inv11;
  Mat2 R;
  R(0, 0) = inv00; R(0, 1) = inv01;
  R(1, 0) = 0.0;   R(1, 1) = inv11;
  return R;
}

// M @ [v1, v2]: multiply 2x2 symmetric matrix by 2-column matrix
// Mv1 = M.d00*v1 + M.d01*v2
// Mv2 = M.d01*v1 + M.d11*v2
inline void sym_mat_vec(const SymMat2 &M, const double *v1, const double *v2,
                        double *Mv1, double *Mv2)
{
  Mv1[0] = M.d00 * v1[0] + M.d01 * v2[0];
  Mv1[1] = M.d00 * v1[1] + M.d01 * v2[1];
  Mv1[2] = M.d00 * v1[2] + M.d01 * v2[2];
  Mv2[0] = M.d01 * v1[0] + M.d11 * v2[0];
  Mv2[1] = M.d01 * v1[1] + M.d11 * v2[1];
  Mv2[2] = M.d01 * v1[2] + M.d11 * v2[2];
}

struct UTMat2 {
  double u00, u01, u11;

  void invert()
  {
    double inv00 = 1.0 / u00;
    double inv11 = 1.0 / u11;
    double inv01 = -u01 * inv00 * inv11;
    u00 = inv00; u01 = inv01; u11 = inv11;
  }

  void mat_vec(const double v[2], double out[2]) const
  {
    out[0] = u00 * v[0] + u01 * v[1];
    out[1] = u11 * v[1];
  }
};

inline Mat2 operator*(const UTMat2 &U, const Mat2 &B)
{
  Mat2 R;
  R(0, 0) = U.u00 * B(0, 0) + U.u01 * B(1, 0);
  R(0, 1) = U.u00 * B(0, 1) + U.u01 * B(1, 1);
  R(1, 0) = U.u11 * B(1, 0);
  R(1, 1) = U.u11 * B(1, 1);
  return R;
}

// S <- U S U^T  where U is upper-triangular Cholesky factor
inline void chol_sandwich(const UTMat2 &U, SymMat2 &S)
{
  double d00 = U.u00 * U.u00 * S.d00 + 2.0 * U.u00 * U.u01 * S.d01 + U.u01 * U.u01 * S.d11;
  double d01 = U.u11 * (U.u00 * S.d01 + U.u01 * S.d11);
  double d11 = U.u11 * U.u11 * S.d11;
  S.d00 = d00;
  S.d01 = d01;
  S.d11 = d11;
}

// M <- (U^T U)^{-1} M  via forward/back substitution with U
inline void chol_left_invmult(const UTMat2 &U, Mat2 &M)
{
  for (int j = 0; j < 2; j++) {
    M(0, j) /= U.u00;
    M(1, j) = (M(1, j) - U.u01 * M(0, j)) / U.u11;
  }
  for (int j = 0; j < 2; j++) {
    M(1, j) /= U.u11;
    M(0, j) = (M(0, j) - U.u01 * M(1, j)) / U.u00;
  }
}

// M <- M (U^T U)^{-1}  via back/forward substitution with U
inline void chol_right_invmult(Mat2 &M, const UTMat2 &U)
{
  for (int i = 0; i < 2; i++) {
    M(i, 0) /= U.u00;
    M(i, 1) = (M(i, 1) - M(i, 0) * U.u01) / U.u11;
  }
  for (int i = 0; i < 2; i++) {
    M(i, 1) /= U.u11;
    M(i, 0) = (M(i, 0) - M(i, 1) * U.u01) / U.u00;
  }
}

}

#endif
