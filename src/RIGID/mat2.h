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

  void invert()
  {
    double inv00 = 1.0 / l00;
    double inv11 = 1.0 / l11;
    double inv10 = -l10 * inv00 * inv11;
    l00 = inv00; l10 = inv10; l11 = inv11;
  }

  void mat_vec(const double v[2], double out[2]) const
  {
    out[0] = l00 * v[0];
    out[1] = l10 * v[0] + l11 * v[1];
  }
};

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

struct SymMat2 {
  double d00, d01, d11;

  SymMat2 operator+(const SymMat2 &B) const {
    return {d00 + B.d00, d01 + B.d01, d11 + B.d11};
  }
  SymMat2 operator-(const SymMat2 &B) const {
    return {d00 - B.d00, d01 - B.d01, d11 - B.d11};
  }

  SymMat2& operator*=(const LTMat2 &L)
  {
    // S <- S * L^T
    double i00 = d00 * L.l00;
    double i01 = d00 * L.l10 + d01 * L.l11;
    double i11 = d01 * L.l10 + d11 * L.l11;
    // <- L * S * L^T
    d00 = L.l00 * i00;
    d01 = L.l00 * i01;
    d11 = L.l10 * i01 + L.l11 * i11;
    return *this;
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

  Mat2& operator*=(const LTMat2 &L)
  {
    // M <- M * L
    d[0][0] = d[0][0] * L.l00 + d[0][1] * L.l10;
    d[0][1] *= L.l11;
    d[1][0] = d[1][0] * L.l00 + d[1][1] * L.l10;
    d[1][1] *= L.l11;
    return *this;
  }

  Mat2& operator*=(const UTMat2 &U)
  {
    // M <- M * U
    d[0][1] = d[0][0] * U.u01 + d[0][1] * U.u11;
    d[0][0] *= U.u00;
    d[1][1] = d[1][0] * U.u01 + d[1][1] * U.u11;
    d[1][0] *= U.u00;
    return *this;
  }
  
  // M <- (U^T U)^{-1} M  via forward/back substitution with U:
  Mat2& operator/=(const UTMat2 &U) { // 
    for (int j = 0; j < 2; j++) {
      d[0][j] /= U.u00;
      d[1][j] -= U.u01 * d[0][j];
      d[1][j] /= U.u11;
    }
    for (int j = 0; j < 2; j++) {
      d[1][j] /= U.u11;
      d[0][j] -= U.u01 * d[1][j];
      d[0][j] /= U.u00;
    }
    return *this;
  }
};

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

inline SymMat2 sym_dot(const Mat2 &M)
{
  return {M(0, 0) * M(0, 0) + M(1, 0) * M(1, 0),
          M(0, 0) * M(0, 1) + M(1, 0) * M(1, 1),
          M(0, 1) * M(0, 1) + M(1, 1) * M(1, 1)};
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

inline UTMat2 operator*(const UTMat2 &U, const UTMat2 &V)
{
  return {U.u00 * V.u00, U.u00 * V.u01 + U.u01 * V.u11, U.u11 * V.u11};
}

inline Mat2 operator*(const UTMat2 &U, const Mat2 &B)
{
  Mat2 R;
  R(0, 0) = U.u00 * B(0, 0) + U.u01 * B(1, 0);
  R(0, 1) = U.u00 * B(0, 1) + U.u01 * B(1, 1);
  R(1, 0) = U.u11 * B(1, 0);
  R(1, 1) = U.u11 * B(1, 1);
  return R;
}

inline Mat2 transpose(const UTMat2 &U)
{
  Mat2 R;
  R(0, 0) = U.u00; R(0, 1) = 0.0;
  R(1, 0) = U.u01; R(1, 1) = U.u11;
  return R;
}

inline UTMat2 chol_upper(const SymMat2 &A)
{
  double u00 = sqrt(A.d00);
  double u01 = A.d01 / u00;
  double u11 = sqrt(A.d11 - u01 * u01);
  return {u00, u01, u11};
}

inline UTMat2 inv_chol_upper(const SymMat2 &A)
{
  double u00 = sqrt(A.d00);
  double u01 = A.d01 / u00;
  double u11 = sqrt(A.d11 - u01 * u01);
  return {1.0 / u00, -u01 / (u00 * u11), 1.0 / u11};
}

inline LTMat2 chol_lower_lt(const SymMat2 &A)
{
  double l11 = sqrt(A.d11);
  double l10 = A.d01 / l11;
  double l00 = sqrt(A.d00 - l10 * l10);
  return {l00, l10, l11};
}

inline LTMat2 operator*(const LTMat2 &A, const LTMat2 &B)
{
  return {A.l00 * B.l00,
          A.l10 * B.l00 + A.l11 * B.l10,
          A.l11 * B.l11};
}

inline Mat2 to_mat(const LTMat2 &L)
{
  Mat2 R;
  R(0, 0) = L.l00; R(0, 1) = 0.0;
  R(1, 0) = L.l10; R(1, 1) = L.l11;
  return R;
}

inline Mat2 transpose(const LTMat2 &L)
{
  Mat2 R;
  R(0, 0) = L.l00; R(0, 1) = L.l10;
  R(1, 0) = 0.0;   R(1, 1) = L.l11;
  return R;
}

// L S L^T  (left- and right-multiply by lower triangular L and its transpose)
inline SymMat2 chol_sandwich(const SymMat2 &S, const LTMat2 &L)
{
  return {L.l00 * L.l00 * S.d00,
           L.l00 * L.l10 * S.d00 + L.l00 * L.l11 * S.d01,
           L.l10 * L.l10 * S.d00 + 2.0 * L.l10 * L.l11 * S.d01 + L.l11 * L.l11 * S.d11};
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

// M <- M L^T L  (right-multiply by product mu * mu^T where L = mu^T)
inline void chol_right_mult(Mat2 &M, const LTMat2 &L)
{
  for (int i = 0; i < 2; i++) {
    double t0 = M(i, 0) * L.l00;
    double t1 = M(i, 0) * L.l10 + M(i, 1) * L.l11;
    M(i, 0) = t0 * L.l00 + t1 * L.l10;
    M(i, 1) = t1 * L.l11;
  }
}

// M <- M U^T U  (right-multiply by Cholesky product)
inline void chol_right_mult(Mat2 &M, const UTMat2 &U)
{
  for (int i = 0; i < 2; i++) {
    double t0 = M(i, 0) * U.u00 + M(i, 1) * U.u01;
    double t1 = M(i, 1) * U.u11;
    M(i, 0) = t0 * U.u00;
    M(i, 1) = t0 * U.u01 + t1 * U.u11;
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
