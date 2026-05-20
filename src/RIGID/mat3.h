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

#ifndef LMP_MAT3_H
#define LMP_MAT3_H

#include <cmath>

struct SymMat3 {
  double d00, d01, d02, d11, d12, d22;

  SymMat3 operator+(const SymMat3 &B) const {
    return {d00 + B.d00, d01 + B.d01, d02 + B.d02,
            d11 + B.d11, d12 + B.d12, d22 + B.d22};
  }
  SymMat3 operator-(const SymMat3 &B) const {
    return {d00 - B.d00, d01 - B.d01, d02 - B.d02,
            d11 - B.d11, d12 - B.d12, d22 - B.d22};
  }
};

struct Mat3 {
  double d[3][3];

  double &operator()(int i, int j) { return d[i][j]; }
  double operator()(int i, int j) const { return d[i][j]; }

  double *operator()(int i) { return d[i]; }
  const double *operator()(int i) const { return d[i]; }

  void set_row(int i, const double v[3]) {
    d[i][0] = v[0]; d[i][1] = v[1]; d[i][2] = v[2];
  }
  void set_col(int j, const double v[3]) {
    d[0][j] = v[0]; d[1][j] = v[1]; d[2][j] = v[2];
  }

  Mat3 operator*(const Mat3 &B) const {
    Mat3 R;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R(i, j) = d[i][0] * B(0, j) + d[i][1] * B(1, j) + d[i][2] * B(2, j);
    return R;
  }

  Mat3 operator*(const SymMat3 &B) const {
    Mat3 R;
    double b[3][3] = {{B.d00, B.d01, B.d02},
                      {B.d01, B.d11, B.d12},
                      {B.d02, B.d12, B.d22}};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R(i, j) = d[i][0] * b[0][j] + d[i][1] * b[1][j] + d[i][2] * b[2][j];
    return R;
  }
};

inline Mat3 operator*(const SymMat3 &A, const Mat3 &B)
{
  Mat3 R;
  for (int j = 0; j < 3; j++)
    R(0, j) = A.d00 * B(0, j) + A.d01 * B(1, j) + A.d02 * B(2, j);
  for (int j = 0; j < 3; j++)
    R(1, j) = A.d01 * B(0, j) + A.d11 * B(1, j) + A.d12 * B(2, j);
  for (int j = 0; j < 3; j++)
    R(2, j) = A.d02 * B(0, j) + A.d12 * B(1, j) + A.d22 * B(2, j);
  return R;
}

inline SymMat3 sym_dot(const double r0[3], const double r1[3], const double r2[3])
{
  return {r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2],
          r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2],
          r0[0] * r2[0] + r0[1] * r2[1] + r0[2] * r2[2],
          r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2],
          r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2],
          r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]};
}

inline Mat3 mat_dot(const double s[][3], const double r[][3])
{
  Mat3 R;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      R(i, j) = s[i][0] * r[j][0] + s[i][1] * r[j][1] + s[i][2] * r[j][2];
  return R;
}

inline SymMat3 mat_mul_tosym(const Mat3 &A, const Mat3 &B)
{
  return {A(0, 0) * B(0, 0) + A(0, 1) * B(1, 0) + A(0, 2) * B(2, 0),
          A(0, 0) * B(0, 1) + A(0, 1) * B(1, 1) + A(0, 2) * B(2, 1),
          A(0, 0) * B(0, 2) + A(0, 1) * B(1, 2) + A(0, 2) * B(2, 2),
          A(1, 0) * B(0, 1) + A(1, 1) * B(1, 1) + A(1, 2) * B(2, 1),
          A(1, 0) * B(0, 2) + A(1, 1) * B(1, 2) + A(1, 2) * B(2, 2),
          A(2, 0) * B(0, 2) + A(2, 1) * B(1, 2) + A(2, 2) * B(2, 2)};
}

inline Mat3 inv(const Mat3 &A)
{
  double a = A(0, 0), b = A(0, 1), c = A(0, 2);
  double d = A(1, 0), e = A(1, 1), f = A(1, 2);
  double g = A(2, 0), h = A(2, 1), i = A(2, 2);

  double C00 =  e * i - f * h;
  double C01 =  f * g - d * i;
  double C02 =  d * h - e * g;
  double C10 =  c * h - b * i;
  double C11 =  a * i - c * g;
  double C12 =  b * g - a * h;
  double C20 =  b * f - c * e;
  double C21 =  c * d - a * f;
  double C22 =  a * e - b * d;

  double det = a * C00 + b * C01 + c * C02;

  Mat3 R;
  R(0, 0) = C00 / det; R(0, 1) = C10 / det; R(0, 2) = C20 / det;
  R(1, 0) = C01 / det; R(1, 1) = C11 / det; R(1, 2) = C21 / det;
  R(2, 0) = C02 / det; R(2, 1) = C12 / det; R(2, 2) = C22 / det;
  return R;
}

inline SymMat3 inv_sym(const SymMat3 &A)
{
  double a = A.d00, b = A.d01, c = A.d02;
  double d = A.d11, e = A.d12, f = A.d22;

  double C00 = d * f - e * e;
  double C01 = c * e - b * f;
  double C02 = b * e - c * d;
  double C11 = a * f - c * c;
  double C12 = b * c - a * e;
  double C22 = a * d - b * b;

  double det = a * C00 + b * C01 + c * C02;
  return {C00 / det, C01 / det, C02 / det, C11 / det, C12 / det, C22 / det};
}

inline Mat3 chol_upper(const SymMat3 &A)
{
  double u00 = sqrt(A.d00);
  double u01 = A.d01 / u00;
  double u02 = A.d02 / u00;
  double u11 = sqrt(A.d11 - u01 * u01);
  double u12 = (A.d12 - u01 * u02) / u11;
  double u22 = sqrt(A.d22 - u02 * u02 - u12 * u12);
  Mat3 R;
  R(0, 0) = u00; R(0, 1) = u01; R(0, 2) = u02;
  R(1, 0) = 0.0; R(1, 1) = u11; R(1, 2) = u12;
  R(2, 0) = 0.0; R(2, 1) = 0.0; R(2, 2) = u22;
  return R;
}

inline Mat3 inv_chol_lower(const SymMat3 &A)
{
  double l00 = sqrt(A.d00);
  double l10 = A.d01 / l00;
  double l20 = A.d02 / l00;
  double l11 = sqrt(A.d11 - l10 * l10);
  double l21 = (A.d12 - l10 * l20) / l11;
  double l22 = sqrt(A.d22 - l20 * l20 - l21 * l21);

  double inv00 = 1.0 / l00;
  double inv11 = 1.0 / l11;
  double inv22 = 1.0 / l22;
  double inv10 = -l10 * inv00 * inv11;
  double inv21 = -l21 * inv11 * inv22;
  double inv20 = (l21 * l10 * inv11 - l20) * inv00 * inv22;

  Mat3 R;
  R(0, 0) = inv00; R(0, 1) = 0.0;   R(0, 2) = 0.0;
  R(1, 0) = inv10; R(1, 1) = inv11; R(1, 2) = 0.0;
  R(2, 0) = inv20; R(2, 1) = inv21; R(2, 2) = inv22;
  return R;
}

inline Mat3 transpose(const Mat3 &A)
{
  Mat3 R;
  R(0, 0) = A(0, 0); R(0, 1) = A(1, 0); R(0, 2) = A(2, 0);
  R(1, 0) = A(0, 1); R(1, 1) = A(1, 1); R(1, 2) = A(2, 1);
  R(2, 0) = A(0, 2); R(2, 1) = A(1, 2); R(2, 2) = A(2, 2);
  return R;
}

#endif
