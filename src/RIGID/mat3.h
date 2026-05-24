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

namespace RigsMath {

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

struct UTMat3 {
  double u00, u01, u02, u11, u12, u22;

  void invert()
  {
    double inv00 = 1.0 / u00;
    double inv11 = 1.0 / u11;
    double inv22 = 1.0 / u22;
    double inv01 = -u01 * inv00 * inv11;
    double inv12 = -u12 * inv11 * inv22;
    double inv02 = (u01 * u12 * inv11 - u02) * inv00 * inv22;
    u00 = inv00; u01 = inv01; u02 = inv02;
    u11 = inv11; u12 = inv12; u22 = inv22;
  }

  void mat_vec(const double v[3], double out[3]) const
  {
    out[0] = u00 * v[0] + u01 * v[1] + u02 * v[2];
    out[1] = u11 * v[1] + u12 * v[2];
    out[2] = u22 * v[2];
  }
};

struct LTMat3 {
  double l00, l10, l20, l11, l21, l22;

  void invert()
  {
    double inv00 = 1.0 / l00;
    double inv11 = 1.0 / l11;
    double inv22 = 1.0 / l22;
    double inv10 = -l10 * inv00 * inv11;
    double inv21 = -l21 * inv11 * inv22;
    double inv20 = (l10 * l21 * inv11 - l20) * inv00 * inv22;
    l00 = inv00; l10 = inv10; l20 = inv20;
    l11 = inv11; l21 = inv21; l22 = inv22;
  }

  void mat_vec(const double v[3], double out[3]) const
  {
    out[0] = l00 * v[0];
    out[1] = l10 * v[0] + l11 * v[1];
    out[2] = l20 * v[0] + l21 * v[1] + l22 * v[2];
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

// M^T M: multiply transpose of M by M, yielding a symmetric matrix
inline SymMat3 sym_dot(const Mat3 &M)
{
  return {M(0, 0) * M(0, 0) + M(1, 0) * M(1, 0) + M(2, 0) * M(2, 0),
           M(0, 0) * M(0, 1) + M(1, 0) * M(1, 1) + M(2, 0) * M(2, 1),
           M(0, 0) * M(0, 2) + M(1, 0) * M(1, 2) + M(2, 0) * M(2, 2),
           M(0, 1) * M(0, 1) + M(1, 1) * M(1, 1) + M(2, 1) * M(2, 1),
           M(0, 1) * M(0, 2) + M(1, 1) * M(1, 2) + M(2, 1) * M(2, 2),
           M(0, 2) * M(0, 2) + M(1, 2) * M(1, 2) + M(2, 2) * M(2, 2)};
}

// S^T R: multiply transpose of S by R
inline Mat3 mat_dot(const Mat3 &S, const Mat3 &R)
{
  Mat3 QR;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      QR(i, j) = S(0, i) * R(0, j) + S(1, i) * R(1, j) + S(2, i) * R(2, j);
  return QR;
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

inline LTMat3 chol_lower(const SymMat3 &A)
{
  double l22 = sqrt(A.d22);
  double l21 = A.d12 / l22;
  double l20 = A.d02 / l22;
  double l11 = sqrt(A.d11 - l21 * l21);
  double l10 = A.d01 / l11;
  double l00 = sqrt(A.d00 - l10 * l10 - l20 * l20);
  return {l00, l10, l20, l11, l21, l22};
}

inline UTMat3 inv_chol_upper(const SymMat3 &A)
{
  double u00 = sqrt(A.d00);
  double u01 = A.d01 / u00;
  double u02 = A.d02 / u00;
  double u11 = sqrt(A.d11 - u01 * u01);
  double u12 = (A.d12 - u01 * u02) / u11;
  double u22 = sqrt(A.d22 - u02 * u02 - u12 * u12);

  double inv00 = 1.0 / u00;
  double inv11 = 1.0 / u11;
  double inv22 = 1.0 / u22;
  double inv01 = -u01 * inv00 * inv11;
  double inv12 = -u12 * inv11 * inv22;
  double inv02 = (u01 * u12 * inv11 - u02) * inv00 * inv22;

  return {inv00, inv01, inv02, inv11, inv12, inv22};
}

inline Mat3 transpose(const Mat3 &A)
{
  Mat3 R;
  R(0, 0) = A(0, 0); R(0, 1) = A(1, 0); R(0, 2) = A(2, 0);
  R(1, 0) = A(0, 1); R(1, 1) = A(1, 1); R(1, 2) = A(2, 1);
  R(2, 0) = A(0, 2); R(2, 1) = A(1, 2); R(2, 2) = A(2, 2);
  return R;
}

inline void mat_vec(const Mat3 &A, const double v[3], double out[3])
{
  out[0] = A(0, 0) * v[0] + A(0, 1) * v[1] + A(0, 2) * v[2];
  out[1] = A(1, 0) * v[0] + A(1, 1) * v[1] + A(1, 2) * v[2];
  out[2] = A(2, 0) * v[0] + A(2, 1) * v[1] + A(2, 2) * v[2];
}

inline void cross(const double a[3], const double b[3], double out[3])
{
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

inline void get_col(const Mat3 &A, int j, double out[3])
{
  out[0] = A(0, j); out[1] = A(1, j); out[2] = A(2, j);
}

inline void skew(const Mat3 &A, double out[3])
{
  out[0] = A(2, 1) - A(1, 2);
  out[1] = A(0, 2) - A(2, 0);
  out[2] = A(1, 0) - A(0, 1);
}

inline void matmul_to(const Mat3 &A, const Mat3 &B, Mat3 &C)
{
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      C(i, j) = A(i, 0) * B(0, j) + A(i, 1) * B(1, j) + A(i, 2) * B(2, j);
}

inline void cayley_rotate(Mat3 &A, const double v[3])
{
  double v_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  double w = sqrt(1.0 - v_sq);

  for (int j = 0; j < 3; j++) {
    double col[3] = {A(0, j), A(1, j), A(2, j)};
    double cross1[3], cross2[3];
    cross(v, col, cross1);
    cross(v, cross1, cross2);
    A(0, j) = col[0] + 2.0 * w * cross1[0] + 2.0 * cross2[0];
    A(1, j) = col[1] + 2.0 * w * cross1[1] + 2.0 * cross2[1];
    A(2, j) = col[2] + 2.0 * w * cross1[2] + 2.0 * cross2[2];
  }
}

inline void skew_ut_mul(const UTMat3 &rc, const Mat3 &sc, double out[3])
{
  double g10 = rc.u11 * sc(1, 0) + rc.u12 * sc(2, 0);
  double g01 = rc.u00 * sc(0, 1) + rc.u01 * sc(1, 1) + rc.u02 * sc(2, 1);
  double g02 = rc.u00 * sc(0, 2) + rc.u01 * sc(1, 2) + rc.u02 * sc(2, 2);
  double g20 = rc.u22 * sc(2, 0);
  double g21 = rc.u22 * sc(2, 1);
  double g12 = rc.u11 * sc(1, 2) + rc.u12 * sc(2, 2);
  out[0] = g21 - g12;
  out[1] = g02 - g20;
  out[2] = g10 - g01;
}

inline void ut_mat_mul_to(const UTMat3 &A, const Mat3 &B, Mat3 &C)
{
  for (int j = 0; j < 3; j++) {
    C(0, j) = A.u00 * B(0, j) + A.u01 * B(1, j) + A.u02 * B(2, j);
    C(1, j) = A.u11 * B(1, j) + A.u12 * B(2, j);
    C(2, j) = A.u22 * B(2, j);
  }
}

inline Mat3 cayley_converge(const UTMat3 &rc, Mat3 &sc, const Mat3 &chi,
                            int max_iters = 10, double tol = 1e-6)
{
  LTMat3 G;
  G.l00 = 2.0 * (rc.u11 * sc(2, 2) + rc.u22 * sc(1, 1));
  G.l10 = 2.0 * (-rc.u22 * sc(1, 0) - rc.u01 * sc(2, 2));
  G.l11 = 2.0 * (rc.u22 * sc(0, 0) + rc.u00 * sc(2, 2));
  G.l20 = 2.0 * (rc.u01 * sc(2, 1) - rc.u02 * sc(1, 1) - (rc.u11 * sc(2, 0) - rc.u12 * sc(1, 0)));
  G.l21 = 2.0 * (-rc.u00 * sc(2, 1) - rc.u12 * sc(0, 0));
  G.l22 = 2.0 * (rc.u00 * sc(1, 1) + rc.u11 * sc(0, 0));

  G.invert();

  double skewChi[3];
  skew(chi, skewChi);

  double negSkewChi[3] = {-skewChi[0], -skewChi[1], -skewChi[2]};
  double neg_Gchi[3];
  G.mat_vec(negSkewChi, neg_Gchi);

  double tol_sq = 3.0 * tol * tol;

  Mat3 gamma;
  double skewGam[3], negSkewGam[3], rotvec[3];
  for (int niter = 0; niter < max_iters; niter++) {
    skew_ut_mul(rc, sc, skewGam);
    double r0 = skewChi[0] + skewGam[0];
    double r1 = skewChi[1] + skewGam[1];
    double r2 = skewChi[2] + skewGam[2];
    if (r0*r0 + r1*r1 + r2*r2 < tol_sq) {
      ut_mat_mul_to(rc, sc, gamma);
      break;
    }
    negSkewGam[0] = -skewGam[0];
    negSkewGam[1] = -skewGam[1];
    negSkewGam[2] = -skewGam[2];
    G.mat_vec(negSkewGam, rotvec);
    rotvec[0] += neg_Gchi[0];
    rotvec[1] += neg_Gchi[1];
    rotvec[2] += neg_Gchi[2];
    cayley_rotate(sc, rotvec);
  }
  ut_mat_mul_to(rc, sc, gamma);

  return gamma;
}

}

#endif
