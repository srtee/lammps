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

  static SymMat3 load(const double *p) {
    return {p[0], p[1], p[2], p[3], p[4], p[5]};
  }
  void store(double *p) const {
    p[0] = d00; p[1] = d01; p[2] = d02;
    p[3] = d11; p[4] = d12; p[5] = d22;
  }

  SymMat3 operator+(const SymMat3 &B) const {
    return {d00 + B.d00, d01 + B.d01, d02 + B.d02,
            d11 + B.d11, d12 + B.d12, d22 + B.d22};
  }
  SymMat3 operator-(const SymMat3 &B) const {
    return {d00 - B.d00, d01 - B.d01, d02 - B.d02,
            d11 - B.d11, d12 - B.d12, d22 - B.d22};
  }
};

struct Mat3;

struct UTMat3 {
  double u00, u01, u02, u11, u12, u22;
  operator Mat3() const;
};

struct Mat3 {
  double d[3][3];

  double &operator()(int i, int j) { return d[i][j]; }
  double operator()(int i, int j) const { return d[i][j]; }

  double *operator()(int i) { return d[i]; }
  const double *operator()(int i) const { return d[i]; }

  Mat3 &operator+=(const Mat3 &B) {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        d[i][j] += B(i, j);
    return *this;
  }
};

inline UTMat3::operator Mat3() const
{
  Mat3 M;
  M(0, 0) = u00; M(0, 1) = u01; M(0, 2) = u02;
  M(1, 0) = 0.0; M(1, 1) = u11; M(1, 2) = u12;
  M(2, 0) = 0.0; M(2, 1) = 0.0; M(2, 2) = u22;
  return M;
}

struct ColMat3 {
  double d[3][3];

  double &operator()(int i, int j) { return d[j][i]; }
  double operator()(int i, int j) const { return d[j][i]; }

  double *operator()(int i) { return d[i]; }
  const double *operator()(int i) const { return d[i]; }

  operator Mat3() const {
    Mat3 M;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        M(i, j) = d[j][i];
    return M;
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
  operator ColMat3() const {
    ColMat3 M;
    M(0, 0) = l00; M(1, 0) = l10; M(2, 0) = l20;
    M(0, 1) = 0.0; M(1, 1) = l11; M(2, 1) = l21;
    M(0, 2) = 0.0; M(1, 2) = 0.0; M(2, 2) = l22;
    return M;
  }
};

inline void ut_mul(const UTMat3 &U, Mat3 &M)
{
  for (int j = 0; j < 3; j++) {
    M(2, j) = U.u02 * M(0, j) + U.u12 * M(1, j) + U.u22 * M(2, j);
    M(1, j) = U.u01 * M(0, j) + U.u11 * M(1, j);
    M(0, j) = U.u00 * M(0, j);
  }
}

inline void u_mul(const UTMat3 &U, Mat3 &M)
{
  for (int j = 0; j < 3; j++) {
    M(0, j) = U.u00 * M(0, j) + U.u01 * M(1, j) + U.u02 * M(2, j);
    M(1, j) = U.u11 * M(1, j) + U.u12 * M(2, j);
    M(2, j) = U.u22 * M(2, j);
  }
}

struct DChol3 {
  double d0, d1, d2, m01, m02, m12;

  static DChol3 load(const double *p) {
    return {p[0], p[1], p[2], p[3], p[4], p[5]};
  }
  void store(double *p) const {
    p[0] = d0; p[1] = d1; p[2] = d2;
    p[3] = m01; p[4] = m02; p[5] = m12;
  }
};

inline SymMat3 mtm(const Mat3 &M) // M^T M
{
  return {M(0, 0) * M(0, 0) + M(1, 0) * M(1, 0) + M(2, 0) * M(2, 0),
           M(0, 0) * M(0, 1) + M(1, 0) * M(1, 1) + M(2, 0) * M(2, 1),
           M(0, 0) * M(0, 2) + M(1, 0) * M(1, 2) + M(2, 0) * M(2, 2),
           M(0, 1) * M(0, 1) + M(1, 1) * M(1, 1) + M(2, 1) * M(2, 1),
           M(0, 1) * M(0, 2) + M(1, 1) * M(1, 2) + M(2, 1) * M(2, 2),
           M(0, 2) * M(0, 2) + M(1, 2) * M(1, 2) + M(2, 2) * M(2, 2)};
}

inline SymMat3 mmt(const Mat3 &M) // M M^T
{
  return {M(0, 0) * M(0, 0) + M(0, 1) * M(0, 1) + M(0, 2) * M(0, 2),
           M(0, 0) * M(1, 0) + M(0, 1) * M(1, 1) + M(0, 2) * M(1, 2),
           M(0, 0) * M(2, 0) + M(0, 1) * M(2, 1) + M(0, 2) * M(2, 2),
           M(1, 0) * M(1, 0) + M(1, 1) * M(1, 1) + M(1, 2) * M(1, 2),
           M(1, 0) * M(2, 0) + M(1, 1) * M(2, 1) + M(1, 2) * M(2, 2),
           M(2, 0) * M(2, 0) + M(2, 1) * M(2, 1) + M(2, 2) * M(2, 2)};
}

inline Mat3 mat_dot(const Mat3 &R, const Mat3 &S) // R S^T
{
  Mat3 QR;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      QR(i, j) = R(i, 0) * S(j, 0) + R(i, 1) * S(j, 1) + R(i, 2) * S(j, 2);
  return QR;
}

inline void lslt_mul(SymMat3 &S, const DChol3 &L)
{
  // S <- S L^T
  S.d02 += S.d01 * L.m12 + S.d00 * L.m02;
  S.d12 += S.d11 * L.m12 + S.d01 * L.m02;
  S.d22 += S.d12 * L.m12 + S.d02 * L.m02;
  S.d01 += S.d00 * L.m01; 
  S.d11 += S.d01 * L.m01; 
  S.d12 += S.d02 * L.m01; 
  // <- L S L^T
  S.d22 += L.m02 * S.d02 + L.m12 * S.d12;
  S.d12 += L.m01 * S.d02;
  S.d11 += L.m01 * S.d01;
}

inline LTMat3 mul_dl(const LTMat3 &L, const DChol3 &DL)
{
  LTMat3 M = L;
  M.l10 += DL.m01 * M.l11;
  M.l20 += DL.m01 * M.l21 + DL.m02 * M.l22;
  M.l21 += DL.m12 * M.l22;
  M.l22 *= DL.d2;
  M.l11 *= DL.d1;
  M.l21 *= DL.d1;
  M.l20 *= DL.d0;
  M.l10 *= DL.d0;
  M.l00 *= DL.d0;
  return M;
}

inline void mul_ltdl(Mat3 &M, const DChol3 &L)
{
  // M <- M L^T <- M L^T D
  for (int i = 0; i < 3; i++) {
    M(i, 2) += M(i, 1) * L.m12 + M(i, 0) * L.m02;
    M(i, 2) *= L.d2;
    M(i, 1) += M(i, 0) * L.m01; 
    M(i, 1) *= L.d1;
    M(i, 0) *= L.d0;
  }
  // <- M L^T D L
  for (int i = 0; i < 3; i++) {
    M(i, 0) += L.m01 * M(i, 1) + L.m02 * M(i, 2);
    M(i, 1) += L.m12 * M(i, 2);
  }
}

inline DChol3 inv_dchol(const SymMat3 &A)
{
  double m01 = A.d01 / A.d00;
  double m12 = (A.d12 - m01 * A.d02) / (A.d11 - m01 * A.d01);
  double m02 = A.d02 / A.d00;
  double d1 = A.d11 - m01 * A.d01;
  double d2 = A.d22 - m02 * A.d02 - m12 * (A.d12 - m01 * A.d02);
  m02 -= m01 * m12; // U^{-1}_{02} = u01*u12 - u02
  return {1.0/A.d00, 1.0/d1, 1.0/d2, -m01, -m02, -m12};
}

inline DChol3 dchol_pivot_one(const SymMat3 &A, int p)
{
  // Semi-pivoted LDL^T: swap row/column p to position 2 only.
  // This ensures the first two columns always correspond to the
  // two non-demoted bonds, while the demoted bond's direction is
  // moved to position 2 where it can be safely eliminated.
  double a00 = A.d00, a01 = A.d01, a02 = A.d02;
  double a11 = A.d11, a12 = A.d12;
  double a22 = A.d22;

  if (p == 1) {
    double tmp = a11; a11 = a22; a22 = tmp;
    tmp = a01; a01 = a12; a12 = tmp;
  } else if (p == 0) {
    double tmp = a00; a00 = a22; a22 = tmp;
    tmp = a01; a01 = a02; a02 = tmp;
  }

  double d0 = a00;
  double m01 = a01 / d0;
  double m02 = a02 / d0;

  double c11 = a11 - m01 * a01;
  double c12 = a12 - m01 * a02;

  double d1 = c11;
  double m12 = c12 / d1;
  double d2 = a22 - m02 * a02 - m12 * c12;
  if (d2 < 0.0) d2 = 0.0;

  return {d0, d1, d2, m01, m02, m12};
}

inline DChol3 dchol_pivot(const SymMat3 &A, int perm[3])
{
  perm[0] = 0; perm[1] = 1; perm[2] = 2;

  double a00 = A.d00, a01 = A.d01, a02 = A.d02;
  double a11 = A.d11, a12 = A.d12;
  double a22 = A.d22;

  {
    double alpha = std::fabs(a00), beta = std::fabs(a11), gamma = std::fabs(a22);
    if (beta > alpha && beta >= gamma) {
      int t = perm[0]; perm[0] = perm[1]; perm[1] = t;
      double tmp = a00; a00 = a11; a11 = tmp;
      tmp = a02; a02 = a12; a12 = tmp;
    } else if (gamma > alpha) {
      int t = perm[0]; perm[0] = perm[2]; perm[2] = t;
      double tmp = a00; a00 = a22; a22 = tmp;
      tmp = a01; a01 = a12; a12 = tmp;
    }
  }

  double d0 = a00;
  double m01 = a01 / d0;
  double m02 = a02 / d0;

  double c11 = a11 - m01 * a01;
  double c12 = a12 - m01 * a02;
  double c22 = a22 - m02 * a02;

  if (std::fabs(c22) > std::fabs(c11)) {
    int t = perm[1]; perm[1] = perm[2]; perm[2] = t;
    double tmp = c11; c11 = c22; c22 = tmp;
    tmp = m01; m01 = m02; m02 = tmp;
  }

  double d1 = c11;
  double m12 = c12 / d1;
  double d2 = c22 - m12 * c12;
  if (d2 < 0.0) d2 = 0.0;

  return {d0, d1, d2, m01, m02, m12};
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

inline void cross(const double a[3], const double b[3], double out[3])
{
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

inline void skew(const Mat3 &A, double out[3])
{
  out[0] = A(2, 1) - A(1, 2);
  out[1] = A(0, 2) - A(2, 0);
  out[2] = A(1, 0) - A(0, 1);
}

inline double dot3(const double a[3], const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double normsq(const double v[3]) {
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

inline void cayley_rotate(const double v[3], ColMat3 &A)
{
  double w = sqrt(1.0 - normsq(v));

  for (int j = 0; j < 3; j++) {
    double* col = A(j);
    double cross1[3], cross2[3];
    cross(v, col, cross1);
    cross(v, cross1, cross2);
    col[0] += 2.0 * w * cross1[0] + 2.0 * cross2[0];
    col[1] += 2.0 * w * cross1[1] + 2.0 * cross2[1];
    col[2] += 2.0 * w * cross1[2] + 2.0 * cross2[2];
  }
}

inline void negskew_ut_mul(const UTMat3 &rc, const ColMat3 &sc, double out[3])
// return -skew(rc * sc) 
{
  double g10 = rc.u11 * sc(1, 0) + rc.u12 * sc(2, 0);
  double g01 = rc.u00 * sc(0, 1) + rc.u01 * sc(1, 1) + rc.u02 * sc(2, 1);
  double g02 = rc.u00 * sc(0, 2) + rc.u01 * sc(1, 2) + rc.u02 * sc(2, 2);
  double g20 = rc.u22 * sc(2, 0);
  double g21 = rc.u22 * sc(2, 1);
  double g12 = rc.u11 * sc(1, 2) + rc.u12 * sc(2, 2);
  out[0] = g12 - g21;
  out[1] = g20 - g02;
  out[2] = g01 - g10;
}

inline Mat3 cayley_converge(const UTMat3 &rc, const LTMat3 &sc, const Mat3 &chi,
                            int max_iters = 10, double tol = 1e-6)
{
  LTMat3 G;
  G.l00 = 2.0 * (rc.u11 * sc.l22 + rc.u22 * sc.l11);
  G.l10 = 2.0 * (-rc.u22 * sc.l10 - rc.u01 * sc.l22);
  G.l11 = 2.0 * (rc.u22 * sc.l00 + rc.u00 * sc.l22);
  G.l20 = 2.0 * (rc.u01 * sc.l21 - rc.u02 * sc.l11 - (rc.u11 * sc.l20 - rc.u12 * sc.l10));
  G.l21 = 2.0 * (-rc.u00 * sc.l21 - rc.u12 * sc.l00);
  G.l22 = 2.0 * (rc.u00 * sc.l11 + rc.u11 * sc.l00);

  G.invert();
  
  ColMat3 scm = sc;

  double skewChi[3];
  skew(chi, skewChi);

  double tol_sq = 3.0 * tol * tol;

  double negSkewGam[3], rotvec[3];
  for (int niter = 0; niter < max_iters; niter++) {
    negskew_ut_mul(rc, scm, negSkewGam);
    negSkewGam[0] -= skewChi[0];
    negSkewGam[1] -= skewChi[1];
    negSkewGam[2] -= skewChi[2];
    if (normsq(negSkewGam) < tol_sq) break;
    G.mat_vec(negSkewGam, rotvec);
    double rvsq = normsq(rotvec);
    if (rvsq > 1.0) {
      double rv = 1.001*sqrt(rvsq);
      rotvec[0] /= rv;
      rotvec[1] /= rv;
      rotvec[2] /= rv;
    }
    cayley_rotate(rotvec, scm);
  }

  Mat3 gamma = scm;
  u_mul(rc, gamma);
  return gamma;
}

struct Mat43 {
  double d[4][3];

  double &operator()(int i, int j) { return d[i][j]; }
  double operator()(int i, int j) const { return d[i][j]; }

  double *operator()(int i) { return d[i]; }
  const double *operator()(int i) const { return d[i]; }

  Mat43 &operator*=(const Mat3 &B) {
    for (int i = 0; i < 4; i++) {
      double row[3] = {d[i][0], d[i][1], d[i][2]};
      for (int j = 0; j < 3; j++) {
        d[i][j] = row[0] * B(0, j) + row[1] * B(1, j) + row[2] * B(2, j);
      }
    }
    return *this;
  }
};

inline Mat43 improper_L_lambda(const Mat3 &lam) {
  Mat43 L;
  L(0,0) = L(0,1) = L(0,2) = 0.0;
  for (int i = 0; i < 3; i++) {
    L(0,0) += lam(i,0); L(0,1) += lam(i,1); L(0,2) += lam(i,2);
    L(i+1,0) = -lam(i,0);
    L(i+1,1) = -lam(i,1);
    L(i+1,2) = -lam(i,2);
  }
  return L;
}

inline Mat43 dihedral_L_lambda(const Mat3 &lam) {
  Mat43 L;
  L(0,0) = L(0,1) = 1.0;  L(0,2) =  0.0;
  L(1,0) = -1.0; L(1,1) = L(1,2) =  0.0;
  L(2,0) = 0.0;  L(2,1) = L(2,2) = -1.0;
  L(3,0) = L(3,1) = 0.0;  L(3,2) =  1.0;
  L *= lam;
  return L;
}

inline void chol_frame3(const double *L, double *R)
{
  double b1 = sqrt(L[0]);
  double r10 = L[1] / b1;
  double b2sq = L[3] - r10 * r10;
  double b2 = sqrt(b2sq);
  double r20 = L[2] / b1;
  double r21 = (L[4] - r10 * r20) / b2;
  double b3sq = L[5] - r20 * r20 - r21 * r21;
  double b3 = sqrt(b3sq);
  R[0] = b1;  R[1] = 0.0; R[2] = 0.0;
  R[3] = r10; R[4] = b2;  R[5] = 0.0;
  R[6] = r20; R[7] = r21; R[8] = b3;
}

}

#endif
