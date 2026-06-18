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

#include "math_extra.h"

namespace RigsMath {

struct SymMat3 {
  double d00, d01, d02, d11, d12, d22;

  static SymMat3 load(const double *p) {
    return {p[0], p[1], p[2], p[3], p[4], p[5]};
  }
};

struct Mat3;
struct ColMat3;

struct UTMat3 {
  double u00, u01, u02, u11, u12, u22;

  static UTMat3 load(const double *p) {
    return {p[0], p[1], p[2], p[3], p[4], p[5]};
  }
  void store(double *p) const {
    p[0] = u00; p[1] = u01; p[2] = u02;
    p[3] = u11; p[4] = u12; p[5] = u22;
  }

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

  operator Mat3() const;
  operator ColMat3() const;
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

  void mat_vec(const double v[3], double out[3]) const
  {
    MathExtra::matvec(d, v, out);
  }
};

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

inline UTMat3::operator Mat3() const
{
  Mat3 M;
  M(0, 0) = u00; M(0, 1) = u01; M(0, 2) = u02;
  M(1, 0) = 0.0; M(1, 1) = u11; M(1, 2) = u12;
  M(2, 0) = 0.0; M(2, 1) = 0.0; M(2, 2) = u22;
  return M;
}

inline UTMat3::operator ColMat3() const {
  ColMat3 M;
  M(0, 0) = u00; M(0, 1) = u01; M(0, 2) = u02;
  M(1, 0) = 0.0; M(1, 1) = u11; M(1, 2) = u12;
  M(2, 0) = 0.0; M(2, 1) = 0.0; M(2, 2) = u22;
  return M;
}

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

  operator Mat3() const {
    Mat3 M;
    M(0, 0) = l00; M(0, 1) = 0.0; M(0, 2) = 0.0;
    M(1, 0) = l10; M(1, 1) = l11; M(1, 2) = 0.0;
    M(2, 0) = l20; M(2, 1) = l21; M(2, 2) = l22;
    return M;
  }

  LTMat3 &operator*=(const double s) {
    l00 *= s; l10 *= s; l11 *= s;
    l20 *= s; l21 *= s; l22 *= s;
    return *this;
  }

  LTMat3 operator*(const double s) const {
    return {l00 * s, l10 * s, l20 * s, l11 * s, l21 * s, l22 * s};
  }
};

struct LTDL3 {
  double d0, d1, d2, l10, l20, l21;
};

inline void l_mul(const LTMat3 &L, Mat3 &M)
{
  for (int j = 0; j < 3; j++) {
    M(2, j) = L.l20 * M(0, j) + L.l21 * M(1, j) + L.l22 * M(2, j);
    M(1, j) = L.l10 * M(0, j) + L.l11 * M(1, j);
    M(0, j) = L.l00 * M(0, j);
  }
}

inline SymMat3 mtm(const Mat3 &M)
{
  return {M(0, 0) * M(0, 0) + M(1, 0) * M(1, 0) + M(2, 0) * M(2, 0),
           M(0, 0) * M(0, 1) + M(1, 0) * M(1, 1) + M(2, 0) * M(2, 1),
           M(0, 0) * M(0, 2) + M(1, 0) * M(1, 2) + M(2, 0) * M(2, 2),
           M(0, 1) * M(0, 1) + M(1, 1) * M(1, 1) + M(2, 1) * M(2, 1),
           M(0, 1) * M(0, 2) + M(1, 1) * M(1, 2) + M(2, 1) * M(2, 2),
           M(0, 2) * M(0, 2) + M(1, 2) * M(1, 2) + M(2, 2) * M(2, 2)};
}

inline LTMat3 ql_decompose(const Mat3 &A, Mat3 &Q)
{
  double a00 = A(0,0), a01 = A(0,1), a02 = A(0,2);
  double a10 = A(1,0), a11 = A(1,1), a12 = A(1,2);
  double a20 = A(2,0), a21 = A(2,1), a22 = A(2,2);

  // QL decomposition: A = Q * L where Q is orthogonal, L is lower triangular.
  // Sign convention: all diagonals of L are forced positive. We apply left
  // Householder reflections to zero upper-triangular entries.
  // H2 * H1 * A = L, so A = H1 * H2 * L = Q * L.

  // Step 1: Zero (0,2) and (1,2) by reflecting column 2
  double col2[3] = {a02, a12, a22};
  double nrm2 = sqrt(col2[0]*col2[0] + col2[1]*col2[1] + col2[2]*col2[2]);
  double l22 = nrm2;
  col2[2] -= l22;
  double inv_n2 = 1.0 / sqrt(col2[0]*col2[0] + col2[1]*col2[1] + col2[2]*col2[2]);
  col2[0] *= inv_n2; col2[1] *= inv_n2; col2[2] *= inv_n2;

  double d;
  d = col2[0]*a00 + col2[1]*a10 + col2[2]*a20;
  a00 -= 2*col2[0]*d; a10 -= 2*col2[1]*d; a20 -= 2*col2[2]*d;
  d = col2[0]*a01 + col2[1]*a11 + col2[2]*a21;
  a01 -= 2*col2[0]*d; a11 -= 2*col2[1]*d; a21 -= 2*col2[2]*d;
  d = col2[0]*a02 + col2[1]*a12 + col2[2]*a22;
  a02 -= 2*col2[0]*d; a12 -= 2*col2[1]*d; a22 -= 2*col2[2]*d;

  // Step 2: Zero (0,1) by reflecting column 1 (rows 0,1)
  double nrm1 = sqrt(a01*a01 + a11*a11);
  double l11 = nrm1;
  double v1_0 = a01, v1_1 = a11 - l11;
  double inv_n1 = 1.0 / sqrt(v1_0*v1_0 + v1_1*v1_1);
  v1_0 *= inv_n1; v1_1 *= inv_n1;

  d = v1_0*a00 + v1_1*a10;
  a00 -= 2*v1_0*d; a10 -= 2*v1_1*d;
  d = v1_0*a01 + v1_1*a11;
  a01 -= 2*v1_0*d; a11 -= 2*v1_1*d;

  // Build Q = H1 * H2
  Q(0,0) = 1; Q(0,1) = 0; Q(0,2) = 0;
  Q(1,0) = 0; Q(1,1) = 1; Q(1,2) = 0;
  Q(2,0) = 0; Q(2,1) = 0; Q(2,2) = 1;
  for (int j = 0; j < 3; j++) {
    d = v1_0*Q(0,j) + v1_1*Q(1,j);
    Q(0,j) -= 2*v1_0*d; Q(1,j) -= 2*v1_1*d;
  }
  for (int j = 0; j < 3; j++) {
    d = col2[0]*Q(0,j) + col2[1]*Q(1,j) + col2[2]*Q(2,j);
    Q(0,j) -= 2*col2[0]*d; Q(1,j) -= 2*col2[1]*d; Q(2,j) -= 2*col2[2]*d;
  }

  // Force l00 > 0: if a00 < 0, flip sign of L row 0 and Q column 0.
  double l00 = a00;
  if (l00 < 0.0) {
    l00 = -l00;
    Q(0,0) = -Q(0,0); Q(1,0) = -Q(1,0); Q(2,0) = -Q(2,0);
  }

  return {l00, a10, a20, l11, a21, l22};
}

inline Mat3 transpose(const Mat3 &A)
{
  Mat3 T;
  MathExtra::transpose3(A.d, T.d);
  return T;
}

inline Mat3 mat_mul(const Mat3 &R, const Mat3 &S)
{
  Mat3 RS;
  MathExtra::times3(R.d, S.d, RS.d);
  return RS;
}

struct LDU3 {
  double d0, d1, d2, u01, u02, u12;

  static LDU3 load(const double *p) {
    return {p[0], p[1], p[2], p[3], p[4], p[5]};
  }
  void store(double *p) const {
    p[0] = d0; p[1] = d1; p[2] = d2;
    p[3] = u01; p[4] = u02; p[5] = u12;
  }
};

inline void UsL3(SymMat3 &S, const LDU3 &ldu)
{
  // s <- upper tri of Us
  S.d00 += ldu.u01 * S.d01 + ldu.u02 * S.d02;
  S.d01 += ldu.u01 * S.d11 + ldu.u02 * S.d12;
  S.d02 += ldu.u01 * S.d12 + ldu.u02 * S.d22;
  S.d11 += ldu.u12 * S.d12;
  S.d12 += ldu.u12 * S.d22;
  // Us <- UsL
  S.d00 += S.d01 * ldu.u01 + S.d02 * ldu.u02;
  S.d01 += S.d02 * ldu.u12;
  S.d11 += S.d12 * ldu.u12;
}

inline UTMat3 mul_du(const UTMat3 &inputU, const LDU3 &ldu)
{
  UTMat3 U = inputU;
  U.u00 *= ldu.d0; U.u01 *= ldu.d1; U.u02 *= ldu.d2;
  U.u11 *= ldu.d1; U.u12 *= ldu.d2; U.u22 *= ldu.d2;
  U.u02 += U.u00 * ldu.u02 + U.u01 * ldu.u12;
  U.u12 += U.u11 * ldu.u12;
  U.u01 += U.u00 * ldu.u01;
  return U;
}

inline Mat3 operator*(const Mat3 &inputM, const LDU3 &ldu) {
  Mat3 M = inputM;
  for (int i = 0; i < 3; i++) {
    M(i, 0) += M(i, 1) * ldu.u01 + M(i, 2) * ldu.u02;
    M(i, 0) *= ldu.d0;
    M(i, 1) += M(i, 2) * ldu.u12;
    M(i, 1) *= ldu.d1;
    M(i, 2) *= ldu.d2;
    M(i, 2) += M(i, 1) * ldu.u12 + M(i, 0) * ldu.u02;
    M(i, 1) += M(i, 0) * ldu.u01;
  }
  return M;
}

inline LTDL3 ltdl_pivot_one(const SymMat3 &A, int p)
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

// Fully-pivoted LDL^T factorization of 3x3 symmetric matrix A.
// Returns P A P^T = L D L^T in compact LTDL3 form, where:
//   - D = diag(d0, d1, d2) is the diagonal
//   - L is unit lower-triangular with sub-diagonal entries m01, m02, m12
//   - P is the row/column permutation recorded in perm[]
// Pivoting selects the largest |diag| at each elimination step for stability.
// Negative d2 is clamped to zero (positive semi-definite projection).
inline LTDL3 ltdl_pivot3(const SymMat3 &A, int perm[3])
{
  perm[0] = 0; perm[1] = 1; perm[2] = 2;

  double a00 = A.d00, a01 = A.d01, a02 = A.d02;
  double a11 = A.d11, a12 = A.d12;
  double a22 = A.d22;

  // Pivot 1: swap largest |diag| to position 0
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

  // Eliminate column 0: compute d0, multipliers m01, m02, and Schur complement
  double d0 = a00;
  double m01 = a01 / d0;
  double m02 = a02 / d0;

  double c11 = a11 - m01 * a01;
  double c12 = a12 - m01 * a02;
  double c22 = a22 - m02 * a02;

  // Pivot 2: swap larger |diag| of 2x2 Schur complement to position 1
  if (std::fabs(c22) > std::fabs(c11)) {
    int t = perm[1]; perm[1] = perm[2]; perm[2] = t;
    double tmp = c11; c11 = c22; c22 = tmp;
    tmp = m01; m01 = m02; m02 = tmp;
  }

  // Eliminate column 1: complete the factorization
  double d1 = c11;
  double m12 = c12 / d1;
  double d2 = c22 - m12 * c12;
  if (d2 < 0.0) d2 = 0.0; // clamp for positive semi-definiteness

  return {d0, d1, d2, m01, m02, m12};
}

inline LTMat3 chol_lower(const SymMat3 &A)
{
  double l00 = sqrt(A.d00);
  double l10 = A.d01 / l00;
  double l20 = A.d02 / l00;
  double l11 = sqrt(A.d11 - l10 * l10);
  double l21 = (A.d12 - l10 * l20) / l11;
  double l22 = sqrt(A.d22 - l20 * l20 - l21 * l21);
  return {l00, l10, l20, l11, l21, l22};
}

inline UTMat3 chol_upper(const SymMat3 &S) {
  LTMat3 L = chol_lower(S);
  return {L.l00, L.l10, L.l20, L.l11, L.l21, L.l22};
}

inline void skew(const Mat3 &A, double out[3])
{
  out[0] = A(2, 1) - A(1, 2);
  out[1] = A(0, 2) - A(2, 0);
  out[2] = A(1, 0) - A(0, 1);
}

inline LDU3 ldu3(const SymMat3 &A)
{
  double u01 = A.d01 / A.d00;
  double u02 = A.d02 / A.d00;
  double d1 = A.d11 - u01 * A.d01;
  double u12 = (A.d12 - A.d01 * u02) / d1;
  double d2 = A.d22 - A.d00 * u02 * u02 - d1 * u12 * u12;
  return {A.d00, d1, d2, u01, u02, u12};
}

inline void cayley_rotate(const double v[3], ColMat3 &A)
{
    double w = sqrt(1.0 - MathExtra::lensq3(v));

    for (int j = 0; j < 3; j++) {
      double* col = A(j);
      double cross1[3], cross2[3];
      MathExtra::cross3(v, col, cross1);
      MathExtra::cross3(v, cross1, cross2);
    col[0] += 2.0 * w * cross1[0] + 2.0 * cross2[0];
    col[1] += 2.0 * w * cross1[1] + 2.0 * cross2[1];
    col[2] += 2.0 * w * cross1[2] + 2.0 * cross2[2];
  }
}

inline void diag_of_product(const LTMat3 &rc, const UTMat3 &sc, double out[3]){
  out[0] = rc.l00*sc.u00;
  out[1] = rc.l11*sc.u11;
  out[2] = rc.l22*sc.u22;
}

inline void diag_of_product(const UTMat3 &rc, const LTMat3 &sc, double out[3]){
  out[0] = rc.u00*sc.l00;
  out[1] = rc.u11*sc.l11;
  out[2] = rc.u22*sc.l22;
}

inline void negskew_lt_mul(const LTMat3 &rc, const ColMat3 &sc, double out[3])
// return -skew(rc * sc)
{
  double g10 = rc.l10 * sc(0, 0) + rc.l11 * sc(1, 0);
  double g01 = rc.l00 * sc(0, 1);
  double g02 = rc.l00 * sc(0, 2);
  double g20 = rc.l20 * sc(0, 0) + rc.l21 * sc(1, 0) + rc.l22 * sc(2, 0);
  double g21 = rc.l20 * sc(0, 1) + rc.l21 * sc(1, 1) + rc.l22 * sc(2, 1);
  double g12 = rc.l10 * sc(0, 2) + rc.l11 * sc(1, 2);
  out[0] = g12 - g21;
  out[1] = g20 - g02;
  out[2] = g01 - g10;
}

// J(A, B) where A is lower-tri, B is upper-tri.
// Result is upper-triangular (document: Case 2). Returns J^{-1} (Cayley
// factor of 2 included, then inverted).
inline UTMat3 cayley_jacobian(const LTMat3 &A, const UTMat3 &B)
{
  // A entries (1-indexed): a11=l00, a21=l10, a31=l20, a22=l11, a32=l21, a33=l22
  // B entries (1-indexed): b11=u00, b12=u01, b13=u02, b22=u11, b23=u12, b33=u22
  // J = 1/2 * upper-tri, rows indexed by crossvec (i,j)=(2,3),(3,1),(1,2):
  //   u00 = a22*b33 + a33*b22
  //   u01 = -a21*b33 - a33*b12     u02 = a21*b23 - a22*b13 - a31*b22 + a32*b12
  //   u11 = a33*b11 + a11*b33      u12 = -a32*b11 - a11*b23
  //   u22 = a11*b22 + a22*b11
  UTMat3 G;
  G.u00 = 2.0 * (A.l11 * B.u22 + A.l22 * B.u11);
  G.u01 = 2.0 * (-A.l10 * B.u22 - A.l22 * B.u01);
  G.u02 = 2.0 * (A.l10 * B.u12 - A.l11 * B.u02 - A.l20 * B.u11 + A.l21 * B.u01);
  G.u11 = 2.0 * (A.l22 * B.u00 + A.l00 * B.u22);
  G.u12 = 2.0 * (-A.l21 * B.u00 - A.l00 * B.u12);
  G.u22 = 2.0 * (A.l00 * B.u11 + A.l11 * B.u00);
  G.invert();
  return G;
}

inline Mat3 cayley_converge(const LTMat3 &rc, const UTMat3 &sc, const Mat3 &chi,
                            int max_iters = 10, double tol = 1e-6, int *niter_out = nullptr)
{
  UTMat3 G_ut = cayley_jacobian(rc, sc);
  Mat3 G = G_ut;
  ColMat3 scm = sc;

  double skewChi[3];
  skew(chi, skewChi);

  double tol_sq = 3.0 * tol * tol;

  double negSkewGam[3], rotvec[3];
  int niter;
  for (niter = 0; niter < max_iters; niter++) {
    negskew_lt_mul(rc, scm, negSkewGam);
    negSkewGam[0] -= skewChi[0];
    negSkewGam[1] -= skewChi[1];
    negSkewGam[2] -= skewChi[2];
    if (MathExtra::lensq3(negSkewGam) < tol_sq) break;
    G.mat_vec(negSkewGam, rotvec);
    double rvsq = MathExtra::lensq3(rotvec);
    if (rvsq > 1.0) {
      double rv = 1.001*sqrt(rvsq);
      rotvec[0] /= rv;
      rotvec[1] /= rv;
      rotvec[2] /= rv;
    }
    cayley_rotate(rotvec, scm);
  }

  if (niter_out) *niter_out = niter;
  Mat3 gamma = scm;
  l_mul(rc, gamma);
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
  L(2,0) =  0.0; L(2,1) = L(2,2) = -1.0;
  L(3,0) = L(3,1) = 0.0;  L(3,2) =  1.0;
  L *= lam;
  return L;
}

inline SymMat3 mass_matrix4(double *m)
{
  double mutot = 1. / (m[0] + m[1] + m[2] + m[3]);
  double u1 = m[1] * mutot;
  double u2 = m[2] * mutot;
  double u3 = m[3] * mutot;
  return SymMat3 { m[1] * (1. - u1), -m[1] * u2, -m[1] * u3,
                               m[2] * (1. - u2), -m[2] * u3,
                                            m[3] * (1. - u3)};
}

}

#endif