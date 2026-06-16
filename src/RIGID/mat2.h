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
  
  Mat2 operator*(const LTMat2 &L) const {
    Mat2 R;
    R(0, 0) = d[0][0] * L.l00 + d[0][1] * L.l10;
    R(0, 1) = d[0][1] * L.l11;
    R(1, 0) = d[1][0] * L.l00 + d[1][1] * L.l10;
    R(1, 1) = d[1][1] * L.l11;
    return R;
  }
};

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

// r1·r1, r1·r2, r2·r2
inline SymMat2 sym_dot(const double r1[3], const double r2[3])
{
  return {r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2],
          r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2],
          r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]};
}

inline SymMat2 sym_dot(const Vec3 &r1, const Vec3 &r2)
{
  return {dot(r1, r1), dot(r1, r2), dot(r2, r2)};
}

inline SymMat2 mmt(const Mat2 &M)
{
  return {M(0, 0) * M(0, 0) + M(0, 1) * M(0, 1),
	  M(0, 0) * M(1, 0) + M(0, 1) * M(1, 1),
	  M(1, 0) * M(1, 0) + M(1, 1) * M(1, 1)};
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

inline UTMat2 inv_chol_upper(const SymMat2 &A)
{
  double u00 = sqrt(A.d00);
  double u01 = A.d01 / u00;
  double u11 = sqrt(A.d11 - u01 * u01);
  return {1.0 / u00, -u01 / (u00 * u11), 1.0 / u11};
}

inline LTMat2 chol_lower(const SymMat2 &A)
{
  double l00 = sqrt(A.d00);
  double l10 = A.d01 / l00;
  double l11 = sqrt(A.d11 - l10 * l10);
  return {l00, l10, l11};
}

// Decomposition A = L^T L (NOT L L^T).
// Conceptually equivalent to a U^T U followed by
// Givens-rotating U into L, but the explicit Givens-rotations
// are unstable. We do this so phi turns out lower triangular.
inline LTMat2 chol_to_ltl_lower(const SymMat2 &A)
{
  double l11 = sqrt(A.d11);
  double l10 = A.d01 / l11;
  double l00 = sqrt(A.d00 - l10 * l10);
  return {l00, l10, l11};
}

struct LTDL2 {
  double d0, d1, l10;
};

inline LTDL2 invert_to_ltdl(const SymMat2 &A)
{ // stores {d0, d1, l10} such that
  // A^(-1) = [[1, l10], [0, 1]][[d0, 0], [0, d1]][[1, 0], [l10, 1]]
  // by calculating A = LDLT and then storing inverses
  double l10 = A.d01 / A.d00;
  double d1 = A.d11 - l10 * A.d01;
  return {1/A.d00, 1/d1, -l10};
}

// S ← L S Lᵀ where L is the unit lower-triangular part of the LTDL struct.
// (When called with inv_ltdl output, L stores L⁻¹ so this computes L⁻¹ S L⁻ᵀ.)
inline void lt_sandwich(SymMat2 &S, const LTDL2 &L)
{
  S.d11 += S.d01 * L.l10;
  S.d01 += S.d00 * L.l10;
  S.d11 += S.d01 * L.l10;
}

inline void l_mul(LTMat2 &A, const LTDL2 &L)
{ // A <- L A
  A.l10 += A.l00 * L.l10;
}

inline LTMat2 mul_dl(const LTMat2 &A, const LTDL2 &L)
{
  double l00 = A.l00 * L.d0;
  double l11 = A.l11 * L.d1;
  double l10 = A.l10 * L.d0 + l11 * L.l10;
  return {l00, l10, l11};
}

inline Mat2 rmul_ltdl(const Mat2 &inputM, const LTDL2 &L)
{
  Mat2 M = inputM;
  M(0, 1) += M(0, 0) * L.l10;
  M(1, 1) += M(1, 0) * L.l10;
  M(0, 0) *= L.d0;
  M(1, 0) *= L.d0;
  M(0, 1) *= L.d1;
  M(1, 1) *= L.d1;
  M(0, 0) += M(0, 1) * L.l10;
  M(1, 0) += M(1, 1) * L.l10;
  return M;
}

}
#endif
