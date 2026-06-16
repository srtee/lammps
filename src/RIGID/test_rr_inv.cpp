#include <cmath>
#include <cstdio>

#include "mat2.h"
#include "vec3.h"

using namespace RigsMath;

int main()
{
  // (1) Random-ish Vec3s
  Vec3 r01 = {1.2, 0.3, -0.7};
  Vec3 r02 = {-0.4, 1.5, 0.9};

  // (2) rr = sym_dot(r01, r02)
  SymMat2 rr = sym_dot(r01, r02);
  printf("rr = {%.10e, %.10e, %.10e}\n", rr.d00, rr.d01, rr.d11);

  // (3) Compute rr_inv explicitly and check rr * rr_inv = I
  Vec3 n = cross(r01, r02);
  double nn = normsq(n);
  double n4 = nn * nn;

  SymMat2 rr_inv = {normsq(r02) / n4,
                    -dot(r01, r02) / n4,
                    normsq(r01) / n4};
  printf("rr_inv = {%.10e, %.10e, %.10e}\n", rr_inv.d00, rr_inv.d01, rr_inv.d11);
  printf("nn = %.10e\n", nn);

  // Check: rr * rr_inv should be I (2x2 identity)
  // (SymMat2 * SymMat2 is not directly defined, so do it elementwise)
  // rr = [a, b; b, c], rr_inv = [A, B; B, C]
  // product: [a*A + b*B,  a*B + b*C; b*A + c*B,  b*B + c*C]
  double a = rr.d00, b = rr.d01, c = rr.d11;
  double A = rr_inv.d00, B = rr_inv.d01, C = rr_inv.d11;
  double p00 = a * A + b * B;
  double p01 = a * B + b * C;
  double p10 = b * A + c * B;
  double p11 = b * B + c * C;
  printf("rr * rr_inv:\n");
  printf("  [%.15e, %.15e]\n", p00, p01);
  printf("  [%.15e, %.15e]\n", p10, p11);
  printf("  (should be identity)\n\n");

  // (4) Compute rnorm = inv_chol_upper(rr)
  UTMat2 rnorm = inv_chol_upper(rr);
  printf("rnorm = {%.10e, %.10e, %.10e}\n", rnorm.u00, rnorm.u01, rnorm.u11);

  // (5) Apply ut_mul then u_mul to identity matrix, check result equals rr_inv
  Mat2 I;
  I(0, 0) = 1.0; I(0, 1) = 0.0;
  I(1, 0) = 0.0; I(1, 1) = 1.0;

  Mat2 result = I;
  ut_mul(rnorm, result);
  u_mul(rnorm, result);

  printf("ut_mul(rnorm, I); u_mul(rnorm, I) = rnorm * I * rnorm^T:\n");
  printf("  [%.15e, %.15e]\n", result(0, 0), result(0, 1));
  printf("  [%.15e, %.15e]\n", result(1, 0), result(1, 1));

  printf("\nrr_inv (for comparison):\n");
  printf("  [%.15e, %.15e]\n", rr_inv.d00, rr_inv.d01);
  printf("  [%.15e, %.15e]\n", rr_inv.d01, rr_inv.d11);

  printf("\nDifference (result - rr_inv):\n");
  printf("  [%.15e, %.15e]\n", result(0, 0) - rr_inv.d00, result(0, 1) - rr_inv.d01);
  printf("  [%.15e, %.15e]\n", result(1, 0) - rr_inv.d01, result(1, 1) - rr_inv.d11);

  return 0;
}