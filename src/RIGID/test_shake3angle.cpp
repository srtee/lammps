#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ===================== Current version (HEAD) =====================
// Uses Cholesky factor types (UTMat2, LTMat2) with chol_left_invmult,
// chol_right_mult, chol_sandwich, etc.

namespace RigsMath {

struct LTMat2 {
  double l00, l10, l11;
};

struct SymMat2 {
  double d00, d01, d11;
  SymMat2 operator+(const SymMat2 &B) const { return {d00 + B.d00, d01 + B.d01, d11 + B.d11}; }
  SymMat2 operator-(const SymMat2 &B) const { return {d00 - B.d00, d01 - B.d01, d11 - B.d11}; }
  SymMat2& operator*=(const SymMat2 &S2)
  {
  // S <- S * S2
  double i00 = d00 * S2.d00 + d01 * S2.d01;
  double i01 = d00 * S2.d01 + d01 * S2.d11;
  double i10 = d01 * S2.d00 + d11 * S2.d01;
  double i11 = d01 * S2.d01 + d11 * S2.d11;
  // S <- S2 * S
  d00 = S2.d00 * i00 + S2.d01 * i10;
  d01 = S2.d00 * i01 + S2.d01 * i11;
  d11 = S2.d01 * i01 + S2.d11 * i11;
  return *this;
  } 
  SymMat2& operator/=(const LTMat2 &L)
{
  // S <- S * L^T
  double i00 = d00 * L.l00;
  double i01 = d00 * L.l10 + d01 * L.l11;
  double i11 = d01 * L.l10 + d11 * L.l11;
  // S <- L * S
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


struct UTMat2 {
  double u00, u01, u11;

  void invert() {
    u00 = 1. / u00;
    u11 = 1. / u11;
    u01 = -u01 * u00 * u11;
  }
};

inline SymMat2 sym_dot(const double r1[3], const double r2[3])
{
  return {r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2],
          r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2],
          r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]};
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

inline Mat2 to_mat(const LTMat2 &L)
{
  Mat2 R;
  R(0, 0) = L.l00; R(0, 1) = 0.0;
  R(1, 0) = L.l10; R(1, 1) = L.l11;
  return R;
}

inline SymMat2 chol_sandwich(const SymMat2 &S, const LTMat2 &L)
{
  return {L.l00 * L.l00 * S.d00,
           L.l00 * L.l10 * S.d00 + L.l00 * L.l11 * S.d01,
           L.l10 * L.l10 * S.d00 + 2.0 * L.l10 * L.l11 * S.d01 + L.l11 * L.l11 * S.d11};
}

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

inline void chol_right_mult(Mat2 &M, const UTMat2 &U)
{
  for (int i = 0; i < 2; i++) {
    double t0 = M(i, 0) * U.u00 + M(i, 1) * U.u01;
    double t1 = M(i, 1) * U.u11;
    M(i, 0) = t0 * U.u00;
    M(i, 1) = t0 * U.u01 + t1 * U.u11;
  }
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

inline LTMat2 operator*(const LTMat2 &A, const LTMat2 &B)
{
  return {A.l00 * B.l00,
          A.l10 * B.l00 + A.l11 * B.l10,
          A.l11 * B.l11};
}

} // namespace RigsMath

// ===================== Old version helpers (HEAD~3) =====================

namespace OldMath {

inline RigsMath::SymMat2 inv_sym(const RigsMath::SymMat2 &A)
{
  double det = A.d00 * A.d11 - A.d01 * A.d01;
  return {A.d11 / det, -A.d01 / det, A.d00 / det};
}

inline RigsMath::SymMat2 sandwich(const RigsMath::SymMat2 &M, const RigsMath::SymMat2 &A)
{
  double MA00 = M.d00 * A.d00 + M.d01 * A.d01;
  double MA01 = M.d00 * A.d01 + M.d01 * A.d11;
  double MA10 = M.d01 * A.d00 + M.d11 * A.d01;
  double MA11 = M.d01 * A.d01 + M.d11 * A.d11;
  return {MA00 * M.d00 + MA01 * M.d01,
          MA00 * M.d01 + MA01 * M.d11,
          MA10 * M.d01 + MA11 * M.d11 };
}

inline RigsMath::Mat2 chol_lower(const RigsMath::SymMat2 &A)
{
  double l11 = sqrt(A.d11);
  double l10 = A.d01 / l11;
  double l00 = sqrt(A.d00 - l10 * l10);
  RigsMath::Mat2 R;
  R(0, 0) = l00; R(0, 1) = 0.0;
  R(1, 0) = l10; R(1, 1) = l11;
  return R;
}

inline RigsMath::Mat2 inv_chol_upper_mat(const RigsMath::SymMat2 &A)
{
  double u00 = sqrt(A.d00);
  double u01 = A.d01 / u00;
  double u11 = sqrt(A.d11 - u01 * u01);
  double inv00 = 1.0 / u00;
  double inv11 = 1.0 / u11;
  double inv01 = -u01 * inv00 * inv11;
  RigsMath::Mat2 R;
  R(0, 0) = inv00; R(0, 1) = inv01;
  R(1, 0) = 0.0;   R(1, 1) = inv11;
  return R;
}

} // namespace OldMath

using namespace RigsMath;
using namespace OldMath;

// ===================== Current algorithm =====================

void compute_lamda_current(const double r01[3], const double r02[3],
                            const double s01[3], const double s02[3],
                            double bond1_sq, double bond2_sq, double bond12,
                            double invmass0, double invmass01, double invmass02,
                            double &lamda01, double &lamda02, double &lamda12,
                            double chi_out[4], double sc_out[4], double sigma_out[3])
{
  SymMat2 rr = sym_dot(r01, r02);
  SymMat2 ss = sym_dot(s01, s02);

  SymMat2 L = {bond1_sq, bond12, bond2_sq};
  SymMat2 diff = L - ss;

  Mat2 RS;
  RS(0, 0) = s01[0] * r01[0] + s01[1] * r01[1] + s01[2] * r01[2];
  RS(1, 0) = s01[0] * r02[0] + s01[1] * r02[1] + s01[2] * r02[2];
  RS(0, 1) = s02[0] * r01[0] + s02[1] * r01[1] + s02[2] * r01[2];
  RS(1, 1) = s02[0] * r02[0] + s02[1] * r02[1] + s02[2] * r02[2];

  SymMat2 mass_inv = inv_sym(SymMat2{invmass01, invmass0, invmass02});

  UTMat2 mu = inv_chol_upper(SymMat2{invmass01, invmass0, invmass02});
  LTMat2 lm = {mu.u00, mu.u01, mu.u11};

  UTMat2 rc = chol_upper(rr);
  Mat2 RS_save = RS;
  chol_left_invmult(rc, RS);
  SymMat2 sigma = diff + mat_mul_tosym(transpose(RS_save), RS);
  //sigma = sandwich(mass_inv, sigma);
  sigma /= lm;
  //sigma *= mass_inv;
  rc.invert();
  sigma_out[0] = sigma.d00; sigma_out[1] = sigma.d01; sigma_out[2] = sigma.d11;

  //LTMat2 sc_lt = chol_lower_lt(sigma);
  LTMat2 sc_lt = chol_lower_lt(sigma) * lm;
  Mat2 sc_mat = to_mat(sc_lt);
  Mat2 chi = RS;
  chol_right_mult(chi, lm);

  sc_out[0] = sc_lt.l00; sc_out[1] = sc_lt.l10; sc_out[2] = 0.0; sc_out[3] = sc_lt.l11;

  chi_out[0] = chi(0, 0); chi_out[1] = chi(0, 1);
  chi_out[2] = chi(1, 0); chi_out[3] = chi(1, 1);

  Mat2 phiC = rc * sc_mat;

  Mat2 J;
  J(0, 0) = 0.0;  J(0, 1) = -1.0;
  J(1, 0) = 1.0;  J(1, 1) = 0.0;
  Mat2 phiS = rc * J * sc_mat;

  double skewC = skew(phiC);
  double skewChi = skew(chi);
  double skewS = skew(phiS);

  double Asq = skewC * skewC + skewS * skewS;
  double sinp = sqrt(Asq - skewChi * skewChi);
  double sskew = -(skewS * skewChi + skewC * sinp) / Asq;
  double cskew = (skewS * sinp - skewChi * skewC) / Asq;

  lamda01 = chi(0, 0) + cskew * phiC(0, 0) + sskew * phiS(0, 0);
  lamda02 = chi(1, 1) + cskew * phiC(1, 1);
  lamda12 = chi(0, 1) + cskew * phiC(0, 1) + sskew * phiS(0, 1);
}

// ===================== Old algorithm (HEAD~3) =====================

void compute_lamda_old(const double r01[3], const double r02[3],
                        const double s01[3], const double s02[3],
                        double bond1_sq, double bond2_sq, double bond12,
                        double invmass0, double invmass01, double invmass02,
                        double &lamda01, double &lamda02, double &lamda12,
                        double chi_out[4], double sc_out[4], double sigma_out[3])
{
  SymMat2 rr = sym_dot(r01, r02);
  SymMat2 ss = sym_dot(s01, s02);

  SymMat2 L = {bond1_sq, bond12, bond2_sq};
  SymMat2 diff = L - ss;

  Mat2 RS;
  RS(0, 0) = s01[0] * r01[0] + s01[1] * r01[1] + s01[2] * r01[2];
  RS(1, 0) = s01[0] * r02[0] + s01[1] * r02[1] + s01[2] * r02[2];
  RS(0, 1) = s02[0] * r01[0] + s02[1] * r01[1] + s02[2] * r01[2];
  RS(1, 1) = s02[0] * r02[0] + s02[1] * r02[1] + s02[2] * r02[2];

  SymMat2 M = inv_sym(SymMat2{invmass01, invmass0, invmass02});

  SymMat2 D = sandwich(M, diff);

  Mat2 K = RS * M;

  SymMat2 rh = inv_sym(rr);

  Mat2 chi = rh * K;
  SymMat2 sigma = mat_mul_tosym(transpose(K), chi) + D;

  Mat2 chiKT = transpose(K) * chi;

  chi_out[0] = chi(0, 0); chi_out[1] = chi(0, 1);
  chi_out[2] = chi(1, 0); chi_out[3] = chi(1, 1);

  Mat2 sc = chol_lower(sigma);

  sc_out[0] = sc(0, 0); sc_out[1] = sc(1, 0); sc_out[2] = sc(0, 1); sc_out[3] = sc(1, 1);

  sigma_out[0] = sigma.d00; sigma_out[1] = sigma.d01; sigma_out[2] = sigma.d11;

  Mat2 rc = inv_chol_upper_mat(rr);

  Mat2 phiC = rc * sc;

  double phiS11 = rc(0, 1) * sc(0, 0) - rc(0, 0) * sc(1, 0);
  double phiS12 = -rc(0, 0) * sc(1, 1);
  double phiS21 = rc(1, 1) * sc(0, 0);

  double skewC = skew(phiC);
  double skewChi = skew(chi);
  double skewS = phiS12 - phiS21;

  double Asq = skewC * skewC + skewS * skewS;
  double sinp = sqrt(Asq - skewChi * skewChi);
  double sskew = -(skewS * skewChi + skewC * sinp) / Asq;
  double cskew = (skewS * sinp - skewChi * skewC) / Asq;

  lamda01 = chi(0, 0) + cskew * phiC(0, 0) + sskew * phiS11;
  lamda02 = chi(1, 1) + cskew * phiC(1, 1);
  lamda12 = chi(0, 1) + cskew * phiC(0, 1) + sskew * phiS12;
}

// ===================== Test harness =====================

// Simple LCG for reproducible random perturbations
static unsigned long lcg_state = 12345;
double rand_perturb()
{
  lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return ((double)(lcg_state >> 33) / (double)(1ULL << 31)) * 2.0 - 1.0; // in [-1, 1)
}

int main(int argc, char **argv)
{
  int ntrials = 100;
  if (argc > 1) ntrials = atoi(argv[1]);

  double r0[3] = {0.0, 0.0, 0.0};
  double R1[3] = {1.0, 0.0, 0.0};
  double R2[3] = {-0.5, 0.8660254037844386, 0.0}; // 120-degree angle

  double r01[3] = {R1[0] - r0[0], R1[1] - r0[1], R1[2] - r0[2]};
  double r02[3] = {R2[0] - r0[0], R2[1] - r0[1], R2[2] - r0[2]};

  // Constraint targets from equilibrium geometry
  double bond1 = sqrt(r01[0]*r01[0] + r01[1]*r01[1] + r01[2]*r01[2]);
  double bond2 = sqrt(r02[0]*r02[0] + r02[1]*r02[1] + r02[2]*r02[2]);
  double bond12 = r01[0]*r02[0] + r01[1]*r02[1] + r01[2]*r02[2]; // dot product

  // Masses: heavy center atom (mass=16), light partners (mass=1)
  double dtfsq = 1.0;
  double invmass0  = dtfsq / 16.0;           // 1/16
  double invmass01 = invmass0 + dtfsq / 1.0;  // 1 + 1/16
  double invmass02 = invmass0 + dtfsq / 1.0;  // 1 + 1/16

  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
  double max_chi_diff = 0.0;
  double max_sc_diff = 0.0;
  double max_sigma_diff = 0.0;
  double max_sst_diff = 0.0;

  printf("=== shake3angle comparison: %d perturbed trials ===\n\n", ntrials);
  printf("invmass0 = %.15e, invmass01 = %.15e, invmass02 = %.15e\n\n",
         invmass0, invmass01, invmass02);
  printf("%-4s  %-22s %-22s %-22s | %-22s %-22s %-22s | %-12s %-12s %-12s | %-12s\n",
         "#", "lamda01_cur", "lamda02_cur", "lamda12_cur",
         "lamda01_old", "lamda02_old", "lamda12_old",
         "max|dlam|", "max|dchi|", "...", "chi_match?");

  for (int trial = 0; trial < ntrials; trial++) {
    double s01[3], s02[3];

    s01[0] = r01[0] + 0.05 * rand_perturb();
    s01[1] = r01[1] + 0.05 * rand_perturb();
    s01[2] = r01[2] + 0.05 * rand_perturb();

    s02[0] = r02[0] + 0.05 * rand_perturb();
    s02[1] = r02[1] + 0.05 * rand_perturb();
    s02[2] = r02[2] + 0.05 * rand_perturb();

    double lam01_c, lam02_c, lam12_c, chi_c[4], sc_c[4], sigma_c[3];
    compute_lamda_current(r01, r02, s01, s02,
                          bond1 * bond1, bond2 * bond2, bond12,
                          invmass0, invmass01, invmass02,
                          lam01_c, lam02_c, lam12_c, chi_c, sc_c, sigma_c);

    double lam01_o, lam02_o, lam12_o, chi_o[4], sc_o[4], sigma_o[3];
    compute_lamda_old(r01, r02, s01, s02,
                      bond1 * bond1, bond2 * bond2, bond12,
                      invmass0, invmass01, invmass02,
                      lam01_o, lam02_o, lam12_o, chi_o, sc_o, sigma_o);

    // Check inv_sym(rr)*RS vs chol_left_invmult(rc, RS)
    {
      SymMat2 rr = sym_dot(r01, r02);
      Mat2 RS;
      RS(0, 0) = s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2];
      RS(1, 0) = s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2];
      RS(0, 1) = s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2];
      RS(1, 1) = s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2];

      SymMat2 rh = inv_sym(rr);
      Mat2 chi_rh = rh * RS;

      Mat2 RS2 = RS;
      UTMat2 rc = chol_upper(rr);
      chol_left_invmult(rc, RS2);

      double d00 = chi_rh(0,0) - RS2(0,0);
      double d01 = chi_rh(0,1) - RS2(0,1);
      double d10 = chi_rh(1,0) - RS2(1,0);
      double d11 = chi_rh(1,1) - RS2(1,1);
      double max_d = fabs(d00);
      if (fabs(d01) > max_d) max_d = fabs(d01);
      if (fabs(d10) > max_d) max_d = fabs(d10);
      if (fabs(d11) > max_d) max_d = fabs(d11);
      printf("  inv_sym(rr)*RS vs chol_left_invmult(rc,RS): max_diff = %.6e\n", max_d);
    }

    // Check chol_right_mult(chi, lm) vs chi * M
    {
      SymMat2 rr_check = sym_dot(r01, r02);
      Mat2 RS_check;
      RS_check(0, 0) = s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2];
      RS_check(1, 0) = s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2];
      RS_check(0, 1) = s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2];
      RS_check(1, 1) = s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2];
      UTMat2 rc_check = chol_upper(rr_check);
      chol_left_invmult(rc_check, RS_check);
      // RS_check = rho^{-1} RS
      SymMat2 mass_inv_check = inv_sym(SymMat2{invmass01, invmass0, invmass02});
      Mat2 M_check;
      M_check(0, 0) = mass_inv_check.d00; M_check(0, 1) = mass_inv_check.d01;
      M_check(1, 0) = mass_inv_check.d01; M_check(1, 1) = mass_inv_check.d11;
      Mat2 chi_direct = RS_check * M_check;
      double d00 = chi_c[0] - chi_direct(0,0);
      double d01 = chi_c[1] - chi_direct(0,1);
      double d10 = chi_c[2] - chi_direct(1,0);
      double d11 = chi_c[3] - chi_direct(1,1);
      double max_d = fabs(d00);
      if (fabs(d01) > max_d) max_d = fabs(d01);
      if (fabs(d10) > max_d) max_d = fabs(d10);
      if (fabs(d11) > max_d) max_d = fabs(d11);
      printf("  chol_right_mult(chi,lm) vs chi*M: max_diff = %.6e\n", max_d);
    }

    // Check: lm * lm^T should equal M for the mass matrix
    // mu = inv_chol_upper of {invmass01, invmass0, invmass02}
    // lm = {mu.u00, mu.u01, mu.u11} as LTMat2
    // lm * lm^T = M (since lm = mu^T and mu*mu^T = M? or mu^T*mu = M?)
    {
      double u00 = sqrt(invmass01);
      double u01 = invmass0 / u00;
      double u11 = sqrt(invmass02 - u01*u01);
      double inv00 = 1.0/u00, inv11 = 1.0/u11;
      double inv01 = -u01*inv00*inv11;
      // mu = {inv00, inv01, inv11} as UTMat2
      // lm = {inv00, inv01, inv11} as LTMat2 (transposed!)
      // So lm represents: [inv00, 0; inv01, inv11] (lower triangular)
      // lm^T = [inv00, inv01; 0, inv11] = mu (upper triangular)
      // lm * lm^T: lower * upper = [inv00,0;inv01,inv11]*[inv00,inv01;0,inv11]
      //  = [inv00^2, inv00*inv01; inv01*inv00, inv01^2+inv11^2]
      double M00 = inv00*inv00;
      double M01 = inv00*inv01;
      double M10 = inv01*inv00;
      double M11 = inv01*inv01 + inv11*inv11;
      double M_expected00 = invmass02 / (invmass01*invmass02 - invmass0*invmass0);
      double M_expected01 = -invmass0 / (invmass01*invmass02 - invmass0*invmass0);
      double M_expected11 = invmass01 / (invmass01*invmass02 - invmass0*invmass0);
      printf("  lm*lm^T:       [%.15e, %.15e; %.15e, %.15e]\n",
             M00, M01, M10, M11);
      printf("  inv_sym(M):    [%.15e, %.15e; %.15e, %.15e]\n",
             M_expected00, M_expected01, M_expected01, M_expected11);
    }


    {
      SymMat2 rr = sym_dot(r01, r02);
      SymMat2 ss = sym_dot(s01, s02);
      double b1sq = bond1*bond1, b2sq = bond2*bond2;
      SymMat2 L = {b1sq, bond12, b2sq};
      SymMat2 diff = L - ss;
      Mat2 RS;
      RS(0, 0) = s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2];
      RS(1, 0) = s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2];
      RS(0, 1) = s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2];
      RS(1, 1) = s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2];
      UTMat2 rc = chol_upper(rr);
      Mat2 RS_orig = RS;
      chol_left_invmult(rc, RS);
      // RS = rho^{-1} RS_orig, RS_orig saved
      SymMat2 sigma_before = diff + mat_mul_tosym(transpose(RS_orig), RS);
      SymMat2 mass_inv = inv_sym(SymMat2{invmass01, invmass0, invmass02});
      SymMat2 MsigmaM = sandwich(mass_inv, sigma_before);

      SymMat2 M_old = inv_sym(SymMat2{invmass01, invmass0, invmass02});
      Mat2 K = RS_orig * M_old;
      SymMat2 rh = inv_sym(rr);
      Mat2 chi_old = rh * K;
      SymMat2 sigma_old = mat_mul_tosym(transpose(K), chi_old) + sandwich(M_old, diff);

      double d00 = fabs(MsigmaM.d00 - sigma_old.d00);
      double d01 = fabs(MsigmaM.d01 - sigma_old.d01);
      double d11 = fabs(MsigmaM.d11 - sigma_old.d11);
      double max_d = d00;
      if (d01 > max_d) max_d = d01;
      if (d11 > max_d) max_d = d11;
      printf("  M*sigma_before*M vs sigma_old: max_diff = %.6e\n", max_d);
    }

    // Check: sc_new (L_s * L_m) should equal sc_old (chol_lower of M*sigma*M)
    // where L_s = chol_lower_lt(L_m * sigma * L_m^T), L_o = chol_lower(M * sigma * M)
    // Theory: L_s * L_m = L_o
    {
      SymMat2 rr = sym_dot(r01, r02);
      SymMat2 ss = sym_dot(s01, s02);
      double b1sq = bond1*bond1, b2sq = bond2*bond2;
      SymMat2 Lb = {b1sq, bond12, b2sq};
      SymMat2 diff = Lb - ss;
      Mat2 RS;
      RS(0, 0) = s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2];
      RS(1, 0) = s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2];
      RS(0, 1) = s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2];
      RS(1, 1) = s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2];
      UTMat2 rc = chol_upper(rr);
      Mat2 RS_orig = RS;
      chol_left_invmult(rc, RS);
      SymMat2 sigma_before = diff + mat_mul_tosym(transpose(RS_orig), RS);
      SymMat2 mass_inv = inv_sym(SymMat2{invmass01, invmass0, invmass02});
      SymMat2 MsigmaM = sandwich(mass_inv, sigma_before);

      // old sc = chol_lower(MsigmaM)
      Mat2 sc_old_mat = chol_lower(MsigmaM);

      // new sc = chol_lower_lt(chol_sandwich(sigma_before, lm)) * lm = L_s * L_m
      double ds00 = sc_c[0] - sc_old_mat(0,0);
      double ds01 = sc_c[1] - sc_old_mat(0,1);
      double ds10 = sc_c[2] - sc_old_mat(1,0);
      double ds11 = sc_c[3] - sc_old_mat(1,1);
      double max_d = fabs(ds00);
      if (fabs(ds01) > max_d) max_d = fabs(ds01);
      if (fabs(ds10) > max_d) max_d = fabs(ds10);
      if (fabs(ds11) > max_d) max_d = fabs(ds11);
      printf("  sc_new (L_s*L_m) vs sc_old (chol_lower(M*sigma*M)): max_diff = %.6e\n", max_d);
    }

    double d01 = lam01_c - lam01_o;
    double d02 = lam02_c - lam02_o;
    double d12 = lam12_c - lam12_o;
    double max_dlam = fabs(d01);
    if (fabs(d02) > max_dlam) max_dlam = fabs(d02);
    if (fabs(d12) > max_dlam) max_dlam = fabs(d12);

    double dc00 = chi_c[0] - chi_o[0];
    double dc01 = chi_c[1] - chi_o[1];
    double dc10 = chi_c[2] - chi_o[2];
    double dc11 = chi_c[3] - chi_o[3];
    double max_dchi = fabs(dc00);
    if (fabs(dc01) > max_dchi) max_dchi = fabs(dc01);
    if (fabs(dc10) > max_dchi) max_dchi = fabs(dc10);
    if (fabs(dc11) > max_dchi) max_dchi = fabs(dc11);

    double ds00 = sc_c[0] - sc_o[0];
    double ds01 = sc_c[1] - sc_o[1];
    double ds10 = sc_c[2] - sc_o[2];
    double ds11 = sc_c[3] - sc_o[3];
    double max_dsc = fabs(ds00);
    if (fabs(ds01) > max_dsc) max_dsc = fabs(ds01);
    if (fabs(ds10) > max_dsc) max_dsc = fabs(ds10);
    if (fabs(ds11) > max_dsc) max_dsc = fabs(ds11);

    double dsig00 = fabs(sigma_c[0] - sigma_o[0]);
    double dsig01 = fabs(sigma_c[1] - sigma_o[1]);
    double dsig11 = fabs(sigma_c[2] - sigma_o[2]);
    double max_dsig = dsig00;
    if (dsig01 > max_dsig) max_dsig = dsig01;
    if (dsig11 > max_dsig) max_dsig = dsig11;

    // Check sc * sc^T matches between old and new
    // new: sc is LTMat2 {l00, l10, l11}, so sc * sc^T:
    //   [l00^2,         l00*l10      ]
    //   [l10*l00,       l10^2+l11^2  ]
    // old: sc is Mat2, so sc * sc^T:
    //   [sc00^2+sc01^2,            sc00*sc10+sc01*sc11]
    //   [sc10*sc00+sc11*sc01,       sc10^2+sc11^2    ]
    double new_sst00 = sc_c[0]*sc_c[0];
    double new_sst01 = sc_c[0]*sc_c[1];
    double new_sst11 = sc_c[1]*sc_c[1] + sc_c[3]*sc_c[3];
    double old_sst00 = sc_o[0]*sc_o[0] + sc_o[2]*sc_o[2];
    double old_sst01 = sc_o[0]*sc_o[1] + sc_o[2]*sc_o[3];
    double old_sst11 = sc_o[1]*sc_o[1] + sc_o[3]*sc_o[3];
    double dsst00 = fabs(new_sst00 - old_sst00);
    double dsst01 = fabs(new_sst01 - old_sst01);
    double dsst11 = fabs(new_sst11 - old_sst11);
    double max_dsst = dsst00;
    if (dsst01 > max_dsst) max_dsst = dsst01;
    if (dsst11 > max_dsst) max_dsst = dsst11;
    if (max_dsst > max_sst_diff) max_sst_diff = max_dsst;

    if (max_dlam > max_abs_diff) max_abs_diff = max_dlam;
    if (max_dchi > max_chi_diff) max_chi_diff = max_dchi;
    if (max_dsc > max_sc_diff) max_sc_diff = max_dsc;

    double denom_cur = fabs(lam01_c) + fabs(lam02_c) + fabs(lam12_c);
    double denom_old = fabs(lam01_o) + fabs(lam02_o) + fabs(lam12_o);
    double denom = denom_cur > denom_old ? denom_cur : denom_old;
    double total_diff = fabs(d01) + fabs(d02) + fabs(d12);
    if (denom > 0) {
      double rd = total_diff / denom;
      if (rd > max_rel_diff) max_rel_diff = rd;
    }

    if (trial < 10 || ntrials <= 20) {
      printf("%-4d  %-22.15e %-22.15e %-22.15e | %-22.15e %-22.15e %-22.15e | %-12.6e %-12s %-12s | %s\n",
             trial, lam01_c, lam02_c, lam12_c, lam01_o, lam02_o, lam12_o,
             max_dlam, "", "",
             max_dchi < 1e-14 ? "MATCH" : "DIFFER");
      if (max_dchi >= 1e-14) {
        printf("     chi_cur: [%.15e, %.15e; %.15e, %.15e]\n",
               chi_c[0], chi_c[1], chi_c[2], chi_c[3]);
        printf("     chi_old: [%.15e, %.15e; %.15e, %.15e]\n",
               chi_o[0], chi_o[1], chi_o[2], chi_o[3]);
        printf("     chi_diff: [%.6e, %.6e; %.6e, %.6e]\n",
               dc00, dc01, dc10, dc11);
      }
    } else if (trial == 10 && ntrials > 20) {
      printf("... (showing first 10 of %d trials) ...\n", ntrials);
    }
  }

  { for (int i = 0; i < 190; i++) putchar('-'); putchar('\n'); }
  printf("Max absolute lamda diff: %.6e\n", max_abs_diff);
  printf("Max relative lamda diff:  %.6e\n", max_rel_diff);
  printf("Max absolute chi diff:    %.6e\n", max_chi_diff);
  printf("Max absolute sc diff:      %.6e\n", max_sc_diff);
  printf("Max absolute sc*sc^T diff:   %.6e\n", max_sst_diff);

  return 0;
}
