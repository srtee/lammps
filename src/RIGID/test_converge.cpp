#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "mat3.h"

using namespace RigsMath;

static double randn()
{
  double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
  double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
  return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static UTMat3 make_ut_rc()
{
  UTMat3 rc;
  rc.u00 = 1.0 + 0.05 * randn();
  rc.u01 = 0.2 * randn();
  rc.u02 = 0.2 * randn();
  rc.u11 = 1.0 + 0.05 * randn();
  rc.u12 = 0.2 * randn();
  rc.u22 = 1.0 + 0.05 * randn();
  return rc;
}

static LTMat3 make_lt_sc()
{
  LTMat3 sc;
  sc.l00 = 1.0 + 0.05 * randn();
  sc.l10 = 0.2 * randn();
  sc.l20 = 0.2 * randn();
  sc.l11 = 1.0 + 0.05 * randn();
  sc.l21 = 0.2 * randn();
  sc.l22 = 1.0 + 0.05 * randn();
  return sc;
}

static Mat3 lt_to_mat(const LTMat3 &sc)
{
  Mat3 m;
  m(0,0)=sc.l00; m(0,1)=0;     m(0,2)=0;
  m(1,0)=sc.l10; m(1,1)=sc.l11; m(1,2)=0;
  m(2,0)=sc.l20; m(2,1)=sc.l21; m(2,2)=sc.l22;
  return m;
}

static Mat3 ut_to_mat(const UTMat3 &rc)
{
  Mat3 m;
  m(0,0)=rc.u00; m(0,1)=rc.u01; m(0,2)=rc.u02;
  m(1,0)=0;     m(1,1)=rc.u11; m(1,2)=rc.u12;
  m(2,0)=0;     m(2,1)=0;     m(2,2)=rc.u22;
  return m;
}

int main()
{
  srand(42);
  int ntrials = 200;
  double tol = 1e-12;

  for (double eps : {0.01, 0.05, 0.1, 0.2}) {
    int converged = 0;
    srand(42);
    for (int t = 0; t < ntrials; t++) {
      UTMat3 rc = make_ut_rc();
      LTMat3 sc_lt = make_lt_sc();
      Mat3 sc = lt_to_mat(sc_lt);

      double rv[3] = {eps*randn(), eps*randn(), eps*randn()};
      cayley_rotate(sc, rv);

      Mat3 chi;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          double s = 0.0;
          for (int k = 0; k < 3; k++) s += ut_to_mat(rc)(i,k) * sc(k,j);
          chi(i,j) = -s;
        }

      Mat3 sc_init = lt_to_mat(sc_lt);
      Mat3 gamma = cayley_converge(rc, sc_init, chi, 200, tol);

      double skewChi[3], skewGamma[3];
      skew(chi, skewChi);
      skew(gamma, skewGamma);
      double maxres = std::fmax(std::fmax(
        fabs(skewChi[0]+skewGamma[0]),
        fabs(skewChi[1]+skewGamma[1])),
        fabs(skewChi[2]+skewGamma[2]));
      if (!std::isnan(maxres) && maxres < 1e-6) converged++;
    }
    std::cout << "eps=" << eps << ": " << converged << "/" << ntrials << " converged\n";
  }

  return 0;
}