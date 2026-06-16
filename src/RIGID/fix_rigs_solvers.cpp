/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Authors: Shern Tee (GU), Stephen Sanderson (UQ)
------------------------------------------------------------------------- */

#include "fix_rigs.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "mat2.h"
#include "mat3.h"

#include <cmath>
#include <cstring>
#include <utils.h>

using namespace LAMMPS_NS;
using namespace RigsMath;

void FixRigs::lookup_or_compute_matrices()
{
  int nsize = nlist;
  if (nsize > (int)ilist_to_idx.size()) ilist_to_idx.resize(nsize);

  for (int ilist = 0; ilist < nlist; ilist++) {
    int m = list[ilist];
    int idx;
    if (shake_flag[m] == 1)
      idx = lookup_or_compute_angle(ilist);
    else if (shake_flag[m] == 5)
      idx = lookup_or_compute_improper(ilist);
    else if (shake_flag[m] == 6)
      idx = lookup_or_compute_dihedral(ilist);
    else
      continue;
    ilist_to_idx[ilist] = idx;
  }

  propagate_demoted_tags();
}

void FixRigs::propagate_demoted_tags()
{
  for (int ilist = 0; ilist < nlist; ilist++) {
    int m = list[ilist];
    if (shake_flag[m] != 5) continue;
    if (demoted_tag[m] == 0) continue;
    if (shake_atom[m][0] != atom->tag[m]) continue;
    for (int k = 1; k <= 3; k++) {
      int pidx = atom->map(shake_atom[m][k]);
      if (pidx >= 0 && pidx < nlocal)
        demoted_tag[pidx] = demoted_tag[m];
    }
  }
}

int FixRigs::lookup_or_compute_angle(int ilist)
{
  int m = list[ilist];
  int bt0 = shake_type[m][0];
  int bt1 = shake_type[m][1];
  int at = shake_type[m][2];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];

  double masses[3];
  get_mass3(closest_list[ilist], masses);
  LDU2 mass_ldu = ldu2(mass_matrix3(masses));

  if (rmass) {
    char key[128];
    std::snprintf(key, sizeof(key), "1:%d:%d:%d", bt0, bt1, at);
    std::string skey(key);
    auto it = cache_key_to_idx.find(skey);
    int idx;
    if (it == cache_key_to_idx.end()) {
      idx = Lsq_cached.size();
      Lsq_cached.emplace_back();
      double bond1 = bond_distance[bt0];
      double bond2 = bond_distance[bt1];
      Lsq_cached[idx].data[0] = bond1 * bond1;
      Lsq_cached[idx].data[1] = rigs_angle[at];
      Lsq_cached[idx].data[2] = bond2 * bond2;
      cache_key_to_idx[skey] = idx;
    } else {
      idx = it->second;
    }

    double *lm = reduced_rmass_ltdl[m];
    lm[0] = mass_ldu.d0;
    lm[1] = mass_ldu.d1;
    lm[2] = mass_ldu.u01;
    return idx;
  }

  int t0 = type[i0], t1 = type[i1], t2 = type[i2];
  char key[128];
  std::snprintf(key, sizeof(key), "1:%d:%d:%d:%d:%d:%d", bt0, bt1, at, t0, t1, t2);
  std::string skey(key);
  auto it = cache_key_to_idx.find(skey);
  if (it != cache_key_to_idx.end()) return it->second;

  int idx = Lsq_cached.size();
  Lsq_cached.emplace_back();
  reduced_mass_ltdl_cached.emplace_back();
  double bond1 = bond_distance[bt0];
  double bond2 = bond_distance[bt1];
  Lsq_cached[idx].data[0] = bond1 * bond1;
  Lsq_cached[idx].data[1] = rigs_angle[at];
  Lsq_cached[idx].data[2] = bond2 * bond2;
  reduced_mass_ltdl_cached[idx].data[0] = mass_ldu.d0;
  reduced_mass_ltdl_cached[idx].data[1] = mass_ldu.d1;
  reduced_mass_ltdl_cached[idx].data[2] = mass_ldu.u01;
  entry_demoted_pivot.push_back(0);
  cache_key_to_idx[skey] = idx;
  return idx;
}

int FixRigs::lookup_or_compute_improper(int ilist)
{
  int m = list[ilist];
  int bt0 = shake_type[m][0];
  int bt1 = shake_type[m][1];
  int bt2 = shake_type[m][2];
  int at0 = rigs_type[m][0];
  int at1 = rigs_type[m][1];
  int at2 = rigs_type[m][2];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  int i3 = closest_list[ilist][3];

  double mu[4];
  get_inv_mass4(closest_list[ilist], mu);
  double mu0 = mu[0];
  double mu01 = mu0 + mu[1];
  double mu02 = mu0 + mu[2];
  double mu03 = mu0 + mu[3];
  // Compute reduced_mass_ltdl = LDL^T of (B M^{-1} B^T)^{-1}.
  // Input: inverse masses. invert_to_ltdl inverts the SymMat of sums of inverse masses,
  // yielding a MASS matrix (not an inverse mass matrix).
  LTDL3 dc = invert_to_ltdl(SymMat3{mu01, mu0, mu0, mu02, mu0, mu03});

  double bond0 = bond_distance[bt0];
  double bond1 = bond_distance[bt1];
  double bond2 = bond_distance[bt2];
  double angle01 = rigs_angle[at0];
  double angle02 = rigs_angle[at1];
  double angle12 = rigs_angle[at2];

  double d0 = bond1;
  double u01 = angle01 / d0;
  double u02 = angle02 / d0;
  double dd1 = sqrt(bond1 * bond1 - u01 * u01);
  double u12 = (angle12 - u01 * u02) / dd1;
  double dd2 = sqrt(bond2 * bond2 - u02 * u02 - u12 * u12);
  Mat3 rt_LM = Mat3(UTMat3{d0, u01, u02, dd1, u12, dd2});
  rmul_ltdl(rt_LM, dc);
  SymMat3 MLM = mtm(rt_LM);
  int perm_mlm[3];
  LTDL3 dc_MLM = ltdl_pivot3(MLM, perm_mlm);
  double ratio_d2d0 = (dc_MLM.d0 > 0.0) ? dc_MLM.d2 / dc_MLM.d0 : 0.0;
  constexpr double demote_threshold = 1e-3;

  if (ratio_d2d0 < demote_threshold) {
    int pos_smallest_d = perm_mlm[2] + 1;
    demoted_tag[m] = shake_atom[m][pos_smallest_d];

    double masses[3];
    masses[0] = get_mass(i0);
    int tri_bt0, tri_bt1, tri_at;
    if (pos_smallest_d == 1) {
      tri_bt0 = bt1; tri_bt1 = bt2; tri_at = at2;
      masses[1] = get_mass(i2);
      masses[2] = get_mass(i3);
    } else if (pos_smallest_d == 2) {
      tri_bt0 = bt0; tri_bt1 = bt2; tri_at = at1;
      masses[1] = get_mass(i1);
      masses[2] = get_mass(i3);
    } else {
      tri_bt0 = bt0; tri_bt1 = bt1; tri_at = at0;
      masses[1] = get_mass(i1);
      masses[2] = get_mass(i2);
    }

    if (rmass) {
      char key[256];
      std::snprintf(key, sizeof(key), "5d:P%d:%d:%d:%d:%d:%d:%d", pos_smallest_d, bt0, bt1, bt2, at0, at1, at2);
      std::string skey(key);
      auto it = cache_key_to_idx.find(skey);
      int idx;
      if (it == cache_key_to_idx.end()) {
        idx = Lsq_cached.size();
        Lsq_cached.emplace_back();
        Lsq_cached[idx].data[0] = bond_distance[tri_bt0] * bond_distance[tri_bt0];
        Lsq_cached[idx].data[1] = rigs_angle[tri_at];
        Lsq_cached[idx].data[2] = bond_distance[tri_bt1] * bond_distance[tri_bt1];
        SymMat3 Lref = {bond0 * bond0, angle01, angle02,
                        bond1 * bond1, angle12, bond2 * bond2};
        LTDL3 dcL = ltdl_pivot_one(Lref, pos_smallest_d - 1);
        Lsq_cached[idx].data[3] = dcL.l20 - dcL.l10 * dcL.l21;
        Lsq_cached[idx].data[4] = dcL.l21;
        Lsq_cached[idx].data[5] = sqrt(dcL.d2);
        cache_key_to_idx[skey] = idx;
      } else {
        idx = it->second;
      }

      LDU2 mass_ldu = ldu2(mass_matrix3(masses));
      double *lm = reduced_rmass_ltdl[m];
      lm[0] = mass_ldu.d0;
      lm[1] = mass_ldu.d1;
      lm[2] = mass_ldu.u01;
      return idx;
    }

    // non-rmass path
    char key[256];
    std::snprintf(key, sizeof(key), "5d:P%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
                  pos_smallest_d, bt0, bt1, bt2, at0, at1, at2,
                  type[i0], type[i1], type[i2], type[i3]);
    std::string skey(key);
    auto it = cache_key_to_idx.find(skey);
    if (it != cache_key_to_idx.end()) {
      int pos_saved = entry_demoted_pivot[it->second];
      if (pos_saved > 0) demoted_tag[m] = shake_atom[m][pos_saved];
      return it->second;
    }

    int idx = Lsq_cached.size();
    Lsq_cached.emplace_back();
    reduced_mass_ltdl_cached.emplace_back();
    Lsq_cached[idx].data[0] = bond_distance[tri_bt0] * bond_distance[tri_bt0];
    Lsq_cached[idx].data[1] = rigs_angle[tri_at];
    Lsq_cached[idx].data[2] = bond_distance[tri_bt1] * bond_distance[tri_bt1];

    LDU2 mass_ldu = ldu2(mass_matrix3(masses));
    reduced_mass_ltdl_cached[idx].data[0] = mass_ldu.d0;
    reduced_mass_ltdl_cached[idx].data[1] = mass_ldu.d1;
    reduced_mass_ltdl_cached[idx].data[2] = mass_ldu.u01;

    SymMat3 Lref = {bond0 * bond0, angle01, angle02,
                    bond1 * bond1, angle12, bond2 * bond2};
    LTDL3 dcL = ltdl_pivot_one(Lref, pos_smallest_d - 1);
    Lsq_cached[idx].data[3] = dcL.l20 - dcL.l10 * dcL.l21;
    Lsq_cached[idx].data[4] = dcL.l21;
    Lsq_cached[idx].data[5] = sqrt(dcL.d2);

    entry_demoted_pivot.push_back(pos_smallest_d);
    cache_key_to_idx[skey] = idx;
    return idx;
  }

  // not demoted
  if (rmass) {
    char key[256];
    std::snprintf(key, sizeof(key), "5:%d:%d:%d:%d:%d:%d", bt0, bt1, bt2, at0, at1, at2);
    std::string skey(key);
    auto it = cache_key_to_idx.find(skey);
    int idx;
    if (it == cache_key_to_idx.end()) {
      idx = Lsq_cached.size();
      Lsq_cached.emplace_back();
      Lsq_cached[idx].data[0] = bond0 * bond0;
      Lsq_cached[idx].data[1] = angle01;
      Lsq_cached[idx].data[2] = angle02;
      Lsq_cached[idx].data[3] = bond1 * bond1;
      Lsq_cached[idx].data[4] = angle12;
      Lsq_cached[idx].data[5] = bond2 * bond2;
      cache_key_to_idx[skey] = idx;
    } else {
      idx = it->second;
    }

    double *lm = reduced_rmass_ltdl[m];
    lm[0] = dc.d0;
    lm[1] = dc.d1;
    lm[2] = dc.d2;
    lm[3] = dc.l10;
    lm[4] = dc.l20;
    lm[5] = dc.l21;
    return idx;
  }

  // non-rmass path
  int t0 = type[i0], t1 = type[i1], t2 = type[i2], t3 = type[i3];
  char key[256];
  std::snprintf(key, sizeof(key), "5:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
                bt0, bt1, bt2, at0, at1, at2, t0, t1, t2, t3);
  std::string skey(key);
  auto it = cache_key_to_idx.find(skey);
  if (it != cache_key_to_idx.end()) {
    int pos_saved = entry_demoted_pivot[it->second];
    if (pos_saved > 0) demoted_tag[m] = shake_atom[m][pos_saved];
    return it->second;
  }

  int idx = Lsq_cached.size();
  Lsq_cached.emplace_back();
  reduced_mass_ltdl_cached.emplace_back();
  Lsq_cached[idx].data[0] = bond0 * bond0;
  Lsq_cached[idx].data[1] = angle01;
  Lsq_cached[idx].data[2] = angle02;
  Lsq_cached[idx].data[3] = bond1 * bond1;
  Lsq_cached[idx].data[4] = angle12;
  Lsq_cached[idx].data[5] = bond2 * bond2;
  dc.store(reduced_mass_ltdl_cached[idx].data);
  entry_demoted_pivot.push_back(0);
  cache_key_to_idx[skey] = idx;
  return idx;
}

int FixRigs::lookup_or_compute_dihedral(int ilist)
{
  int m = list[ilist];
  int bt0 = shake_type[m][0];
  int bt1 = shake_type[m][1];
  int bt2 = shake_type[m][2];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  int i3 = closest_list[ilist][3];

  double mu[4];
  get_inv_mass4(closest_list[ilist], mu);
  double mu0 = mu[0];
  double mu2 = mu[2];
  double mu10 = mu0 + mu[1];
  double mu02 = mu0 + mu2;
  double mu23 = mu2 + mu[3];
  // Compute reduced_mass_ltdl = LDL^T of (B M^{-1} B^T)^{-1}.
  // Input: inverse masses. invert_to_ltdl inverts the SymMat of sums of inverse masses,
  // yielding a MASS matrix (not an inverse mass matrix).
  LTDL3 dc = invert_to_ltdl(SymMat3{mu10, mu0, 0, mu02, mu2, mu23});

  if (rmass) {
    char key[128];
    std::snprintf(key, sizeof(key), "6:%d:%d:%d", bt0, bt1, bt2);
    std::string skey(key);
    auto it = cache_key_to_idx.find(skey);
    int idx;
    if (it == cache_key_to_idx.end()) {
      idx = Lsq_cached.size();
      Lsq_cached.emplace_back();
      double bond1 = bond_distance[bt0];
      double bond2 = bond_distance[bt1];
      double bond3 = bond_distance[bt2];
      Lsq_cached[idx].data[0] = bond1 * bond1;
      Lsq_cached[idx].data[1] = bond1 * bond2;
      Lsq_cached[idx].data[2] = bond1 * bond3;
      Lsq_cached[idx].data[3] = bond2 * bond2;
      Lsq_cached[idx].data[4] = bond2 * bond3;
      Lsq_cached[idx].data[5] = bond3 * bond3;
      cache_key_to_idx[skey] = idx;
    } else {
      idx = it->second;
    }

    double *lm = reduced_rmass_ltdl[m];
    dc.store(lm);
    return idx;
  }

  int t0 = type[i0], t1 = type[i1], t2 = type[i2], t3 = type[i3];
  char key[128];
  std::snprintf(key, sizeof(key), "6:%d:%d:%d:%d:%d:%d:%d",
                bt0, bt1, bt2, t0, t1, t2, t3);
  std::string skey(key);
  auto it = cache_key_to_idx.find(skey);
  if (it != cache_key_to_idx.end()) return it->second;

  int idx = Lsq_cached.size();
  Lsq_cached.emplace_back();
  reduced_mass_ltdl_cached.emplace_back();
  double bond1 = bond_distance[bt0];
  double bond2 = bond_distance[bt1];
  double bond3 = bond_distance[bt2];
  Lsq_cached[idx].data[0] = bond1 * bond1;
  Lsq_cached[idx].data[1] = bond1 * bond2;
  Lsq_cached[idx].data[2] = bond1 * bond3;
  Lsq_cached[idx].data[3] = bond2 * bond2;
  Lsq_cached[idx].data[4] = bond2 * bond3;
  Lsq_cached[idx].data[5] = bond3 * bond3;
  dc.store(reduced_mass_ltdl_cached[idx].data);
  entry_demoted_pivot.push_back(0);
  cache_key_to_idx[skey] = idx;
  return idx;
}

void FixRigs::shake4(int ilist)
{
  int m = list[ilist];
  if (shake_flag[m] == 5) {
    if (demoted_tag[m] != 0)
      shake4demoted(ilist);
    else
      solve3x3(ilist, IMPROPER);
  } else if (shake_flag[m] == 6) {
    solve3x3(ilist, DIHEDRAL);
  } else {
    FixShake::shake4(ilist);
  }
}

/* ----------------------------------------------------------------------
   demoted improper solver: solve 3-atom triangle then push 4th atom
   ------------------------------------------------------------------------- */

static bool has_nan3(const double v[3]) {
  return std::isnan(v[0]) || std::isnan(v[1]) || std::isnan(v[2]);
}

void FixRigs::shake4demoted(int ilist)
{
  const int m = list[ilist];
  const int i0 = closest_list[ilist][0];
  const tagint dtag = demoted_tag[m];
  int k, i1, i2, i3; // reorder: 3 is always demoted

  if (atom->tag[closest_list[ilist][1]] == dtag) {
    i1 = closest_list[ilist][2];
    i2 = closest_list[ilist][3];
    i3 = closest_list[ilist][1];
  } else {
      i1 = closest_list[ilist][1];
      if (atom->tag[closest_list[ilist][2]] == dtag) {
        i2 = closest_list[ilist][3];
        i3 = closest_list[ilist][2];
      } else if (atom->tag[closest_list[ilist][3]] == dtag) {
        i2 = closest_list[ilist][2];
        i3 = closest_list[ilist][3];
      } // TODO: throw an error!
  }

  int idx = ilist_to_idx[ilist];
  double a1 = Lsq_cached[idx].data[3];
  double a2 = Lsq_cached[idx].data[4];
  double l22 = Lsq_cached[idx].data[5];

  double mass0, mass1, mass2, mass3;

  if (rmass) {
    mass0 = rmass[i0]; mass1 = rmass[i1];
    mass2 = rmass[i2]; mass3 = rmass[i3];
  } else {
    mass0 = mass[type[i0]]; mass1 = mass[type[i1]];
    mass2 = mass[type[i2]]; mass3 = mass[type[i3]];
  }
  
  //redistribute_forcemom_linear(ilist, i0, i1, i2, i3);
  double M012 = mass0 + mass1 + mass2;
  double ratio012 = M012 / (M012 + mass3);
  Vec3 fchange = Vec3(f[i3]) * ratio012;
  Vec3 vchange = Vec3(v[i3]) * mass3 * ratio012;
  Vec3 xchange = dtv * vchange + dtfsq * fchange;

  double mult0 = (1 - a1 - a2) / mass0;
  double mult1 = a1 / mass1;
  double mult2 = a2 / mass2;

  for (k = 0; k < 3; k++)
    xshake[i0][k] += xchange[k] * mult0;
    if (i0 < nlocal)
      for (k = 0; k < 3; k++) {
      v[i0][k] += vchange[k] * mult0;
      f[i0][k] += fchange[k] * mult0 * mass0;
    }
  for (k = 0; k < 3; k++)
    xshake[i1][k] += xchange[k] * mult1;
    if (i1 < nlocal)
      for (k = 0; k < 3; k++) {
      v[i1][k] += vchange[k] * mult1;
      f[i1][k] += fchange[k] * mult1 * mass1;
    }
  for (k = 0; k < 3; k++)
    xshake[i2][k] += xchange[k] * mult2;
    if (i2 < nlocal)
      for (k = 0; k < 3; k++) {
      v[i2][k] += vchange[k] * mult2;
      f[i2][k] += fchange[k] * mult2 * mass2;
    }
  for (int k = 0; k < 3; k++) {
    f[i3][k] -= fchange[k];
    v[i3][k] -= vchange[k] / mass3;
  }

  store_lamda_corrections = true;

  shake3angle_solve(i0, i1, i2, ilist);
  store_lamda_corrections = false;

  if (output_every) {
    iter_b_count[shake_type[m][0]]++; iter_b_total[shake_type[m][0]]++;
    iter_b_count[shake_type[m][1]]++; iter_b_total[shake_type[m][1]]++;
    iter_b_count[shake_type[m][2]]++; iter_b_total[shake_type[m][2]]++;
    if (rigs_type[m][0] > 0) { iter_a_count[rigs_type[m][0]]++; iter_a_total[rigs_type[m][0]]++; }
    if (rigs_type[m][1] > 0) { iter_a_count[rigs_type[m][1]]++; iter_a_total[rigs_type[m][1]]++; }
    if (rigs_type[m][2] > 0) { iter_a_count[rigs_type[m][2]]++; iter_a_total[rigs_type[m][2]]++; }
  }

  Vec3 r01 = Vec3(xshake[i0]) - xshake[i1];
  Vec3 r02 = Vec3(xshake[i0]) - xshake[i2];
  Vec3 r03 = Vec3(xshake[i0]) - x[i3];
  Vec3 n = cross(r01, r02);
  double nnorm = sqrt(normsq(n));

  double sgn = (dot(r03, n) < 0) ? -1.0 : 1.0;

  Vec3 xcorr = (a1 * r01 + a2 * r02 + (sgn * l22 / nnorm) * n) - r03;
  Vec3 vcorr, fcorr;
  if (in_setup) {
    vcorr = -1.0 * Vec3(v[i3]); // zero out velocities
    fcorr = -4.0 * mass3 * xcorr / dtfsq - Vec3(f[i3]);
  } else {
      vcorr = (xcorr / dtv + Vec3(v[i3]));
      fcorr = -2.0 * mass3 * vcorr * dtv / dtfsq - Vec3(f[i3]);
  }

  if (i3 < nlocal) {
    f[i3][0] += fcorr.x; f[i3][1] += fcorr.y; f[i3][2] += fcorr.z;
    v[i3][0] += vcorr.x; v[i3][1] += vcorr.y; v[i3][2] += vcorr.z;
  }

  vcorr *= mass3 / M012;
  if (i0 < nlocal)
    for (k = 0; k < 3; k++) {
    v[i0][k] -= vcorr[k];
    f[i0][k] -= fcorr[k] * mass0 / M012;
  }
  if (i1 < nlocal)
    for (k = 0; k < 3; k++) {
    v[i1][k] -= vcorr[k];
    f[i1][k] -= fcorr[k] * mass1 / M012;
  }
  if (i2 < nlocal)
    for (k = 0; k < 3; k++) {
    v[i2][k] -= vcorr[k];
    f[i2][k] -= fcorr[k] * mass2 / M012;
  }
}

/* ----------------------------------------------------------------------
   calculate RIGS constraint forces for size 3 cluster = two bonds + angle
   ------------------------------------------------------------------------- */

void FixRigs::shake3angle(int ilist)
{
  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];

  shake3angle_solve(i0, i1, i2, ilist);
  if (output_every) {
    iter_b_count[shake_type[m][0]]++; iter_b_total[shake_type[m][0]]++;
    iter_b_count[shake_type[m][1]]++; iter_b_total[shake_type[m][1]]++;
    iter_a_count[shake_type[m][2]]++; iter_a_total[shake_type[m][2]]++;
  }
}

void FixRigs::shake3angle_solve(int i0, int i1, int i2, int ilist)
{
  int m = list[ilist];
  int idx = ilist_to_idx[ilist];
  const double *L_ptr = Lsq_cached[idx].data;
  const double *lm_ptr = rmass ? reduced_rmass_ltdl[m] : reduced_mass_ltdl_cached[idx].data;
  int atomlist[3];
  double v[6];

  double lamda01, lamda02, lamda12;
  lamda01 = lamda02 = lamda12 = 0.0;
  
  SymMat2 Lsq = {L_ptr[0], L_ptr[1], L_ptr[2]};
  LDU2 mass_ldu = {lm_ptr[0], lm_ptr[1], lm_ptr[2]};
  
  Vec3 r01 = Vec3(x[i0]) - x[i1];
  Vec3 r02 = Vec3(x[i0]) - x[i2];
  Vec3 n = cross(r01, r02);
  double nn_inv = 1. / normsq(n);

  double r01sq = normsq(r01);
  double r0102 = dot(r01, r02);
  SymMat2 rr_inv = {  normsq(r02) * nn_inv,
                           -r0102 * nn_inv,
                            r01sq * nn_inv };
  LTMat2 rnorm = chol_lower(rr_inv);

  Vec3 s01 = Vec3(xshake[i0]) - xshake[i1];
  Vec3 s02 = Vec3(xshake[i0]) - xshake[i2];

  Mat2 rPs;
  rPs(0, 0) = dot(r01, s01);
  rPs(0, 1) = dot(r01, s02);
  rPs(1, 0) = dot(r02, s01);
  rPs(1, 1) = dot(r02, s02);
  rPs = rr_inv * rPs;

  Mat2 chi = rPs * mass_ldu;

  double p1 = dot(s01, n);
  double p2 = dot(s02, n);
  SymMat2 ss_perp = {p1 * p1 * nn_inv, p1 * p2 * nn_inv, 
	             p2 * p2 * nn_inv};
  SymMat2 Lres = Lsq - ss_perp;

  UsL(Lres, mass_ldu);
  UTMat2 pre_phi = chol_upper(Lres);
  UTMat2 phi = mul_du(pre_phi, mass_ldu);

  Mat2 phiCos = rnorm * phi;
  Mat2 J;
  J(0, 0) = 0.0;  J(0, 1) = -1.0;
  J(1, 0) = 1.0;  J(1, 1) = 0.0;
  Mat2 phiSin = rnorm * (J * phi);

  double skewCos = skew(phiCos);
  double skewSin = skew(phiSin);
  double skewChi = skew(chi);

  double Asq = skewCos * skewCos + skewSin * skewSin;
  double sinsqp = Asq - skewChi * skewChi;
  double sinp = sinsqp > 0.0 ? sqrt(sinsqp) : 0.0;

  double signp = (skewCos * trace_of_product(chi, phiSin)
                < skewSin * trace_of_product(chi, phiCos)) ? sinp : -sinp;

  double ccos = -(skewCos * skewChi + skewSin * signp) / Asq;
  double ssin = (skewCos * signp - skewSin * skewChi) / Asq;

  lamda01 += chi(0, 0) + ccos * phiCos(0, 0) + ssin * phiSin(0, 0);
  lamda02 += chi(1, 1) + ccos * phiCos(1, 1) + ssin * phiSin(1, 1);
  lamda12 += chi(0, 1) + ccos * phiCos(0, 1) + ssin * phiSin(0, 1);

  if (store_lamda_corrections) {
    double m0 = get_mass(i0);
    double m1 = get_mass(i1);
    double m2 = get_mass(i2);
    Vec3 corr = -(lamda01 + lamda12) * r01 - (lamda02 + lamda12) * r02;
    for (int i = 0; i < 3; i++) xshake[i0][i] += corr[i] / m0;
    if (i0 < nlocal)
      for (int i = 0; i < 3; i++) f[i0][i] += corr[i] / dtfsq;
    corr = lamda01 * r01 + lamda12 * r02;
    for (int i = 0; i < 3; i++) xshake[i1][i] += corr[i] / m1;
    if (i1 < nlocal)
      for (int i = 0; i < 3; i++) f[i1][i] += corr[i] / dtfsq;
    corr = lamda12 * r01 + lamda02 * r02;
    for (int i = 0; i < 3; i++) xshake[i2][i] += corr[i] / m2;
    if (i2 < nlocal)
      for (int i = 0; i < 3; i++) f[i2][i] += corr[i] / dtfsq;
    lamda01 /= dtfsq;
    lamda02 /= dtfsq;
    lamda12 /= dtfsq;
  } else {
    lamda01 /= dtfsq;
    lamda02 /= dtfsq;
    lamda12 /= dtfsq;
    if (i0 < nlocal) {
      Vec3 fcorr = -(lamda01 + lamda12) * r01 - (lamda02 + lamda12) * r02;
      for (int i = 0; i < 3; i++) f[i0][i] += fcorr[i];
    }
    if (i1 < nlocal) {
      Vec3 fcorr = lamda01 * r01 + lamda12 * r02;
      for (int i = 0; i < 3; i++) f[i1][i] += fcorr[i];
    }
    if (i2 < nlocal) {
      Vec3 fcorr = lamda02 * r02 + lamda12 * r01;
      for (int i = 0; i < 3; i++) f[i2][i] += fcorr[i];
    }
  }
  if (evflag) {
    int count = 0;
    if (i0 < nlocal) atomlist[count++] = i0;
    if (i1 < nlocal) atomlist[count++] = i1;
    if (i2 < nlocal) atomlist[count++] = i2;

    Vec3 r12 = r02 - r01;

    double lamda01_shake = -lamda01 - lamda12;
    double lamda02_shake = -lamda02 - lamda12;
    double lamda12_shake = lamda12;

    v[0] = lamda01_shake * r01[0] * r01[0] + lamda02_shake * r02[0] * r02[0] + lamda12_shake * r12[0] * r12[0];
    v[1] = lamda01_shake * r01[1] * r01[1] + lamda02_shake * r02[1] * r02[1] + lamda12_shake * r12[1] * r12[1];
    v[2] = lamda01_shake * r01[2] * r01[2] + lamda02_shake * r02[2] * r02[2] + lamda12_shake * r12[2] * r12[2];
    v[3] = lamda01_shake * r01[0] * r01[1] + lamda02_shake * r02[0] * r02[1] + lamda12_shake * r12[0] * r12[1];
    v[4] = lamda01_shake * r01[0] * r01[2] + lamda02_shake * r02[0] * r02[2] + lamda12_shake * r12[0] * r12[2];
    v[5] = lamda01_shake * r01[1] * r01[2] + lamda02_shake * r02[1] * r02[2] + lamda12_shake * r12[1] * r12[2];

    double fpairlist[] = {lamda01_shake, lamda02_shake, lamda12_shake};
    double dellist[][3] = {{r01[0], r01[1], r01[2]},
                           {r02[0], r02[1], r02[2]},
                           {r12[0], r12[1], r12[2]}};
    int pairlist[][2] = {{i0, i1}, {i0, i2}, {i1, i2}};
    v_tally(count, atomlist, 3.0, v, nlocal, 3, pairlist, fpairlist, dellist);
  }
}

/* ----------------------------------------------------------------------
   Unified 3x3 constraint solver for improper (star) and dihedral (chain) topologies

   IMPROPER: star topology, center atom 0 + partners 1,2,3
     R column k = x[i0] - x[partner_k]  (center minus partner)
     S column k = xshake version of same
     force sign = -1/dtfsq

   DIHEDRAL: chain topology A-B-C-D stored as 1-0-2-3
     R column 0 = x[i1]-x[i0], column 1 = x[i0]-x[i2], column 2 = x[i2]-x[i3]
     S column k = xshake version of same
     force sign = +1 (use dtf^2 multiplier internally)
   ------------------------------------------------------------------------- */

void FixRigs::solve3x3(int ilist, Topology topo)
{
  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  int i3 = closest_list[ilist][3];
  int niter = 0;

  int col_ia[3], col_ib[3];
  if (topo == IMPROPER) {
    col_ia[0] = i0; col_ib[0] = i1;
    col_ia[1] = i0; col_ib[1] = i2;
    col_ia[2] = i0; col_ib[2] = i3;
  } else {
    col_ia[0] = i1; col_ib[0] = i0;
    col_ia[1] = i0; col_ib[1] = i2;
    col_ia[2] = i2; col_ib[2] = i3;
  }

  Mat3 R, S;
  for (int k = 0; k < 3; k++) {
    Vec3 rv = Vec3(x[col_ia[k]]) - x[col_ib[k]];
    Vec3 sv = Vec3(xshake[col_ia[k]]) - xshake[col_ib[k]];
    R(0, k) = rv.x; R(1, k) = rv.y; R(2, k) = rv.z;
    S(0, k) = sv.x; S(1, k) = sv.y; S(2, k) = sv.z;
  }

  SymMat3 rr = mtm(R);

  int idx = ilist_to_idx[ilist];
  const double *Lp = Lsq_cached[idx].data;
  const double *Lmp = rmass ? reduced_rmass_ltdl[m] : reduced_mass_ltdl_cached[idx].data;
  SymMat3 Lsq = SymMat3::load(Lp);
  LTDL3 reduced_mass_ltdl = LTDL3::load(Lmp);

  Mat3 chi = mat_mul(inv_mat3(R), S);
  rmul_ltdl(chi, reduced_mass_ltdl);

  SymMat3 rr_target_M = Lsq;
  lt_sandwich(rr_target_M, reduced_mass_ltdl);
  LTMat3 phi = mul_dl(chol_to_ltl_lower(rr_target_M), reduced_mass_ltdl);

  UTMat3 rnorm = inv_chol_upper(rr);
  Mat3 lamda = cayley_converge(rnorm, phi, chi, max_iter, tolerance, &niter);
  if (output_every) {
    iter_b_count[shake_type[m][0]]++; iter_b_total[shake_type[m][0]] += niter;
    iter_b_count[shake_type[m][1]]++; iter_b_total[shake_type[m][1]] += niter;
    iter_b_count[shake_type[m][2]]++; iter_b_total[shake_type[m][2]] += niter;
    if (topo == IMPROPER) {
      if (rigs_type[m][0] > 0) { iter_a_count[rigs_type[m][0]]++; iter_a_total[rigs_type[m][0]] += niter; }
      if (rigs_type[m][1] > 0) { iter_a_count[rigs_type[m][1]]++; iter_a_total[rigs_type[m][1]] += niter; }
      if (rigs_type[m][2] > 0) { iter_a_count[rigs_type[m][2]]++; iter_a_total[rigs_type[m][2]] += niter; }
    } else {
      if (rigs_type[m][0] > 0) { iter_a_count[rigs_type[m][0]]++; iter_a_total[rigs_type[m][0]] += niter; }
      if (rigs_type[m][1] > 0) { iter_a_count[rigs_type[m][1]]++; iter_a_total[rigs_type[m][1]] += niter; }
    }
  }
  lamda += chi;

  Mat3 Rt = transpose(R);
  Mat43 L_lam = (topo == IMPROPER) ? improper_L_lambda(lamda)
                                      : dihedral_L_lambda(lamda);
  L_lam *= Rt;
  
  double force_scale = -1.0 / dtfsq;

  if (i0 < nlocal)
    for (int i = 0; i < 3; i++) f[i0][i] += force_scale * L_lam(0, i);
  if (i1 < nlocal)
    for (int i = 0; i < 3; i++) f[i1][i] += force_scale * L_lam(1, i);
  if (i2 < nlocal)
    for (int i = 0; i < 3; i++) f[i2][i] += force_scale * L_lam(2, i);
  if (i3 < nlocal)
    for (int i = 0; i < 3; i++) f[i3][i] += force_scale * L_lam(3, i);
}

/* ----------------------------------------------------------------------
   calculate SHAKE constraint forces for size 2 cluster = single bond
------------------------------------------------------------------------- */

void FixRigs::shake(int ilist)
{
  int atomlist[2];
  double v[6];
  double invmass0,invmass1;

  // local atom IDs and constraint distances

  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  double bond1 = bond_distance[shake_type[m][0]];

  // r01 = distance vec between atoms

  double r01[3];
  r01[0] = x[i0][0] - x[i1][0];
  r01[1] = x[i0][1] - x[i1][1];
  r01[2] = x[i0][2] - x[i1][2];

  // s01 = distance vec after unconstrained update

  double s01[3];
  s01[0] = xshake[i0][0] - xshake[i1][0];
  s01[1] = xshake[i0][1] - xshake[i1][1];
  s01[2] = xshake[i0][2] - xshake[i1][2];

  // scalar distances between atoms
  double rr = r01[0]*r01[0] + r01[1]*r01[1] + r01[2]*r01[2];
  double ss = s01[0]*s01[0] + s01[1]*s01[1] + s01[2]*s01[2];
  double rs = r01[0]*s01[0] + r01[1]*s01[1] + r01[2]*s01[2];

  double mu;

  if (rmass) {
    mu = 1.0 / rmass[i0] + 1.0 / rmass[i1];
  } else {
    mu = 1.0 / mass[type[i0]] + 1.0 / mass[type[i1]];
  }

  double project = rs / rr;
  double determ = 1 - (ss - bond1*bond1) / (project * rs);

  // error check

  if (determ < 0.0) {
    error->warning(FLERR,"RIGS determinant < 0.0");
    determ = 0.0;
  }

  // exact quadratic solution for lamda

  double mult = sqrt(determ) - 1.; 
  double lamda = project * mult / (mu * dtfsq);

  // update forces if atom is owned by this processor

  if (output_every) {
    int bt = shake_type[m][0];
    iter_b_count[bt]++; iter_b_total[bt]++;
  }

  if (i0 < nlocal) {
    f[i0][0] += lamda*r01[0];
    f[i0][1] += lamda*r01[1];
    f[i0][2] += lamda*r01[2];
  }

  if (i1 < nlocal) {
    f[i1][0] -= lamda*r01[0];
    f[i1][1] -= lamda*r01[1];
    f[i1][2] -= lamda*r01[2];
  }

  if (evflag) {
    int count = 0;
    if (i0 < nlocal) atomlist[count++] = i0;
    if (i1 < nlocal) atomlist[count++] = i1;

    v[0] = lamda*r01[0]*r01[0];
    v[1] = lamda*r01[1]*r01[1];
    v[2] = lamda*r01[2]*r01[2];
    v[3] = lamda*r01[0]*r01[1];
    v[4] = lamda*r01[0]*r01[2];
    v[5] = lamda*r01[1]*r01[2];

    double fpairlist[] = {lamda};
    double dellist[][3]  = {{r01[0], r01[1], r01[2]}};
    int pairlist[][2] = {{i0,i1}};
    v_tally(count,atomlist,2.0,v,nlocal,1,pairlist,fpairlist,dellist);
  }
}

void FixRigs::get_inv_mass3(int *i, double *mu) {
  if (rmass) {
    mu[0] = 1.0 / rmass[i[0]];
    mu[1] = 1.0 / rmass[i[1]];
    mu[2] = 1.0 / rmass[i[2]];
  } else {
    mu[0] = 1.0 / mass[type[i[0]]];
    mu[1] = 1.0 / mass[type[i[1]]];
    mu[2] = 1.0 / mass[type[i[2]]];
  }
}

void FixRigs::get_inv_mass4(int *i, double *m) {
  if (rmass) {
    m[0] = 1.0 / rmass[i[0]];
    m[1] = 1.0 / rmass[i[1]];
    m[2] = 1.0 / rmass[i[2]];
    m[3] = 1.0 / rmass[i[3]];
  } else {
    m[0] = 1.0 / mass[type[i[0]]];
    m[1] = 1.0 / mass[type[i[1]]];
    m[2] = 1.0 / mass[type[i[2]]];
    m[3] = 1.0 / mass[type[i[3]]];
  }
}

void FixRigs::get_mass3(int *i, double *m) {
  if (rmass) {
    m[0] = rmass[i[0]];
    m[1] = rmass[i[1]];
    m[2] = rmass[i[2]];
  } else {
    m[0] = mass[type[i[0]]];
    m[1] = mass[type[i[1]]];
    m[2] = mass[type[i[2]]];
  }
}

void FixRigs::get_mass4(int *i, double *m) {
  if (rmass) {
    m[0] = rmass[i[0]];
    m[1] = rmass[i[1]];
    m[2] = rmass[i[2]];
    m[3] = rmass[i[3]];
  } else {
    m[0] = mass[type[i[0]]];
    m[1] = mass[type[i[1]]];
    m[2] = mass[type[i[2]]];
    m[3] = mass[type[i[3]]];
  }
}

