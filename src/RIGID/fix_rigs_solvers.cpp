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

  if (rmass) {
    for (int ilist = 0; ilist < nlist; ilist++) {
      int m = list[ilist];
      int idx;

      if (shake_flag[m] == 1) {
        int bt0 = shake_type[m][0];
        int bt1 = shake_type[m][1];
        int at = shake_type[m][2];

        char key[128];
        std::snprintf(key, sizeof(key), "1:%d:%d:%d", bt0, bt1, at);
        std::string skey(key);
        auto it = cache_key_to_idx.find(skey);
        if (it == cache_key_to_idx.end()) {
          idx = L_entries.size();
          L_entries.emplace_back();
          double bond1 = bond_distance[bt0];
          double bond2 = bond_distance[bt1];
          L_entries[idx].data[0] = bond1 * bond1;
          L_entries[idx].data[1] = rigs_angle[at];
          L_entries[idx].data[2] = bond2 * bond2;
          cache_key_to_idx[skey] = idx;
        } else {
          idx = it->second;
        }
        ilist_to_idx[ilist] = idx;

        double *lm = rigs_lm_atom[m];
        int i0 = closest_list[ilist][0];
        int i1 = closest_list[ilist][1];
        int i2 = closest_list[ilist][2];
        double invmass0 = 1.0 / rmass[i0];
        double invmass01 = invmass0 + 1.0 / rmass[i1];
        double invmass02 = invmass0 + 1.0 / rmass[i2];
        DChol2 dc = inv_dchol(SymMat2{invmass01, invmass0, invmass02});
        lm[0] = dc.d0;
        lm[1] = dc.d1;
        lm[2] = dc.m01;

      } else if (shake_flag[m] == 5) {
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

        double mu0 = 1.0 / rmass[i0];
        double mu01 = mu0 + 1.0 / rmass[i1];
        double mu02 = mu0 + 1.0 / rmass[i2];
        double mu03 = mu0 + 1.0 / rmass[i3];
        DChol3 dc = inv_dchol(SymMat3{mu01, mu0, mu0, mu02, mu0, mu03});

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
        mul_ltdl(rt_LM, dc);
        SymMat3 MLM = mtm(rt_LM);
        int perm_mlm[3];
        DChol3 dc_MLM = dchol_pivot(MLM, perm_mlm);
        double ratio_d2d0 = (dc_MLM.d0 > 0.0) ? dc_MLM.d2 / dc_MLM.d0 : 0.0;
        constexpr double demote_threshold = 1e-3;

        if (ratio_d2d0 < demote_threshold) {
          int pos_smallest_d = perm_mlm[2] + 1;
          demoted_tag[m] = shake_atom[m][pos_smallest_d];

          char key[256];
          std::snprintf(key, sizeof(key), "5d:P%d:%d:%d:%d:%d:%d", pos_smallest_d, bt0, bt1, bt2, at0, at1, at2);
          std::string skey(key);
          auto it = cache_key_to_idx.find(skey);
          if (it == cache_key_to_idx.end()) {
            idx = L_entries.size();
            L_entries.emplace_back();
            double im0 = 1.0 / rmass[i0];
            double im1, im2;
            int tri_bt0, tri_bt1, tri_at;
            if (pos_smallest_d == 1) {
              tri_bt0 = bt1; tri_bt1 = bt2; tri_at = at2;
              im1 = 1.0 / rmass[i2]; im2 = 1.0 / rmass[i3];
            } else if (pos_smallest_d == 2) {
              tri_bt0 = bt0; tri_bt1 = bt2; tri_at = at1;
              im1 = 1.0 / rmass[i1]; im2 = 1.0 / rmass[i3];
            } else {
              tri_bt0 = bt0; tri_bt1 = bt1; tri_at = at0;
              im1 = 1.0 / rmass[i1]; im2 = 1.0 / rmass[i2];
            }
            L_entries[idx].data[0] = bond_distance[tri_bt0] * bond_distance[tri_bt0];
            L_entries[idx].data[1] = rigs_angle[tri_at];
            L_entries[idx].data[2] = bond_distance[tri_bt1] * bond_distance[tri_bt1];
            SymMat3 Lref = {bond0 * bond0, angle01, angle02,
                            bond1 * bond1, angle12, bond2 * bond2};
            DChol3 dcL = dchol_pivot_one(Lref, pos_smallest_d - 1);
            L_entries[idx].data[3] = dcL.m02 - dcL.m01 * dcL.m12;
            L_entries[idx].data[4] = dcL.m12;
            L_entries[idx].data[5] = sqrt(dcL.d2);
            cache_key_to_idx[skey] = idx;
          } else {
            idx = it->second;
          }
          ilist_to_idx[ilist] = idx;

          double *lm = rigs_lm_atom[m];
          double im0 = 1.0 / rmass[i0];
          double im1, im2;
          int tri_bt0, tri_bt1, tri_at;
          if (pos_smallest_d == 1) {
            tri_bt0 = bt1; tri_bt1 = bt2; tri_at = at2;
            im1 = 1.0 / rmass[i2]; im2 = 1.0 / rmass[i3];
          } else if (pos_smallest_d == 2) {
            tri_bt0 = bt0; tri_bt1 = bt2; tri_at = at1;
            im1 = 1.0 / rmass[i1]; im2 = 1.0 / rmass[i3];
          } else {
            tri_bt0 = bt0; tri_bt1 = bt1; tri_at = at0;
            im1 = 1.0 / rmass[i1]; im2 = 1.0 / rmass[i2];
          }
          DChol2 dc3 = inv_dchol(SymMat2{im0 + im1, im0, im0 + im2});
          lm[0] = dc3.d0;
          lm[1] = dc3.d1;
          lm[2] = dc3.m01;

        } else {
          char key[256];
          std::snprintf(key, sizeof(key), "5:%d:%d:%d:%d:%d:%d", bt0, bt1, bt2, at0, at1, at2);
          std::string skey(key);
          auto it = cache_key_to_idx.find(skey);
          if (it == cache_key_to_idx.end()) {
            idx = L_entries.size();
            L_entries.emplace_back();
            L_entries[idx].data[0] = bond0 * bond0;
            L_entries[idx].data[1] = angle01;
            L_entries[idx].data[2] = angle02;
            L_entries[idx].data[3] = bond1 * bond1;
            L_entries[idx].data[4] = angle12;
            L_entries[idx].data[5] = bond2 * bond2;
            cache_key_to_idx[skey] = idx;
          } else {
            idx = it->second;
          }
          ilist_to_idx[ilist] = idx;

          double *lm = rigs_lm_atom[m];
          lm[0] = dc.d0;
          lm[1] = dc.d1;
          lm[2] = dc.d2;
          lm[3] = dc.m01;
          lm[4] = dc.m02;
          lm[5] = dc.m12;
        }

      } else if (shake_flag[m] == 6) {
        int bt0 = shake_type[m][0];
        int bt1 = shake_type[m][1];
        int bt2 = shake_type[m][2];

        char key[128];
        std::snprintf(key, sizeof(key), "6:%d:%d:%d", bt0, bt1, bt2);
        std::string skey(key);
        auto it = cache_key_to_idx.find(skey);
        if (it == cache_key_to_idx.end()) {
          idx = L_entries.size();
          L_entries.emplace_back();
          double bond1 = bond_distance[bt0];
          double bond2 = bond_distance[bt1];
          double bond3 = bond_distance[bt2];
          L_entries[idx].data[0] = bond1 * bond1;
          L_entries[idx].data[1] = bond1 * bond2;
          L_entries[idx].data[2] = bond1 * bond3;
          L_entries[idx].data[3] = bond2 * bond2;
          L_entries[idx].data[4] = bond2 * bond3;
          L_entries[idx].data[5] = bond3 * bond3;
          cache_key_to_idx[skey] = idx;
        } else {
          idx = it->second;
        }
        ilist_to_idx[ilist] = idx;

        int i0 = closest_list[ilist][0];
        int i1 = closest_list[ilist][1];
        int i2 = closest_list[ilist][2];
        int i3 = closest_list[ilist][3];
        double mu0 = 1.0 / rmass[i0];
        double mu2 = 1.0 / rmass[i2];
        double mu10 = 1.0 / rmass[i1] + mu0;
        double mu02 = mu0 + mu2;
        double mu23 = mu2 + 1.0 / rmass[i3];
        DChol3 dc = inv_dchol(SymMat3{mu10, mu0, 0, mu02, mu2, mu23});
        double *lm = rigs_lm_atom[m];
        lm[0] = dc.d0;
        lm[1] = dc.d1;
        lm[2] = dc.d2;
        lm[3] = dc.m01;
        lm[4] = dc.m02;
        lm[5] = dc.m12;
      }
    }
    return;
  }

  for (int ilist = 0; ilist < nlist; ilist++) {
    int m = list[ilist];
    int idx;

    if (shake_flag[m] == 1) {
      int bt0 = shake_type[m][0];
      int bt1 = shake_type[m][1];
      int at = shake_type[m][2];
      int i0 = closest_list[ilist][0];
      int i1 = closest_list[ilist][1];
      int i2 = closest_list[ilist][2];
      int t0 = type[i0], t1 = type[i1], t2 = type[i2];

      char key[128];
      std::snprintf(key, sizeof(key), "1:%d:%d:%d:%d:%d:%d", bt0, bt1, at, t0, t1, t2);
      std::string skey(key);
      auto it = cache_key_to_idx.find(skey);
      if (it == cache_key_to_idx.end()) {
        idx = L_entries.size();
        L_entries.emplace_back();
        lm_entries.emplace_back();
        double bond1 = bond_distance[bt0];
        double bond2 = bond_distance[bt1];
        L_entries[idx].data[0] = bond1 * bond1;
        L_entries[idx].data[1] = rigs_angle[at];
        L_entries[idx].data[2] = bond2 * bond2;
        double invmass0 = 1.0 / mass[t0];
        double invmass01 = invmass0 + 1.0 / mass[t1];
        double invmass02 = invmass0 + 1.0 / mass[t2];
        DChol2 dc = inv_dchol(SymMat2{invmass01, invmass0, invmass02});
        lm_entries[idx].data[0] = dc.d0;
        lm_entries[idx].data[1] = dc.d1;
        lm_entries[idx].data[2] = dc.m01;
        entry_demoted_pivot.push_back(0);
        cache_key_to_idx[skey] = idx;
      } else {
        idx = it->second;
      }
      ilist_to_idx[ilist] = idx;

     } else if (shake_flag[m] == 5) {
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
      int t0 = type[i0], t1 = type[i1], t2 = type[i2], t3 = type[i3];

      char key[256];
      std::snprintf(key, sizeof(key), "5:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
                    bt0, bt1, bt2, at0, at1, at2, t0, t1, t2, t3);
      std::string skey(key);
      auto it = cache_key_to_idx.find(skey);
      if (it == cache_key_to_idx.end()) {
        idx = L_entries.size();
        L_entries.emplace_back();
        lm_entries.emplace_back();
        double mu0 = 1.0 / mass[t0];
        double mu01 = mu0 + 1.0 / mass[t1];
        double mu02 = mu0 + 1.0 / mass[t2];
        double mu03 = mu0 + 1.0 / mass[t3];
        DChol3 dc = inv_dchol(SymMat3{mu01, mu0, mu0, mu02, mu0, mu03});
        double bond0 = bond_distance[bt0];
        double bond1 = bond_distance[bt1];
        double bond2 = bond_distance[bt2];
        double angle01 = rigs_angle[at0];
        double angle02 = rigs_angle[at1];
        double angle12 = rigs_angle[at2];

        double d0 = bond1;
        double u01 = angle01 / d0;
        double u02 = angle02 / d0;
        double d1 = sqrt(bond1 * bond1 - u01 * u01);
        double u12 = (angle12 - u01 * u02) / d1;
        double d2 = sqrt(bond2 * bond2 - u02 * u02 - u12 * u12);
        Mat3 rt_LM = Mat3(UTMat3{d0, u01, u02, d1, u12, d2});
        mul_ltdl(rt_LM, dc);
        SymMat3 MLM = mtm(rt_LM);
        int perm_mlm[3];
        DChol3 dc_MLM = dchol_pivot(MLM, perm_mlm);

        double ratio_d2d0 = (dc_MLM.d0 > 0.0) ? dc_MLM.d2 / dc_MLM.d0 : 0.0;
        constexpr double demote_threshold = 1e-3;
        if (ratio_d2d0 < demote_threshold) {
          int pos_smallest_d = perm_mlm[2] + 1;

          double im0 = 1.0 / mass[t0];
          double im1, im2;
          int tri_bt0, tri_bt1, tri_at;
          if (pos_smallest_d == 1) {
            tri_bt0 = bt1; tri_bt1 = bt2; tri_at = at2;
            im1 = 1.0 / mass[t2]; im2 = 1.0 / mass[t3];
          } else if (pos_smallest_d == 2) {
            tri_bt0 = bt0; tri_bt1 = bt2; tri_at = at1;
            im1 = 1.0 / mass[t1]; im2 = 1.0 / mass[t3];
          } else {
            tri_bt0 = bt0; tri_bt1 = bt1; tri_at = at0;
            im1 = 1.0 / mass[t1]; im2 = 1.0 / mass[t2];
          }

          L_entries[idx].data[0] = bond_distance[tri_bt0] * bond_distance[tri_bt0];
          L_entries[idx].data[1] = rigs_angle[tri_at];
          L_entries[idx].data[2] = bond_distance[tri_bt1] * bond_distance[tri_bt1];

          DChol2 dc3 = inv_dchol(SymMat2{im0 + im1, im0, im0 + im2});
          lm_entries[idx].data[0] = dc3.d0;
          lm_entries[idx].data[1] = dc3.d1;
          lm_entries[idx].data[2] = dc3.m01;

          // Geometric parameters for the demoted virtual particle.
          //
          // Lref is the Gram matrix of the constraint vectors (bond0-bond2),
          // stored in the ORIGINAL bond ordering to avoid reordering sensitive
          // lists.  dchol_pivot_one swaps only the demoted bond's row/column
          // to position 2, so the first two columns always correspond to the
          // two non-demoted bonds in their original order.
          //
          // The semi-pivoted factorisation gives:
          //   P Lref P^T = D^{1/2} Ltilde Ltilde^T D^{1/2}
          // where P swaps only the demoted bond to position 2,
          // D^{1/2} = diag(sqrt(d0), sqrt(d1), sqrt(d2)) and
          // Ltilde = [[1, 0, 0], [m01, 1, 0], [m02, m12, 1]].
          //
          // The diagonal entries l00=d0^{1/2}, l11=d1^{1/2}, l22=d2^{1/2}
          // and off-diagonal products m02*l00, m12*l11 are stored together
          // with m01 as the decomposition used by shake4demoted to construct
          // an orthonormal frame (e1, e2, n) from the two non-demoted
          // constraint vectors and then position the demoted particle at
          // xshake[i0] - (l20*e1 + l21*e2 + sgn*l22*n).
          SymMat3 Lref = {bond0 * bond0, angle01, angle02,
                          bond1 * bond1, angle12, bond2 * bond2};
          DChol3 dcL = dchol_pivot_one(Lref, pos_smallest_d - 1);
          L_entries[idx].data[3] = dcL.m02 - dcL.m01 * dcL.m12;
          L_entries[idx].data[4] = dcL.m12;
          L_entries[idx].data[5] = sqrt(dcL.d2);

          entry_demoted_pivot.push_back(pos_smallest_d);
          demoted_tag[m] = shake_atom[m][pos_smallest_d];
        } else {
          L_entries[idx].data[0] = bond0 * bond0;
          L_entries[idx].data[1] = angle01;
          L_entries[idx].data[2] = angle02;
          L_entries[idx].data[3] = bond1 * bond1;
          L_entries[idx].data[4] = angle12;
          L_entries[idx].data[5] = bond2 * bond2;
          lm_entries[idx].data[0] = dc.d0;
          lm_entries[idx].data[1] = dc.d1;
          lm_entries[idx].data[2] = dc.d2;
          lm_entries[idx].data[3] = dc.m01;
          lm_entries[idx].data[4] = dc.m02;
          lm_entries[idx].data[5] = dc.m12;
          entry_demoted_pivot.push_back(0);
        }
        cache_key_to_idx[skey] = idx;
      } else {
        idx = it->second;
        int pos_smallest_d = entry_demoted_pivot[idx];
        if (pos_smallest_d > 0) demoted_tag[m] = shake_atom[m][pos_smallest_d];
      }
      ilist_to_idx[ilist] = idx;

    } else if (shake_flag[m] == 6) {
      int bt0 = shake_type[m][0];
      int bt1 = shake_type[m][1];
      int bt2 = shake_type[m][2];
      int i0 = closest_list[ilist][0];
      int i1 = closest_list[ilist][1];
      int i2 = closest_list[ilist][2];
      int i3 = closest_list[ilist][3];
      int t0 = type[i0]; int t1 = type[i1]; int t2 = type[i2]; int t3 = type[i3];

      char key[128];
      std::snprintf(key, sizeof(key), "6:%d:%d:%d:%d:%d:%d:%d",
                    bt0, bt1, bt2, t0, t1, t2, t3);
      std::string skey(key);
      auto it = cache_key_to_idx.find(skey);
      if (it == cache_key_to_idx.end()) {
        idx = L_entries.size();
        L_entries.emplace_back();
        lm_entries.emplace_back();
        double bond1 = bond_distance[bt0];
        double bond2 = bond_distance[bt1];
        double bond3 = bond_distance[bt2];
        L_entries[idx].data[0] = bond1 * bond1;
        L_entries[idx].data[1] = bond1 * bond2;
        L_entries[idx].data[2] = bond1 * bond3;
        L_entries[idx].data[3] = bond2 * bond2;
        L_entries[idx].data[4] = bond2 * bond3;
        L_entries[idx].data[5] = bond3 * bond3;
        double mu0 = 1.0 / mass[t0];
        double mu2 = 1.0 / mass[t2];
        double mu10 = 1.0 / mass[t1] + mu0;
        double mu02 = mu0 + mu2;
        double mu23 = mu2 + 1.0 / mass[t3];
        DChol3 dc = inv_dchol(SymMat3{mu10, mu0, 0, mu02, mu2, mu23});
        lm_entries[idx].data[0] = dc.d0;
        lm_entries[idx].data[1] = dc.d1;
        lm_entries[idx].data[2] = dc.d2;
        lm_entries[idx].data[3] = dc.m01;
        lm_entries[idx].data[4] = dc.m02;
        lm_entries[idx].data[5] = dc.m12;
        entry_demoted_pivot.push_back(0);
        cache_key_to_idx[skey] = idx;
      } else {
        idx = it->second;
      }
      ilist_to_idx[ilist] = idx;
    }
  }

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
  double a1 = L_entries[idx].data[3];
  double a2 = L_entries[idx].data[4];
  double l22 = L_entries[idx].data[5];
  double a0 = 1 - a1 - a2;

  double mass0, mass1, mass2, mass3;

  if (rmass) {
    mass0 = rmass[i0]; mass1 = rmass[i1];
    mass2 = rmass[i2]; mass3 = rmass[i3];
  } else {
    mass0 = mass[type[i0]]; mass1 = mass[type[i1]];
    mass2 = mass[type[i2]]; mass3 = mass[type[i3]];
  }

  // re-project forces + momenta from i3 onto i0, i1, i2:

  double mxcorr[3], pcorr[3], fcorr[3];

  for (k = 0; k < 3; k++) {
    fcorr[k] = f[i3][k];
    pcorr[k] = mass3 * v[i3][k];
  }

  for (k = 0; k < 3; k++) {
    mxcorr[k] = pcorr[k] * dtv + fcorr[k] * dtfsq;
  }

  for (k = 0; k < 3; k++)
    xshake[i0][k] += a0 * mxcorr[k] / mass0;
  if (i0 < nlocal) {
      for (k = 0; k < 3; k++)
      v[i0][k] += a0 * pcorr[k] / mass0;
      f[i0][k] += a0 * fcorr[k];
    }
  for (k = 0; k < 3; k++)
    xshake[i1][k] += a1 * mxcorr[k] / mass1;
  if (i1 < nlocal) {
      for (k = 0; k < 3; k++)
      v[i1][k] += a1 * pcorr[k] / mass1;
      f[i1][k] += a1 * fcorr[k];
    }
  for (k = 0; k < 3; k++)
    xshake[i2][k] += a2 * mxcorr[k] / mass2;
  if (i2 < nlocal) {
      for (k = 0; k < 3; k++)
      v[i2][k] += a2 * pcorr[k] / mass2;
      f[i2][k] += a2 * fcorr[k];
    }
  //if (i3 < nlocal)
    for (int k = 0; k < 3; k++) {
      f[i3][k] = 0.0;
      v[i3][k] = 0.0;
    }

  store_lamda_corrections = true;

  shake3angle_solve(i0, i1, i2, ilist);
  store_lamda_corrections = false;

  double r01[3], r02[3], r03[3], n[3];
  for (k = 0; k < 3; k++) {
    r01[k] = xshake[i0][k] - xshake[i1][k];
    r02[k] = xshake[i0][k] - xshake[i2][k];
    r03[k] = xshake[i0][k] - x[i3][k];
  }

  cross(r01, r02, n);
  double nnorm = sqrt(dot3(n, n));

  double sgn = (dot3(r03, n) < 0) ? -1.0 : 1.0;
  double M012 = mass0 + mass1 + mass2;

  for (k = 0; k < 3; k++) {
    fcorr[k] = r03[k] - (a1 * r01[k] + a2 * r02[k] + sgn * l22 * n[k] / nnorm);
    fcorr[k] *= mass3;
    pcorr[k] = -fcorr[k] / dtv;
    fcorr[k] *= 2 / dtfsq;
  }
  //if (i3 < nlocal)
    for (k = 0; k < 3; k++) {
      f[i3][k] += fcorr[k];
      v[i3][k] += pcorr[k] / mass3;
    }
  //if (i0 < nlocal) {
  //  for (k = 0; k < 3; k++) {
  //    f[i0][k] -= fcorr[k] * mass0 / M012;
  //    v[i0][k] -= pcorr[k] / M012;
  //    f[i0][k] -= a0 * fcorr[k];
  //    v[i0][k] -= a0 * pcorr[k] / mass0;
  //  }
  //}
  //if (i1 < nlocal) {
  //  for (k = 0; k < 3; k++) {
  //    f[i1][k] -= a1 * fcorr[k];
  //    v[i1][k] -= a1 * pcorr[k] / mass1;
  //  }
  //}
  //if (i2 < nlocal) {
  //  for (k = 0; k < 3; k++) {
  //    f[i2][k] -= a2 * fcorr[k];
  //    v[i2][k] -= a2 * pcorr[k] / mass2;
  // }
  //}
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
}

void FixRigs::shake3angle_solve(int i0, int i1, int i2, int ilist)
{
  int m = list[ilist];
  int idx = ilist_to_idx[ilist];
  const double *L_ptr = L_entries[idx].data;
  const double *lm_ptr = rmass ? rigs_lm_atom[m] : lm_entries[idx].data;
  int atomlist[3];
  double v[6];

  double r01[3];
  r01[0] = x[i0][0] - x[i1][0];
  r01[1] = x[i0][1] - x[i1][1];
  r01[2] = x[i0][2] - x[i1][2];

  double r02[3];
  r02[0] = x[i0][0] - x[i2][0];
  r02[1] = x[i0][1] - x[i2][1];
  r02[2] = x[i0][2] - x[i2][2];

  double s01[3];
  s01[0] = xshake[i0][0] - xshake[i1][0];
  s01[1] = xshake[i0][1] - xshake[i1][1];
  s01[2] = xshake[i0][2] - xshake[i1][2];

  double s02[3];
  s02[0] = xshake[i0][0] - xshake[i2][0];
  s02[1] = xshake[i0][1] - xshake[i2][1];
  s02[2] = xshake[i0][2] - xshake[i2][2];

  SymMat2 rr = sym_dot(r01, r02);
  SymMat2 ss = sym_dot(s01, s02);

  SymMat2 Lm = {L_ptr[0], L_ptr[1], L_ptr[2]};
  SymMat2 diff = Lm - ss;

  Mat2 chi;
  chi(0, 0) = s01[0] * r01[0] + s01[1] * r01[1] + s01[2] * r01[2];
  chi(1, 0) = s01[0] * r02[0] + s01[1] * r02[1] + s01[2] * r02[2];
  chi(0, 1) = s02[0] * r01[0] + s02[1] * r01[1] + s02[2] * r01[2];
  chi(1, 1) = s02[0] * r02[0] + s02[1] * r02[1] + s02[2] * r02[2];

  DChol2 lmc = {lm_ptr[0], lm_ptr[1], lm_ptr[2]};

  UTMat2 rc = inv_chol_upper(rr);
  ut_mul(rc, chi);
  SymMat2 sigma = diff + mtm(chi);
  u_mul(rc, chi);

  lslt_mul(sigma, lmc);
  LTMat2 sc = mul_dl(chol_lower(sigma),lmc);
  mul_ltdl(chi, lmc);

  Mat2 phiC = rc * sc;

  Mat2 J;
  J(0, 0) = 0.0;  J(0, 1) = -1.0;
  J(1, 0) = 1.0;  J(1, 1) = 0.0;
  Mat2 phiS = rc * (J * sc);

  double skewC = skew(phiC);
  double skewChi = skew(chi);
  double skewS = skew(phiS);

  double Asq = skewC * skewC + skewS * skewS;
  double sinsqp = Asq - skewChi*skewChi;
  double sinp = sinsqp > 0.0 ? sqrt(sinsqp) : 0.0;
  double sskew = -(skewS * skewChi + skewC * sinp) / Asq;
  double cskew = (skewS * sinp - skewChi * skewC) / Asq;

  double lamda01 = chi(0, 0) + cskew * phiC(0, 0) + sskew * phiS(0, 0);
  double lamda02 = chi(1, 1) + cskew * phiC(1, 1);
  double lamda12 = chi(0, 1) + cskew * phiC(0, 1) + sskew * phiS(0, 1);

  if (store_lamda_corrections) {
    double m0, m1, m2;
    if (rmass) {
      m0 = rmass[i0]; m1 = rmass[i1]; m2 = rmass[i2];
    } else {
      m0 = mass[type[i0]]; m1 = mass[type[i1]];
      m2 = mass[type[i2]];
    }
    double corr[3];
    for (int i = 0; i < 3; i++) {
      corr[i] = -(lamda01 + lamda12) * r01[i] - (lamda02 + lamda12) * r02[i];
      xshake[i0][i] += corr[i] / m0;
    }
    if (i0 < nlocal)
      for (int i = 0; i < 3; i++) f[i0][i] += corr[i] / dtfsq;
    for (int i = 0; i < 3; i++) {
      corr[i] = lamda01 * r01[i] + lamda12 * r02[i];
      xshake[i1][i] += corr[i] / m1;
    }
    if (i1 < nlocal)
      for (int i = 0; i < 3; i++) f[i1][i] += corr[i] / dtfsq;
    for (int i = 0; i < 3; i++) {
      corr[i] = lamda12 * r01[i] + lamda02 * r02[i];
      xshake[i2][i] += corr[i] / m2;
    }
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
      f[i0][0] -= (lamda01 + lamda12) * r01[0] + (lamda02 + lamda12) * r02[0];
      f[i0][1] -= (lamda01 + lamda12) * r01[1] + (lamda02 + lamda12) * r02[1];
      f[i0][2] -= (lamda01 + lamda12) * r01[2] + (lamda02 + lamda12) * r02[2];
    }
    if (i1 < nlocal) {
      f[i1][0] += lamda01 * r01[0] + lamda12 * r02[0];
      f[i1][1] += lamda01 * r01[1] + lamda12 * r02[1];
      f[i1][2] += lamda01 * r01[2] + lamda12 * r02[2];
    }
    if (i2 < nlocal) {
      f[i2][0] += lamda02 * r02[0] + lamda12 * r01[0];
      f[i2][1] += lamda02 * r02[1] + lamda12 * r01[1];
      f[i2][2] += lamda02 * r02[2] + lamda12 * r01[2];
    }
  }
  if (evflag) {
    int count = 0;
    if (i0 < nlocal) atomlist[count++] = i0;
    if (i1 < nlocal) atomlist[count++] = i1;
    if (i2 < nlocal) atomlist[count++] = i2;

    double r12[3];
    r12[0] = r02[0] - r01[0];
    r12[1] = r02[1] - r01[1];
    r12[2] = r02[2] - r01[2];

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
     R row k = x[i0] - x[partner_k]  (center minus partner)
     S row k = xshake[i0] - xshake[partner_k]
     force sign = -1/dtfsq

   DIHEDRAL: chain topology A-B-C-D stored as 1-0-2-3
     R row 0 = x[i1]-x[i0], row 1 = x[i0]-x[i2], row 2 = x[i2]-x[i3]
     S row k = xshake version of same
     force sign = +1 (use dtf^2 multiplier internally)
   ------------------------------------------------------------------------- */

void FixRigs::solve3x3(int ilist, Topology topo)
{
  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  int i3 = closest_list[ilist][3];

  // row_source[k] = {ia, ib} means R(k,*) = x[ia] - x[ib]
  // dihedral: row 0 = B-A (i1-i0), row 1 = A-C (i0-i2), row 2 = C-D (i2-i3)
  int row_ia[3], row_ib[3];
  if (topo == IMPROPER) {
    row_ia[0] = i0; row_ib[0] = i1;
    row_ia[1] = i0; row_ib[1] = i2;
    row_ia[2] = i0; row_ib[2] = i3;
  } else {
    row_ia[0] = i1; row_ib[0] = i0;
    row_ia[1] = i0; row_ib[1] = i2;
    row_ia[2] = i2; row_ib[2] = i3;
  }

  double force_scale = (topo == IMPROPER) ? -1.0 / dtfsq : dtfsq;

  Mat3 R, S;
  for (int row = 0; row < 3; row++) {
    R(row, 0) = x[row_ia[row]][0] - x[row_ib[row]][0];
    R(row, 1) = x[row_ia[row]][1] - x[row_ib[row]][1];
    R(row, 2) = x[row_ia[row]][2] - x[row_ib[row]][2];
    S(row, 0) = xshake[row_ia[row]][0] - xshake[row_ib[row]][0];
    S(row, 1) = xshake[row_ia[row]][1] - xshake[row_ib[row]][1];
    S(row, 2) = xshake[row_ia[row]][2] - xshake[row_ib[row]][2];
  }

  SymMat3 rr = mmt(R);
  SymMat3 ss = mmt(S);

  int idx = ilist_to_idx[ilist];
  const double *Lp = L_entries[idx].data;
  const double *lmp = rmass ? rigs_lm_atom[m] : lm_entries[idx].data;
  SymMat3 L_mat = {Lp[0], Lp[1], Lp[2], Lp[3], Lp[4], Lp[5]};
  SymMat3 diff = L_mat - ss;
  DChol3 lm_chol = {lmp[0], lmp[1], lmp[2], lmp[3], lmp[4], lmp[5]};

  Mat3 chi = mat_dot(R, S);
  UTMat3 rc = inv_chol_upper(rr);
  ut_mul(rc, chi);
  SymMat3 sigma = diff + mtm(chi);
  u_mul(rc, chi);
  lslt_mul(sigma, lm_chol);
  LTMat3 sc = mul_dl(chol_lower(sigma), lm_chol);
  mul_ltdl(chi, lm_chol);

  Mat3 lamda = cayley_converge(rc, sc, chi, max_iter, tolerance);
  lamda += chi;

  Mat43 L_lam = (topo == IMPROPER) ? improper_L_lambda(lamda)
                                      : dihedral_L_lambda(lamda);
  L_lam *= R;

  if (i0 < nlocal)
    for (int i = 0; i < 3; i++) f[i0][i] += force_scale * L_lam(0, i);
  if (i1 < nlocal)
    for (int i = 0; i < 3; i++) f[i1][i] += force_scale * L_lam(1, i);
  if (i2 < nlocal)
    for (int i = 0; i < 3; i++) f[i2][i] += force_scale * L_lam(2, i);
  if (i3 < nlocal)
    for (int i = 0; i < 3; i++) f[i3][i] += force_scale * L_lam(3, i);
}


