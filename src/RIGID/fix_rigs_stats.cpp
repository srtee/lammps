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

#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "math_const.h"
#include "molecule.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

int FixRigs::bondtype_find(int i, tagint partner, int setflag)
{
  return FixShake::bondtype_findset(i, atom->tag[i], partner, setflag);
}

int FixRigs::improper_check(int i)
{
  tagint a1 = shake_atom[i][1];
  tagint a2 = shake_atom[i][2];
  tagint a3_atom = shake_atom[i][3];

  if (molecular == Atom::MOLECULAR) {
    int nimp = atom->num_improper[i];
    for (int m = 0; m < nimp; m++) {
      if (atom->improper_type[i][m] <= 0) continue;
      if (!improper_flag[atom->improper_type[i][m]]) continue;
      tagint b1 = atom->improper_atom1[i][m];
      tagint b2 = atom->improper_atom2[i][m];
      tagint b3 = atom->improper_atom3[i][m];
      tagint b4 = atom->improper_atom4[i][m];

      if (b1 != atom->tag[i]) continue;
      if ((b2 == a1 && b3 == a2 && b4 == a3_atom) ||
          (b2 == a1 && b3 == a3_atom && b4 == a2) ||
          (b2 == a2 && b3 == a1 && b4 == a3_atom) ||
          (b2 == a2 && b3 == a3_atom && b4 == a1) ||
          (b2 == a3_atom && b3 == a1 && b4 == a2) ||
          (b2 == a3_atom && b3 == a2 && b4 == a1))
        return 1;
    }
  } else {
    int imol = atom->molindex[i];
    int iatom = atom->molatom[i];
    tagint tagprev = atom->tag[i] - iatom - 1;
    int nimp = atommols[imol]->num_improper[iatom];
    for (int m = 0; m < nimp; m++) {
      if (atommols[imol]->improper_type[iatom][m] <= 0) continue;
      if (!improper_flag[atommols[imol]->improper_type[iatom][m]]) continue;
      tagint b1 = atommols[imol]->improper_atom1[iatom][m] + tagprev;
      tagint b2 = atommols[imol]->improper_atom2[iatom][m] + tagprev;
      tagint b3 = atommols[imol]->improper_atom3[iatom][m] + tagprev;
      tagint b4 = atommols[imol]->improper_atom4[iatom][m] + tagprev;
      if (b1 != atom->tag[i]) continue;
      if ((b2 == a1 && b3 == a2 && b4 == a3_atom) ||
          (b2 == a1 && b3 == a3_atom && b4 == a2) ||
          (b2 == a2 && b3 == a1 && b4 == a3_atom) ||
          (b2 == a2 && b3 == a3_atom && b4 == a1) ||
          (b2 == a3_atom && b3 == a1 && b4 == a2) ||
          (b2 == a3_atom && b3 == a2 && b4 == a1))
        return 1;
    }
  }
  return 0;
}

void FixRigs::fill_improper_types(int i)
{
  tagint a1 = shake_atom[i][1];
  tagint a2 = shake_atom[i][2];
  tagint a3_atom = shake_atom[i][3];

  rigs_type[i][0] = 0;
  rigs_type[i][1] = 0;
  rigs_type[i][2] = 0;

  int n = angletype_findset(i, a1, a2, 0);
  if (n > 0 && angle_flag[n]) rigs_type[i][0] = n;

  n = angletype_findset(i, a1, a3_atom, 0);
  if (n > 0 && angle_flag[n]) rigs_type[i][1] = n;

  n = angletype_findset(i, a2, a3_atom, 0);
  if (n > 0 && angle_flag[n]) rigs_type[i][2] = n;

  int nimp = impropertype_findset(i, shake_atom[i][0], a1, a2, a3_atom, 0);
  if (nimp > 0 && improper_flag[nimp]) {
    if (rigs_type[i][2] == 0) rigs_type[i][2] = -nimp;
  }
}

int FixRigs::impropertype_findset(int i, tagint n0, tagint n1, tagint n2, tagint n3, int setflag)
{
  int m, nimp;

  if (molecular == Atom::MOLECULAR) {
    nimp = atom->num_improper[i];
    for (m = 0; m < nimp; m++) {
      tagint b0 = atom->improper_atom1[i][m];
      tagint b1 = atom->improper_atom2[i][m];
      tagint b2 = atom->improper_atom3[i][m];
      tagint b3 = atom->improper_atom4[i][m];
      if (b0 != n0) continue;
      if (b1 == n1 && b2 == n2 && b3 == n3) break;
      if (b1 == n1 && b2 == n3 && b3 == n2) break;
      if (b1 == n2 && b2 == n1 && b3 == n3) break;
    }
  } else {
    int imol = atom->molindex[i];
    int iatom = atom->molatom[i];
    tagint tagprev = atom->tag[i] - iatom - 1;
    nimp = atommols[imol]->num_improper[iatom];
    int *itype = atommols[imol]->improper_type[iatom];
    for (m = 0; m < nimp; m++) {
      tagint b0 = atommols[imol]->improper_atom1[iatom][m] + tagprev;
      tagint b1 = atommols[imol]->improper_atom2[iatom][m] + tagprev;
      tagint b2 = atommols[imol]->improper_atom3[iatom][m] + tagprev;
      tagint b3 = atommols[imol]->improper_atom4[iatom][m] + tagprev;
      if (b0 != n0) continue;
      if (b1 == n1 && b2 == n2 && b3 == n3) break;
      if (b1 == n1 && b2 == n3 && b3 == n2) break;
      if (b1 == n2 && b2 == n1 && b3 == n3) break;
    }
  }

  if (m < nimp) {
    if (setflag == 0) {
      if (molecular == Atom::MOLECULAR) return atom->improper_type[i][m];
      else {
        int *itype = atommols[atom->molindex[i]]->improper_type[atom->molatom[i]];
        return itype[m];
      }
    }
    if (molecular == Atom::MOLECULAR) {
      if ((setflag < 0 && atom->improper_type[i][m] > 0) ||
          (setflag > 0 && atom->improper_type[i][m] < 0))
        atom->improper_type[i][m] = -atom->improper_type[i][m];
    } else {
      int *itype = atommols[atom->molindex[i]]->improper_type[atom->molatom[i]];
      if ((setflag < 0 && itype[m] > 0) ||
          (setflag > 0 && itype[m] < 0))
        itype[m] = -itype[m];
    }
  }

  return 0;
}

int FixRigs::dihedraltype_findset(int i, tagint n1, tagint n2, tagint n3, tagint n4, int setflag)
{
  int m, ndih;
  int *dtype;

  if (molecular == Atom::MOLECULAR) {
    ndih = atom->num_dihedral[i];
    for (m = 0; m < ndih; m++) {
      tagint d1 = atom->dihedral_atom1[i][m];
      tagint d2 = atom->dihedral_atom2[i][m];
      tagint d3 = atom->dihedral_atom3[i][m];
      tagint d4 = atom->dihedral_atom4[i][m];
      if (d1 == n1 && d2 == n2 && d3 == n3 && d4 == n4) break;
      if (d1 == n4 && d2 == n3 && d3 == n2 && d4 == n1) break;
    }
  } else {
    int imol = atom->molindex[i];
    int iatom = atom->molatom[i];
    tagint tagprev = atom->tag[i] - iatom - 1;
    ndih = atommols[imol]->num_dihedral[iatom];
    dtype = atommols[imol]->dihedral_type[iatom];
    for (m = 0; m < ndih; m++) {
      tagint d1 = atommols[imol]->dihedral_atom1[iatom][m] + tagprev;
      tagint d2 = atommols[imol]->dihedral_atom2[iatom][m] + tagprev;
      tagint d3 = atommols[imol]->dihedral_atom3[iatom][m] + tagprev;
      tagint d4 = atommols[imol]->dihedral_atom4[iatom][m] + tagprev;
      if (d1 == n1 && d2 == n2 && d3 == n3 && d4 == n4) break;
      if (d1 == n4 && d2 == n3 && d3 == n2 && d4 == n1) break;
    }
  }

  if (m < ndih) {
    if (setflag == 0) {
      if (molecular == Atom::MOLECULAR) return atom->dihedral_type[i][m];
      else return dtype[m];
    }
    if (molecular == Atom::MOLECULAR) {
      if ((setflag < 0 && atom->dihedral_type[i][m] > 0) ||
          (setflag > 0 && atom->dihedral_type[i][m] < 0))
        atom->dihedral_type[i][m] = -atom->dihedral_type[i][m];
    } else {
      if ((setflag < 0 && dtype[m] > 0) ||
          (setflag > 0 && dtype[m] < 0))
        dtype[m] = -dtype[m];
    }
  }

  return 0;
}

void FixRigs::min_post_force(int vflag)
{
  FixShake::min_post_force(vflag);

  int atom1, atom2, atom3, atom4;

  for (int i = 0; i < nlocal; i++) {
    if (shake_flag[i] != 5 && shake_flag[i] != 6) continue;
    atom1 = atom->map(shake_atom[i][0]);
    atom2 = atom->map(shake_atom[i][1]);
    atom3 = atom->map(shake_atom[i][2]);
    atom4 = atom->map(shake_atom[i][3]);
    if (atom1 == -1 || atom2 == -1 || atom3 == -1 || atom4 == -1)
      error->one(FLERR, "RIGS atoms {} {} {} missing on proc {} at step {}{}",
                 shake_atom[i][1], shake_atom[i][2], shake_atom[i][3],
                 comm->me, update->ntimestep, utils::errorurl(5));
    atom1 = domain->closest_image(i, atom1);
    atom2 = domain->closest_image(i, atom2);
    atom3 = domain->closest_image(i, atom3);
    atom4 = domain->closest_image(i, atom4);
    if (i <= atom1 && i <= atom2 && i <= atom3 && i <= atom4) {
      if (rigs_type[i][0] > 0)
        bond_force(atom2, atom3, rigs_angle_distance[rigs_type[i][0]]);
      if (rigs_type[i][1] > 0)
        bond_force(atom2, atom4, rigs_angle_distance[rigs_type[i][1]]);
      if (rigs_type[i][2] > 0)
        bond_force(atom3, atom4, rigs_angle_distance[rigs_type[i][2]]);
    }
  }
}

void FixRigs::stats()
{
  int nb = atom->nbondtypes + 1;
  int na = atom->nangletypes + 1;
  static constexpr double BIG = 1.0e20;
  for (int i = 0; i < nb; i++) {
    b_count[i] = 0;
    b_ave[i] = b_max[i] = 0.0;
    b_min[i] = BIG;
  }
  for (int i = 0; i < na; i++) {
    a_count[i] = 0;
    a_ave[i] = a_max[i] = 0.0;
    a_min[i] = BIG;
  }

  double **xx = atom->x;
  int nlocal = atom->nlocal;

  for (int ii = 0; ii < nlist; ++ii) {
    int i = list[ii];
    int n = shake_flag[i];
    if (n == 0) continue;

    // bond stats for all cluster types
    if (n == 1) n = 3;
    else if (n > 4) n = 4;
    int iatom = closest_list[ii][0];
    for (int j = 1; j < n; j++) {
      int jatom = closest_list[ii][j];
      if (jatom >= nlocal) continue;
      double delx = xx[iatom][0] - xx[jatom][0];
      double dely = xx[iatom][1] - xx[jatom][1];
      double delz = xx[iatom][2] - xx[jatom][2];
      double r = sqrt(delx*delx + dely*dely + delz*delz);
      int m = shake_type[i][j-1];
      b_count[m]++;
      b_ave[m] += r;
      b_max[m] = MAX(b_max[m],r);
      b_min[m] = MIN(b_min[m],r);
    }

    // angle stats for shake_flag==1 (standard 3-atom angle clusters)
    if (shake_flag[i] == 1) {
      int i0 = closest_list[ii][0];
      int i1 = closest_list[ii][1];
      int i2 = closest_list[ii][2];
      int nc = 0;
      if (i0 < nlocal) ++nc;
      if (i1 < nlocal) ++nc;
      if (i2 < nlocal) ++nc;

      double delx = xx[i0][0] - xx[i1][0];
      double dely = xx[i0][1] - xx[i1][1];
      double delz = xx[i0][2] - xx[i1][2];
      double r1 = sqrt(delx*delx + dely*dely + delz*delz);

      delx = xx[i0][0] - xx[i2][0];
      dely = xx[i0][1] - xx[i2][1];
      delz = xx[i0][2] - xx[i2][2];
      double r2 = sqrt(delx*delx + dely*dely + delz*delz);

      delx = xx[i1][0] - xx[i2][0];
      dely = xx[i1][1] - xx[i2][1];
      delz = xx[i1][2] - xx[i2][2];
      double r3 = sqrt(delx*delx + dely*dely + delz*delz);

      double angle = acos((r1*r1 + r2*r2 - r3*r3) / (2.0*r1*r2)) * (180.0/MY_PI);
      int m = shake_type[i][2];
      a_count[m] += nc;
      a_ave[m] += nc*angle;
      a_max[m] = MAX(a_max[m],angle);
      a_min[m] = MIN(a_min[m],angle);
    }

    // angle stats for shake_flag==5 (improper clusters)
    if (shake_flag[i] == 5) {
      int i0 = closest_list[ii][0];
      int i1 = closest_list[ii][1];
      int i2 = closest_list[ii][2];
      int i3 = closest_list[ii][3];

      if (rigs_type[i][0] > 0) {
        double r01 = 0.0, r02 = 0.0, r12 = 0.0;
        for (int d = 0; d < 3; d++) {
          double d01 = xx[i0][d] - xx[i1][d];
          double d02 = xx[i0][d] - xx[i2][d];
          double d12 = xx[i1][d] - xx[i2][d];
          r01 += d01 * d01;
          r02 += d02 * d02;
          r12 += d12 * d12;
        }
        r01 = sqrt(r01); r02 = sqrt(r02); r12 = sqrt(r12);
        double ang = acos((r01*r01 + r02*r02 - r12*r12) / (2.0*r01*r02)) * (180.0/MY_PI);
        int m = rigs_type[i][0];
        int nc = 0;
        if (i0 < nlocal) nc++;
        if (i1 < nlocal) nc++;
        if (i2 < nlocal) nc++;
        a_count[m] += nc;
        a_ave[m] += nc * ang;
        a_max[m] = MAX(a_max[m], ang);
        a_min[m] = MIN(a_min[m], ang);
      }

      if (rigs_type[i][1] > 0) {
        double r01 = 0.0, r03 = 0.0, r13 = 0.0;
        for (int d = 0; d < 3; d++) {
          double d01 = xx[i0][d] - xx[i1][d];
          double d03 = xx[i0][d] - xx[i3][d];
          double d13 = xx[i1][d] - xx[i3][d];
          r01 += d01 * d01;
          r03 += d03 * d03;
          r13 += d13 * d13;
        }
        r01 = sqrt(r01); r03 = sqrt(r03); r13 = sqrt(r13);
        double ang = acos((r01*r01 + r03*r03 - r13*r13) / (2.0*r01*r03)) * (180.0/MY_PI);
        int m = rigs_type[i][1];
        int nc = 0;
        if (i0 < nlocal) nc++;
        if (i1 < nlocal) nc++;
        if (i3 < nlocal) nc++;
        a_count[m] += nc;
        a_ave[m] += nc * ang;
        a_max[m] = MAX(a_max[m], ang);
        a_min[m] = MIN(a_min[m], ang);
      }

      if (rigs_type[i][2] > 0) {
        double r02 = 0.0, r03 = 0.0, r23 = 0.0;
        for (int d = 0; d < 3; d++) {
          double d02 = xx[i0][d] - xx[i2][d];
          double d03 = xx[i0][d] - xx[i3][d];
          double d23 = xx[i2][d] - xx[i3][d];
          r02 += d02 * d02;
          r03 += d03 * d03;
          r23 += d23 * d23;
        }
        r02 = sqrt(r02); r03 = sqrt(r03); r23 = sqrt(r23);
        double ang = acos((r02*r02 + r03*r03 - r23*r23) / (2.0*r02*r03)) * (180.0/MY_PI);
        int m = rigs_type[i][2];
        int nc = 0;
        if (i0 < nlocal) nc++;
        if (i2 < nlocal) nc++;
        if (i3 < nlocal) nc++;
        a_count[m] += nc;
        a_ave[m] += nc * ang;
        a_max[m] = MAX(a_max[m], ang);
        a_min[m] = MIN(a_min[m], ang);
      }
    }

    // angle stats for shake_flag==6 (dihedral clusters)
    if (shake_flag[i] == 6) {
      int i0 = closest_list[ii][0];
      int i1 = closest_list[ii][1];
      int i2 = closest_list[ii][2];
      int i3 = closest_list[ii][3];

      if (rigs_type[i][0] > 0) {
        double r01 = 0.0, r02 = 0.0, r12 = 0.0;
        for (int d = 0; d < 3; d++) {
          double d01 = xx[i1][d] - xx[i0][d];
          double d02 = xx[i0][d] - xx[i2][d];
          double d12 = xx[i2][d] - xx[i1][d];
          r01 += d01 * d01;
          r02 += d02 * d02;
          r12 += d12 * d12;
        }
        r01 = sqrt(r01); r02 = sqrt(r02); r12 = sqrt(r12);
        double ang = acos((r01*r01 + r02*r02 - r12*r12) / (2.0*r01*r02)) * (180.0/MY_PI);
        int m = rigs_type[i][0];
        int nc = 0;
        if (i0 < nlocal) nc++;
        if (i1 < nlocal) nc++;
        if (i2 < nlocal) nc++;
        a_count[m] += nc;
        a_ave[m] += nc * ang;
        a_max[m] = MAX(a_max[m], ang);
        a_min[m] = MIN(a_min[m], ang);
      }

      if (rigs_type[i][1] > 0) {
        double r02 = 0.0, r23 = 0.0, r03 = 0.0;
        for (int d = 0; d < 3; d++) {
          double d02 = xx[i0][d] - xx[i2][d];
          double d23 = xx[i2][d] - xx[i3][d];
          double d03 = xx[i3][d] - xx[i0][d];
          r02 += d02 * d02;
          r23 += d23 * d23;
          r03 += d03 * d03;
        }
        r02 = sqrt(r02); r23 = sqrt(r23); r03 = sqrt(r03);
        double ang = acos((r02*r02 + r23*r23 - r03*r03) / (2.0*r02*r23)) * (180.0/MY_PI);
        int m = rigs_type[i][1];
        int nc = 0;
        if (i0 < nlocal) nc++;
        if (i2 < nlocal) nc++;
        if (i3 < nlocal) nc++;
        a_count[m] += nc;
        a_ave[m] += nc * ang;
        a_max[m] = MAX(a_max[m], ang);
        a_min[m] = MIN(a_min[m], ang);
      }
    }
  }

  MPI_Allreduce(b_count, b_count_all, nb, MPI_LMP_BIGINT, MPI_SUM, world);
  MPI_Allreduce(b_ave, b_ave_all, nb, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(b_max, b_max_all, nb, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(b_min, b_min_all, nb, MPI_DOUBLE, MPI_MIN, world);

  MPI_Allreduce(a_count, a_count_all, na, MPI_LMP_BIGINT, MPI_SUM, world);
  MPI_Allreduce(a_ave, a_ave_all, na, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(a_max, a_max_all, na, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(a_min, a_min_all, na, MPI_DOUBLE, MPI_MIN, world);

  MPI_Allreduce(iter_b_count, iter_b_count_all, nb, MPI_LMP_BIGINT, MPI_SUM, world);
  MPI_Allreduce(iter_b_total, iter_b_total_all, nb, MPI_LMP_BIGINT, MPI_SUM, world);
  MPI_Allreduce(iter_a_count, iter_a_count_all, na, MPI_LMP_BIGINT, MPI_SUM, world);
  MPI_Allreduce(iter_a_total, iter_a_total_all, na, MPI_LMP_BIGINT, MPI_SUM, world);

  if (comm->me == 0) {
    const int width = (int) log10((double)(MAX(MAX(1,nb),na))) + 2;
    auto mesg = fmt::format("{} stats (type/ave/delta/count/ave_iters) on step {}\n",
                            utils::uppercase(style), update->ntimestep);
    for (int i = 1; i < nb; i++) {
      const auto bcnt = b_count_all[i];
      if (bcnt) {
        double ave_iters = (iter_b_count_all[i] > 0) ?
          (double) iter_b_total_all[i] / (double) iter_b_count_all[i] : 0.0;
        mesg += fmt::format("Bond:  {:>{}d}   {:<9.6} {:<11.6} {:>8d}   {:<9.3f}\n",i,width,
                            b_ave_all[i]/bcnt,b_max_all[i]-b_min_all[i],bcnt,ave_iters);
      }
    }
    for (int i = 1; i < na; i++) {
      const auto acnt = a_count_all[i];
      if (acnt) {
        double ave_iters = (iter_a_count_all[i] > 0) ?
          (double) iter_a_total_all[i] / (double) iter_a_count_all[i] : 0.0;
        mesg += fmt::format("Angle: {:>{}d}   {:<9.6} {:<11.6} {:>8d}   {:<9.3f}\n",i,width,
                            a_ave_all[i]/acnt,a_max_all[i]-a_min_all[i],acnt/3,ave_iters);
      }
    }
    utils::logmesg(lmp,mesg);
  }

  // reset iteration accumulators for next stats period
  for (int i = 0; i < nb; i++) {
    iter_b_count[i] = 0;
    iter_b_total[i] = 0;
  }
  for (int i = 0; i < na; i++) {
    iter_a_count[i] = 0;
    iter_a_total[i] = 0;
  }

  next_output += output_every;
}
