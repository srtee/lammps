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
#include "atom_vec.h"
#include "comm.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "modify.h"
#include "molecule.h"
#include "update.h"

using namespace LAMMPS_NS;

void FixRigs::post_constructor()
{
  grow_arrays(atom->nmax);

  int i, m;
  int nlocal = atom->nlocal;
  int *mask_atom = atom->mask;

  atommols = atom->avec->onemols;

  int dihedrals_allow = atom->avec->dihedrals_allow;

  for (i = 0; i < nlocal; i++) {
    rigs_type[i][0] = 0;
    rigs_type[i][1] = 0;
    rigs_type[i][2] = 0;
  }

  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] != -1 && shake_flag[i] != 6) shake_flag[i] = 0;
  }

  if (dihedrals_allow && molecular == Atom::MOLECULAR) {
    for (i = 0; i < nlocal; i++) {
      if (!(mask_atom[i] & groupbit)) continue;
      int ndih = atom->num_dihedral[i];
      for (m = 0; m < ndih; m++) {
        if (atom->dihedral_type[i][m] <= 0) continue;
        if (!dihedral_flag[atom->dihedral_type[i][m]]) continue;

        tagint da1 = atom->dihedral_atom1[i][m];
        tagint da2 = atom->dihedral_atom2[i][m];
        tagint da3 = atom->dihedral_atom3[i][m];
        tagint da4 = atom->dihedral_atom4[i][m];

        tagint b, c;
        if (atom->tag[i] == da2) {
          b = da2; c = da3;
        } else if (atom->tag[i] == da3) {
          b = da3; c = da2;
        } else {
          continue;
        }

        tagint a, d;
        if (b == da2 && c == da3) {
          a = da1; d = da4;
        } else {
          a = da4; d = da1;
        }

        tagint owner_tag = MIN(da2, da3);
        if (atom->tag[i] != owner_tag) continue;

        int a_idx = atom->map(a);
        int b_idx = atom->map(b);
        int c_idx = atom->map(c);
        int d_idx = atom->map(d);
        if (a_idx < 0 || b_idx < 0 || c_idx < 0 || d_idx < 0) continue;
        if (!(mask_atom[a_idx] & groupbit)) continue;
        if (!(mask_atom[b_idx] & groupbit)) continue;
        if (!(mask_atom[c_idx] & groupbit)) continue;
        if (!(mask_atom[d_idx] & groupbit)) continue;

        int bond_ab = bondtype_find(b_idx, a, 0);
        int bond_bc = bondtype_find(b_idx, c, 0);
        int bond_cd = bondtype_find(c_idx, d, 0);
        if (bond_ab <= 0 || bond_cd <= 0) continue;

        int bond_bc_type = bondtype_find(b_idx, c, 0);
        if (bond_bc_type <= 0) bond_bc_type = bondtype_find(c_idx, b, 0);
        if (bond_bc_type <= 0) continue;

        if (atom->avec->angles_allow) {
          int angle1 = angletype_findset(b_idx, a, c, 0);
          if (angle1 <= 0 || !angle_flag[angle1]) continue;
          int angle2 = angletype_findset(c_idx, b, d, 0);
          if (angle2 <= 0 || !angle_flag[angle2]) continue;
        }

        if (shake_flag[b_idx] != 0) continue;

        shake_flag[b_idx] = 6;
        shake_atom[b_idx][0] = b;
        shake_atom[b_idx][1] = a;
        shake_atom[b_idx][2] = c;
        shake_atom[b_idx][3] = d;
        shake_type[b_idx][0] = bond_ab;
        shake_type[b_idx][1] = bond_bc_type;
        shake_type[b_idx][2] = bond_cd;

        rigs_type[b_idx][0] = angletype_findset(b_idx, a, c, 0);
        rigs_type[b_idx][1] = angletype_findset(c_idx, b, d, 0);
        rigs_type[b_idx][2] = atom->dihedral_type[i][m];

        if (c_idx < nlocal && shake_flag[c_idx] == 0) {
          shake_flag[c_idx] = 6;
          shake_atom[c_idx][0] = b;
          shake_atom[c_idx][1] = a;
          shake_atom[c_idx][2] = c;
          shake_atom[c_idx][3] = d;
          shake_type[c_idx][0] = bond_ab;
          shake_type[c_idx][1] = bond_bc_type;
          shake_type[c_idx][2] = bond_cd;
          rigs_type[c_idx][0] = rigs_type[b_idx][0];
          rigs_type[c_idx][1] = rigs_type[b_idx][1];
          rigs_type[c_idx][2] = rigs_type[b_idx][2];
        }

        if (a_idx < nlocal) shake_flag[a_idx] = -1;
        if (d_idx < nlocal) shake_flag[d_idx] = -1;
      }
    }
  }

  find_clusters();

  int impropers_allow = atom->avec->impropers_allow;
  int nimproper = 0;

  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] != 4) continue;
    if (!(mask_atom[i] & groupbit)) continue;
    if (shake_atom[i][0] != atom->tag[i]) continue;

    int angles_allow_flag = atom->avec->angles_allow;
    int nangle_found = 0;
    if (angles_allow_flag) {
      tagint a1 = shake_atom[i][1];
      tagint a2 = shake_atom[i][2];
      tagint a3_atom = shake_atom[i][3];
      int n1 = angletype_findset(i, a1, a2, 0);
      int n2 = angletype_findset(i, a1, a3_atom, 0);
      int n3 = angletype_findset(i, a2, a3_atom, 0);
      if (n1 > 0 && angle_flag[n1]) nangle_found++;
      if (n2 > 0 && angle_flag[n2]) nangle_found++;
      if (n3 > 0 && angle_flag[n3]) nangle_found++;
    }
    if (nangle_found == 3) {
      shake_flag[i] = 5;
      nimproper++;
    } else if (impropers_allow) {
      int impflag = improper_check(i);
      if (impflag) {
        shake_flag[i] = 5;
        nimproper++;
      }
    }
  }

  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] == 5) {
      fill_improper_types(i);
    }
  }

  {
    bigint count4 = 0, count5 = 0;
    for (i = 0; i < nlocal; i++) {
      if (shake_flag[i] == 4 && shake_atom[i][0] == atom->tag[i]) count4++;
      if (shake_flag[i] == 5 && shake_atom[i][0] == atom->tag[i]) count5++;
    }
    bigint tmp4, tmp5;
    MPI_Allreduce(&count4, &tmp4, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    MPI_Allreduce(&count5, &tmp5, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    if (comm->me == 0)
      utils::logmesg(lmp, "{:>8} = # of remaining size 4 clusters\n"
                     "{:>8} = # of upgraded improper clusters\n",
                     tmp4, tmp5);
  }

  transform_clusters_global(4, 5);
  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] == 5) {
      if (rigs_type[i][0] > 0)
        angletype_findset(i, shake_atom[i][1], shake_atom[i][2], -1);
      if (rigs_type[i][1] > 0)
        angletype_findset(i, shake_atom[i][1], shake_atom[i][3], -1);
      if (rigs_type[i][2] > 0)
        angletype_findset(i, shake_atom[i][2], shake_atom[i][3], -1);
    } else if (shake_flag[i] == 6) {
      if (atom->tag[i] == shake_atom[i][0]) {
        bondtype_findset(i, shake_atom[i][0], shake_atom[i][1], -1);
        bondtype_findset(i, shake_atom[i][0], shake_atom[i][2], -1);
        if (rigs_type[i][0] > 0)
          angletype_findset(i, shake_atom[i][1], shake_atom[i][2], -1);
        dihedraltype_findset(i, shake_atom[i][0], shake_atom[i][1],
                             shake_atom[i][2], shake_atom[i][3], -1);
      } else {
        bondtype_findset(i, shake_atom[i][2], shake_atom[i][3], -1);
        if (rigs_type[i][1] > 0)
          angletype_findset(i, shake_atom[i][0], shake_atom[i][3], -1);
      }
    }
  }
}

void FixRigs::transform_clusters(int from_flag, int to_flag, bool global, bool propagate_shake_data)
{
  int i;
  int nlocal = atom->nlocal;
  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] != to_flag) continue;
    if (shake_atom[i][0] != atom->tag[i]) continue;
    for (int k = 1; k <= 3; k++) {
      int pidx = atom->map(shake_atom[i][k]);
      if (pidx >= 0 && pidx < nlocal && shake_flag[pidx] == from_flag) {
        shake_flag[pidx] = to_flag;
        rigs_type[pidx][0] = rigs_type[i][0];
        rigs_type[pidx][1] = rigs_type[i][1];
        rigs_type[pidx][2] = rigs_type[i][2];
        if (propagate_shake_data) {
          shake_type[pidx][0] = shake_type[i][0];
          shake_type[pidx][1] = shake_type[i][1];
          shake_type[pidx][2] = shake_type[i][2];
          shake_atom[pidx][0] = shake_atom[i][0];
          shake_atom[pidx][1] = shake_atom[i][1];
          shake_atom[pidx][2] = shake_atom[i][2];
          shake_atom[pidx][3] = shake_atom[i][3];
        }
      }
    }
  }

  if (!global) return;

  int nsend = 0;
  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] == to_flag && shake_atom[i][0] == atom->tag[i])
      nsend++;
  }

  if (propagate_shake_data) {
    struct UpgradedCluster {
      tagint central;
      int rt0, rt1, rt2;
      int st0, st1, st2;
      tagint sa0, sa1, sa2, sa3;
    };
    const int sizeof_uc = sizeof(UpgradedCluster);
    UpgradedCluster *sendbuf = new UpgradedCluster[nsend > 0 ? nsend : 1];
    nsend = 0;
    for (i = 0; i < nlocal; i++) {
      if (shake_flag[i] == to_flag && shake_atom[i][0] == atom->tag[i]) {
        sendbuf[nsend].central = atom->tag[i];
        sendbuf[nsend].rt0 = rigs_type[i][0];
        sendbuf[nsend].rt1 = rigs_type[i][1];
        sendbuf[nsend].rt2 = rigs_type[i][2];
        sendbuf[nsend].st0 = shake_type[i][0];
        sendbuf[nsend].st1 = shake_type[i][1];
        sendbuf[nsend].st2 = shake_type[i][2];
        sendbuf[nsend].sa0 = shake_atom[i][0];
        sendbuf[nsend].sa1 = shake_atom[i][1];
        sendbuf[nsend].sa2 = shake_atom[i][2];
        sendbuf[nsend].sa3 = shake_atom[i][3];
        nsend++;
      }
    }

    int nsend_bytes = nsend * sizeof_uc;
    int *recvcounts = new int[comm->nprocs];
    MPI_Allgather(&nsend_bytes, 1, MPI_INT, recvcounts, 1, MPI_INT, world);

    int totalrecv = 0;
    int *displs = new int[comm->nprocs];
    for (int p = 0; p < comm->nprocs; p++) {
      displs[p] = totalrecv;
      totalrecv += recvcounts[p];
    }

    int totalrecv_n = totalrecv / sizeof_uc;
    UpgradedCluster *recvbuf = new UpgradedCluster[totalrecv_n > 0 ? totalrecv_n : 1];
    MPI_Allgatherv(sendbuf, nsend_bytes, MPI_CHAR,
                   recvbuf, recvcounts, displs, MPI_CHAR, world);

    for (int c = 0; c < totalrecv_n; c++) {
      tagint ctag = recvbuf[c].central;
      for (i = 0; i < nlocal; i++) {
        if (shake_flag[i] != from_flag) continue;
        if (shake_atom[i][0] != ctag) continue;
        shake_flag[i] = to_flag;
        rigs_type[i][0] = recvbuf[c].rt0;
        rigs_type[i][1] = recvbuf[c].rt1;
        rigs_type[i][2] = recvbuf[c].rt2;
        shake_type[i][0] = recvbuf[c].st0;
        shake_type[i][1] = recvbuf[c].st1;
        shake_type[i][2] = recvbuf[c].st2;
        shake_atom[i][0] = recvbuf[c].sa0;
        shake_atom[i][1] = recvbuf[c].sa1;
        shake_atom[i][2] = recvbuf[c].sa2;
        shake_atom[i][3] = recvbuf[c].sa3;
      }
    }

    delete[] sendbuf;
    delete[] recvbuf;
    delete[] recvcounts;
    delete[] displs;
  } else {
    struct UpgradedCluster { tagint central; int rt0, rt1, rt2; };
    const int sizeof_uc = sizeof(UpgradedCluster);
    UpgradedCluster *sendbuf = new UpgradedCluster[nsend > 0 ? nsend : 1];
    nsend = 0;
    for (i = 0; i < nlocal; i++) {
      if (shake_flag[i] == to_flag && shake_atom[i][0] == atom->tag[i]) {
        sendbuf[nsend].central = atom->tag[i];
        sendbuf[nsend].rt0 = rigs_type[i][0];
        sendbuf[nsend].rt1 = rigs_type[i][1];
        sendbuf[nsend].rt2 = rigs_type[i][2];
        nsend++;
      }
    }

    int nsend_bytes = nsend * sizeof_uc;
    int *recvcounts = new int[comm->nprocs];
    MPI_Allgather(&nsend_bytes, 1, MPI_INT, recvcounts, 1, MPI_INT, world);

    int totalrecv = 0;
    int *displs = new int[comm->nprocs];
    for (int p = 0; p < comm->nprocs; p++) {
      displs[p] = totalrecv;
      totalrecv += recvcounts[p];
    }

    int totalrecv_n = totalrecv / sizeof_uc;
    UpgradedCluster *recvbuf = new UpgradedCluster[totalrecv_n > 0 ? totalrecv_n : 1];
    MPI_Allgatherv(sendbuf, nsend_bytes, MPI_CHAR,
                   recvbuf, recvcounts, displs, MPI_CHAR, world);

    for (int c = 0; c < totalrecv_n; c++) {
      tagint ctag = recvbuf[c].central;
      for (i = 0; i < nlocal; i++) {
        if (shake_flag[i] != from_flag) continue;
        if (shake_atom[i][0] != ctag) continue;
        shake_flag[i] = to_flag;
        rigs_type[i][0] = recvbuf[c].rt0;
        rigs_type[i][1] = recvbuf[c].rt1;
        rigs_type[i][2] = recvbuf[c].rt2;
      }
    }

    delete[] sendbuf;
    delete[] recvbuf;
    delete[] recvcounts;
    delete[] displs;
  }
}

