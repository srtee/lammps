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
#include "dihedral.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "mat2.h"
#include "mat3.h"
#include "memory.h"
#include "molecule.h"
#include "modify.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace RigsMath;

FixRigs::FixRigs(LAMMPS *lmp, int narg, char **arg) :
    FixShake(lmp, narg, arg), rigs_type(nullptr), rigs_angle(nullptr),
    rigs_angle_distance(nullptr), rigs_improper_distance(nullptr),
    rigs_dihedral_distance(nullptr)
{
  restart_peratom = 1;
  atom->add_callback(Atom::RESTART);
}

FixRigs::~FixRigs()
{
  if (modify->get_fix_by_id(id)) atom->delete_callback(id, Atom::RESTART);
  memory->destroy(rigs_type);
  delete[] rigs_angle;
  delete[] rigs_angle_distance;
  delete[] rigs_improper_distance;
  delete[] rigs_dihedral_distance;
}

void FixRigs::post_constructor()
{
  grow_arrays(atom->nmax);

  int i, m;
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  int *mask_atom = atom->mask;

  atommols = atom->avec->onemols;

  int dihedrals_allow = atom->avec->dihedrals_allow;

  // dihedral detection: scan atom topology for chains A-B-C-D
  // where all 3 bonds are SHAKE-eligible, both angles are constrained,
  // and the dihedral type is in dihedral_flag
  // set shake_flag = 6 on the owner atom B, and -1 on partners A, C, D
  // find_clusters() will skip pre-assigned atoms

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

        // atom i must be one of the interior atoms (B or C)
        // ownership convention: B = min(da2, da3), stored first in shake_atom
        // check that i is an interior atom
        tagint b, c;
        if (tag[i] == da2) {
          b = da2; c = da3;
        } else if (tag[i] == da3) {
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

        // owner atom: lowest interior ID
        tagint owner_tag = MIN(da2, da3);
        if (tag[i] != owner_tag) continue;

        // check all 4 atoms are in the group
        int a_idx = atom->map(a);
        int b_idx = atom->map(b);
        int c_idx = atom->map(c);
        int d_idx = atom->map(d);
        if (a_idx < 0 || b_idx < 0 || c_idx < 0 || d_idx < 0) continue;
        if (!(mask_atom[a_idx] & groupbit)) continue;
        if (!(mask_atom[b_idx] & groupbit)) continue;
        if (!(mask_atom[c_idx] & groupbit)) continue;
        if (!(mask_atom[d_idx] & groupbit)) continue;

        // check 3 bonds are SHAKE-eligible
        int bond_ab = bondtype_find(b_idx, a, 0);
        int bond_bc = bondtype_find(b_idx, c, 0);
        int bond_cd = bondtype_find(c_idx, d, 0);
        if (bond_ab <= 0 || bond_cd <= 0) continue;

        // bond B-C: could be stored on either B or C
        int bond_bc_type = bondtype_find(b_idx, c, 0);
        if (bond_bc_type <= 0) bond_bc_type = bondtype_find(c_idx, b, 0);
        if (bond_bc_type <= 0) continue;

        // check both angles are in angle_flag
        if (atom->avec->angles_allow) {
          int angle1 = angletype_findset(b_idx, a, c, 0);
          if (angle1 <= 0 || !angle_flag[angle1]) continue;
          int angle2 = angletype_findset(c_idx, b, d, 0);
          if (angle2 <= 0 || !angle_flag[angle2]) continue;
        }

        // found a valid dihedral cluster
        // set owner atom (B, the lower interior ID)
        if (shake_flag[b_idx] != 0) continue;

        shake_flag[b_idx] = 6;
        shake_atom[b_idx][0] = b;
        shake_atom[b_idx][1] = a;
        shake_atom[b_idx][2] = c;
        shake_atom[b_idx][3] = d;
        shake_type[b_idx][0] = bond_ab;
        shake_type[b_idx][1] = bond_bc_type;
        shake_type[b_idx][2] = bond_cd;

        // set rig_type for angles and dihedral on B
        rigs_type[b_idx][0] = angletype_findset(b_idx, a, c, 0);
        rigs_type[b_idx][1] = angletype_findset(c_idx, b, d, 0);
        rigs_type[b_idx][2] = atom->dihedral_type[i][m];

        // pre-assign C with same cluster data (B->C hop)
        // so find_clusters skips C and shake_info can propagate C->D
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

        // mark end atoms A and D as "don't touch" sentinels
        if (a_idx < nlocal) shake_flag[a_idx] = -1;
        if (d_idx < nlocal) shake_flag[d_idx] = -1;
      }
    }
  }

  find_clusters();

  // improper detection: after find_clusters, upgrade flag 4 star clusters
  // to flag 5 if they match an improper topology or have 3 constrained angles

  int impropers_allow = atom->avec->impropers_allow;
  int nimproper = 0;

  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] != 4) continue;
    if (!(mask_atom[i] & groupbit)) continue;
    if (shake_atom[i][0] != tag[i]) continue;

    int angles_allow_flag = atom->avec->angles_allow;
    int nangle_found = 0;
    if (angles_allow_flag) {
      tagint a1 = shake_atom[i][1];
      tagint a2 = shake_atom[i][2];
      tagint a3_atom = shake_atom[i][3];
      int n = angletype_findset(i, a1, a2, 0);
      if (n > 0 && angle_flag[n]) nangle_found++;
      n = angletype_findset(i, a1, a3_atom, 0);
      if (n > 0 && angle_flag[n]) nangle_found++;
      n = angletype_findset(i, a2, a3_atom, 0);
      if (n > 0 && angle_flag[n]) nangle_found++;
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

  // propagate flag 5 to local partner atoms in upgraded clusters

  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] != 5) continue;
    if (shake_atom[i][0] != tag[i]) continue;
    tagint ct = tag[i];
    for (int k = 1; k <= 3; k++) {
      int pidx = atom->map(shake_atom[i][k]);
      if (pidx >= 0 && pidx < nlocal && shake_flag[pidx] == 4) {
        shake_flag[pidx] = 5;
        rigs_type[pidx][0] = rigs_type[i][0];
        rigs_type[pidx][1] = rigs_type[i][1];
        rigs_type[pidx][2] = rigs_type[i][2];
      }
    }
  }

  // propagate flag 5 + rigs_type to remote partner atoms
  // gather all upgraded central-atom records, broadcast, update local atoms

  int nsend = 0;
  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] == 5 && shake_atom[i][0] == tag[i])
      nsend++;
  }

  struct UpgradedCluster { tagint central; int rt0, rt1, rt2; };
  const int sizeof_uc = sizeof(UpgradedCluster);
  UpgradedCluster *sendbuf = new UpgradedCluster[nsend > 0 ? nsend : 1];
  nsend = 0;
  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] == 5 && shake_atom[i][0] == tag[i]) {
      sendbuf[nsend].central = tag[i];
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
      if (shake_flag[i] != 4) continue;
      if (shake_atom[i][0] != ctag) continue;
      shake_flag[i] = 5;
      rigs_type[i][0] = recvbuf[c].rt0;
      rigs_type[i][1] = recvbuf[c].rt1;
      rigs_type[i][2] = recvbuf[c].rt2;
    }
  }

  delete[] sendbuf;
  delete[] recvbuf;
  delete[] recvcounts;
  delete[] displs;

  for (i = 0; i < nlocal; i++) {
    if (shake_flag[i] == 5) {
      if (rigs_type[i][0] > 0)
        angletype_findset(i, shake_atom[i][1], shake_atom[i][2], -1);
      if (rigs_type[i][1] > 0)
        angletype_findset(i, shake_atom[i][1], shake_atom[i][3], -1);
      if (rigs_type[i][2] > 0)
        angletype_findset(i, shake_atom[i][2], shake_atom[i][3], -1);
    } else if (shake_flag[i] == 6) {
      // dihedral chain A-B-C-D
      // shake_atom[i] = {B, A, C, D} for both B and C
      // B (owner, tag == shake_atom[i][0]) handles: bonds A-B, B-C;
      //   angle A-B-C; dihedral type
      // C (tag == shake_atom[i][2]) handles: bond C-D; angle B-C-D
      if (tag[i] == shake_atom[i][0]) {
        // B's half
        bondtype_findset(i, shake_atom[i][0], shake_atom[i][1], -1);
        bondtype_findset(i, shake_atom[i][0], shake_atom[i][2], -1);
        if (rigs_type[i][0] > 0)
          angletype_findset(i, shake_atom[i][1], shake_atom[i][2], -1);
        dihedraltype_findset(i, shake_atom[i][0], shake_atom[i][1],
                             shake_atom[i][2], shake_atom[i][3], -1);
      } else {
        // C's half
        bondtype_findset(i, shake_atom[i][2], shake_atom[i][3], -1);
        if (rigs_type[i][1] > 0)
          angletype_findset(i, shake_atom[i][0], shake_atom[i][3], -1);
      }
    }
  }
}

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

void FixRigs::grow_arrays(int nmax)
{
  FixShake::grow_arrays(nmax);
  memory->grow(rigs_type, nmax, 3, "rigs:rigs_type");
}

void FixRigs::copy_arrays(int i, int j, int delflag)
{
  FixShake::copy_arrays(i, j, delflag);
  if (shake_flag[j] == 5 || shake_flag[j] == 6) {
    rigs_type[j][0] = rigs_type[i][0];
    rigs_type[j][1] = rigs_type[i][1];
    rigs_type[j][2] = rigs_type[i][2];
  }
}

int FixRigs::pack_exchange(int i, double *buf)
{
  int m = FixShake::pack_exchange(i, buf);
  if (shake_flag[i] == 5 || shake_flag[i] == 6) {
    buf[m++] = rigs_type[i][0];
    buf[m++] = rigs_type[i][1];
    buf[m++] = rigs_type[i][2];
  }
  return m;
}

int FixRigs::unpack_exchange(int nlocal, double *buf)
{
  int m = FixShake::unpack_exchange(nlocal, buf);
  if (shake_flag[nlocal] == 5 || shake_flag[nlocal] == 6) {
    rigs_type[nlocal][0] = static_cast<int>(buf[m++]);
    rigs_type[nlocal][1] = static_cast<int>(buf[m++]);
    rigs_type[nlocal][2] = static_cast<int>(buf[m++]);
  }
  return m;
}

int FixRigs::pack_restart(int i, double *buf)
{
  int m = 0;
  if (shake_flag[i] == 5 || shake_flag[i] == 6) {
    buf[m++] = 4;
    buf[m++] = rigs_type[i][0];
    buf[m++] = rigs_type[i][1];
    buf[m++] = rigs_type[i][2];
  } else {
    buf[m++] = 1;
  }
  return m;
}

void FixRigs::unpack_restart(int i, int nth)
{
  double **extra = atom->extra;

  int m = 0;
  for (int j = 0; j < nth; j++) m += static_cast<int>(extra[i][m]);
  m++;

  int count = static_cast<int>(extra[i][m++]);
  if (count == 4) {
    rigs_type[i][0] = static_cast<int>(extra[i][m++]);
    rigs_type[i][1] = static_cast<int>(extra[i][m++]);
    rigs_type[i][2] = static_cast<int>(extra[i][m++]);
  }
}

int FixRigs::size_restart(int i)
{
  if (shake_flag[i] == 5 || shake_flag[i] == 6) return 4;
  return 1;
}

int FixRigs::maxsize_restart()
{
  return 4;
}

void FixRigs::init()
{
  FixShake::init();

  delete[] rigs_angle;
  rigs_angle = new double[atom->nangletypes + 1];

  int nlocal = atom->nlocal;
  for (int i = 1; i <= atom->nangletypes; i++) {
    if (angle_flag[i] == 0) continue;
    if (force->angle == nullptr) continue;

    int bond1_type = 0, bond2_type = 0;
    for (int m = 0; m < nlocal; m++) {
      if (shake_flag[m] != 1) continue;
      if (shake_type[m][2] != i) continue;
      int type1 = MIN(shake_type[m][0], shake_type[m][1]);
      int type2 = MAX(shake_type[m][0], shake_type[m][1]);
      bond1_type = type1;
      bond2_type = type2;
      break;
    }

    int flag_all;
    MPI_Allreduce(&bond1_type, &flag_all, 1, MPI_INT, MPI_MAX, world);
    bond1_type = flag_all;
    MPI_Allreduce(&bond2_type, &flag_all, 1, MPI_INT, MPI_MAX, world);
    bond2_type = flag_all;

    if (bond1_type == 0) {
      rigs_angle[i] = 0.0;
      continue;
    }

    double b1 = bond_distance[bond1_type];
    double b2 = bond_distance[bond2_type];
    double angle = force->angle->equilibrium_angle(i);
    rigs_angle[i] = b1 * b2 * cos(angle);
  }

  // compute rigs_angle_distance: non-bond pair distances from angles
  // for each constrained angle type, find the equilibrium angle and
  // the two bond types it connects, then compute the endpoint distance

  delete[] rigs_angle_distance;
  rigs_angle_distance = new double[atom->nangletypes + 1];
  for (int i = 1; i <= atom->nangletypes; i++) {
    if (angle_flag[i] == 0) {
      rigs_angle_distance[i] = 0.0;
      continue;
    }
    if (force->angle == nullptr) {
      rigs_angle_distance[i] = 0.0;
      continue;
    }

    // find the two bond types for this angle type
    // scan clusters with this angle type
    int bond1_type = 0, bond2_type = 0;
    for (int m = 0; m < nlocal; m++) {
      if (shake_flag[m] == 1 && shake_type[m][2] == i) {
        bond1_type = MIN(shake_type[m][0], shake_type[m][1]);
        bond2_type = MAX(shake_type[m][0], shake_type[m][1]);
        break;
      }
      if (shake_flag[m] == 5 && rigs_type[m][0] == i) {
        bond1_type = MIN(shake_type[m][0], shake_type[m][1]);
        bond2_type = MAX(shake_type[m][0], shake_type[m][1]);
        break;
      }
      if (shake_flag[m] == 5 && rigs_type[m][1] == i) {
        // need to determine which two bonds this angle connects
        // for improper cluster, angle between partners j and k
        // connects bond j and bond k through center
        break;
      }
      if (shake_flag[m] == 6 && rigs_type[m][0] == i) {
        bond1_type = MIN(shake_type[m][0], shake_type[m][1]);
        bond2_type = MAX(shake_type[m][0], shake_type[m][1]);
        break;
      }
      if (shake_flag[m] == 6 && rigs_type[m][1] == i) {
        bond1_type = MIN(shake_type[m][1], shake_type[m][2]);
        bond2_type = MAX(shake_type[m][1], shake_type[m][2]);
        break;
      }
    }

    int flag_all;
    MPI_Allreduce(&bond1_type, &flag_all, 1, MPI_INT, MPI_MAX, world);
    bond1_type = flag_all;
    MPI_Allreduce(&bond2_type, &flag_all, 1, MPI_INT, MPI_MAX, world);
    bond2_type = flag_all;

    if (bond1_type == 0) {
      rigs_angle_distance[i] = 0.0;
      continue;
    }

    double b1 = bond_distance[bond1_type];
    double b2 = bond_distance[bond2_type];
    double ang = force->angle->equilibrium_angle(i);
    double rsq = b1*b1 + b2*b2 - 2.0*b1*b2*cos(ang);
    rigs_angle_distance[i] = sqrt(rsq);
  }

  // compute rigs_improper_distance: non-bond pair distance from improper type
  // find the 3 bond types of the improper, then compute 12/13/23 distances
  // for now we only need the distance for the pair not covered by 2 of the 3 angles

  delete[] rigs_improper_distance;
  rigs_improper_distance = new double[atom->nimpropertypes + 1];
  // TODO: compute improper equilibrium distances

  // compute rigs_dihedral_distance: end-to-end distance from dihedral type
  // A-B-C-D chain: d(A,D) from the 3 bonds and the dihedral angle

  delete[] rigs_dihedral_distance;
  rigs_dihedral_distance = new double[atom->ndihedraltypes + 1];
  // TODO: compute dihedral equilibrium distances
}

void FixRigs::shake4(int ilist)
{
  int m = list[ilist];
  if (shake_flag[m] == 6) {
    error->one(FLERR,"RIGS dihedral constraint solver not yet implemented");
  } else {
    FixShake::shake4(ilist);
  }
}

/* ----------------------------------------------------------------------
   calculate RIGS constraint forces for size 3 cluster = two bonds + angle
   ------------------------------------------------------------------------- */

void FixRigs::shake3angle(int ilist)
{
  int atomlist[3];
  double v[6];

  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  double bond1 = bond_distance[shake_type[m][0]];
  double bond2 = bond_distance[shake_type[m][1]];
  double bond12 = rigs_angle[shake_type[m][2]];
  
  double invmass0, invmass01, invmass02;
  if (rmass) {
    invmass0 = dtfsq / rmass[i0];
    invmass01 = invmass0 + dtfsq / rmass[i1];
    invmass02 = invmass0 + dtfsq / rmass[i2];
  } else {
    invmass0 = dtfsq / mass[type[i0]];
    invmass01 = invmass0 + dtfsq / mass[type[i1]];
    invmass02 = invmass0 + dtfsq / mass[type[i2]];
  }

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

  SymMat2 L = {bond1 * bond1, bond12, bond2 * bond2};
  SymMat2 diff = L - ss;

  Mat2 RS;
  RS(0, 0) = s01[0] * r01[0] + s01[1] * r01[1] + s01[2] * r01[2];
  RS(1, 0) = s01[0] * r02[0] + s01[1] * r02[1] + s01[2] * r02[2];
  RS(0, 1) = s02[0] * r01[0] + s02[1] * r01[1] + s02[2] * r01[2];
  RS(1, 1) = s02[0] * r02[0] + s02[1] * r02[1] + s02[2] * r02[2];

  LTMat2 lm = trans_inv_chol_upper(SymMat2{invmass01, invmass0, invmass02});

  UTMat2 rc = inv_chol_upper(rr);
  Mat2 chi = uut_mul(rc, RS);
  SymMat2 sigma = diff + mat_mul_tosym(transpose(RS), chi);
  lslt_mul(sigma, lm);

  LTMat2 sc = chol_lower(sigma) * lm;
  mul_ltl(chi, lm);
  Mat2 phiC = rc * sc;

  Mat2 J;
  J(0, 0) = 0.0;  J(0, 1) = -1.0;
  J(1, 0) = 1.0;  J(1, 1) = 0.0;
  Mat2 phiS = rc * J * sc;

  double skewC = skew(phiC);
  double skewChi = skew(chi);
  double skewS = skew(phiS);

  double Asq = skewC * skewC + skewS * skewS;
  double sinp = sqrt(Asq - skewChi * skewChi);
  double sskew = -(skewS * skewChi + skewC * sinp) / Asq;
  double cskew = (skewS * sinp - skewChi * skewC) / Asq;

  double lamda01 = chi(0, 0) + cskew * phiC(0, 0) + sskew * phiS(0, 0);
  double lamda02 = chi(1, 1) + cskew * phiC(1, 1);
  double lamda12 = chi(0, 1) + cskew * phiC(0, 1) + sskew * phiS(0, 1);

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
   calculate RIGS constraint forces for flag 5 = improper cluster
   star topology: center atom 0 + partners 1,2,3
   3×3 Gram matrices with r01, r02, r03 on diagonal
   ------------------------------------------------------------------------- */

void FixRigs::shake4improper(int ilist)
{
  int m = list[ilist];
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  int i3 = closest_list[ilist][3];

  double bond1 = bond_distance[shake_type[m][0]];
  double bond2 = bond_distance[shake_type[m][1]];
  double bond3 = bond_distance[shake_type[m][2]];

  // equilibrium non-bond distances from angles/improper
  double dist12 = rigs_angle_distance[rigs_type[m][0]];
  double dist13 = rigs_angle_distance[rigs_type[m][1]];
  double dist23;
  if (rigs_type[m][2] > 0)
    dist23 = rigs_angle_distance[rigs_type[m][2]];
  else
    dist23 = rigs_improper_distance[-rigs_type[m][2]];

  // current displacement vectors
  
  Mat3 R, S;
  R(0, 0) = x[i0][0] - x[i1][0];
  R(0, 1) = x[i0][1] - x[i1][1];
  R(0, 2) = x[i0][2] - x[i1][2];
  R(1, 0) = x[i0][0] - x[i2][0];
  R(1, 1) = x[i0][1] - x[i2][1];
  R(1, 2) = x[i0][2] - x[i2][2];
  R(2, 0) = x[i0][0] - x[i3][0];
  R(2, 1) = x[i0][1] - x[i3][1];
  R(2, 2) = x[i0][2] - x[i3][2];

  S(0, 0) = xshake[i0][0] - xshake[i1][0];
  S(0, 1) = xshake[i0][1] - xshake[i1][1];
  S(0, 2) = xshake[i0][2] - xshake[i1][2];
  S(1, 0) = xshake[i0][0] - xshake[i2][0];
  S(1, 1) = xshake[i0][1] - xshake[i2][1];
  S(1, 2) = xshake[i0][2] - xshake[i2][2];
  S(2, 0) = xshake[i0][0] - xshake[i3][0];
  S(2, 1) = xshake[i0][1] - xshake[i3][1]; 
  S(2, 2) = xshake[i0][2] - xshake[i3][2];

  // Gram matrices

  SymMat3 rr = sym_dot(R);
  SymMat3 ss = sym_dot(S); 

  SymMat3 L = {bond1 * bond1, bond1 * bond2, bond1 * bond3,
               bond2 * bond2, bond2 * bond3, bond3 * bond3};

  // add non-bond equilibrium distances to L off-diagonals
  // L becomes the full 6-constraint target: 3 bonds on diagonal,
  // 3 non-bond pair distances on off-diagonal

  // mass matrix

  double invmass0, invmass01, invmass02, invmass03, invmass12, invmass13, invmass23;
  if (rmass) {
    invmass0 = dtfsq / rmass[i0];
    invmass01 = invmass0 + dtfsq / rmass[i1];
    invmass02 = invmass0 + dtfsq / rmass[i2];
    invmass03 = invmass0 + dtfsq / rmass[i3];
    invmass12 = dtfsq / rmass[i1] + dtfsq / rmass[i2];
    invmass13 = dtfsq / rmass[i1] + dtfsq / rmass[i3];
    invmass23 = dtfsq / rmass[i2] + dtfsq / rmass[i3];
  } else {
    invmass0 = dtfsq / mass[type[i0]];
    invmass01 = invmass0 + dtfsq / mass[type[i1]];
    invmass02 = invmass0 + dtfsq / mass[type[i2]];
    invmass03 = invmass0 + dtfsq / mass[type[i3]];
    invmass12 = dtfsq / mass[type[i1]] + dtfsq / mass[type[i2]];
    invmass13 = dtfsq / mass[type[i1]] + dtfsq / mass[type[i3]];
    invmass23 = dtfsq / mass[type[i2]] + dtfsq / mass[type[i3]];
  }

  SymMat3 M = inv_sym(SymMat3 {invmass01, invmass12, invmass13, invmass02,invmass23, invmass03}); 

  Mat3 RS = mat_dot(R, S);
  Mat3 chi_mu = inv_sym(rr) * RS;
  SymMat3 mu_sig_mu = mat_mul_tosym(transpose(RS), chi_mu) + L - ss;
  SymMat3 sigma = mu_sig_mu;

  Mat3 chi = chi_mu * M;

  LTMat3 sc = chol_lower(sigma);
  UTMat3 rc = inv_chol_upper(rr);

  Mat3 gamma = cayley_converge(rc, sc, chi, 11, tolerance);
  // M = 3x3 inverse mass matrix for non-bond pairs (12, 13, 23)
  // D = M (L - S^T S) M
  // K = M (S^T R)

  // orthogonal solve: chi, phiC, phiS -> cskew, sskew -> lamda

  // TODO: user will implement the 3x3 orthogonal matrix solve

  // force application (improper-specific)
}

/* ----------------------------------------------------------------------
   calculate RIGS constraint forces for flag 6 = dihedral cluster
   chain topology: atoms A-B-C-D stored as 1-0-2-3
   3×3 Gram matrices with r10, r02, r23 on diagonal
   ------------------------------------------------------------------------- */

void FixRigs::shake4dihedral(int ilist)
{
  int m = list[ilist];
  // dihedral chain A-B-C-D, shake_atom = {B, A, C, D}
  // relabel: 0=B, 1=A, 2=C, 3=D
  // diagonal displacements: r10 (A-B), r02 (B-C), r23 (C-D)
  int i0 = closest_list[ilist][0];
  int i1 = closest_list[ilist][1];
  int i2 = closest_list[ilist][2];
  int i3 = closest_list[ilist][3];

  double bond1 = bond_distance[shake_type[m][0]];
  double bond2 = bond_distance[shake_type[m][1]];
  double bond3 = bond_distance[shake_type[m][2]];

  double dist12 = rigs_angle_distance[rigs_type[m][0]];
  double dist23 = rigs_angle_distance[rigs_type[m][1]];
  double dist13 = rigs_dihedral_distance[rigs_type[m][2]];

  // current displacement vectors: diagonal elements are r10, r02, r23

  double r10[3], r02[3], r23[3];
  r10[0] = x[i1][0] - x[i0][0]; r10[1] = x[i1][1] - x[i0][1]; r10[2] = x[i1][2] - x[i0][2];
  r02[0] = x[i0][0] - x[i2][0]; r02[1] = x[i0][1] - x[i2][1]; r02[2] = x[i0][2] - x[i2][2];
  r23[0] = x[i2][0] - x[i3][0]; r23[1] = x[i2][1] - x[i3][1]; r23[2] = x[i2][2] - x[i3][2];

  double s10[3], s02[3], s23[3];
  s10[0] = xshake[i1][0] - xshake[i0][0]; s10[1] = xshake[i1][1] - xshake[i0][1]; s10[2] = xshake[i1][2] - xshake[i0][2];
  s02[0] = xshake[i0][0] - xshake[i2][0]; s02[1] = xshake[i0][1] - xshake[i2][1]; s02[2] = xshake[i0][2] - xshake[i2][2];
  s23[0] = xshake[i2][0] - xshake[i3][0]; s23[1] = xshake[i2][1] - xshake[i3][1]; s23[2] = xshake[i2][2] - xshake[i3][2];

  // Gram matrices

  //SymMat3 rr = sym_dot(r10, r02, r23);
  //SymMat3 ss = sym_dot(s10, s02, s23);

  SymMat3 L = {bond1 * bond1, bond1 * bond2, bond1 * bond3,
               bond2 * bond2, bond2 * bond3, bond3 * bond3};

  // mass matrix: for chain A-B-C-D with atoms 0=B, 1=A, 2=C, 3=D
  // pair 1-0 (A-B): mass_A + mass_B
  // pair 0-2 (B-C): mass_B + mass_C
  // pair 2-3 (C-D): mass_C + mass_D

  double invmass10, invmass02, invmass23;
  if (rmass) {
    invmass10 = dtfsq / rmass[i1] + dtfsq / rmass[i0];
    invmass02 = dtfsq / rmass[i0] + dtfsq / rmass[i2];
    invmass23 = dtfsq / rmass[i2] + dtfsq / rmass[i3];
  } else {
    invmass10 = dtfsq / mass[type[i1]] + dtfsq / mass[type[i0]];
    invmass02 = dtfsq / mass[type[i0]] + dtfsq / mass[type[i2]];
    invmass23 = dtfsq / mass[type[i2]] + dtfsq / mass[type[i3]];
  }

  // M, D, K, chi, phiC, phiS construction
  // orthogonal solve
  // TODO: user will implement the 3x3 orthogonal matrix solve

  // force application (dihedral-specific)
}
