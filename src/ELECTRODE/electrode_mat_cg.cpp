/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Ludwig Ahrens-Iwers (TUHH), Shern Tee (GU), Robert Meissner (Hereon, TUHH)
------------------------------------------------------------------------- */

#include "electrode_mat_cg.h"
#include "atom.h"
#include "comm.h"
#include "electrode_math.h"
#include "error.h"
#include "fix_electrode_conp.h"
#include <cassert>

using namespace LAMMPS_NS;

ElectrodeMatCG::ElectrodeMatCG(LAMMPS *lmp) : ElectrodeCG(lmp)
{
  matrix_set = fragmented = false;
}

/* ---------------------------------------------------------------------- */

ElectrodeMatCG::~ElectrodeMatCG() noexcept
{
  el_frag.clear();
  el_frag.shrink_to_fit();
}

/* ---------------------------------------------------------------------- */

double ElectrodeMatCG::memory_use()
{
  double bytes = ElectrodeCG::memory_use();
  // retained matrix storage: per-rank row fragments after setup
  bytes += el_frag.size() * nele_world * sizeof(double);
  for (const auto &row : el_frag) bytes += (row.capacity() - row.size()) * sizeof(double);
  bytes += qele_world.capacity() * sizeof(double);
  bytes += iele_local.capacity() * sizeof(double);
  bytes += tag_to_iele.bucket_count() * (sizeof(tagint) + sizeof(int));    // TODO check
  return bytes;
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatCG::set_elastance(int nele_world, double **elastance)
{
  matrix_set = true;
  fragmented = false;
  this->nele_world = nele_world;
  n_mat = nele_world;
  qele_world = std::vector<double>(nele_world);
  this->elastance = elastance;
}

/* ----------------------------------------------------------------------
    Scatter the replicated elastance matrix into per-rank row fragments
    and release the full matrix. Fragment r holds the row of electrode
    index iele_local[r]; every electrode index is owned by exactly one rank.
    Ownership is derived from the gathered iele lists built in update_solver.
------------------------------------------------------------------------- */


void ElectrodeMatCG::fragmentize()
{
  assert(matrix_set);
  assert(!fragmented);
  // gather global electrode index lists once (rank-ordered)
  const int nprocs = comm->nprocs;
  std::vector<int> recvcounts(nprocs);
  MPI_Allgather(&nele, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);
  std::vector<int> displs(nprocs);
  displs[0] = 0;
  for (int p = 1; p < nprocs; p++) displs[p] = displs[p - 1] + recvcounts[p - 1];
  std::vector<int> iele_gathered(nele_world);
  MPI_Allgatherv(iele_local.data(), nele, MPI_INT, iele_gathered.data(), recvcounts.data(),
                 displs.data(), MPI_INT, world);

  std::vector<int> rowowner(nele_world);
  for (int p = 0, idx = 0; p < nprocs; p++)
    for (int k = 0; k < recvcounts[p]; k++, idx++) rowowner[iele_gathered[idx]] = p;

  // supplier of row i is rank i % nprocs (fixed cyclic assignment)
  el_frag.resize(nele);
  for (int r = 0; r < nele; r++)
    el_frag[r] = std::vector<double>(elastance[iele_local[r]],
                                     elastance[iele_local[r]] + nele_world);

  std::vector<int> scnt(nprocs, 0), sdis(nprocs), rcnt(nprocs), rdis(nprocs);
  std::vector<size_t> srow;    // ascending electrode index I must send
  for (int i = 0; i < nele_world; i++) {
    if (i % nprocs != comm->me || rowowner[i] == comm->me) continue;
    scnt[rowowner[i]]++;
    srow.push_back(i);
  }
  int ssize = 0;
  for (int p = 0; p < nprocs; p++) {
    sdis[p] = ssize;
    ssize += scnt[p];
  }
  for (int p = 0; p < nprocs; p++) scnt[p] *= (int)nele_world;    // in doubles now
  for (int p = 0; p < nprocs; p++) sdis[p] *= (int)nele_world;

  std::vector<double> sendbuf((size_t)ssize * nele_world);
  {
    size_t off = 0;
    for (int i : srow) {
      std::copy(elastance[i], elastance[i] + nele_world, sendbuf.begin() + off);
      off += (size_t)nele_world;
    }
  }

  // counts of doubles I will receive from each rank
  MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, world);
  int rsize = 0;
  for (int p = 0; p < nprocs; p++) {
    rdis[p] = rsize;
    rsize += rcnt[p];
  }
  for (int p = 0; p < nprocs; p++) {
    rcnt[p] *= (int)nele_world;    // rows -> doubles
    rdis[p] *= (int)nele_world;
  }

  std::vector<double> recvbuf((size_t)rsize * nele_world);
  MPI_Alltoallv(sendbuf.data(), scnt.data(), sdis.data(), MPI_DOUBLE, recvbuf.data(), rcnt.data(),
                rdis.data(), MPI_DOUBLE, world);

  // local fragment position of each electrode index I own (-1 otherwise)
  std::vector<int> frag_of_iele(nele_world, -1);
  for (int r = 0; r < nele; r++) frag_of_iele[iele_local[r]] = r;

  // unpack: block p holds rows with ascending i, i%nprocs==p, owned by me
  {
    size_t ioff = 0;
    auto next = [&](int p, int start) {    // first i >= start with i%nprocs==p, owned by me
      int i = start + ((p - start % nprocs + nprocs) % nprocs);
      for (; i < nele_world; i += nprocs)
        if (frag_of_iele[i] >= 0) return i;
      return -1;
    };
    for (int p = 0; p < nprocs; p++) {
      const int nrows = rcnt[p] / (int)nele_world;
      int i = p;
      for (int k = 0; k < nrows; k++) {
        i = next(p, i);
        el_frag[frag_of_iele[i]].assign(recvbuf.begin() + ioff, recvbuf.begin() + ioff + nele_world);
        ioff += (size_t)nele_world;
        i += 1;
      }
    }
  }

  fragmented = true;
  elastance = nullptr;    // full replicated copy released
}

void ElectrodeMatCG::setup_solver(double cg_threshold, std::unordered_map<tagint, int> tag_to_iele,
                                  int predictor_cols)
{
  assert(matrix_set);
  ElectrodeCG::setup_cg(cg_threshold, predictor_cols);

  // electrode atoms are identified by membership in the global sorted tag
  // list: gather all electrode tags (rank-ordered) and sort. This is
  // independent of the passed-in tag_to_iele map, whose tag->index mapping
  // may be rank-local.
  tagint *tag = atom->tag;
  std::vector<tagint> local_tags;
  for (int i = 0; i < atom->nlocal; i++) {
    if (tag_to_iele.count(tag[i])) local_tags.push_back(tag[i]);
  }
  std::vector<int> gather_counts(comm->nprocs);
  MPI_Allgather(&nele, 1, MPI_INT, gather_counts.data(), 1, MPI_INT, world);
  std::vector<int> gather_displs(comm->nprocs);
  gather_displs[0] = 0;
  for (int p = 1; p < comm->nprocs; p++)
    gather_displs[p] = gather_displs[p - 1] + gather_counts[p - 1];
  std::vector<tagint> all_tags(nele_world);
  MPI_Allgatherv(local_tags.data(), nele, MPI_LMP_TAGINT, all_tags.data(), gather_counts.data(),
                 gather_displs.data(), MPI_LMP_TAGINT, world);
  std::sort(all_tags.begin(), all_tags.end());
  std::unordered_map<tagint, int> iele_of_tag;
  iele_of_tag.reserve(nele_world);
  for (int i = 0; i < nele_world; i++) iele_of_tag.emplace(all_tags[i], i);
  iele_local.clear();
  iele_local.reserve(nele);
  for (tagint t : local_tags) iele_local.push_back(iele_of_tag[t]);
  tag_to_iele = iele_of_tag;
  nele = static_cast<int>(iele_local.size());
  MPI_Allreduce(&nele, &nele_world, 1, MPI_INT, MPI_SUM, world);
  qele_world = std::vector<double>(nele_world);
  fragmentize();
}

/* ---------------------------------------------------------------------- */

void ElectrodeMatCG::update_solver(std::vector<tagint> taglist_local,
                                   std::vector<int> iele_to_group)
{
  ElectrodeCG::update_solver(taglist_local, iele_to_group);
  if (n_mat != nele_world) error->all(FLERR, "Number of electrode atoms has changed");
  iele_local.clear();
  iele_local.reserve(nele);
  // electrode index = position of the tag in the globally sorted tag list
  // (rank-ordered gather + sort; every rank computes the same ordering)
  std::vector<int> gather_counts(comm->nprocs);
  MPI_Allgather(&nele, 1, MPI_INT, gather_counts.data(), 1, MPI_INT, world);
  std::vector<int> gather_displs(comm->nprocs);
  gather_displs[0] = 0;
  for (int p = 1; p < comm->nprocs; p++)
    gather_displs[p] = gather_displs[p - 1] + gather_counts[p - 1];
  std::vector<tagint> all_tags(nele_world);
  MPI_Allgatherv(taglist_local.data(), nele, MPI_LMP_TAGINT, all_tags.data(),
                 gather_counts.data(), gather_displs.data(), MPI_LMP_TAGINT, world);
  std::sort(all_tags.begin(), all_tags.end());
  std::unordered_map<tagint, int> iele_of_tag;
  iele_of_tag.reserve(nele_world);
  for (int i = 0; i < nele_world; i++) iele_of_tag.emplace(all_tags[i], i);
  for (tagint t : taglist_local) iele_local.push_back(iele_of_tag[t]);
  tag_to_iele = iele_of_tag;
}

/* ----------------------------------------------------------------------
   Calculate ele ele interaction by multiplying elastance matrix with charges
------------------------------------------------------------------------- */

std::vector<double> ElectrodeMatCG::ele_ele_interaction(const std::vector<double> &q_vec)
{
  MPI_Barrier(world);
  double mult_start = MPI_Wtime();
  auto a = std::vector<double>(nele, 0.);
  std::fill(qele_world.begin(), qele_world.end(), 0.);
  for (int i = 0; i < nele; i++) qele_world[iele_local[i]] = q_vec[i];
  MPI_Allreduce(MPI_IN_PLACE, qele_world.data(), nele_world, MPI_DOUBLE, MPI_SUM, world);
  for (int i = 0; i < nele; i++) {
    double a_tmp = 0.;
    const double *_noalias row = el_frag[i].data();    // row of iele_local[i]
    for (int j = 0; j < nele_world; j++) a_tmp += row[j] * qele_world[j];
    a[i] = a_tmp;
  }
  MPI_Barrier(world);
  mult_time += MPI_Wtime() - mult_start;
  return a;
}

