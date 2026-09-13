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

#include "electrode_inv.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "update.h"

#include <cassert>
#include <unordered_map>

using namespace LAMMPS_NS;

static constexpr double SMALL = 1e-16;

extern "C" {
void dgetrf_(const int *M, const int *N, double *A, const int *lda, int *ipiv, int *info);
void dgetri_(const int *N, double *A, const int *lda, const int *ipiv, double *work,
             const int *lwork, int *info);
void dgetrs_(const char *TRANS, const int *N, const int *NRHS, double *A, const int *LDA,
             const int *IPIV, double *B, const int *LDB, int *INFO);
}

#if defined(FFT_MKL) || defined(FFT_MKL_THREADS)
extern "C" void MKL_Set_Num_Threads(int nth);
extern "C" int MKL_Get_Max_Threads(void);
#endif

ElectrodeInv::ElectrodeInv(LAMMPS *lmp) : Pointers(lmp), ChargeSolver()
{
  setup = cap_set = vac_cap_computed = fragmented = false;
  nmax = 0;
  memory->create(potential_i, nmax, "ElectrodeInv:potential_i");
  const int nprocs = comm->nprocs;
  recvcounts = new int[nprocs];
  displs = new int[nprocs];
  elyt_step = -1;
}

/* ---------------------------------------------------------------------- */

ElectrodeInv::~ElectrodeInv() noexcept
{
  memory->destroy(potential_i);
  if (setup) {
    memory->destroy(iele_gathered);
    memory->destroy(buf_gathered);
    memory->destroy(potential_iele);
  }
  cap_frag.clear();
  cap_frag.shrink_to_fit();
  delete[] recvcounts;
  delete[] displs;
}

/* ---------------------------------------------------------------------- */

double ElectrodeInv::memory_use()
{
  double bytes = 0.;
  if (setup) bytes += 3 * nele_world * sizeof(double);
  // retained matrix storage: per-rank row fragments after setup
  bytes += cap_frag.size() * nele_world * sizeof(double);
  for (const auto &row : cap_frag) bytes += (row.capacity() - row.size()) * sizeof(double);
  bytes += qvec.capacity() * sizeof(double);
  bytes += iele_to_group.capacity() * sizeof(int);
  bytes += taglist_local.capacity() * sizeof(tagint);
  bytes += iele_local.capacity() * sizeof(int);
  bytes += buf_iele.capacity() * sizeof(double);
  bytes += tag_to_iele.bucket_count() * (sizeof(tagint) + sizeof(int));    // TODO check
  return bytes;
}

/* ---------------------------------------------------------------------- */

void ElectrodeInv::set_elastance(int nele_world, double **elastance, bool timer_flag,
                                 const std::vector<int> &frag_iele)
{
  cap_set = true;
  this->nele_world = nele_world;
  this->capacitance = elastance;
  // invert elastance to obtain capacitance
  MPI_Barrier(world);
  double invert_time = MPI_Wtime();
  if (timer_flag && (comm->me == 0)) utils::logmesg(lmp, "CONP inverting matrix\n");
  int m = nele_world, n = nele_world, lda = nele_world;
  std::vector<int> ipiv(nele_world);

  int info_rf;
#if defined(FFT_MKL) || defined(FFT_MKL_THREADS)
  int mkl_threads = MKL_Get_Max_Threads();
  MKL_Set_Num_Threads(1);
#endif
  dgetrf_(&m, &n, &capacitance[0][0], &lda, ipiv.data(), &info_rf);
#if defined(FFT_MKL) || defined(FFT_MKL_THREADS)
  MKL_Set_Num_Threads(mkl_threads);
#endif
  if (info_rf != 0) error->all(FLERR, "CONP matrix factorization failed!");

  // S3.1: each rank solves A^T x = e_i for its OWNED rows i (dgetrs,
  // trans='T'), giving row i of A^{-1} directly into the fragment — no
  // N x N dgetri work array, no fragmentize() redistribution.
  iele_local = frag_iele;
  nlocalele = static_cast<int>(frag_iele.size());
  cap_frag.resize(nlocalele);
  // the matrix is stored row-major; column-major LAPACK sees A_cm = A^T,
  // so trans='N' (solve A_cm x = b) yields x = A^{-T} e_i = row i of A^{-1}
  const char trans = 'N';
  const int nrhs = 1;
  std::vector<double> rhs(nele_world, 0.0);
  int info_rs;
  for (int r = 0; r < nlocalele; r++) {
    const int iele = frag_iele[r];
    std::fill(rhs.begin(), rhs.end(), 0.0);
    rhs[iele] = 1.0;
    dgetrs_(&trans, &n, &nrhs, &capacitance[0][0], &lda, ipiv.data(), rhs.data(), &n,
            &info_rs);
    if (info_rs != 0) error->all(FLERR, "CONP matrix solve failed!");
    cap_frag[r] = rhs;
  }
  fragmented = true;
  // NOTE: capacitance (the factorized elastance) stays alive for now —
  // symmetrize/compute_sd_vectors still consume the full matrix. S3.2/S3.3
  // convert those to fragment-local form, then the release moves here.
  MPI_Barrier(world);
  if (timer_flag && (comm->me == 0))
    utils::logmesg(lmp, "Invert time: {:.4g} s\n", MPI_Wtime() - invert_time);
}

/* ---------------------------------------------------------------------- */

void ElectrodeInv::set_capacitance(int nele_world, double **capacitance,
                                   const std::vector<int> &frag_iele)
{
  cap_set = true;
  this->nele_world = nele_world;
  this->capacitance = capacitance;
  // S3.4: the file provides the already-inverted capacitance; slice the
  // owned rows into fragments directly and release the full copy
  iele_local = frag_iele;
  nlocalele = static_cast<int>(frag_iele.size());
  cap_frag.resize(nlocalele);
  for (int r = 0; r < nlocalele; r++)
    cap_frag[r] = std::vector<double>(capacitance[frag_iele[r]],
                                      capacitance[frag_iele[r]] + nele_world);
  fragmented = true;
  capacitance = nullptr;
}

/* ---------------------------------------------------------------------- */

void ElectrodeInv::setup_solver(int groupbit, std::unordered_map<tagint, int> tag_to_iele,
                                std::vector<int> group_bits, bool ffield, bool timer_flag)
{
  assert(cap_set);
  setup = true;
  evscale = force->qe2f / force->qqrd2e;
  this->groupbit = groupbit;
  if (ffield) symmetrize();
  this->tag_to_iele = tag_to_iele;
  memory->create(iele_gathered, nele_world, "ElectrodeInv:iele_gathered");
  memory->create(buf_gathered, nele_world, "ElectrodeInv:buf_gathered");
  memory->create(potential_iele, nele_world, "ElectrodeInv:potential_iele");
  const int nlocal = atom->nlocal;
  ngroups = group_bits.size();
  group_pot = std::vector<double>(ngroups, 0.);
  int *mask = atom->mask;
  tagint *tag = atom->tag;

  // build the local electrode list (normally updated by update_solver, but
  // the first gather has not happened yet at setup time). The electrode
  // index is the position of the tag in the globally sorted tag list:
  // gather tags (rank-ordered), sort, and map - independent of the
  // tag_to_iele map, whose mapping may be rank-local.
  std::vector<tagint> local_tags;
  nlocalele = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      local_tags.push_back(tag[i]);
      nlocalele++;
    }
  }
  {
    std::vector<int> gather_counts(comm->nprocs);
    MPI_Allgather(&nlocalele, 1, MPI_INT, gather_counts.data(), 1, MPI_INT, world);
    std::vector<int> gather_displs(comm->nprocs);
    gather_displs[0] = 0;
    for (int p = 1; p < comm->nprocs; p++)
      gather_displs[p] = gather_displs[p - 1] + gather_counts[p - 1];
    std::vector<tagint> all_tags(nele_world);
    MPI_Allgatherv(local_tags.data(), nlocalele, MPI_LMP_TAGINT, all_tags.data(),
                   gather_counts.data(), gather_displs.data(), MPI_LMP_TAGINT, world);
    std::sort(all_tags.begin(), all_tags.end());
    std::unordered_map<tagint, int> iele_of_tag;
    iele_of_tag.reserve(nele_world);
    for (int i = 0; i < nele_world; i++) iele_of_tag.emplace(all_tags[i], i);
    iele_local.clear();
    iele_local.reserve(nlocalele);
    for (tagint t : local_tags) iele_local.push_back(iele_of_tag[t]);
    this->tag_to_iele = iele_of_tag;
  }

  iele_to_group = std::vector<int>(nele_world, -1);
  for (int i = 0; i < nlocal; i++) {
    for (int g = 0; g < ngroups; g++) {
      if (mask[i] & group_bits[g]) { iele_to_group[tag_to_iele[tag[i]]] = g; }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, iele_to_group.data(), nele_world, MPI_INT, MPI_MAX, world);
  sb_charges = std::vector<double>(ngroups);
  MPI_Barrier(world);
  double start = MPI_Wtime();
  if (ffield) {
    compute_sd_vectors_ffield(group_bits);
  } else
    compute_sd_vectors();
  compute_macro_matrices(ffield);
  MPI_Barrier(world);
  if (timer_flag && (comm->me == 0))
    utils::logmesg(lmp, "SD-vector and macro matrices time: {:.4g} s\n", MPI_Wtime() - start);

  if (!fragmented) fragmentize();    // S3.1: fragments built by set_elastance
}

/* ----------------------------------------------------------------------
    Scatter the replicated capacitance matrix into per-rank row fragments
    and release the full matrix. Fragment r holds the row of electrode
    index iele_local[r]; every electrode index is owned by exactly one rank.
    Uses the iele_gathered mapping produced by update_solver.
------------------------------------------------------------------------- */

void ElectrodeInv::fragmentize()
{
  assert(setup);
  assert(!fragmented);
  // owner of electrode index i is the rank whose iele_local contains i.
  // Build the global ownership map here: gather the local electrode index
  // lists (rank-ordered) instead of relying on update_solver state, because
  // fragmentize runs inside setup_solver, before the first update_solver.
  const int nprocs = comm->nprocs;
  std::vector<int> recvcounts_frag(nprocs);
  MPI_Allgather(&nlocalele, 1, MPI_INT, recvcounts_frag.data(), 1, MPI_INT, world);
  std::vector<int> displs_frag(nprocs);
  displs_frag[0] = 0;
  for (int p = 1; p < nprocs; p++) displs_frag[p] = displs_frag[p - 1] + recvcounts_frag[p - 1];
  std::vector<int> iele_gathered_frag(nele_world);
  MPI_Allgatherv(iele_local.data(), nlocalele, MPI_INT, iele_gathered_frag.data(),
                 recvcounts_frag.data(), displs_frag.data(), MPI_INT, world);

  std::vector<int> rowowner(nele_world);
  for (int p = 0, idx = 0; p < nprocs; p++)
    for (int k = 0; k < recvcounts_frag[p]; k++, idx++)
      rowowner[iele_gathered_frag[idx]] = p;

  // my own rows are already correct; wire traffic only for remote rows.
  // supplier of row i is rank i % nprocs (fixed cyclic assignment).
  cap_frag.resize(nlocalele);
  for (int r = 0; r < nlocalele; r++)
    cap_frag[r] = std::vector<double>(capacitance[iele_local[r]],
                                      capacitance[iele_local[r]] + nele_world);

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
      std::copy(capacitance[i], capacitance[i] + nele_world, sendbuf.begin() + off);
      off += (size_t)nele_world;
    }
  }

  // counts of doubles I will receive from each rank: scnt was already
  // scaled to doubles before the Alltoall, so rcnt/rdis are in doubles —
  // no further scaling (a second scaling overflowed recvbuf by a factor
  // of nele_world)
  MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, world);
  int rsize = 0;
  for (int p = 0; p < nprocs; p++) {
    rdis[p] = rsize;
    rsize += rcnt[p];
  }

  std::vector<double> recvbuf((size_t) rsize);
  MPI_Alltoallv(sendbuf.data(), scnt.data(), sdis.data(), MPI_DOUBLE, recvbuf.data(), rcnt.data(),
                rdis.data(), MPI_DOUBLE, world);

  // unpack: block p holds rows with ascending i, i%nprocs==p, owned by me
  // local fragment position of each electrode index I own (-1 otherwise)
  std::vector<int> frag_of_iele(nele_world, -1);
  for (int r = 0; r < nlocalele; r++) frag_of_iele[iele_local[r]] = r;
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
        cap_frag[frag_of_iele[i]].assign(recvbuf.begin() + ioff, recvbuf.begin() + ioff + nele_world);
        ioff += (size_t)nele_world;
        i += 1;
      }
    }
  }

  fragmented = true;
  capacitance = nullptr;    // full replicated copy released
}
/* ---------------------------------------------------------------------- */

void ElectrodeInv::update_solver(std::vector<tagint> taglist_local,
                                 std::vector<int> /*iele_to_group_local*/)
{
  assert(setup);
  this->taglist_local = taglist_local;
  // refresh the communication bookkeeping; iele_local rebuilds from the tag
  // list exactly as pre-fragment (fragments are keyed by electrode index)
  nlocalele = taglist_local.size();
  const int nprocs = comm->nprocs;
  qvec.assign(nlocalele, 0.);
  delete[] recvcounts;
  delete[] displs;
  recvcounts = new int[nprocs];
  displs = new int[nprocs];
  MPI_Allgather(&nlocalele, 1, MPI_INT, recvcounts, 1, MPI_INT, world);
  displs[0] = 0;
  for (int i = 1; i < nprocs; i++) displs[i] = displs[i - 1] + recvcounts[i - 1];
  iele_local.clear();
  iele_local.reserve(nlocalele);
  for (tagint t : taglist_local) iele_local.push_back(tag_to_iele[t]);
  MPI_Allgatherv(iele_local.data(), nlocalele, MPI_INT, iele_gathered, recvcounts, displs, MPI_INT,
                 world);
}

/* ---------------------------------------------------------------------- */

void ElectrodeInv::set_elyt_pot(double *b_nall)
{
  elyt_step = update->ntimestep;
  buffer_and_gather(b_nall, potential_iele);
  // calculate charges due to electrolyte
  std::fill(sb_charges.begin(), sb_charges.end(), 0.);
  MPI_Barrier(world);
  double mult_start = MPI_Wtime();
  for (int i = 0; i < nlocalele; i++) {
    double q_tmp = 0.;
    const double *_noalias caprow = cap_frag[i].data();    // row of iele_local[i]
    for (int j = 0; j < nele_world; j++) { q_tmp -= caprow[j] * potential_iele[j]; }
    sb_charges[iele_to_group[iele_local[i]]] += q_tmp;
    qvec[i] = q_tmp;
  }
  MPI_Allreduce(MPI_IN_PLACE, sb_charges.data(), ngroups, MPI_DOUBLE, MPI_SUM, world);
  MPI_Barrier(world);
  mult_time += MPI_Wtime() - mult_start;
}

/* ---------------------------------------------------------------------- */

std::vector<double> ElectrodeInv::solve(std::vector<double> v)
{
  assert(setup);
  assert(update->ntimestep == elyt_step);    // qvec already has charges due to electrolyte
  MPI_Barrier(world);
  double mult_start = MPI_Wtime();
  group_pot = apply_constraint(v);
  // calculate final charges
  for (int g = 0; g < ngroups; g++)
    for (int j = 0; j < nlocalele; j++) qvec[j] += sd_vectors[g][iele_local[j]] * group_pot[g];
  MPI_Barrier(world);
  mult_time += MPI_Wtime() - mult_start;
  return qvec;
}

/* ---------------------------------------------------------------------- */

std::vector<double> ElectrodeInv::apply_constraint(std::vector<double> v)
{
  switch (constraint) {
    case ChargeConstraint::NONE:
      break;
    case ChargeConstraint::SINGLE: {
      double q_current = 0.;
      for (int i = 0; i < ngroups; i++) {
        q_current += sb_charges[i];
        for (int j = 0; j < ngroups; j++) q_current += macro_capacitance[i][j] * v[j];
      }
      double add_psi = (qtotal - q_current) / macro_capacitance_sum;
      for (int i = 0; i < ngroups; i++) v[i] += add_psi;
      break;
    }
    case ChargeConstraint::GROUP: {
      std::vector<double> group_remainder_q(ngroups);
      for (int g = 0; g < ngroups; g++) group_remainder_q[g] = qtotal_group[g] - sb_charges[g];
      for (int g = 0; g < ngroups; g++) {
        double vtmp = 0;
        for (int h = 0; h < ngroups; h++) { vtmp += macro_elastance[g][h] * group_remainder_q[h]; }
        v[g] = vtmp;
      }
      break;
    }
    default:
      error->all(FLERR, "Constraint not implemented");
  }
  return v;
}

/* ----------------------------------------------------------------------
    calculate potentials that would give current charges
------------------------------------------------------------------------- */

std::vector<double> ElectrodeInv::compute_potentials()
{
  assert(setup);
  assert(update->ntimestep == elyt_step);    // assert sb_charges up to date
  // sum charges for each group
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  double *q = atom->q;
  auto group_q = std::vector<double>(ngroups, 0.);
  for (int i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) group_q[iele_to_group[tag_to_iele[tag[i]]]] += q[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, group_q.data(), ngroups, MPI_DOUBLE, MPI_SUM, world);

  // compute potentials
  for (int g = 0; g < ngroups; g++) group_q[g] -= sb_charges[g];
  for (int g = 0; g < ngroups; g++) {
    double vtmp = 0.;
    for (int h = 0; h < ngroups; h++) vtmp += macro_elastance[g][h] * group_q[h];
    group_pot[g] = vtmp;
  }
  return group_pot;
}

/* ---------------------------------------------------------------------- */

double ElectrodeInv::get_sb_charges(int igroup)
{
  assert(setup);
  assert(igroup < ngroups);
  assert(update->ntimestep == elyt_step);
  return sb_charges[igroup];
}

/* ---------------------------------------------------------------------- */

double ElectrodeInv::get_macro_capacitance(int igroup, int jgroup)
{
  assert(setup);
  assert(igroup < ngroups && jgroup < ngroups);
  return macro_capacitance[igroup][jgroup];
}

/* ---------------------------------------------------------------------- */

double ElectrodeInv::get_macro_elastance(int igroup, int jgroup)
{
  assert(setup);
  assert(igroup < ngroups && jgroup < ngroups);
  return macro_elastance[igroup][jgroup];
}

/* ---------------------------------------------------------------------- */

double ElectrodeInv::get_potential(int igroup)
{
  assert(setup);
  assert(igroup < ngroups);
  return group_pot[igroup];
}

/* ---------------------------------------------------------------------- */

double ElectrodeInv::vacuum_capacitance()
{
  assert(ngroups == 2);
  if (!vac_cap_computed) {
    vac_cap_computed = true;
    vac_cap = (macro_capacitance[0][0] * macro_capacitance[1][1] -
               macro_capacitance[0][1] * macro_capacitance[0][1]) /
        (macro_capacitance[0][0] + macro_capacitance[1][1] + 2 * macro_capacitance[0][1]);
  }
  return vac_cap;
}

/* ---------------------------------------------------------------------- */

void ElectrodeInv::buffer_and_gather(double const *ivec, double *elevec)
{
  buf_iele.resize(nlocalele);
  for (int i_iele = 0; i_iele < nlocalele; i_iele++) {
    buf_iele[i_iele] = ivec[atom->map(taglist_local[i_iele])];
  }
  MPI_Allgatherv(buf_iele.data(), nlocalele, MPI_DOUBLE, buf_gathered, recvcounts, displs,
                 MPI_DOUBLE, world);

  for (int i = 0; i < nele_world; i++) elevec[iele_gathered[i]] = buf_gathered[i];
}

/* ----------------------------------------------------------------------
    S matrix to enforce charge neutrality constraint
------------------------------------------------------------------------- */

void ElectrodeInv::symmetrize()
{
  // S3.3: fragment-local rank-1 update; AinvE/EAinvE are exchanged so each
  // rank can update its own rows without the full matrix
  std::vector<double> AinvE(nele_world, 0.);
  double EAinvE_local = 0.0;
  for (int r = 0; r < nlocalele; r++) {
    const int iele = iele_local[r];
    double AinvEtmp = 0.0;
    for (int j = 0; j < nele_world; j++) AinvEtmp += cap_frag[r][j];
    AinvE[iele] = AinvEtmp;    // use temp accumulator to enable vectorization
    EAinvE_local += AinvEtmp;
  }
  double EAinvE = 0.0;
  MPI_Allreduce(&EAinvE_local, &EAinvE, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, AinvE.data(), nele_world, MPI_DOUBLE, MPI_SUM, world);
  for (int r = 0; r < nlocalele; r++) {
    const double iAinvE = AinvE[iele_local[r]];
    for (int j = 0; j < nele_world; j++) cap_frag[r][j] -= AinvE[j] * iAinvE / EAinvE;
  }
}

/* ---------------------------------------------------------------------- */

void ElectrodeInv::compute_sd_vectors()
{
  // S3.2: fragment-local accumulation; sd[g][k] gets row k's entries at
  // columns j in group g — row k is owned here, so accumulate row-locally
  // and Allreduce over ranks
  sd_vectors = std::vector<std::vector<double>>(ngroups, std::vector<double>(nele_world, 0.));
  for (int r = 0; r < nlocalele; r++) {
    const int k = iele_local[r];
    for (int j = 0; j < nele_world; j++) {
      int g = iele_to_group[j];
      sd_vectors[g][k] += cap_frag[r][j] * evscale;
    }
  }
  for (int g = 0; g < ngroups; g++)
    MPI_Allreduce(MPI_IN_PLACE, sd_vectors[g].data(), nele_world, MPI_DOUBLE, MPI_SUM, world);
}
/* ---------------------------------------------------------------------- */

void ElectrodeInv::compute_sd_vectors_ffield(std::vector<int> group_bits)
{
  int top_group = get_top_group(group_bits);
  sd_vectors = std::vector<std::vector<double>>(ngroups, std::vector<double>(nele_world, 0.));
  double **x = atom->x;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  double zprd = domain->prd[2];
  // S3.2: two-pass fragment-local form. Pass 1 builds coef[g][j] =
  // gmult(g) * w_j summed over ranks, where w_j is the z-weight of the
  // atom whose column is j. Pass 2 contracts each owned row against coef.
  std::vector<std::vector<double>> coef(ngroups, std::vector<double>(nele_world, 0.));
  for (int i = 0; i < atom->nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const int j = tag_to_iele[tag[i]];
    double const zoff = (mask[i] & group_bits[top_group]) ? 0.0 : 1.0;
    double const w = evscale * (x[i][2] / zprd + zoff);
    for (int g = 0; g < ngroups; g++) {
      double gmult = (g == top_group) ? -1.0 : 1.0;
      coef[g][j] += gmult * w;
    }
  }
  for (int g = 0; g < ngroups; g++)
    MPI_Allreduce(MPI_IN_PLACE, coef[g].data(), nele_world, MPI_DOUBLE, MPI_SUM, world);

  for (int r = 0; r < nlocalele; r++) {
    const int k = iele_local[r];
    for (int g = 0; g < ngroups; g++) {
      double row_sum = 0.0;
      for (int j = 0; j < nele_world; j++) row_sum += cap_frag[r][j] * coef[g][j];
      sd_vectors[g][k] += row_sum;
    }
  }
  for (int g = 0; g < ngroups; g++) {
    MPI_Allreduce(MPI_IN_PLACE, sd_vectors[g].data(), nele_world, MPI_DOUBLE, MPI_SUM, world);
  }
}

/* ---------------------------------------------------------------------- */

int ElectrodeInv::get_top_group(std::vector<int> group_bits)
{
  auto *zmax = new double[ngroups];
  double **x = atom->x;
  for (int g = 0; g < ngroups; g++) { zmax[g] = domain->boxlo[2]; }
  int *mask = atom->mask;
  for (int i = 0; i < atom->nlocal; i++) {
    for (int g = 0; g < ngroups; g++) {
      if (mask[i] & group_bits[g]) {
        if (x[i][2] > zmax[g]) zmax[g] = x[i][2];
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, zmax, ngroups, MPI_DOUBLE, MPI_MAX, world);
  int gmax = 0;
  for (int g = 0; g < ngroups; g++) { gmax = (zmax[g] > zmax[gmax]) ? g : gmax; }
  delete[] zmax;
  return gmax;
}

/* ---------------------------------------------------------------------- */

void ElectrodeInv::compute_macro_matrices(bool symm)
{
  // capacitance
  macro_capacitance = std::vector<std::vector<double>>(ngroups, std::vector<double>(ngroups, 0.));
  for (int g = 0; g < ngroups; g++) {
    for (int k = 0; k < nele_world; k++) {
      macro_capacitance[iele_to_group[k]][g] += sd_vectors[g][k];
    }
  }
  if (symm) {    // scaling with C[0][0] improves numerical stability
    double scalar = macro_capacitance[0][0];
    macro_capacitance.back() = std::vector<double>(ngroups, scalar);
  }

  macro_capacitance_sum = 0.;
  for (int i = 0; i < ngroups; i++)
    for (int j = 0; j < ngroups; j++) macro_capacitance_sum += macro_capacitance[i][j];

  // elastance
  macro_elastance = std::vector<std::vector<double>>(ngroups, std::vector<double>(ngroups));
  switch (ngroups) {
    case 1: {
      macro_elastance[0][0] = 1. / macro_capacitance[0][0];
      break;
    }
    case 2: {
      double const det = macro_capacitance[0][0] * macro_capacitance[1][1] -
          macro_capacitance[0][1] * macro_capacitance[1][0];
      if (fabs(det) < SMALL) error->all(FLERR, "ELECTRODE macro matrix inversion failed!");
      double const detinv = 1 / det;
      macro_elastance[0][0] = macro_capacitance[1][1] * detinv;
      macro_elastance[1][1] = macro_capacitance[0][0] * detinv;
      macro_elastance[0][1] = -macro_capacitance[0][1] * detinv;
      macro_elastance[1][0] = -macro_capacitance[1][0] * detinv;
      break;
    }
    default:
      int m = ngroups;
      int n = m, lda = m;
      std::vector<int> ipiv(m);
      const int lwork = m * m;
      std::vector<double> work(lwork);
      std::vector<double> tmp(lwork);
      for (int i = 0; i < ngroups; i++) {
        for (int j = 0; j < ngroups; j++) {
          int idx = i * ngroups + j;
          tmp[idx] = macro_capacitance[i][j];
        }
      }
      int info_rf, info_ri;
#if defined(FFT_MKL) || defined(FFT_MKL_THREADS)
      int mkl_threads = MKL_Get_Max_Threads();
      MKL_Set_Num_Threads(1);
#endif
      dgetrf_(&m, &n, tmp.data(), &lda, ipiv.data(), &info_rf);
      dgetri_(&n, tmp.data(), &lda, ipiv.data(), work.data(), &lwork, &info_ri);
#if defined(FFT_MKL) || defined(FFT_MKL_THREADS)
      MKL_Set_Num_Threads(mkl_threads);
#endif
      if (info_rf != 0 || info_ri != 0)
        error->all(FLERR, "ELECTRODE macro matrix inversion failed!");
      for (int i = 0; i < ngroups; i++) {
        for (int j = 0; j < ngroups; j++) {
          int idx = i * ngroups + j;
          macro_elastance[i][j] = tmp[idx];
        }
      }
      break;
  }
}
