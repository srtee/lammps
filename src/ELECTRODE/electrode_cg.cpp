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

#include "electrode_cg.h"
#include "atom.h"
#include "comm.h"
#include "electrode_math.h"
#include "error.h"
#include "fix_electrode_conp.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string>

static constexpr double SMALL = 1e-16;

using namespace LAMMPS_NS;
using namespace ElectrodeMath;

ElectrodeCG::ElectrodeCG(LAMMPS *lmp, FixElectrodeConp *fix) : Pointers(lmp), ChargeSolver()
{
  setup = a_cached_flag = false;
  macro_computed = vac_cap_computed = sb_stale = false;
  vac_cap = 0.;
  nstep = ncall = 0;
  nmax = 0;
  memory->create(potential_i, nmax, "ElectrodeCG:potential_i");
  elyt_step = -1;
  predictor_cols = predictor_count = 0;
  predictor_index = -1;
  this->fix = fix;
}

/* ---------------------------------------------------------------------- */

ElectrodeCG::~ElectrodeCG() noexcept
{
  memory->destroy(potential_i);
  if (comm->me == 0)
    utils::logmesg(lmp, "Average conjugate gradient steps: {:.3g}\n", nstep * 1. / ncall);
}

/* ---------------------------------------------------------------------- */

double ElectrodeCG::memory_use()
{
  double bytes = 0.;
  bytes += q_ele.capacity() * sizeof(double);
  bytes += taglist.capacity() * sizeof(tagint);
  bytes += iele_to_group.capacity() * sizeof(int);
  bytes += bvec.capacity() * sizeof(double);
  bytes += a_cached.capacity() * sizeof(double);
  if (macro_computed) {
    for (const auto &v : sd_vectors) bytes += v.capacity() * sizeof(double);
    for (const auto &v : macro_capacitance) bytes += v.capacity() * sizeof(double);
    for (const auto &v : macro_elastance) bytes += v.capacity() * sizeof(double);
    bytes += sb_charges.capacity() * sizeof(double);
  }
  return bytes;
}

/* ---------------------------------------------------------------------- */

void ElectrodeCG::setup_solver(double cg_threshold, ElectrodeVector *vec, int predictor_cols)
{
  setup_cg(cg_threshold, predictor_cols);
  elec_vec = vec;
}

/* ---------------------------------------------------------------------- */

void ElectrodeCG::setup_cg(double cg_threshold, int predictor_cols)
{
  setup = true;
  evscale = force->qe2f / force->qqrd2e;
  threshold = cg_threshold;
  this->predictor_cols = predictor_cols;
  // setup atom/property array to store prior charges
  if (predictor_cols) {
    std::string property_call = "fx_electrode_cg_predictor all property/atom d2_predict_array " +
        std::to_string(predictor_cols);
    modify->add_fix(property_call, 1);
    int is_double, cols;
    predictor_index = atom->find_custom("predict_array", is_double, cols);
    if (predictor_index == -1)
      error->all(FLERR, "Failed to setup property/atom array for conjugate gradient predictor");
    assert(is_double);
    assert(predictor_cols == cols);
  }
  // prepare predictor weights of ASPC, cf. Kolafa 2003
  predictor_weights = std::vector<std::vector<double>>();
  for (int k = 0; k < predictor_cols; k++) {
    auto weights = std::vector<double>();    // weights[0] = B_1, ...
    int sign = 1;
    for (int i = 1; i <= k + 2; i++) {
      double num = 1;
      double denom = k + 3;
      for (int j = 0; j < i - 1; j++) {
        num *= k + 1 - j;
        denom *= k + 4 + j;
      }
      weights.push_back(sign * i * (4 * k + 6) * num / denom);
      sign *= -1;
    }
    predictor_weights.push_back(weights);
  }
}

/* ---------------------------------------------------------------------- */

void ElectrodeCG::update_solver(std::vector<tagint> taglist_local, std::vector<int> iele_to_group)
{
  assert(setup);
  taglist = taglist_local;
  nele = taglist.size();    // local number of electrode atoms
  q_ele.resize(nele);
  MPI_Allreduce(&nele, &nele_world, 1, MPI_INT, MPI_SUM, world);
  this->iele_to_group = iele_to_group;
}

/* ---------------------------------------------------------------------- */

void ElectrodeCG::set_elyt_pot(double *b_nall)
{
  elyt_step = update->ntimestep;
  bvec = pot_to_vector(b_nall);
}

std::vector<double> ElectrodeCG::solve(std::vector<double> v)
{
  assert(setup);
  assert(update->ntimestep == elyt_step);    // bvec is up to date
  a_cached_flag = false;
  ncall++;
  auto b = std::vector<double>(nele);
  for (int i = 0; i < nele; i++) b[i] = evscale * v[iele_to_group[i]] - bvec[i];
  applied_psi = v;
  sb_stale = true;
  predict_q();
  q_ele = cg_solve(std::move(b), q_ele, true);
  return q_ele;
}

/* ----------------------------------------------------------------------
   Core CG iteration on (evscale*A): x converges to the solution of
   M x = b. When constrain, project iterates onto the charge constraint
   subspace (qtotal / group constraints). Does not touch predictor state.
------------------------------------------------------------------------- */

std::vector<double> ElectrodeCG::cg_solve(std::vector<double> b, const std::vector<double> &x_init,
                                          bool constrain)
{
  auto project = [&](std::vector<double> x, bool correction) -> std::vector<double> {
    return constrain ? constraint_projection(std::move(x), correction) : x;
  };
  auto q = project(x_init, constrain && (constraint != ChargeConstraint::NONE));
  auto r = b - ele_ele_interaction(q);
  auto d = project(r, false);
  double dot_old = dot_product(r, d);
  double delta = dot_old;
  for (int k = 0; k < nele_world && delta > threshold; k++, nstep++) {
    auto y = ele_ele_interaction(d);
    double alpha = dot_old / dot_product(d, y);
    q += alpha * d;
    // prepare next step
    if ((k + 1) % 20 == 0) {
      // avoid shifting residual. This rarely happens.
      q = project(q, constrain);
      r = b - ele_ele_interaction(q);
    } else {
      r -= alpha * std::move(y);
    }
    auto p = project(r, false);
    double dot_new = dot_product(r, p);
    d = std::move(p) + (dot_new / dot_old) * d;
    delta = dot_product(r, d);
    dot_old = dot_new;
  }
  if ((delta > threshold) && (comm->me == 0)) error->warning(FLERR, "CG threshold not reached");
  return q;
}

/* ---------------------------------------------------------------------- */

double ElectrodeCG::get_potential(int igroup)
{
  assert(update->ntimestep == elyt_step);    // bvec is up to date
  if (!a_cached_flag) {
    a_cached_flag = true;
    a_cached = ele_ele_interaction(q_ele);
  }
  double pot = 0.;
  int count = 0;
  for (int i = 0; i < nele; i++) {
    if (iele_to_group[i] == igroup) {
      pot += (a_cached[i] + bvec[i]) / evscale;
      count++;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &pot, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM, world);
  return pot / count;
}

/* ----------------------------------------------------------------------
    if possible, extrapolate charges based on previous steps, else predict with current charges
------------------------------------------------------------------------- */

void ElectrodeCG::predict_q()
{
  assert(predictor_count <= predictor_cols);
  assert(predictor_weights.size() == predictor_cols);
  double *q = atom->q;
  for (int i = 0; i < nele; i++) q_ele[i] = q[atom->map(taglist[i])];
  if (!predictor_count) {    // predict with current charges
    for (int i = 0; i < nele; i++) q_ele[i] = q[atom->map(taglist[i])];
  } else {    // ASPC method, cf. Kolafa 2003
    const int k = predictor_count - 1;
    double **qold = atom->darray[predictor_index];
    auto weights = predictor_weights[k];    // weights[0] = B_1, ...
    for (int i = 0; i < nele; i++) {
      const int ii = atom->map(taglist[i]);
      double qi = weights[0] * q[ii];
      for (int j = 1; j <= k + 1; j++) qi += weights[j] * qold[ii][j - 1];
      q_ele[i] = qi;
    }
  }

  // move current charges to predictor array for following steps
  if (predictor_cols) {
    const int nlocal = atom->nlocal;
    double **qold = atom->darray[predictor_index];
    for (int i = predictor_count; i > 0; i--) {
      if (i == predictor_cols) continue;    // forget last column
      for (int j = 0; j < nlocal; j++) { qold[j][i] = qold[j][i - 1]; }
    }
    for (int j = 0; j < nlocal; j++) qold[j][0] = q[j];
    if (predictor_count < predictor_cols) predictor_count++;
  }
}

/* ----------------------------------------------------------------------
   Calculate ele ele interaction by calling ElectrodeVector and translate
   from nlocal indices to local electrode indices
------------------------------------------------------------------------- */

std::vector<double> ElectrodeCG::ele_ele_interaction(const std::vector<double> &q_vec)
{
  MPI_Barrier(world);
  double mult_start = MPI_Wtime();
  fix->set_charges(q_vec);
  if (atom->nmax > nmax) {
    memory->destroy(potential_i);
    nmax = atom->nmax;
    memory->create(potential_i, nmax, "ElectrodeCG:potential_i");
  }
  memset(potential_i, 0, atom->nmax * sizeof(double));
  elec_vec->compute_pot(potential_i);
  auto a = pot_to_vector(potential_i);
  MPI_Barrier(world);
  mult_time += MPI_Wtime() - mult_start;
  return a;
}

/* ---------------------------------------------------------------------- */

std::vector<double> ElectrodeCG::pot_to_vector(double *pot)
{
  auto vec = std::vector<double>(nele, 0.);
  for (int i = 0; i < nele; i++) vec[i] = pot[atom->map(taglist[i])];
  return vec;
}

/* ---------------------------------------------------------------------- */

double ElectrodeCG::dot_product(const std::vector<double> &a, const std::vector<double> &b)
{
  assert(((int) a.size() == nele) && ((int) b.size() == nele));
  double out = 0.;
  for (int i = 0; i < nele; i++) out += a[i] * b[i];
  MPI_Allreduce(MPI_IN_PLACE, &out, 1, MPI_DOUBLE, MPI_SUM, world);
  return out;
}

/* ----------------------------------------------------------------------
   project into direction that conserves total charge (cf. Gingrich master thesis)
   or correct total electrode charge to qtotal if correction
------------------------------------------------------------------------- */

std::vector<double> ElectrodeCG::constraint_projection(std::vector<double> x, bool correction)
{
  switch (constraint) {
    case ChargeConstraint::NONE:
      return x;
    case ChargeConstraint::SINGLE: {
      double sum = 0.;
      for (double xi : x) sum += xi;
      MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, world);
      if (correction) sum -= qtotal;
      sum /= nele_world;
      for (double &xi : x) xi -= sum;
      return x;
    }
    case ChargeConstraint::GROUP: {
      const int n = x.size();
      const int ngroups = qtotal_group.size();
      auto counts = std::vector<int>(ngroups, 0);
      auto sums = std::vector<double>(ngroups, 0);
      for (int i = 0; i < n; i++) {
        const int g = iele_to_group[i];
        sums[g] += x[i];
        counts[g]++;
      }
      MPI_Allreduce(MPI_IN_PLACE, sums.data(), ngroups, MPI_DOUBLE, MPI_SUM, world);
      MPI_Allreduce(MPI_IN_PLACE, counts.data(), ngroups, MPI_INT, MPI_SUM, world);
      for (int g = 0; g < ngroups; g++) {
        if (correction) sums[g] -= qtotal_group[g];
        sums[g] /= counts[g];
      }
      for (int i = 0; i < n; i++) x[i] -= sums[iele_to_group[i]];
      return x;
    }
    default:
      error->all(FLERR, "Constraint not implemented");
  }
}

/* ---------------------------------------------------------------------- */

std::vector<double> ElectrodeCG::compute_potentials()
{
  assert(setup);
  assert(update->ntimestep == elyt_step);    // bvec is up to date
  if (!a_cached_flag) {
    a_cached_flag = true;
    a_cached = ele_ele_interaction(q_ele);
  }
  const int ngroups = *std::max_element(iele_to_group.begin(), iele_to_group.end()) + 1;
  auto pots = std::vector<double>(ngroups, 0.);
  auto counts = std::vector<int>(ngroups, 0);
  for (int i = 0; i < nele; i++) {
    const int g = iele_to_group[i];
    pots[g] += (a_cached[i] + bvec[i]) / evscale;
    counts[g]++;
  }
  MPI_Allreduce(MPI_IN_PLACE, pots.data(), ngroups, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, counts.data(), ngroups, MPI_INT, MPI_SUM, world);
  for (int g = 0; g < ngroups; g++) pots[g] /= counts[g];
  return pots;
}

/* ---------------------------------------------------------------------- */

void ElectrodeCG::compute_macro_calibration()
{
  if (macro_computed) return;
  const int ngroups = *std::max_element(iele_to_group.begin(), iele_to_group.end()) + 1;
  sd_vectors = std::vector<std::vector<double>>(ngroups, std::vector<double>(nele, 0.));
  macro_capacitance = std::vector<std::vector<double>>(ngroups, std::vector<double>(ngroups, 0.));
  const auto constraint_save = constraint;
  constraint = ChargeConstraint::NONE;
  // unconstrained calibration solves: x_g = A^{-1} e_g via CG on (evscale*A)
  for (int g = 0; g < ngroups; g++) {
    auto b = std::vector<double>(nele, 0.);
    for (int i = 0; i < nele; i++) {
      if (iele_to_group[i] == g) b[i] = evscale;
    }
    auto x_g = cg_solve(std::move(b), std::vector<double>(nele, 0.), false);
    for (int i = 0; i < nele; i++) sd_vectors[g][i] = evscale * x_g[i];
    fix->set_charges(q_ele);    // restore electrode charges clobbered by the calibration
  }
  constraint = constraint_save;
  // macro_capacitance[g0][g1] = evscale * sum_{i in g0} (A^{-1} e_{g1})_i
  for (int g0 = 0; g0 < ngroups; g0++) {
    for (int g1 = 0; g1 < ngroups; g1++) {
      double c = 0.;
      for (int i = 0; i < nele; i++) {
        if (iele_to_group[i] == g0) c += sd_vectors[g1][i];
      }
      macro_capacitance[g0][g1] = c;
    }
  }
  // macro_elastance = inverse of macro_capacitance (small dense solve)
  auto tmp = macro_capacitance;
  macro_elastance = std::vector<std::vector<double>>(ngroups, std::vector<double>(ngroups, 0.));
  for (int g = 0; g < ngroups; g++) macro_elastance[g][g] = 1.;
  for (int col = 0; col < ngroups; col++) {
    // find pivot
    int piv = col;
    for (int r = col + 1; r < ngroups; r++) {
      if (std::fabs(tmp[r][col]) > std::fabs(tmp[piv][col])) piv = r;
    }
    if (std::fabs(tmp[piv][col]) < SMALL)
      error->all(FLERR, "ELECTRODE macro matrix inversion failed!");
    std::swap(tmp[col], tmp[piv]);
    std::swap(macro_elastance[col], macro_elastance[piv]);
    const double diag = tmp[col][col];
    for (int r = col + 1; r < ngroups; r++) {
      const double f = tmp[r][col] / diag;
      if (f == 0.) continue;
      for (int c2 = col; c2 < ngroups; c2++) tmp[r][c2] -= f * tmp[col][c2];
      for (int c2 = 0; c2 < ngroups; c2++) macro_elastance[r][c2] -= f * macro_elastance[col][c2];
    }
  }
  for (int r = ngroups - 1; r >= 0; r--) {
    const double diag = tmp[r][r];
    for (int c2 = 0; c2 < ngroups; c2++) macro_elastance[r][c2] /= diag;
    for (int r2 = 0; r2 < r; r2++) {
      const double f = tmp[r2][r] / diag;
      if (f == 0.) continue;
      for (int c2 = 0; c2 < ngroups; c2++)
        macro_elastance[r2][c2] -= f * macro_elastance[r][c2];
    }
  }
  macro_computed = true;
}

/* ---------------------------------------------------------------------- */

double ElectrodeCG::get_sb_charges(int igroup)
{
  assert(setup);
  if (!macro_computed) compute_macro_calibration();
  if (sb_stale) {
    sb_stale = false;
    const int ngroups = (int) sd_vectors.size();
    // q_sb = q_ele - sum_g sd[g] * psi_g (linearity of the constrained solve)
    sb_charges = std::vector<double>(ngroups, 0.);
    for (int i = 0; i < nele; i++) {
      double q_sb_i = q_ele[i];
      for (int g = 0; g < ngroups; g++) q_sb_i -= sd_vectors[g][i] * applied_psi[g];
      sb_charges[iele_to_group[i]] += q_sb_i;
    }
    MPI_Allreduce(MPI_IN_PLACE, sb_charges.data(), ngroups, MPI_DOUBLE, MPI_SUM, world);
  }
  return sb_charges[igroup];
}

/* ---------------------------------------------------------------------- */

double ElectrodeCG::get_macro_capacitance(int igroup, int jgroup)
{
  assert(setup);
  if (!macro_computed) compute_macro_calibration();
  return macro_capacitance[igroup][jgroup];
}

/* ---------------------------------------------------------------------- */

double ElectrodeCG::get_macro_elastance(int igroup, int jgroup)
{
  assert(setup);
  if (!macro_computed) compute_macro_calibration();
  return macro_elastance[igroup][jgroup];
}

/* ---------------------------------------------------------------------- */

double ElectrodeCG::vacuum_capacitance()
{
  if (!macro_computed) compute_macro_calibration();
  const int ngroups = (int) sd_vectors.size();
  if (ngroups != 2) error->all(FLERR, "vacuum_capacitance requires exactly two electrode groups");
  if (!vac_cap_computed) {
    vac_cap_computed = true;
    vac_cap = (macro_capacitance[0][0] * macro_capacitance[1][1] -
               macro_capacitance[0][1] * macro_capacitance[0][1]) /
        (macro_capacitance[0][0] + macro_capacitance[1][1] + 2 * macro_capacitance[0][1]);
  }
  return vac_cap;
}

/* ---------------------------------------------------------------------- */

void ElectrodeCG::buffer_and_gather(double const * /*ivec*/, double * /*elevec*/)
{
  error->all(FLERR, "Method not implemented");
}
