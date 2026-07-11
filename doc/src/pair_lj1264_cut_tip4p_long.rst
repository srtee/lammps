.. index:: pair_style lj1264/cut/tip4p/long

pair_style lj1264/cut/tip4p/long command
==========================================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style lj1264/cut/tip4p/long otype htype btype atype qdist cutoff (cutoff2)

* otype,htype = atom types (numeric or type label) for TIP4P O and H
* btype,atype = bond and angle types (numeric or type label) for TIP4P waters
* qdist = distance from O atom to massless charge (distance units)
* cutoff = global cutoff for LJ (and Coulombic if only 1 arg) (distance units)
* cutoff2 = global cutoff for Coulombic (optional) (distance units)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style lj1264/cut/tip4p/long 5 6 4 5 0.15 10.0
   pair_style lj1264/cut/tip4p/long 5 6 4 5 0.15 10.0 8.0
   pair_coeff * * 100.0 3.0 50.0
   pair_coeff 1 1 100.0 3.5 80.0 9.0
   pair_coeff 1 2 mix mix -20.0

   pair_style lj1264/cut/tip4p/long OW HW HW-OW HW-OW-HW 0.15 12.0
   labelmap atom 1 OW 2 HW
   labelmap bond 1 HW-OW
   labelmap angle 1 HW-OW-HW
   pair_coeff * * 100.0 3.0 50.0
   pair_coeff OW OW 100.0 3.5 80.0 9.0

Description
"""""""""""

The *lj1264/cut/tip4p/long* style combines the 12/6/4 Lennard-Jones
potential from the :doc:`lj1264/cut/coul/long <pair_lj1264_cut_coul_long>`
pair style with the TIP4P water model treatment from the
:doc:`lj/cut/tip4p/long <pair_lj_cut_tip4p>` pair style.  It is designed
for use with four-point water models such as TIP4P, TIP4P-Ew, TIP4P/2005,
TIP4P-FB, and OPC, in which a massless charge site M is located a short
distance away from the oxygen atom along the bisector of the HOH angle.

The pair potential is given by

.. math::

   E = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
       \left(\frac{\sigma}{r}\right)^6 \right]
       - \frac{C_4}{r^4}
                       \qquad r < r_c

where :math:`\epsilon` and :math:`\sigma` are the standard Lennard-Jones
parameters, :math:`C_4` is the coefficient of the :math:`r^{-4}` term
(in energy :math:`\times` distance\ :sup:`4` units), and :math:`r_c` is
the cutoff.  The :math:`r^{-4}` term is subtractive (attractive);
:math:`C_4` is supplied as a positive value and the potential evaluates
the term as :math:`-C_4 / r^4`.  The force contribution of the
:math:`r^{-4}` term is :math:`-4 C_4 / r^6`.

The Lennard-Jones and :math:`r^{-4}` interactions use the true
interatomic distance (between oxygen atoms for water-water interactions),
while the Coulombic interaction uses the position of the massless M site
for water oxygen atoms, as in the standard TIP4P treatment.  The Coulombic
part is treated identically to the :doc:`lj/cut/tip4p/long <pair_lj_cut_tip4p>`
style: pairwise interactions within the Coulombic cutoff are computed
directly, and interactions outside that distance are computed in
reciprocal space by a :doc:`kspace_style <kspace_style>` solver (*ewald*
or *pppm*).  See the :doc:`lj/cut/tip4p/long <pair_lj_cut_tip4p>`
documentation for details of the TIP4P Coulombic treatment.

.. note::

   For each TIP4P water molecule in your system, the atom IDs for
   the O and 2 H atoms must be consecutive, with the O atom first.  This
   is to enable LAMMPS to "find" the 2 H atoms associated with each O
   atom.  For example, if the atom ID of an O atom in a TIP4P water
   molecule is 500, then its 2 H atoms must have IDs 501 and 502.

.. note::

   If using type labels, the type labels must be defined before calling
   the :doc:`pair_coeff <pair_coeff>` command.

See the :doc:`Howto tip4p <Howto_tip4p>` page for more information
on how to use the TIP4P pair styles and lists of parameters to set.

The 12-6-4 model was developed by Li and Merz to account for
ion-induced dipole interactions that are missing in the standard 12-6
Lennard-Jones model :ref:`(Li and Merz) <Li1264-1>`.  The :math:`C_4`
parameters for metal ions in conjunction with various water models
are tabulated in :ref:`(Li and Merz, 2014) <Li1264-1>`,
:ref:`(Li, Song, and Merz, 2015) <Li1264-2>`,
:ref:`(Li, Song, Li, and Merz, 2020) <Li1264-3>`, and
:ref:`(Sengupta, Li, Song, Li, and Merz, 2021) <Li1264-4>`.
Positive :math:`C_4` values (attractive :math:`r^{-4}` term) correct for
the missing ion-induced dipole interaction in cations, while negative
:math:`C_4` values (repulsive :math:`r^{-4}` term) correct for
overestimated charge-transfer effects in anions.

Coefficients
""""""""""""

The following coefficients must be defined for each pair of atoms types
via the :doc:`pair_coeff <pair_coeff>` command as in the examples above,
or in the data file or restart files read by the :doc:`read_data <read_data>`
or :doc:`read_restart <read_restart>` commands, or by mixing as described
below:

* :math:`\epsilon` (energy units)
* :math:`\sigma` (distance units)
* :math:`C_4` (energy :math:`\times` distance\ :sup:`4` units)
* LJ cutoff (distance units)

Note that :math:`\sigma` is defined in the LJ formula as the zero-crossing
distance for the 12/6 part of the potential, not as the energy minimum at
:math:`2^{\frac{1}{6}} \sigma`.

The last coefficient is optional.  If not specified, the global LJ
cutoff specified in the pair_style command is used.  Only the LJ cutoff
can be specified for an individual I,J type pair; all type pairs use the
same global Coulombic cutoff specified in the pair_style command.

A warning is issued if a negative :math:`C_4` is supplied, since
the :math:`r^{-4}` term is designed to be attractive with a positive
:math:`C_4`.

The *mix* keyword
^^^^^^^^^^^^^^^^^

For off-diagonal pairs (:math:`I \neq J`), the :math:`\epsilon` and/or
:math:`\sigma` arguments may be replaced by the string ``mix``.  When
``mix`` is given, the corresponding coefficient is not set directly but is
instead computed from the diagonal (:math:`I,I` and :math:`J,J`) entries
using the current mixing rule at initialization time.  This allows you to
specify a pair interaction that inherits mixed :math:`\epsilon` and/or
:math:`\sigma` while imposing an explicit :math:`C_4`.  For example:

.. code-block:: LAMMPS

   pair_coeff 1 2 mix mix -20.0

mixes both :math:`\epsilon` and :math:`\sigma` for the 1-2 pair but sets
:math:`C_4 = -20.0`.  The ``mix`` keyword is not permitted for
diagonal (:math:`I = J`) pairs.

.. warning::

   Because of how this pair style implements the coulomb interactions
   by implicitly defining a fourth site for the negative charge
   of the TIP4P and similar water models, special care must be taken
   when using this pair style with other computations that also use
   charges.  Unless they are specially set up to also handle the implicit
   definition of the 4th site, results are likely incorrect.  Example:
   :doc:`compute dipole/chunk <compute_dipole_chunk>`.  For the same
   reason, when using this pair style with
   :doc:`pair_style hybrid <pair_hybrid>`, **all** coulomb interactions
   should be handled by a single sub-style with TIP4P support.  All other
   instances and styles will "see" the M point charges at the position
   of the Oxygen atom and thus compute incorrect forces and energies.
   LAMMPS will print a warning when it detects one of these issues.

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, the :math:`\epsilon` and
:math:`\sigma` coefficients and cutoff distance can be mixed.  Both
:math:`\epsilon` and :math:`\sigma` are mixed using the standard energy
and distance mixing rules (geometric by default).  See the
:doc:`pair_modify <pair_modify>` command for details.

The :math:`C_4` coefficient is **never** mixed.  For off-diagonal
pairs that were not explicitly set via :doc:`pair_coeff <pair_coeff>`,
:math:`C_4` defaults to zero, which reduces the potential to the
standard 12/6 Lennard-Jones form.  To supply a non-zero
:math:`C_4` for a mixed pair, use the ``mix`` keyword described
above.

This pair style supports the :doc:`pair_modify <pair_modify>` shift
option for the energy of the LJ portion (including the :math:`r^{-4}`
term) of the pair interaction.

This pair style supports the :doc:`pair_modify <pair_modify>` table
option since it can tabulate the short-range portion of the long-range
Coulombic interaction.

This pair style supports the :doc:`pair_modify <pair_modify>` tail
option for adding a long-range tail correction to the energy and
pressure for the standard 12/6 Lennard-Jones portion of the pair
interaction.  No tail correction is applied for the :math:`r^{-4}` term,
which is short-ranged and makes a negligible contribution beyond the
cutoff.

This pair style writes its information to :doc:`binary restart files <restart>`, so pair_style and pair_coeff commands do not need to be
specified in an input script that reads a restart file.

This pair style does not support the *inner*, *middle*, and *outer*
keywords of the :doc:`run_style respa <run_style>` command.  Only the
*pair* keyword of run_style respa is supported.

----------

Restrictions
""""""""""""

This pair style is part of the KSPACE package.  It is only enabled if
LAMMPS was built with that package.  See the
:doc:`Build package <Build_package>` page for more info.

This pair style requires use of a :doc:`kspace_style <kspace_style>`
command.

This pair style requires atom IDs to be enabled and newton pair to be
on.

This pair style requires a :doc:`bond_style <bond_style>` and
:doc:`angle_style <angle_style>` to be defined (for the TIP4P water
geometry).

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`,
:doc:`pair_style lj1264/cut/coul/long <pair_lj1264_cut_coul_long>`,
:doc:`pair_style lj/cut/tip4p/long <pair_lj_cut_tip4p>`

Default
"""""""

none

----------

.. _Li1264-1:

**(Li and Merz, 2014)** Li, Merz, J Chem Theory Comput, 10, 289 (2014).

.. _Li1264-2:

**(Li, Song, and Merz, 2015)** Li, Song, Merz, J Chem Theory Comput, 11, 1645 (2015).

.. _Li1264-3:

**(Li, Song, Li, and Merz, 2020)** Li, Song, Li, Merz, J Chem Theory Comput, 16, 4429 (2020).

.. _Li1264-4:

**(Sengupta, Li, Song, Li, and Merz, 2021)** Sengupta, Li, Song, Li, Merz, J Chem Inf Model, 61, 869 (2021).