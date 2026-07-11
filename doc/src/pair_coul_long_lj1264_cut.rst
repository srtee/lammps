.. index:: pair_style coul/long/lj1264/cut

pair_style coul/long/lj1264/cut command
========================================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style coul/long/lj1264/cut cutoff (cutoff2)

* cutoff = global cutoff for LJ (and Coulombic if only 1 arg) (distance units)
* cutoff2 = global cutoff for Coulombic (optional) (distance units)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style coul/long/lj1264/cut 10.0
   pair_style coul/long/lj1264/cut 10.0 8.0
   pair_coeff * * 100.0 3.0 0.5
   pair_coeff 1 1 100.0 3.5 0.8 9.0
   pair_coeff 1 2 mix mix -2.0

Description
"""""""""""

The *coul/long/lj1264/cut* style computes a 12/6/4 Lennard-Jones potential
combined with long-range Coulombic interactions, given by

.. math::

   E = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
       \left(\frac{\sigma}{r}\right)^6 \right]
       - \epsilon_4 \left(\frac{\sigma}{r}\right)^{4}
                       \qquad r < r_c

where :math:`\epsilon`, :math:`\sigma`, and :math:`\epsilon_4` are the
three energy parameters of the potential, and :math:`r_c` is the cutoff.
The :math:`r^{-4}` term is subtractive (attractive); :math:`\epsilon_4`
is supplied as a positive energy and the potential evaluates the term as
:math:`-\epsilon_4 (\sigma/r)^4`.  Internally the code precomputes
:math:`C_4 = \epsilon_4 \sigma^4` (in energy :math:`\times` distance\ :sup:`4`
units) so that the energy contribution is :math:`-C_4 / r^4` and the force
contribution is :math:`-4 C_4 / r^6`.

The Coulombic part is treated identically to the
:doc:`lj/cut/coul/long <pair_lj_cut_coul>` style: pairwise interactions within
the Coulombic cutoff are computed directly, and interactions outside that
distance are computed in reciprocal space by a
:doc:`kspace_style <kspace_style>` solver (*ewald* or *pppm*).  See the
:doc:`lj/cut/coul/long <pair_lj_cut_coul>` documentation for details of the
Coulombic treatment.

Coefficients
""""""""""""

The following coefficients must be defined for each pair of atoms types
via the :doc:`pair_coeff <pair_coeff>` command as in the examples above,
or in the data file or restart files read by the :doc:`read_data <read_data>`
or :doc:`read_restart <read_restart>` commands, or by mixing as described
below:

* :math:`\epsilon` (energy units)
* :math:`\sigma` (distance units)
* :math:`\epsilon_4` (energy units)
* cutoff1 (distance units)

Note that :math:`\sigma` is defined in the LJ formula as the zero-crossing
distance for the 12/6 part of the potential, not as the energy minimum at
:math:`2^{\frac{1}{6}} \sigma`.

The last coefficient is optional.  If not specified, the global LJ
cutoff specified in the pair_style command is used.  Only the LJ cutoff
can be specified for an individual I,J type pair; all type pairs use the
same global Coulombic cutoff specified in the pair_style command.

A warning is issued if a negative :math:`\epsilon_4` is supplied, since
the :math:`r^{-4}` term is designed to be attractive with a positive
:math:`\epsilon_4`.

The *mix* keyword
^^^^^^^^^^^^^^^^^

For off-diagonal pairs (:math:`I \neq J`), the :math:`\epsilon` and/or
:math:`\sigma` arguments may be replaced by the string ``mix``.  When
``mix`` is given, the corresponding coefficient is not set directly but is
instead computed from the diagonal (:math:`I,I` and :math:`J,J`) entries
using the current mixing rule at initialization time.  This allows you to
specify a pair interaction that inherits mixed :math:`\epsilon` and/or
:math:`\sigma` while imposing an explicit :math:`\epsilon_4`.  For example:

.. code-block:: LAMMPS

   pair_coeff 1 2 mix mix -2.0

mixes both :math:`\epsilon` and :math:`\sigma` for the 1-2 pair but sets
:math:`\epsilon_4 = -2.0`.  The ``mix`` keyword is not permitted for
diagonal (:math:`I = J`) pairs.

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, the :math:`\epsilon` and
:math:`\sigma` coefficients and cutoff distance can be mixed.  Both
:math:`\epsilon` and :math:`\sigma` are mixed using the standard energy
and distance mixing rules (geometric by default).  See the
:doc:`pair_modify <pair_modify>` command for details.

The :math:`\epsilon_4` coefficient is **never** mixed.  For off-diagonal
pairs that were not explicitly set via :doc:`pair_coeff <pair_coeff>`,
:math:`\epsilon_4` defaults to zero, which reduces the potential to the
standard 12/6 Lennard-Jones form.  To supply a non-zero
:math:`\epsilon_4` for a mixed pair, use the ``mix`` keyword described
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

This pair style supports the use of the *inner*, *middle*, and *outer*
keywords of the :doc:`run_style respa <run_style>` command, meaning the
pairwise forces can be partitioned by distance at different levels of
the rRESPA hierarchy.  See the :doc:`run_style <run_style>` command for
details.

----------

Restrictions
""""""""""""

This pair style is part of the KSPACE package.  It is only enabled if
LAMMPS was built with that package.  See the
:doc:`Build package <Build_package>` page for more info.

This pair style requires use of a :doc:`kspace_style <kspace_style>`
command.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`,
:doc:`pair_style lj/cut/coul/long <pair_lj_cut_coul>`

Default
"""""""

none