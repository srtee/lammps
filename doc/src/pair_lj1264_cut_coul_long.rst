.. index:: pair_style lj1264/cut/coul/long

pair_style lj1264/cut/coul/long command
========================================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style lj1264/cut/coul/long cutoff (cutoff2)

* cutoff = global cutoff for LJ (and Coulombic if only 1 arg) (distance units)
* cutoff2 = global cutoff for Coulombic (optional) (distance units)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style lj1264/cut/coul/long 10.0
   pair_style lj1264/cut/coul/long 10.0 8.0
   pair_coeff * * 100.0 3.0 50.0
   pair_coeff 1 1 100.0 3.5 80.0 9.0
   pair_coeff 1 2 mix mix -20.0

Description
"""""""""""

The *lj1264/cut/coul/long* style computes a 12/6/4 Lennard-Jones potential
combined with long-range Coulombic interactions, given by

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

The :math:`r^{-4}` term arises from charge-induced-dipole and
dipole-induced-dipole interactions and is particularly important for
modeling ions in polar solvents.  The 12-6-4 potential was
parametrized for monovalent, divalent, trivalent, and tetravalent metal
ions in several water models by the Merz group
:ref:`(Sengupta et al.) <Sengupta2021>`,
:ref:`(Li et al., 2020) <Li2020>`, and
:ref:`(Li et al., 2021) <Li2021>`.

In typical ion--water simulations, :math:`C_4` should be **positive** for
cation--oxygen (OW) interactions, **negative** for anion--oxygen (OW)
interactions, and **zero** for all other interactions, including
ion--hydrogen (HW).  This reflects the physical origin of the
:math:`r^{-4}` term as a charge--induced-dipole interaction that depends
on the polarizability of the water oxygen site.

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
* :math:`C_4` (energy :math:`\times` distance\ :sup:`4` units)
* cutoff1 (distance units)

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

----------

.. _Sengupta2021:

**(Sengupta et al.)** A. Sengupta, Z. Li, L. F. Song, K. M. Merz Jr.,
"Parameterization of Monovalent Ions for the OPC3, OPC, TIP3P-FB, and
TIP4P-FB Water Models", J. Chem. Inf. Model. **61**, 869--880 (2021).
doi:10.1021/acs.jcim.0c01390

.. _Li2020:

**(Li et al., 2020)** Z. Li, L. F. Song, K. M. Merz Jr., "Systematic
Parametrization of Divalent Metal Ions for the OPC3, OPC, TIP3P-FB, and
TIP4P-FB Water Models", J. Chem. Theory Comput. **16**, 4429--4442 (2020).
doi:10.1021/acs.jctc.0c00194

.. _Li2021:

**(Li et al., 2021)** Z. Li, L. F. Song, K. M. Merz Jr., "Parametrization
of Trivalent and Tetravalent Metal Ions for the OPC3, OPC, TIP3P-FB, and
TIP4P-FB Water Models", J. Chem. Theory Comput. **17**, 2342--2354 (2021).
doi:10.1021/acs.jctc.0c01320