# -*- coding: utf-8 -*-
#
# The slipping fluid-solid boundary on a wall that is not a level surface (test class).
#
# This file is part of earth-tides 1.1.
#
# earth-tides is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 14/11/2026 Machiel Bos
#===============================================================================
"""SlipTermSolver: TidalSolver with the normal slip condition of the non-hydrostatic Earth (docs/notes_loading.md
6.3, born.slip_terms, solver._bnd with p['slip']) switched on for the hydrostatic state (lab notes 2026-11-14).

The solver's conditions at a slipping wall (_fs_interface: n.tau.n of the Lagrangian stress increment continuous)
are valid on level surfaces only. With consistent_reference=True, Phi0 is the potential that Poisson's equation
gives for the density on the mapped surfaces; a wall that is a coordinate surface of a first-order map, or of a
bumped map, is then off the level surface of Phi0 at O(eps^2) (metres). The fluid and the solid particle in
contact after the slip started at different places along the wall, at different fluid pressures, and
    n.tau.n(fluid) - n.tau.n(solid) = -(u_fluid - u_solid) . grad_S dp,
dp the fluid pressure along the wall: d(dp)/dtheta = -rho_fluid d(Phi0)/dtheta along the wall.
Left out: the solid's tangential condition (n.tau)_t = (tau0 . eps . n)_t, with tau0 the deviatoric stress of the
solid that balances dp where the density jumps (~ (rho_f - rho_s) g h, 1e5 .. 1e6 Pa, times the strain: second
order in the small quantities; the slip in the normal condition is the large one), and the state of the fluid
itself, whose density surfaces are off level by the same O(eps^2)."""
from earthtides.solver import TidalSolver


class SlipTermSolver(TidalSolver):
    """Since 2026-11-14 the term is in the solver itself (TidalSolver(slip_nonlevel=True), the default); this class
    only switches it on, for the scripts that were written before the option existed."""

    def __init__(self, *args, **kw):
        kw['slip_nonlevel'] = True
        super().__init__(*args, **kw)
