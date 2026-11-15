# -*- coding: utf-8 -*-
#
# The sign of the radial rho_2 term of Smith (1974) eq. (5.26): printed, corrected and dropped, against
# the solver at decreasing flattening (Supplementary Material of P1).
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
# 08/11/2026 Machiel Bos
#===============================================================================
"""Smith's theory is first order in the flattening: against an exact solution its latitude terms (themselves
proportional to eps) must agree to a relative error that falls in proportion to eps. A wrong first-order term
leaves a relative error that does not fall. The self-gravitating two-layer body of check_smith1974.py, part 4
(data/TWOLAYER.dat: density jump and density gradients, solid, first-order Clairaut ellipticity), with the
radial rho_2 term of (5.26) as printed (+), corrected (-, the default of smith1974.SmithGrav) and dropped (0),
against the solver, at reference rotations 1, 0.3 and 0.1 times the Earth's (eps ~ rotation^2), Coriolis force
at the Earth's rate. Relative differences of the terms of Dehant et al. (1999, eq. 20) l+/l0, h+/h0, w+/l0.
Usage: python scripts/smith1974_sign_test.py  ->  results/smith1974_sign_test.txt"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry
from earthtides.smith1974 import SmithGrav
from check_smith1974 import layered_terms, TIDES

SIGNS = (('printed +', 1.0), ('corrected -', -1.0), ('dropped 0', 0.0))
SPINS = (1.0, 0.3, 0.1)


def rel(P, Q, k, r):
    a, b = (P[k] / P[r]).real, (Q[k] / Q[r]).real
    return abs(a / b - 1)


def main():
    from provenance import Run
    md = EarthModel('data/TWOLAYER.dat')
    clairaut = ClairautGeometry(md)
    with Run('Smith (1974) eq. 5.26 radial rho2 term: printed vs corrected vs dropped, two-layer body',
             params=dict(signs=SIGNS, spins=SPINS, tides=TIDES), inputs=['data/TWOLAYER.dat']) as run_:
        lines = ['# relative difference Smith vs solver of l+/l0, h+/h0, w+/l0; eps = Clairaut ellipticity at the '
                 'surface', '# tide      Coriolis  rotation  eps        sign          l+/l0     h+/h0     w+/l0']
        for tide, m, fr in TIDES:
            for cor in (False, True):
                for spin in SPINS:
                    for name, sg in SIGNS:
                        SmithGrav.RHO2_SIGN = sg
                        P, Q, eps = layered_terms(md, clairaut, spin, m, fr, cor)
                        lines.append('%-4s      %-8s  %.1f       %.2e   %-12s  %.1e   %.1e   %.1e' % (
                            tide, 'yes' if cor else 'no', spin, eps, name, rel(P, Q, 'lp', 'l0'),
                            rel(P, Q, 'hp', 'h0'), rel(P, Q, 'wp', 'l0')))
                        print(lines[-1], flush=True)
        SmithGrav.RHO2_SIGN = -1.0
        run_.write_text('results/smith1974_sign_test.txt', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
