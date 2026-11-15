# -*- coding: utf-8 -*-
#
# Degree-2 zonal non-hydrostatic state: three-shell fit to the targets (Table S9).
#
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
# 17/10/2026 Machiel Bos
#===============================================================================
"""Fit of three mantle load shells to the targets of the non-hydrostatic Earth (Table S9 of P1 PEPI v3):
excess J2 (Chambat et al. 2010, 1.14e-5), excess equatorial minus polar radius of the core-mantle boundary
(450 m) and of the surface (residual topography of Holdt et al. 2022, +0.1 +- 0.25 km), exact and
regularised (target sigmas, prior 0.5 % on the relative load), elastic support, measured on the supported
state on the figure of EARTHTIDES_FIGURE (FlatSupport; the paper: level). For a given support J2, CMB and
surface are nearly dependent, so the exact fit needs large loads; the regularised one shows what elastic
support can reach.
Until 2026-11-15 the script also measured unit responses and fits on a spherical background
(nonhydro.Support, the paper-2 Born machinery) as a cross-check; that part was dropped for the release
(it needs born.py) and the paper no longer quotes it; the older outputs remain in results/.
Usage: EARTHTIDES_FIGURE=level python scripts/nonhydro_support.py  ->  results/nonhydro_support_level.txt
"""
import sys
import warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')

from earthtides.model import EarthModel
sys.path.insert(0, 'scripts')
from figure_choice import FIGURE, SUF   # EARTHTIDES_FIGURE=level adds the three-shell fit on the level figure

SHELLS3 = [(3480e3, 5701e3), (5701e3, 5971e3), (5971e3, 6346.6e3)]
TARGET3 = {'J2': 1.14e-5, 'cmb': 450.0, 'surface': 100.0}
SIGMA3 = {'J2': 0.1e-6, 'cmb': 50.0, 'surface': 250.0}
PRIOR = 5e-3
lines = []


def out(t=''):
    print(t, flush=True)
    lines.append(t)


def main():
    from earthtides.geometry import figure, OMEGA_EARTH
    from earthtides.nonhydro import FlatSupport
    md, mds = EarthModel('data/PREM.dat'), EarthModel('data/PREMs.dat')
    base = figure(md, FIGURE)
    S = FlatSupport(md, mds, SHELLS3, base, OMEGA_EARTH)
    ca, a_ = S._CA(base, None, md.r0[S.ioc], md.r1[S.ioc])
    out(f'Three shells (LM, transition zone, shallow UM), targets J2 1.14e-5, CMB 450 m, surface +100 m')
    out('(sigmas 1e-7, 50 m, 250 m; prior 0.5 % on the relative load). Exact fit, then regularised.')
    out(f'Measured on the {FIGURE} figure (FlatSupport, elastic), hydrostatic e_f {ca / a_:.7f}')
    out('support            fit     a_LM        a_TZ        a_UM        J2         CMB [m]  surface [m]  e_f [%]')
    for lab, a in (('exact', S.fit(TARGET3)), ('reg.', S.fit(TARGET3, SIGMA3, PRIOR))):
        r = S.response(a)
        out(f'{"elastic":18s} {lab:6s}  {a[0]:+.3e}  {a[1]:+.3e}  {a[2]:+.3e}  {r["J2"]:.3e}  {r["cmb"]:7.0f}  '
            f'{r["surface"]:9.0f}    {100 * r["e_f"] * a_ / ca:.2f}')
    open(f'results/nonhydro_support{SUF}.txt', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    sys.path.insert(0, 'scripts')
    from provenance import Run
    _run = Run('elastic and viscous support of the mantle loads: targets (S10)', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
    main()
    _run.output(f'results/nonhydro_support{SUF}.txt')
    _run.__exit__(None, None, None)
