# -*- coding: utf-8 -*-
#
# Consistent non-hydrostatic Earth in the diurnal band: FCN and Love numbers, route A vs B.
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
"""Consistent non-hydrostatic Earth in the diurnal band (docs/notes_nonhydrostatic.md, steps 3-4).

Elastic PREM; mantle loads (lower mantle, 670 km-Moho) fitted to the excess J2 of Chambat et al. (2010,
1.14e-5) and an excess CMB flattening of 0 / 225 / 450 / 500 m (nonhydro.FlatSupport, measured on the supported state), supported elastically
on the flattened rotating background (nonhydro.FlatState) and put into the rotating tidal solver
(TidalSolver(nonhydro=..., geometry=FlatMap)). Two routes:
  A  the route of Dehant & Defraigne (1997) and Dehant et al. (1999): the non-hydrostatic geometry,
     density and potential, hydrostatic initial stress (no tau, no slip terms);
     slip_nonlevel=False since 2026-11-14: the slope term of the slipping wall is part of what this route
     leaves out (with the solver's default the route would get it from the reference potential);
  B  consistent: plus the deviatoric initial stress tau that supports the shape (C(u) in the field
     equation, P1(tau).n tractions, slip terms at the CMB and ICB).
Printed: FCN period (sidereal days, space), the e_f estimate (A/A_m)(e_f - beta) from the hydrostatic
FCN and the e_f excess of the state, Love numbers h, l, k (Wahr frame) at O1, P1, psi1, OO1, and the
angular-momentum balance |dH/dt + Omega x H - N| / |N| (scripts/check_angular_momentum.py).
Usage: python scripts/nonhydro_diurnal.py [--figure level]  ->  results/nonhydro_diurnal[_level].txt
--figure level: everything on the level figure (geometry.figure; default: the first-order figure of paper 1);
the e_f of the hydrostatic core is then computed from the figure (first-order: the constant 0.0025452).
"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH, figure
from earthtides.solver import TidalSolver
from earthtides.nonhydro import FlatSupport, FlatState, FlatMap
import check_angular_momentum as cam

FIGURE = sys.argv[sys.argv.index('--figure') + 1] if '--figure' in sys.argv else 'first-order'
LINES = (('O1', 0.92700), ('P1', 0.99454), ('psi1', 1.00273), ('OO1', 1.07300))     # cpsd (psi1, OO1 fixed 2026-11-10)
SHELLS = [(3480e3, 5701e3), (5701e3, 6346.6e3)]
lines = []


def out(t=''):
    print(t, flush=True)
    lines.append(t)


class RouteA:
    """Geometry, density and potential of the state, hydrostatic initial stress."""

    def __init__(self, st):
        self.st = st

    def drho(self, *a):
        return self.st.drho(*a)

    def tau(self, *a):
        return None


def slip_on(nh):
    """The slope term of the slipping wall (TidalSolver slip_nonlevel) is one of the slip terms: off for a
    state that hides them (route A here; routes A and C of nonhydro_stress_split.py)."""
    return nh is None or hasattr(nh, 'slip')


def run(md, geo, nh, sig):
    T = TidalSolver(md, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle', nonhydro=nh, slip_nonlevel=slip_on(nh))
    T.solve()
    return T


def fcn(md, geo, nh, sig=(1.0012, 1.0016, 1.0030, 1.0036)):
    x = np.array(sig)
    k = np.array([run(md, geo, nh, s).results()['kk'] for s in x]).real
    return np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0]


def row(md, name, geo, nh, est=None):
    c = fcn(md, geo, nh)
    t = f'{name:22s} FCN {1 / (c - 1):6.1f}' + (f' (e_f estimate {1 / (est - 1):6.1f})' if est else ' ' * 22)
    for ln, sg in LINES:
        T = run(md, geo, nh, sg)
        r = T.results()
        Hc, Nc, lhs = cam.balance(T, sg * OMEGA_EARTH)
        t += (f' | {ln} {r["h"].real:.5f} {r["l"].real:.5f} {r["kk"].real:.5f} '
              f'AM {np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc):.0e}')
    out(t)
    return c


def main():
    md, mds = EarthModel('data/PREM.dat'), EarthModel('data/PREMs.dat')
    base = figure(md, FIGURE)
    S = FlatSupport(md, mds, SHELLS, base, OMEGA_EARTH)
    out('Elastic PREM, J2 excess 1.14e-5; per line: h, l, k (Wahr frame) and the angular-momentum balance')
    c0 = row(md, 'hydrostatic', base, None)
    e_f_hyd, AAm = 0.0025452, None
    if FIGURE != 'first-order':
        ca, a_ = S._CA(base, None, md.r0[S.ioc], md.r1[S.ioc])
        e_f_hyd = ca / a_
        out(f'figure: {FIGURE}; e_f of the hydrostatic core {e_f_hyd:.7f} (1/{1 / e_f_hyd:.2f})')
    for cmb in (0.0, 225.0, 450.0, 500.0):
        a = S.fit({'J2': 1.14e-5, 'cmb': cmb})
        r = S.response(a)
        st = FlatState(S, a, base, OMEGA_EARTH)
        geo = FlatMap(base, st)
        # (A/A_m)(e_f - beta): scale the hydrostatic offset by the e_f excess, beta = 6.07e-4
        est = 1 + (c0 - 1) * (e_f_hyd + r['e_f'] - 6.07e-4) / (e_f_hyd - 6.07e-4)
        out(f'-- CMB excess {cmb:.0f} m (e_f +{100 * r["e_f"] / e_f_hyd:.2f} %, surface {r["surface"]:.0f} m)')
        row(md, '   A (DDW route)', geo, RouteA(st))
        row(md, '   B (consistent)', geo, st, est)
    from provenance import Run
    with Run('non-hydrostatic Earth in the diurnal band: routes A (DDW, slip_nonlevel off) and B (consistent)',
             params=dict(K=4, N=24, cmb=[0, 225, 450, 500], figure=FIGURE), inputs=['data/PREM.dat', 'data/PREMs.dat']) as run_:
        run_.write_text('results/nonhydro_diurnal' + ('' if FIGURE == 'first-order' else '_' + FIGURE) + '.txt', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    sys.path.insert(0, 'scripts')
    from provenance import Run
    _run = Run('non-hydrostatic Earth in the diurnal band: FCN, Love numbers, routes A and B (Table 4, Fig. 9)', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
    main()
    _run.output('results/nonhydro_diurnal.txt')
    _run.__exit__(None, None, None)
