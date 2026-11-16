# -*- coding: utf-8 -*-
#
# Which part of the deviatoric stress moves the FCN and balances angular momentum? (P1 v2 Sect. 6.3, review)
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
# 10/11/2026 Machiel Bos
#===============================================================================
"""The state of Sect. 6 enters the tidal solve through three pieces (nonhydro.FlatState): its density (drho),
the deviatoric stress tau_0 of the solids (tau: volume terms and the tractions at the interfaces) and the
pressure terms of the slip conditions at the fluid-solid boundaries (slip, Woodhouse & Dahlen 1978). Route A of
nonhydro_diurnal.py keeps drho only, route B all three. This script adds C (drho + tau, no slip terms) and
D (drho + slip terms, no tau) for elastic PREM with the excess J2 and CMB_M of excess CMB flattening, inviscid
slipping core as nonhydro_diurnal.py (K 4, N 24), and gives per route the FCN period, the angular-momentum
residual at four diurnal frequencies and the Love numbers (h0 h2 l0 l1 l2 k, IERS fit) at Mf, O1, P1, K1,
psi1, OO1 and M2, with the differences from route B.
Since 2026-11-14 'slip terms' includes the slope term of the wall (TidalSolver slip_nonlevel): off in A and C.
Usage: python scripts/nonhydro_stress_split.py [CMB_M] [NPROC] [--figure level]
       ->  results/nonhydro_stress_split_<CMB>m[_level].txt"""
import sys
import warnings
from multiprocessing import Pool
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH, figure
from earthtides.solver import TidalSolver
from earthtides.nonhydro import FlatSupport, FlatState, FlatMap
from earthtides.iers import iers_fit
import check_angular_momentum as cam
import nonhydro_diurnal as nd

FIGURE = sys.argv[sys.argv.index('--figure') + 1] if '--figure' in sys.argv else 'first-order'   # or 'level'
SUF = '' if FIGURE == 'first-order' else '_' + FIGURE
_pos = [a for a in sys.argv[1:] if a not in ('--figure', 'level', 'first-order')]   # argv stays intact for the workers
CMB = float(_pos[0]) if len(_pos) > 0 else 450.0
NPROC = int(_pos[1]) if len(_pos) > 1 else 4
WAVES = (('Mf', 0, 0.07300), ('O1', 1, 0.92700), ('P1', 1, 0.99454), ('K1', 1, 1.0), ('psi1', 1, 1.002730421),
         ('OO1', 1, 1.07300), ('M2', 2, 1.92700))
AM_SIG = (0.92700, 0.99454, 1.00273, 1.07300)          # O1, P1, psi1, OO1 as nonhydro_diurnal.py
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k')          # k: degree-2 potential Love number as solved (kk)
ROUTES = ('A', 'B', 'C', 'D')
LABEL = {'A': 'A: density only', 'B': 'B: tau and slip terms', 'C': 'C: tau, no slip terms',
         'D': 'D: slip terms, no tau'}


class Part:
    """Hides pieces of the state: tau (volume and interface tractions of the solids) and/or the slip terms."""

    def __init__(self, st, tau, slip):
        self.st, self._tau = st, tau
        if slip:
            self.slip = st.slip

    def drho(self, *a):
        return self.st.drho(*a)

    def tau(self, *a):
        t = self.st.tau(*a)
        if self._tau or t is None:
            return t if self._tau else None
        if hasattr(self, 'slip'):                    # the slip branch adds dp to tau: give it a zero tau
            return {k: v * 0.0 for k, v in t.items()}
        return None


def state():
    md, mds = EarthModel('data/PREM.dat'), EarthModel('data/PREMs.dat')
    base = figure(md, FIGURE)
    S = FlatSupport(md, mds, nd.SHELLS, base, OMEGA_EARTH)
    a = S.fit({'J2': 1.14e-5, 'cmb': CMB})
    st = FlatState(S, a, base, OMEGA_EARTH)
    return md, FlatMap(base, st), st, S.response(a)


def route(st, r):
    return {'A': Part(st, False, False), 'B': st, 'C': Part(st, True, False), 'D': Part(st, False, True)}[r]


def solve(md, geo, nh, m, sig):
    T = TidalSolver(md, geo, m=m, K=4 if m != 2 else 5, N=24 if m != 2 else 30, omega_f=0.0,
                    coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2),
                    dyn_fluid=True, consistent_reference=True, frame='mantle', nonhydro=nh,
                    slip_nonlevel=nd.slip_on(nh))
    T.solve()
    return T


_S = {}


def init():
    _S['md'], _S['geo'], _S['st'], _ = state()


def job(arg):
    r, kind, m, sig = arg
    md, geo, st = _S['md'], _S['geo'], _S['st']
    nh = route(st, r)
    if kind == 'fcn':
        return arg, nd.fcn(md, geo, nh)
    if kind == 'am':
        T = solve(md, geo, nh, 1, sig)
        Hc, Nc, lhs = cam.balance(T, sig * OMEGA_EARTH)
        return arg, float(np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc))
    sigs = (1 - 1e-5, 1 + 1e-5) if (m == 1 and sig == 1.0) else (sig,)     # K1: across the tilt-over pole
    f = []
    for s in sigs:
        T = solve(md, geo, nh, m, s)
        x = iers_fit(T, geo)
        x['k'] = T.results()['kk']
        f.append(x)
    return arg, {k: float(np.mean([x[k].real for x in f])) for k in KEYS if k in f[0]}


def main():
    from provenance import Run
    with Run(f'deviatoric stress split into tau and slip terms, {CMB:.0f} m, inviscid slipping core',
             params=dict(cmb=CMB, waves=WAVES, am_sig=AM_SIG, figure=FIGURE), inputs=['data/PREM.dat', 'data/PREMs.dat']) as run:
        jobs = [(r, 'fcn', 1, 0.0) for r in ROUTES] + [(r, 'am', 1, s) for r in ROUTES for s in AM_SIG] + \
               [(r, 'love', m, s) for r in ROUTES for _, m, s in WAVES]
        with Pool(NPROC, initializer=init) as p:
            res = dict(p.map(job, jobs, chunksize=1))
        _, _, _, resp = state()
        lines = [f'elastic PREM, excess J2 1.14e-5, CMB {CMB:.0f} m (e_f excess {100 * resp["e_f"] / 0.0025452:.2f} %), '
                 f'inviscid slipping core (K 4, N 24; M2 K 5, N 30); figure: {FIGURE}', '',
                 f'{"route":24s} {"FCN sd":>7s}  AM residual at ' + ' '.join(f'{s:.5f}' for s in AM_SIG)]
        for r in ROUTES:
            c = res[(r, 'fcn', 1, 0.0)]
            lines.append(f'{LABEL[r]:24s} {1 / (c - 1):7.1f}  ' + ' '.join(f'{res[(r, "am", 1, s)]:7.0e}' for s in AM_SIG))
        lines += ['', 'Love numbers of route B, and route minus B (1e-6)',
                  f'{"wave":6s} {"route":3s} ' + ' '.join(f'{k:>9s}' for k in KEYS)]
        for w, m, s in WAVES:
            b = res[('B', 'love', m, s)]
            lines.append(f'{w:6s} {"B":3s} ' + ' '.join(f'{b[k]:9.6f}' if k in b else ' ' * 9 for k in KEYS))
            for r in ('A', 'C', 'D'):
                x = res[(r, 'love', m, s)]
                lines.append(f'{"":6s} {r:3s} ' + ' '.join(f'{1e6 * (x[k] - b[k]):+9.1f}' if k in b else ' ' * 9
                                                           for k in KEYS))
        run.write_text(f'results/nonhydro_stress_split_{CMB:.0f}m{SUF}.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
