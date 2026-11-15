# -*- coding: utf-8 -*-
#
# Angular-momentum balance of the non-hydrostatic Earth through the diurnal band: with and without the stress
# that supports the shape (P1 v2, Fig. am_sweep).
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
# 01/11/2026 Machiel Bos
#===============================================================================
"""|dH/dt + Omega x H - N| / |N| (scripts/check_angular_momentum.py) through the diurnal band for elastic
PREM: hydrostatic, and the elastically supported state with the excess J2 of Chambat et al. (2010) and CMB_M m
of excess CMB flattening, route A (geometry, density, potential; hydrostatic stress), route B (plus the
stress that supports the shape) and route D (A plus the slip terms of the non-level boundaries, no tau_0: the
slipping core without the deviatoric stress, as nonhydro_stress_split.py; added 2026-11-15). Inviscid core slipping along its boundaries, K 4 N 24, as
scripts/nonhydro_diurnal.py (Table 6 of P1). Extra frequencies next to the tilt-over mode (1 cpsd) and the FCN.
Usage: python scripts/nonhydro_am_sweep.py CMB_M [WORKERS]  ->  results/nonhydro_am_sweep_<CMB>m.{txt,npz}
"""
import os
import sys
import time
import warnings
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
from multiprocessing import Pool

import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.nonhydro import FlatSupport, FlatState, FlatMap
import check_angular_momentum as cam
from nonhydro_diurnal import RouteA, run, SHELLS
from provenance import Run

SIG = np.unique(np.concatenate([np.linspace(0.88, 1.08, 41),
                                [0.9996, 0.9998, 0.9999, 1.0001, 1.0002, 1.0004],
                                np.linspace(1.0016, 1.0030, 8)]))
SIG = SIG[np.abs(SIG - 1.0) > 1e-9]          # K1 itself: the tilt-over mode, singular
CASES = ('hydrostatic', 'A', 'B', 'D')
_W = {}


class SlipOnly:
    """Route D: density and the slip terms of the state, no tau_0 (Part(st, False, True) of nonhydro_stress_split.py)."""

    def __init__(self, st):
        self.st, self.slip = st, st.slip

    def drho(self, *a):
        return self.st.drho(*a)

    def tau(self, *a):
        t = self.st.tau(*a)
        return None if t is None else {k: v * 0.0 for k, v in t.items()}


def init(cmb):
    md, mds = EarthModel('data/PREM.dat'), EarthModel('data/PREMs.dat')
    import figure_choice as fc
    base = fc.base(md)
    S = FlatSupport(md, mds, SHELLS, base, OMEGA_EARTH)
    st = FlatState(S, S.fit({'J2': 1.14e-5, 'cmb': cmb}), base, OMEGA_EARTH)
    _W.update(md=md, base=base, geo=FlatMap(base, st), st=st)


def one(job):
    case, sg = job
    w = _W
    geo, nh = {'hydrostatic': (w['base'], None), 'A': (w['geo'], RouteA(w['st'])), 'B': (w['geo'], w['st']),
               'D': (w['geo'], SlipOnly(w['st']))}[case]
    T = run(w['md'], geo, nh, sg)
    Hc, Nc, lhs = cam.balance(T, sg * OMEGA_EARTH)
    return case, sg, np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc), T.results()['h'].real


def main():
    cmb = float(sys.argv[1])
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    jobs = [(c, s) for c in CASES for s in SIG]
    with Run(f'P1 v2: AM balance through the diurnal band, hydrostatic / routes A, B, D, {cmb:.0f} m',
             params=dict(cmb=cmb, sig=SIG.tolist(), K=4, N=24, core='inviscid, slipping', workers=nproc),
             inputs=['data/PREM.dat', 'data/PREMs.dat']) as run_:
        t0 = time.time()
        with Pool(nproc, initializer=init, initargs=(cmb,)) as p:
            res = p.map(one, jobs, chunksize=1)
        am = {c: np.array([r[2] for r in res if r[0] == c]) for c in CASES}
        h = {c: np.array([r[3] for r in res if r[0] == c]) for c in CASES}
        lines = [f'AM balance |dH/dt + Omega x H - N| / |N|, elastic PREM, excess J2 1.14e-5, CMB {cmb:.0f} m, '
                 f'inviscid slipping core, K 4 N 24 ({time.time() - t0:.0f} s)',
                 f'{"sigma (cpsd)":>12s} ' + ' '.join(f'{c:>12s}' for c in CASES)]
        for i, s in enumerate(SIG):
            lines.append(f'{s:12.5f} ' + ' '.join(f'{am[c][i]:12.2e}' for c in CASES))
        lines.append('max over the band: ' + ', '.join(f'{c} {am[c].max():.1e}' for c in CASES))
        lines.append('median: ' + ', '.join(f'{c} {np.median(am[c]):.1e}' for c in CASES))
        print('\n'.join(lines))
        import figure_choice as fc
        out = f'results/nonhydro_am_sweep_{cmb:.0f}m{fc.SUF}'
        run_.write_text(out + '.txt', '\n'.join(lines) + '\n')
        run_.savez(out + '.npz', sig=SIG, **{f'am_{c}': am[c] for c in CASES}, **{f'h_{c}': h[c] for c in CASES})


if __name__ == '__main__':
    main()
