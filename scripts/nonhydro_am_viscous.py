# -*- coding: utf-8 -*-
#
# Angular-momentum balance of the non-hydrostatic PREM with the viscous (welded) core, routes A and B:
# the viscous counterpart of the AM column of results/nonhydro_diurnal.txt (slipping core). Plan 2026-10-19, task C.
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
# 31/10/2026 Machiel Bos
#===============================================================================
"""Angular-momentum balance |dH/dt + Omega x H - N| / |N| (scripts/check_angular_momentum.py) with the viscous core.

With the slipping core, route A (geometry without stress) broke the balance (2e-3..2e-2) while route B kept the
hydrostatic 8e-4 (results/nonhydro_diurnal.txt). With the welded viscous core both routes follow the e_f formula
(lab notes 10-31, TABLE 4 WITH THE VISCOUS CORE); does route A still break the balance? Same state and tidal run as
scripts/nonhydro_fcn_viscous.py (k_at), elastic PREM, J2 excess 1.14e-5; lines O1, P1, psi1, OO1 and two
frequencies near the FCN; Ekman thickness DELTA_KM.
Resolution test (10-31): --fluid-N (core and Ekman sub-layers), --cases / --lines (substrings) to test whether
route B's 3e-3..4e-3 next to the FCN at delta 0.5 km is the Ekman layer's resolution; amp = |H| Omega / |N|.
Usage: python scripts/nonhydro_am_viscous.py K N DELTA_KM [DELTA_KM ...] [--fluid-N FN] [--cases A,B] [--lines L,..]
       -> results/nonhydro_am_viscous_K<K>_N<N>[_fN<FN>][_<cases>][_<lines>].txt"""
import argparse
import os
import re
import sys
import time
import warnings
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.nonhydro import FlatSupport, FlatState, FlatMap
import check_angular_momentum as cam
import nonhydro_fcn_viscous as nv
from provenance import Run

LINES = (('O1', 0.92700), ('P1', 0.99455), ('psi1', 1.00546), ('OO1', 1.07546), ('s1.0016', 1.0016),
         ('s1.0030', 1.0030))


def solve(md, geo, st, sig, delta, K, N, fluid_N=None):
    """As nonhydro_fcn_viscous.k_at (elastic, CORE_DP static), returning the solver."""
    mv = md.viscous_core(sig * OMEGA_EARTH, delta=delta)
    par = nv.parent_map(md, mv)
    g = nv._ByParent(geo, par)
    nh = None
    if st is not None:
        nh = nv._ByParent(st, par) if isinstance(st, nv._RouteA) else nv._CoreStress(st, par)
    T = TidalSolver(mv, g, m=1, K=K, N=N, fluid_N=fluid_N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle', nonhydro=nh)
    T.solve()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('K', type=int)
    ap.add_argument('N', type=int)
    ap.add_argument('delta_km', type=float, nargs='+')
    ap.add_argument('--fluid-N', type=int, default=None)
    ap.add_argument('--cases', default='')
    ap.add_argument('--lines', default='')
    o = ap.parse_args()
    K, N, fN = o.K, o.N, o.fluid_N
    deltas = [v * 1e3 for v in o.delta_km]
    want_c = [c for c in o.cases.split(',') if c]
    want_l = [c for c in o.lines.split(',') if c]
    lines_run = [l for l in LINES if not want_l or l[0] in want_l]
    nv.CORE_DP = os.environ.get('CORE_DP', 'static')   # 'gauge': the old form (degree-0 imbalance at the CMB)
    with Run('task C: angular-momentum balance, non-hydrostatic PREM, viscous core, routes A and B',
             params=dict(K=K, N=N, fluid_N=fN, deltas=deltas, lines=lines_run, cases=want_c, core_dp=nv.CORE_DP, cmb=[0, 225, 450, 500]),
             inputs=['data/PREM.dat', 'data/PREMs.dat']) as run:
        lines = []

        def say(t):
            print(t, flush=True)
            lines.append(t)
        md, mds = EarthModel('data/PREM.dat'), EarthModel('data/PREMs.dat')
        import figure_choice as fc
        base = fc.base(md)
        S = FlatSupport(md, mds, nv.SHELLS, base, OMEGA_EARTH)
        cases = [('hydrostatic', base, None)]
        for cmb in (0.0, 225.0, 450.0, 500.0):
            st = FlatState(S, S.fit({'J2': 1.14e-5, 'cmb': cmb}), base, OMEGA_EARTH)
            geo = FlatMap(base, st)
            cases += [(f'{cmb:.0f} m A', geo, nv._RouteA(st)), (f'{cmb:.0f} m B', geo, st)]
        if want_c:
            cases = [c for c in cases if c[0] in want_c]
        say(f'Elastic PREM, J2 excess 1.14e-5, viscous (welded) core, K {K}, N {N}, fluid_N {fN or N}, core dp '
            f'{nv.CORE_DP}; |dH/dt + Omega x H - N| / |N| per line (and k, amp = |H| Omega / |N|)')
        worst = {}
        for dl in deltas:
            say(f'-- Ekman thickness delta {dl / 1e3:.2f} km')
            for name, geo, st in cases:
                t0, row = time.time(), f'  {name:12s}'
                for ln, sg in lines_run:
                    T = solve(md, geo, st, sg, dl, K, N, fN)
                    Hc, Nc, lhs = cam.balance(T, sg * OMEGA_EARTH)
                    am = np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc)
                    worst[(name, dl)] = max(worst.get((name, dl), 0.0), am)
                    row += f' | {ln} k {T.results()["kk"].real:.5f} AM {am:.1e}'
                    if want_l or want_c:
                        row += f' amp {np.linalg.norm(Hc) * OMEGA_EARTH / np.linalg.norm(Nc):.1f}'
                say(row + f' | {time.time() - t0:.0f} s')
        run.note(**{f'max_AM_{n.replace(" ", "_")}_d{d / 1e3:g}km': v for (n, d), v in worst.items()})
        tag = (f'_fN{fN}' if fN else '') + (f'_{nv.CORE_DP}' if nv.CORE_DP != 'static' else '') + ''.join('_' + re.sub(r'[^A-Za-z0-9.]+', '', x) for x in (o.cases, o.lines) if x)
        if want_c or want_l:
            tag += '_d' + '_'.join(f'{d / 1e3:g}' for d in deltas)
        import figure_choice as fc
        run.write_text(f'results/nonhydro_am_viscous_K{K}_N{N}{tag}{fc.SUF}.txt', '\n'.join(lines))


if __name__ == '__main__':
    main()
