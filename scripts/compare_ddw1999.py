# -*- coding: utf-8 -*-
#
# Benchmark against Dehant, Defraigne & Wahr (1999, JGR 104, B1, 1035-1058).
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
# 5/10/2026 Machiel Bos
#===============================================================================
"""Benchmark against Dehant, Defraigne & Wahr (1999, JGR 104, B1, 1035-1058).

Their Table 11 (Love numbers in the Mathews et al. definition = IERS form,
eq. 21) for "Degree 2 ... Elastic Hydrostatic" PREM, i.e. Wahr's full theory
(ellipsoidal, rotating, fluid outer core, normal-mode resonances) without
anelasticity. Compared with this solver: elastic, hydrostatic flattening,
Coriolis + inertia in the solid layers, static weak-solid outer core.
Results -> results/ddw1999_<model>.txt
"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from provenance import Run
_run = Run('comparison with Dehant et al. (1999), inviscid core, PREMs (S4)', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'PREMs'
# DDW1999 Table 11, elastic hydrostatic (zonal row has no frequency: static)
DDW = {0: ('zonal', 0.0, dict(h0=0.59892, h2=-0.00047, hp=0.00014, l0=0.08304, l2=0.00008)),
       1: ('O1', 6.7598e-5, dict(h0=0.59553, h2=-0.00048, l0=0.08298, l1=0.00070, l2=0.00008,
                                 lp=0.00020)),
       2: ('M2', 1.4053e-4, dict(h0=0.60175, h2=-0.00047, l0=0.08359, l1=0.00100, l2=0.00008))}
IERS = {0: dict(h2=-0.0006, hp=0.0001, l2=0.0002), 1: dict(h2=-0.0006, l1=0.0012, l2=0.0002, lp=-0.0002),
        2: dict(h2=-0.0006, l1=0.0024, l2=0.0002)}
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


md = EarthModel(f'data/{MODEL}.dat')
g = ClairautGeometry(md)
out(f'{MODEL}, elastic, hydrostatic ellipsoid, Coriolis + inertia in solids, vs DDW1999 Table 11')
out('Tisserand: rigid rotation of every region removed; Wahr frame: solid regions rotate freely,')
out('rigid rotation removed from the surface displacement only (Wahr 1981 eqs. 4.3, 4.6)')
out()
WZ = 2 * np.pi / (13.661 * 86400)      # zonal: use Mf for the dynamics (DDW: frequency independent)
for m, (name, w, ref) in DDW.items():
    fits = {}
    for frame in ('tisserand', 'wahr'):
        S = TidalSolver(md, g, m=m, K=5, N=30, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=w or WZ, omega_rot=OMEGA_EARTH, wahr_frame=(frame == 'wahr'))
        S.solve()
        fits[frame] = iers_fit(S, g)
    out(f'---- {name} (m = {m}) ----')
    out(f"  {'':8s}{'Tisserand':>12s}{'Wahr frame':>12s}{'DDW1999':>12s}{'diff(W)':>12s}{'IERS 2010':>12s}")
    for k, v in ref.items():
        t, wv = (float(np.real(fits[fr][k])) for fr in ('tisserand', 'wahr'))
        iv = IERS[m].get(k)
        out(f"  {k:8s}{t:12.5f}{wv:12.5f}{v:12.5f}{wv - v:+12.5f}" + (f'{iv:12.4f}' if iv is not None else ''))
    out()

open(f'results/ddw1999_{MODEL}.txt', 'w').write('\n'.join(lines) + '\n')
_run.output(f'results/ddw1999_{MODEL}.txt')
_run.__exit__(None, None, None)
