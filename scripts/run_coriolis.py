# -*- coding: utf-8 -*-
#
# How much of the gap to IERS does the Coriolis force (solid regions) close?
#
# This file is part of earth-tides 1.0.
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
"""How much of the gap to IERS does the Coriolis force (solid regions) close?

Per band (Mf, O1, M2), PREMs, fitted to the IERS 2010 displacement model:
  ellipsoid, static                      (previous result)
  sphere, inertia only                   (rho omega^2 u in the solids)
  sphere, Coriolis + inertia
  ellipsoid, Coriolis + inertia
  ellipsoid, Coriolis + inertia, anelastic
The outer core stays a static weak solid (no Coriolis, no inertia); m = 0, 1
are solved in per-region Tisserand frames. Results -> results/coriolis_<model>.txt
"""
import sys
import warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'PREMs'
BANDS = {0: ('Mf', 2 * np.pi / (13.661 * 86400)), 1: ('O1', 6.7598e-5), 2: ('M2', 1.4053e-4)}
TABLE = {0: dict(h0=0.6109 - 0.0037j, h2=-0.0006, hp=0.0001, l0=0.0864 - 0.0011j, l1=0.0, l2=0.0002),
         1: dict(h0=0.6028 - 0.0025j, h2=-0.0006, l0=0.0846 - 0.0006j, l1=0.0012, l2=0.0002,
                 lp=-0.0002),
         2: dict(h0=0.6078 - 0.0022j, h2=-0.0006, l0=0.0847 - 0.0007j, l1=0.0024, l2=0.0002)}
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def c(z):
    z = complex(z)
    return f'{z.real:+.6f}' if abs(z.imag) < 5e-7 else f'{z.real:+.6f}{z.imag:+.6f}i'


md0 = EarthModel(f'data/{MODEL}.dat')
out(f'Model {MODEL}; Coriolis + inertia in solid layers only; outer core static weak solid')
out()
for m, (name, w) in BANDS.items():
    mda = md0.anelastic(w)
    cases = [('ellipsoid, static', md0, 1.0, 0.0, 0.0),
             ('sphere, inertia', md0, 0.0, 0.0, w),
             ('sphere, Coriolis+inertia', md0, 0.0, OMEGA_EARTH, w),
             ('ellipsoid, Cor+inertia', md0, 1.0, OMEGA_EARTH, w),
             ('  + anelastic', mda, 1.0, OMEGA_EARTH, w)]
    out(f'==== m = {m} ({name}) ====')
    keys = ['h0', 'h2', 'l0', 'l2', 'l1'] + (['hp', 'lp'] if m == 0 else ['lp'] if m == 1 else [])
    out(f"  {'':26s}" + ''.join(f'{k:>21s}' for k in keys))
    for label, model, scale, cor, wt in cases:
        g = ClairautGeometry(model, scale=scale)
        S = TidalSolver(model, g, m=m, K=5, N=30, omega_f=0.0, coriolis=cor, omega_t=wt,
                        omega_rot=OMEGA_EARTH * np.sqrt(scale))
        S.solve()
        f = iers_fit(S, g)
        out(f'  {label:26s}' + ''.join(f'{c(f.get(k, 0)):>21s}' for k in keys)
            + f"   misfit {max(f['misR'] / f['scaleR'], f['misT'] / f['scaleT']):.0e}")
    T = TABLE[m]
    out(f"  {'IERS Table 7.2':26s}" + ''.join(f"{(c(T[k]) if k in T else ''):>21s}" for k in keys))
    out()

open(f'results/coriolis_{MODEL}.txt', 'w').write('\n'.join(lines) + '\n')
