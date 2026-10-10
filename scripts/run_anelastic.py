# -*- coding: utf-8 -*-
#
# Degree-2 body tide on an anelastic, hydrostatically flattened PREM.
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
"""Degree-2 body tide on an anelastic, hydrostatically flattened PREM.

Per band the shear modulus is evaluated at a representative tidal frequency
(Mf, O1, M2) with the absorption-band model of astyanax (T0 = 1 s,
Tm = 200 s, alpha = 0.15). Static otherwise: no inertia, no Coriolis.

Compares elastic vs anelastic and sphere vs ellipsoid, all fitted to the IERS
2010 displacement model, and lists TN36 Table 7.2 for reference.
Results -> results/anelastic_<model>.txt
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
ALPHA, TM = 0.15, 200.0
BANDS = {0: ('Mf', 2 * np.pi / (13.661 * 86400)), 1: ('O1', 6.7598e-5), 2: ('M2', 1.4053e-4)}
# TN36 Table 7.2 (h(0) complex, h(2); l(0) complex, l(1), l(2); primes)
TABLE = {0: dict(h0=0.6109 - 0.0037j, h2=-0.0006, hp=0.0001, l0=0.0864 - 0.0011j, l1=0.0, l2=0.0002),
         1: dict(h0=0.6028 - 0.0025j, h2=-0.0006, l0=0.0846 - 0.0006j, l1=0.0012, l2=0.0002,
                 lp=-0.0002),
         2: dict(h0=0.6078 - 0.0022j, h2=-0.0006, l0=0.0847 - 0.0007j, l1=0.0024, l2=0.0002)}
lines = []


def out(s=''):
    print(s)
    lines.append(s)


def c(z, w=7):
    z = complex(z)
    return f'{z.real:+.{w - 1}f}{z.imag:+.{w - 1}f}i'


md0 = EarthModel(f'data/{MODEL}.dat')
out(f'Model {MODEL}; anelasticity: absorption band T0 = 1 s .. Tm = {TM:.0f} s, alpha = {ALPHA}')
out('Static (no inertia, no Coriolis); m = 1 in per-region Tisserand frames.')
out()
for m, (name, w) in BANDS.items():
    md = md0.anelastic(w, alpha=ALPHA, Tm=TM)
    d = md0.dmu_factor(w, alpha=ALPHA, Tm=TM)
    out(f'==== m = {m} ({name}, T = {2 * np.pi / w / 3600:.2f} h): '
        f'dmu/mu upper mantle {c(d[7], 5)}, lower mantle {c(d[3], 5)}')
    rows = {}
    for label, model, scale in (('elastic sphere', md0, 0.0), ('elastic ellipsoid', md0, 1.0),
                                ('anelastic sphere', md, 0.0), ('anelastic ellipsoid', md, 1.0)):
        g = ClairautGeometry(model, scale=scale)
        S = TidalSolver(model, g, m=m, K=4, N=30, omega_f=0.0,
                        omega_rot=OMEGA_EARTH * np.sqrt(scale))
        r = S.solve()
        f = iers_fit(S, g)
        f['k2'] = r['kk']
        rows[label] = f
    keys = ['h0', 'h2', 'l0', 'l2'] + (['l1'] if m else []) + (['hp'] if m == 0 else []) \
        + (['lp'] if m == 1 else []) + ['k2']
    out(f"  {'':22s}" + ''.join(f'{k:>24s}' for k in keys))
    for label, f in rows.items():
        out(f'  {label:22s}' + ''.join(f'{c(f[k]):>24s}' for k in keys))
    T = TABLE[m]
    out(f"  {'IERS Table 7.2':22s}" + ''.join(f"{(c(T[k]) if k in T else ''):>24s}" for k in keys))
    ea, aa = rows['elastic ellipsoid'], rows['anelastic ellipsoid']
    es, as_ = rows['elastic sphere'], rows['anelastic sphere']
    out('  anelastic change of the ellipticity terms: '
        + ', '.join(f'{k} {c(aa[k] - ea[k], 6)}' for k in keys if k not in ('h0', 'l0', 'k2')))
    out(f"  anelastic change of h0 (sphere) {c(as_['h0'] - es['h0'], 6)}, "
        f"l0 {c(as_['l0'] - es['l0'], 6)}, k2 {c(as_['k2'] - es['k2'], 6)}")
    out()

open(f'results/anelastic_{MODEL}.txt', 'w').write('\n'.join(lines) + '\n')
