# -*- coding: utf-8 -*-
#
# Degree-2 Love numbers and free core nutation: PREM vs STW105 (Kustowski et
# al. 2008).
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
# 6/10/2026 Machiel Bos
#===============================================================================
"""Degree-2 Love numbers and free core nutation: PREM vs STW105 (Kustowski et al. 2008).

Same settings as run_diurnal.py (Table 2 of the paper): fluid outer core with
slip, Coriolis + inertia everywhere, Poisson-consistent reference potential,
mantle frame, k in the DDW normalisation, hydrostatic flattening (Clairaut) of
each model. Transverse isotropy (STW105, 24-220 km) is Voigt-averaged.
FCN: sign change of det H (as in examples/ex5). Anelastic M2 (alpha = 0.15,
Tm = 200 s) on the same ellipsoid.
Usage: python scripts/compare_models.py [PREM STW105 ...]
Results -> results/compare_models.txt (default models) or results/compare_models_<names>.txt
"""
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import brentq

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

TIDES = [('Mf', 0.0730, 0), ('O1', 0.92700, 1), ('P1', 0.99454, 1), ('K1', 1.0, 1),
         ('psi1', 1.00273, 1), ('phi1', 1.00546, 1), ('OO1', 1.07300, 1), ('M2', 1.93227, 2)]
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k0')
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def solver(md, g, m, fr, K=4, N=24):
    return TidalSolver(md, g, m=m, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH,
                       omega_t=fr * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2),
                       dyn_fluid=True, consistent_reference=True, frame='mantle')


def love(md, g, m, fr):
    K, N = (5, 30) if m == 2 else (4, 24)
    if m == 1 and fr == 1.0:                     # K1: average across the tilt-over pole
        a, b = love(md, g, m, 1 - 1e-5), love(md, g, m, 1 + 1e-5)
        return {k: 0.5 * (a[k] + b[k]) for k in a}
    S = solver(md, g, m, fr, K, N)
    r = S.solve()
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5
    return f


def fcn_period(md, g):
    def sgn(P):
        S = solver(md, g, 1, 1 + 1 / P)
        S.solve()
        sd = np.linalg.slogdet(S.H / np.abs(S.H).max(axis=1)[:, None])
        return sd[0] * np.exp(sd[1] - ref)
    S = solver(md, g, 1, 1 + 1 / 450)
    S.solve()
    ref = np.linalg.slogdet(S.H / np.abs(S.H).max(axis=1)[:, None])[1]
    Ps = np.array([380., 420., 445., 470., 500., 560.])
    v = [sgn(P) for P in Ps]
    return [brentq(sgn, Ps[i], Ps[i + 1], xtol=1e-6) for i in range(len(Ps) - 1)
            if np.sign(v[i]) != np.sign(v[i + 1])]


models = sys.argv[1:] or ['PREM', 'STW105']
res = {}
for name in models:
    md = EarthModel(f'data/{name}.dat')
    g = ClairautGeometry(md)
    res[name] = {t: love(md, g, m, fr) for t, fr, m in TIDES}
    w = 1.4053e-4
    S = TidalSolver(md.anelastic(w), g, m=2, K=5, N=30, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=w, omega_rot=OMEGA_EARTH, dyn_fluid=True, consistent_reference=True)
    S.solve()
    res[name]['M2 anelastic'] = iers_fit(S, g)
    res[name]['M2 anelastic']['k0'] = S.results()['kk'] * (1 + g.flattening_surface / 3) ** -5
    res[name]['_fcn'] = fcn_period(md, g)
    res[name]['_f'] = (1 / g.flattening_surface, md.M, md.a)
    out(f'{name}: 1/f = {1 / g.flattening_surface:.2f}, M = {md.M:.5e} kg, '
        f'FCN period = {", ".join(f"{p:.1f}" for p in res[name]["_fcn"])} sidereal days')

out()
out(f"{'tide':14s}{'model':>8s}" + ''.join(f'{k:>10s}' for k in KEYS))
for t in [x[0] for x in TIDES] + ['M2 anelastic']:
    for name in models:
        f = res[name][t]
        out(f"{t if name == models[0] else '':14s}{name:>8s}" + ''.join(f'{f[k].real:+10.5f}' for k in KEYS))
    if len(models) == 2:
        a, b = res[models[0]][t], res[models[1]][t]
        out(f"{'':14s}{'diff':>8s}" + ''.join(f'{(b[k] - a[k]).real:+10.5f}' for k in KEYS))
open('results/compare_models.txt' if not sys.argv[1:] else 'results/compare_models_' + '_'.join(models) + '.txt', 'w').write('\n'.join(lines) + '\n')
