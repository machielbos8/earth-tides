# -*- coding: utf-8 -*-
#
# Degree-2 Love numbers with a true fluid outer core, all bands, vs DDW1999
# and Mathews et al. (1997).
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
"""Degree-2 Love numbers with a true fluid outer core, all bands, vs DDW1999 and
Mathews et al. (1997).

PREM (fluid outer core), hydrostatic flattening, elastic; Coriolis and inertia in
all layers; slip at the fluid-solid interfaces; Poisson-consistent reference
potential; displacements in the mantle (Tisserand) frame with the tilt potential
removed from k (m = 1). Love numbers in the IERS/Mathews form; k converted to the
DDW normalisation (tidal potential at the equatorial radius).
Results -> results/diurnal_PREM.txt, results/figdata_fcn.npz
"""
import sys
import warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import least_squares

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

md = EarthModel('data/PREM.dat')
g = ClairautGeometry(md)
conv = (1 + g.flattening_surface / 3) ** -5
# DDW1999 Tables 11, 12 (elastic hydrostatic PREM): h0, h2, l0, l1, l2, k0
DDW = {'O1': (0.92700, 0.59553, -0.00048, 0.08298, 0.00070, 0.00008, 0.29335),
       'P1': (0.99454, 0.57411, -0.00053, 0.08367, 0.00070, 0.00009, 0.28249),
       'K1': (1.00000, 0.51400, -0.00065, 0.08558, 0.00060, 0.00011, 0.25205),
       'psi1': (1.00273, 0.93394, 0.00024, 0.07227, 0.00130, -0.00004, 0.46477),
       'phi1': (1.00546, 0.65422, -0.00036, 0.08113, 0.00084, 0.00006, 0.32308),
       'OO1': (1.07300, 0.60092, -0.00047, 0.08283, 0.00077, 0.00008, 0.29607)}
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def solve(m, fr, K=4, N=24):
    S = TidalSolver(md, g, m=m, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=fr * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2),
                    dyn_fluid=True, consistent_reference=True, frame='mantle')
    r = S.solve()
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * conv
    return f


def avg(a, b):
    return {k: 0.5 * (a[k] + b[k]) for k in a if k in b}


keys = ('h0', 'h2', 'l0', 'l1', 'l2', 'k0')
out('PREM, fluid core, Coriolis + inertia everywhere, mantle frame (DDW normalisation for k0)')
out(f"{'':6s}" + ''.join(f'{k:>10s}' for k in keys))
res = {}
for name, (fr, *ref) in DDW.items():
    f = avg(solve(1, fr - 1e-5), solve(1, fr + 1e-5)) if name == 'K1' else solve(1, fr)
    res[name] = f
    out(f'{name:6s}' + ''.join(f'{f[k].real:+10.5f}' for k in keys))
    out(f"{'  DDW':6s}" + ''.join(f'{v:+10.5f}' for v in ref))
fM2 = solve(2, 1.4053e-4 / OMEGA_EARTH, K=5, N=30)
out(f"{'M2':6s}" + ''.join(f'{fM2[k].real:+10.5f}' for k in keys))
out(f"{'  DDW':6s}" + ''.join(f'{v:+10.5f}' for v in (0.60175, -0.00047, 0.08359, 0.00100, 0.00008, 0.29751)))
# FCN eigenfrequency from the resonance in h0
frs = np.array([1.0012, 1.0015, 1.0018, 1.0020, 1.0024, 1.0027, 1.0032, 1.0040, 1.0055])
h = np.array([solve(1, x, K=3)['h0'].real for x in frs])
fit = least_squares(lambda p: p[0] + p[1] / (p[2] - frs) - h, [0.6, -1e-3, 1.00218]).x
out(f'FCN: {fit[2]:.6f} cpsd = 1 + 1/{1 / (fit[2] - 1):.1f}   (DDW Table 13: 1.0021832 = 1 + 1/458.0)')
# sweep for the figure
sweep = np.concatenate([np.linspace(0.985, 0.9995, 30), np.linspace(1.0005, 1.0019, 20),
                        np.linspace(1.00225, 1.012, 40)])
hs, ks, l1s = [], [], []
for x in sweep:
    f = solve(1, x, K=3)
    hs.append(f['h0'].real); ks.append(f['k0'].real); l1s.append(f['l1'].real)
np.savez('results/figdata_fcn.npz', sweep=sweep, h0=np.array(hs), k0=np.array(ks),
         l1=np.array(l1s), fcn=fit[2],
         ddw_f=np.array([v[0] for v in DDW.values()]), ddw_h0=np.array([v[1] for v in DDW.values()]),
         ddw_k0=np.array([v[6] for v in DDW.values()]))
# l' (frame-dependent; mantle frame) against DDW Table 11 (elastic hydrostatic)
DDW_LP = {'O1': 0.00020, 'P1': 0.00018, 'K1': 0.00015, 'psi1': 0.00034, 'phi1': 0.00021, 'OO1': 0.00018}
out("l' (mantle frame):  " + '  '.join(f"{n} {res[n]['lp'].real:+.5f} (DDW {v:+.5f})" for n, v in DDW_LP.items()))
open('results/diurnal_PREM.txt', 'w').write('\n'.join(lines) + '\n')
