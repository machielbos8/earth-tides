# -*- coding: utf-8 -*-
#
# Diurnal band with the viscous core: fit the FCN resonance at each Ekman thickness and extrapolate
# the resonance parameters, not the Love numbers, to delta -> 0.
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
"""Near the FCN the Ekman damping of a resolvable viscous core (delta >= 0.5 km) swamps the resonance, so
the Love numbers are not linear in delta there. The resonance PARAMETERS should be: at each delta fit
    L(s) = a + b x + c x^2 + d x^3 + A / (s - s_f) + C / (s - s_i) + B / x,   x = s - 1
(s in cpsd; s_f FCN and s_i FICN, complex, shared by all six parameters; B: tilt-over mode at 1 cpsd) to
a sweep of the diurnal band, then extrapolate the poles and coefficients linearly to delta -> 0 from 1 and
0.5 km (2 km: check)
and evaluate at the table frequencies (K1: mean of 1 -+ 1e-5). Elastic hydrostatic PREM, viscous core
(EarthModel.viscous_core, 40 km boundary layers), K = 6, N = 30, mantle frame. v2 (10-17): nested boundary layers (5 delta inside 40 km; the 40 km
layer alone is biased near the FCN for delta < 0.75 km), five deltas, quadratic extrapolation from
0.35, 0.5, 0.75 km, check from 0.5, 0.75, 1 km.
Run from the project root with .venv/bin/python -> results/fcn_resonance_extrapolation_v2.{txt,npz}"""
import sys
import warnings
from multiprocessing import Pool
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import least_squares

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k0')
DELTAS = (2e3, 1e3, 7.5e2, 5e2, 3.5e2)
SWEEP = (0.90, 0.92700, 0.96, 0.98, 0.99454, 0.9965, 0.9972, 0.9978, 0.9984, 0.9990, 0.9995, 1.0010, 1.0015, 1.0020, 1.00245, 1.00273,
         1.0035, 1.00546, 1.02, 1.07300)
TABLE = {'O1': 0.92700, 'P1': 0.99454, 'K1': 1.0, 'psi1': 1.00273, 'phi1': 1.00546, 'OO1': 1.07300}
# inviscid (null pair removed, K 7, N 30), for comparison
INVISCID = {'O1': (0.59626, -0.00050, 0.08305, 0.00048, 0.00009, 0.29384),
            'P1': (0.57523, -0.00059, 0.08372, 0.00045, 0.00011, 0.28326),
            'K1': (0.51490, -0.00082, 0.08564, 0.00034, 0.00017, 0.25302),
            'psi1': (0.93517, 0.00084, 0.07238, 0.00112, -0.00024, 0.46402),
            'phi1': (0.65457, -0.00027, 0.08123, 0.00060, 0.00004, 0.32312),
            'OO1': (0.60160, -0.00047, 0.08291, 0.00051, 0.00009, 0.29652)}


def solve(job):
    s, delta = job
    w = s * OMEGA_EARTH
    md = EarthModel('data/PREM.dat').viscous_core(w, delta=delta)
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=1, K=6, N=30, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle')
    r = S.solve()
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5
    print('s %.5f delta %.1f km  h0 %+.6f %+.2e' % (s, delta / 1e3, f['h0'].real, f['h0'].imag), flush=True)
    return job, np.array([complex(f[k]) for k in KEYS])


def basis(s, sf):
    """sf = (FCN pole, FICN pole)."""
    x = s - 1.0
    return np.stack([x ** k for k in range(4)] + [1.0 / (s - p) for p in sf] + [1.0 / x],
                    axis=1).astype(complex)


def fit(s, L, delta):
    """s (n,), L (n, 6) complex -> (s_f, s_i), coefficients (7, 6), max absolute residual (6,)."""
    def coefs(sf):
        B = basis(s, sf)
        return B, np.linalg.lstsq(B, L, rcond=None)[0]

    def res(p):
        B, c = coefs((p[0] + 1j * p[1], p[2] + 1j * p[3]))
        r = B @ c - L
        return np.concatenate([r.real.ravel(), r.imag.ravel()])
    best = None
    for a in (0.9975, 0.9979, 0.9983):
        for b in (1e-4, 1e-3, -1e-3):
            sol = least_squares(res, [1.00225 + 1e-4 * delta / 1e3, 8e-4 * delta / 1e3, a, b],
                                x_scale=1e-4, xtol=1e-15, ftol=1e-15)
            if best is None or sol.cost < best.cost:
                best = sol
    p = best.x
    sf = (p[0] + 1j * p[1], p[2] + 1j * p[3])
    B, c = coefs(sf)
    return sf, c, np.abs(B @ c - L).max(axis=0)


def evaluate(sf, c, s):
    fs = (s - 1e-5, s + 1e-5) if s == 1.0 else (s,)
    return np.mean([basis(np.array([f]), sf)[0] @ c for f in fs], axis=0)


if __name__ == '__main__':
    R = {}
    try:                                    # reuse earlier solves (same set-up)
        old = np.load('results/fcn_resonance_extrapolation_v2.npz')
        for j, d in enumerate(old['deltas']):
            for i, x in enumerate(old['sweep']):
                R[(float(x), float(d))] = old['L'][j, i]
    except FileNotFoundError:
        pass
    jobs = [(s, d) for d in DELTAS for s in SWEEP if (s, d) not in R]
    with Pool(2, maxtasksperchild=1) as p:
        R.update(dict(p.map(solve, jobs, chunksize=1)))
    s = np.array(SWEEP)
    lines, P = [], {}
    for d in DELTAS:
        L = np.array([R[(x, d)] for x in SWEEP])
        sf, c, mx = fit(s, L, d)
        P[d] = (np.array(sf), c)
        lines.append('delta %.1f km: FCN %.7f %+.2ei (%.1f sd)  FICN %.7f %+.2ei (%.0f sd)  max |resid| %s' % (
            d / 1e3, sf[0].real, sf[0].imag, 1 / (sf[0].real - 1), sf[1].real, sf[1].imag,
            1 / (sf[1].real - 1), np.array2string(mx, precision=1)))
    def lagrange0(ds):
        """Quadratic in delta through the three deltas, evaluated at delta = 0."""
        w = [np.prod([-dj / (di - dj) for dj in ds if dj != di]) for di in ds]
        return (sum(wi * P[d][0] for wi, d in zip(w, ds)), sum(wi * P[d][1] for wi, d in zip(w, ds)))
    sf0, c0 = lagrange0((3.5e2, 5e2, 7.5e2))
    sfc, cc = lagrange0((5e2, 7.5e2, 1e3))
    lines.append('delta -> 0: FCN %.7f %+.2ei (%.1f sd), FICN %.7f %+.2ei (%.0f sd); check (0.5, 0.75, 1 km): FCN %.1f sd, FICN %.0f sd; inviscid FCN fit 457.2 sd' % (
        sf0[0].real, sf0[0].imag, 1 / (sf0[0].real - 1), sf0[1].real, sf0[1].imag, 1 / (sf0[1].real - 1),
        1 / (sfc[0].real - 1), 1 / (sfc[1].real - 1)))
    lines.append('%-6s %-9s ' % ('', '') + ''.join('%11s' % k for k in KEYS))
    for name, x in TABLE.items():
        v, vc = evaluate(sf0, c0, x), evaluate(sfc, cc, x)
        lines.append('%-6s %-9s ' % (name, 'delta->0') + ''.join('%+11.5f' % z.real for z in v))
        lines.append('%-6s %-9s ' % ('', 'Im') + ''.join('%+11.1e' % z.imag for z in v))
        lines.append('%-6s %-9s ' % ('', 'chk') + ''.join('%11.1e' % abs(a - b) for a, b in zip(v, vc)))
        lines.append('%-6s %-9s ' % ('', 'inviscid') + ''.join('%+11.5f' % z for z in INVISCID[name]))
        for d in DELTAS:
            lines.append('%-6s %-9s ' % ('', 'fit %.1fkm' % (d / 1e3)) + ''.join('%+11.5f' % z.real for z in evaluate(*P[d], x)))
    out = '\n'.join(lines)
    print(out)
    open('results/fcn_resonance_extrapolation_v2.txt', 'w').write(out + '\n')
    np.savez('results/fcn_resonance_extrapolation_v2.npz', sweep=s, deltas=np.array(DELTAS),
             L=np.array([[R[(x, d)] for x in SWEEP] for d in DELTAS]))
