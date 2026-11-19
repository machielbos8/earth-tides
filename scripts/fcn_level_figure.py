# -*- coding: utf-8 -*-
#
# Free core nutation of hydrostatic PREM on the first-order figure, exact ellipses and true level surfaces.
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
# 14/11/2026 Machiel Bos
#===============================================================================
"""How much do the terms of second order in the flattening of the figure move the free core nutation?
Elastic hydrostatic PREM, slipping core with the slope term (TidalSolver default), FCN from the fit of
nonhydro_diurnal.fcn (k at 1.0012, 1.0016, 1.0030, 1.0036 cpsd) on three figures: the first-order (Clairaut)
map of the paper, exact ellipses of the same flattening (exact_levels_share_test.ExactLevelsGeometry) and
coordinate surfaces that are level surfaces of the reference potential (level_geometry.py). Also the polar
flattening of the surface and of the core-mantle boundary of each figure.
Usage: python scripts/fcn_level_figure.py [--K 4,6] [--N 24]  ->  results/fcn_level_figure.txt"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

_argv, sys.argv = sys.argv, sys.argv[:1]
import exact_levels_share_test as els
sys.argv = _argv
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH, mapping, mapping_from_R
from earthtides.solver import TidalSolver
import level_geometry as lg
from provenance import Run


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def fcn(md, geo, K, N, sig=(1.0012, 1.0016, 1.0030, 1.0036)):
    x = np.array(sig)
    k = []
    for s in x:
        T = TidalSolver(md, geo, m=1, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=s * OMEGA_EARTH,
                        omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                        frame='mantle')
        T.solve()
        k.append(T.results()['kk'].real)
    k = np.array(k)
    return np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0]


def flattening(md, g, s):
    """(equatorial - polar radius) / equatorial radius of the coordinate surface s."""
    i = int(np.flatnonzero(np.isclose(md.r1, s))[0])
    th = np.array([1e-9, np.pi / 2])
    s2 = np.array([[s]])
    R = (g.map_derivs(i, s2, th)[0] if hasattr(g, 'map_derivs') else mapping(s2, th, *g.eps(i, s2))['R'])[0]
    return (R[1] - R[0]) / R[1]


def moments(md, geo, n_gl=48, n_th=64):
    """A (Earth), A_f (fluid core), A_s (inner core) and e_f = (C_f - A_f)/A_f on any figure, integrated over the
    mapped layers (R, R_s of the figure): C - A = int rho R^2 (sin^2/2 - cos^2) dV, A = int rho R^2 (sin^2/2 +
    cos^2) dV, dV = R^2 R_s ds dmu dlambda. (2026-11-19: was computed inline for the paper, now under provenance.)"""
    from earthtides.nonhydro import _bg_R
    x, wg = np.polynomial.legendre.leggauss(n_gl)
    mu, wq = np.polynomial.legendre.leggauss(n_th)
    th = np.arccos(mu)
    c2, sn2 = mu**2, 1 - mu**2
    A = Af = As = CAf = 0.0
    fl = [i for i in range(md.n_layers) if abs(float(md.mu[i](0.5 * (md.r0[i] + md.r1[i])))) < 1.0]
    for i in range(md.n_layers):
        r0, r1 = max(md.r0[i], 1.0), md.r1[i]
        s = r0 + (x + 1) * (r1 - r0) / 2
        R, R_s = _bg_R(geo, i, s, th)
        dV = R**2 * R_s * (wg * (r1 - r0) / 2)[:, None] * wq[None, :] * 2 * np.pi
        rho = md.rho[i](s)[:, None]
        Ai = np.sum(rho * R**2 * (0.5 * sn2 + c2) * dV)
        A += Ai
        if i in fl:
            Af += Ai
            CAf += np.sum(rho * R**2 * (0.5 * sn2 - c2) * dV)
        elif i < min(fl):
            As += Ai
    return A, Af, As, CAf / Af


def main():
    Ks, N = [int(x) for x in _opt('--K', '4,6').split(',')], int(_opt('--N', 24))
    with Run('FCN of hydrostatic PREM on the first-order figure, exact ellipses and true level surfaces',
             params=dict(K=Ks, N=N), inputs=['data/PREM.dat']) as run:
        md = EarthModel('data/PREM.dat')
        lines = ['elastic hydrostatic PREM, slipping core with the slope term; FCN in sidereal days',
                 f'{"figure":18s} {"1/f surface":>12s} {"1/f CMB":>10s} ' + ' '.join(f'{"K " + str(K):>9s}' for K in Ks)]
        glev, hist = lg.build(md, OMEGA_EARTH, log=lambda t: None)
        lines.insert(1, f'level surfaces: {len(hist)} iterations, max height left {hist[-1][1]:.1e} m')
        hough = []
        for name, g in (('first-order map', ClairautGeometry(md)), ('exact ellipses', els.ExactLevelsGeometry(md)),
                        ('level surfaces', glev)):
            P = [1 / (fcn(md, g, K, N) - 1) for K in Ks]
            lines.append(f'{name:18s} {1 / flattening(md, g, md.r1[-1]):12.3f} {1 / flattening(md, g, 3480e3):10.3f} '
                         + ' '.join(f'{p:9.2f}' for p in P))
            print(lines[-1], flush=True)
            if name != 'exact ellipses':
                A, Af, As, ef = moments(md, g)
                Am = A - Af - As
                beta = ef - Am / (A * P[-1])          # eq. (hough): sigma - 1 = (A / A_m)(e_f - beta), K largest
                hough.append(f'{name:18s} e_f 1/{1 / ef:.2f}  A_m/A {Am / A:.4f}  rigid mantle (beta = 0) '
                             f'{Am / (A * ef):.1f} sd  beta {beta:.2e} (beta/e_f {beta / ef:.2f})')
        lines += ['', "eq. (hough), Hough's formula with the moments of the figure; beta from the FCN of the largest K:"]
        lines += hough
        # mantle and inner core stiffened 1e4 times (the 'rigid mantle' row of Table 1 of paper 1), K 4
        stiff = EarthModel('data/PREM.dat')
        for i in range(stiff.n_layers):
            if float(stiff.mu[i](0.5 * (stiff.r0[i] + stiff.r1[i]))) != 0.0:
                stiff.mu[i], stiff.lam[i] = stiff.mu[i] * 1e4, stiff.lam[i] * 1e4
        from earthtides.geometry import figure
        lines += ['', 'mantle and inner core stiffened 1e4 times, K 4:']
        for kind in ('first-order', 'level'):
            lines.append(f'{kind:18s} {1 / (fcn(stiff, figure(stiff, kind), 4, N) - 1):9.2f}')
            print(lines[-1], flush=True)
        run.write_text('results/fcn_level_figure.txt', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
