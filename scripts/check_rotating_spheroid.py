# -*- coding: utf-8 -*-
#
# Rotating homogeneous spheroid: solver vs polynomial series solution (w+, l+,
# h+, l0).
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
# 16/10/2026 Machiel Bos
#===============================================================================
"""Rotating homogeneous spheroid: solver vs polynomial series solution (w+, l+, h+, l0).

The degree-3 toroidal term w+ of Dehant et al. (1999, eq. 20) has two causes, the flattening and the
Coriolis force. Both are tested here, for m = 1 (O1, OO1) and m = 2 (M2), on a sphere and on exact
spheroids of flattening 1/298 and 1/30, against src/earthtides/exact_rotating.py, which shares no
code with the solver. Solver: one homogeneous layer, gravity=False, the exact spheroid map; the rigid
rotation that best fits the surface (R(theta) pattern, Wahr frame) is removed from both solutions.
Usage: python scripts/check_rotating_spheroid.py  ->  results/rotating_spheroid.txt
"""
import sys
import warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
from scipy.special import lpmv, factorial

from earthtides.model import EarthModel
from earthtides.geometry import OMEGA_EARTH, ExactSpheroidGeometry
from earthtides.solver import TidalSolver
from earthtides import exact_rotating as er

th = np.linspace(0.05, np.pi - 0.05, 181)
c, s = np.cos(th), np.sin(th)
lines = []


def out(t=''):
    print(t, flush=True)
    lines.append(t)


def Y(l, m):
    return np.sqrt((2*l + 1)/(4*np.pi)*factorial(l - m)/factorial(l + m))*lpmv(m, l, c)


def dY(l, m, d=1e-6):
    N = np.sqrt((2*l + 1)/(4*np.pi)*factorial(l - m)/factorial(l + m))
    return N*(lpmv(m, l, np.cos(th + d)) - lpmv(m, l, np.cos(th - d)))/(2*d)


def fit(m, ur, ut, ul):
    """Terms of Dehant et al. (1999) eq. (20); ul is the coefficient of i e_lambda."""
    pr = np.linalg.lstsq(np.column_stack([Y(2, m), Y(4, m)]).astype(complex), ur, rcond=None)[0]
    ct = [dY(2, m), dY(4, m), m/s*Y(3, m)] + ([m/s*Y(1, m)] if m == 1 else [])
    cl = [m/s*Y(2, m), m/s*Y(4, m), dY(3, m)] + ([dY(1, m)] if m == 1 else [])
    pt = np.linalg.lstsq(np.vstack([np.column_stack(ct), np.column_stack(cl)]).astype(complex),
                         np.concatenate([ut, ul]), rcond=None)[0]
    return dict(h0=pr[0], hp=pr[1], l0=pt[0], lp=pt[1], wp=pt[2])


def unrigid(m, R, ut, ul):
    """Remove the best-fitting rigid rotation of the surface (rotation x R(theta), Wahr 1981)."""
    if m != 1:
        return ut, ul
    bt, bl = R, R*np.cos(th)
    w = np.sin(th)*np.gradient(th)
    xi = np.sum(w*(ut*bt + ul*bl))/np.sum(w*(bt*bt + bl*bl))
    return ut - xi*bt, ul - xi*bl


def compare(md, fl, tide, m, fr, coriolis, K=10):
    a = md.a
    rho, lam, mu = (float(f(a/2)) for f in (md.rho[0], md.lam[0], md.mu[0]))
    G = ExactSpheroidGeometry(fl) if fl > 0 else None
    A, C = (G.A, G.C) if G else (1.0, 1.0)
    sig = fr*OMEGA_EARTH
    kap = 2*rho*OMEGA_EARTH*sig*a**2/mu if coriolis else 0.0
    u, res = er.solve(m, lam/mu, rho*sig**2*a**2/mu, kap, A, C)
    ur, ut, ulam, R = er.surface(u, A, C, th)
    ut, ul = unrigid(m, R, ut, ulam/1j)
    P = fit(m, ur, ut, ul)
    S = TidalSolver(md, G, m=m, K=K if G else 2, N=12, omega_f=0.0, gravity=False,
                    coriolis=OMEGA_EARTH if coriolis else 0.0, omega_t=sig)
    S.solve()
    sr, st, sp_ = S.surface_field(th, remove_rigid=False)
    Rs = S.geom.map_derivs(0, a, th)[0]/a if G else np.ones_like(th)
    st, sl = unrigid(m, Rs, st, -sp_)            # solver: u_lambda x sin(m lambda)
    Q = fit(m, sr, st, sl)
    return P, Q, res


def main():
    md = EarthModel('data/HOMOG.dat')
    out('Homogeneous non-gravitating rotating spheroid (lambda = mu = 1e11 Pa, rho = 5500 kg/m3):')
    out('ratios of the terms of Dehant et al. (1999, eq. 20), solver / series solution')
    for fl in (0.0, 1/298.26, 1/30):
        out(f'\nflattening {fl:.5f}')
        for tide, m, fr in (('O1', 1, 0.92700), ('OO1', 1, 1.07300), ('M2', 2, 1.4053e-4/OMEGA_EARTH)):
            for cor in (False, True):
                P, Q, res = compare(md, fl, tide, m, fr, cor)
                wq, wp = (Q['wp']/Q['l0']).real, (P['wp']/P['l0']).real
                dev = max(abs(Q[k]/Q[r]/(P[k]/P[r]) - 1) for k, r in (('lp', 'l0'), ('hp', 'h0'), ('l0', 'h0'))
                          if abs(P[k]/P[r]) > 1e-6)
                out(f'  {tide:3s} {"inertia+Coriolis" if cor else "inertia only    "}  w+/l0 solver {wq:+.6e}'
                    f'  series {wp:+.6e}  rel. diff {abs(wq/wp - 1) if abs(wp) > 1e-9 else 0:.0e}'
                    f'   l+, h+, l0: {dev:.0e}   (series residual {res:.0e})')
    open('results/rotating_spheroid.txt', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
