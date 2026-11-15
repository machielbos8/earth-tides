# -*- coding: utf-8 -*-
#
# Smith (1974) equations against the exact series and against the solver, for
# m = 1 and m = 2.
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
"""Smith (1974) equations against the exact series and against the solver, for m = 1 and m = 2.

The Love numbers of Wahr (1981) and Dehant et al. (1999) come from integrating Smith's scalar
equations. src/earthtides/smith1974.py transcribes them for a non-gravitating, rotating, slightly
elliptical solid body. Part 1: homogeneous body against the exact series of exact_rotating.py (tests the
Coriolis couplings 5.34-5.36 and the boundary conditions on the flattened surface 5.44-5.46). Part 2:
a body with smooth radial profiles (data/SMOOTH.dat) against the solver (tests the interior flattening
terms lambda_2, mu_2, beta_2, rho_2 of 5.22-5.30). Part 3: a self-gravitating, hydrostatic, homogeneous
spheroid against the solver (tests self-gravitation 5.25-5.26, the initial gravity in 5.31-5.32, the initial
potential eta_3 5.37-5.39 and the gravity boundary conditions 5.42-5.43); the hydrostatic ellipticity
5q/4 is changed through the rate of the reference rotation (1 and 0.3 times the Earth's), with the Coriolis
force kept at the Earth's rate. Part 4: a self-gravitating body of two solid layers with a density jump
and density gradients (data/TWOLAYER.dat) on its first-order Clairaut ellipticity (tests the welded-interface
conditions 5.41-5.45 and the rho_2 terms of 5.26 and 5.28-5.30 with gravity; the radial rho_2 term of 5.26
needs the opposite sign to the printed one). Same truncation in both: spheroidal 2, 4; toroidal 1, 3
(m = 1) or 3 (m = 2). Compared: the terms of Dehant et al. (1999, eq. 20) after removing the best-fitting
rigid rotation (R(theta) pattern). Smith's theory is first order in the flattening, so the differences
should fall in proportion to the flattening.
Usage: python scripts/check_smith1974.py  ->  results/smith1974.txt
"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel, G
from earthtides.geometry import OMEGA_EARTH, ExactSpheroidGeometry, ClairautGeometry
from earthtides.solver import TidalSolver
from earthtides.smith1974 import Smith, SmithGrav, SmithLayered
from earthtides import exact_rotating as er
import check_rotating_spheroid as crs

th = crs.th
lines = []
TIDES = (('O1', 1, 0.92700), ('M2', 2, 1.4053e-4/OMEGA_EARTH))


def out(t=''):
    print(t, flush=True)
    lines.append(t)


class FirstOrder:
    """Constant ellipticity of the equidensity surfaces, first-order map r = s [1 - 2/3 eps P2]."""

    def __init__(self, eps):
        self.eps_ = eps

    def eps(self, i, s):
        s = np.asarray(s, float)
        return self.eps_ + 0*s, 0*s, 0*s


def smith_terms(md, fl, m, fr, cor):
    a = md.a
    mu_ref, rho_ref = float(md.mu[0](a)), float(md.rho[0](a))
    lam = lambda r: float(md.lam[0](r*a))/mu_ref
    mu = lambda r: float(md.mu[0](r*a))/mu_ref
    rho = lambda r: float(md.rho[0](r*a))/rho_ref
    sig = fr*OMEGA_EARTH
    S = Smith(m, lam, mu, rho, rho_ref*sig**2*a**2/mu_ref,
              2*rho_ref*OMEGA_EARTH*sig*a**2/mu_ref if cor else 0.0, fl,
              sph=(2, 4), tor=(1, 3) if m == 1 else (3,))
    S.solve()
    ur, ut, ul, R = S.surface(th)
    ut, ul = crs.unrigid(m, R, ut, ul)
    return crs.fit(m, ur, ut, ul)


def solver_terms(md, fl, m, fr, cor):
    S = TidalSolver(md, FirstOrder(fl), m=m, N=16, omega_f=0.0, gravity=False,
                    coriolis=OMEGA_EARTH if cor else 0.0, omega_t=fr*OMEGA_EARTH,
                    degrees=([2, 4], [1, 3] if m == 1 else [3]))
    S.solve()
    sr, st, sp_ = S.surface_field(th, remove_rigid=False)
    R = 1 - 2/3*fl*0.5*(3*np.cos(th)**2 - 1)
    st, sl = crs.unrigid(m, R, st, -sp_)
    return crs.fit(m, sr, st, sl)


def grav_terms(md, spin, m, fr, cor):
    """Self-gravitating homogeneous spheroid: Smith (1974) and the solver, hydrostatic eps = 5q/4."""
    a = md.a
    mu_ref, rho_ref = float(md.mu[0](a)), float(md.rho[0](a))
    eps = 1.25*(spin*OMEGA_EARTH)**2/(G*4/3*np.pi*rho_ref)
    sig = fr*OMEGA_EARTH
    S = SmithGrav(m, lambda r: float(md.lam[0](r*a))/mu_ref, lambda r: float(md.mu[0](r*a))/mu_ref,
                  lambda r: float(md.rho[0](r*a))/rho_ref, rho_ref*sig**2*a**2/mu_ref,
                  2*rho_ref*OMEGA_EARTH*sig*a**2/mu_ref if cor else 0.0, lambda r: eps,
                  G*rho_ref**2*a**2/mu_ref, rho_ref*(spin*OMEGA_EARTH)**2*a**2/mu_ref,
                  lambda r: 4/3*np.pi*G*rho_ref*r*a*a*rho_ref/mu_ref,
                  sph=(2, 4), tor=(1, 3) if m == 1 else (3,))
    S.solve()
    ur, ut, ul, R = S.surface(th)
    ut, ul = crs.unrigid(m, R, ut, ul)
    T = TidalSolver(md, FirstOrder(eps), m=m, N=16, omega_f=0.0, coriolis=OMEGA_EARTH if cor else 0.0,
                    omega_t=sig, omega_rot=spin*OMEGA_EARTH, consistent_reference=True,
                    degrees=([2, 4], [1, 3] if m == 1 else [3]))
    T.solve()
    sr, st, sp_ = T.surface_field(th, remove_rigid=False)
    st, sl = crs.unrigid(m, 1 - 2/3*eps*0.5*(3*np.cos(th)**2 - 1), st, -sp_)
    return crs.fit(m, ur, ut, ul), crs.fit(m, sr, st, sl), eps


class Scaled:
    """A ClairautGeometry with its ellipticity scaled by k (the hydrostatic eps goes as Omega^2)."""

    def __init__(self, geom, k):
        self.g, self.k = geom, k

    def eps(self, i, s):
        return tuple(self.k*x for x in self.g.eps(i, s))


def layered_terms(md, clairaut, spin, m, fr, cor):
    """Self-gravitating two-layer body: Smith (1974) with welded-interface conditions and the solver,
    both on the same first-order Clairaut ellipticity."""
    a = md.a
    mu_ref, rho_ref = float(md.mu[-1](a)), float(md.rho[-1](a))
    geom = Scaled(clairaut, spin**2)
    layers = [(md.r1[i]/a, lambda r, i=i: float(md.lam[i](r*a))/mu_ref,
               lambda r, i=i: float(md.mu[i](r*a))/mu_ref, lambda r, i=i: float(md.rho[i](r*a))/rho_ref)
              for i in range(md.n_layers)]
    def eps3(i, r):
        e, de, dde = geom.eps(i, np.asarray(r*a))
        return float(e), float(de)*a, float(dde)*a*a
    sig = fr*OMEGA_EARTH
    S = SmithLayered(m, layers, rho_ref*sig**2*a**2/mu_ref,
                     2*rho_ref*OMEGA_EARTH*sig*a**2/mu_ref if cor else 0.0, eps3,
                     G*rho_ref**2*a**2/mu_ref, rho_ref*(spin*OMEGA_EARTH)**2*a**2/mu_ref,
                     lambda i, r: float(md.gravity(i, r*a))*a*rho_ref/mu_ref,
                     sph=(2, 4), tor=(1, 3) if m == 1 else (3,))
    S.solve()
    ur, ut, ul, R = S.surface(th)
    ut, ul = crs.unrigid(m, R, ut, ul)
    T = TidalSolver(md, geom, m=m, N=16, omega_f=0.0, coriolis=OMEGA_EARTH if cor else 0.0,
                    omega_t=sig, omega_rot=spin*OMEGA_EARTH, consistent_reference=True,
                    degrees=([2, 4], [1, 3] if m == 1 else [3]))
    T.solve()
    sr, st, sp_ = T.surface_field(th, remove_rigid=False)
    eps_a = float(geom.eps(md.n_layers - 1, np.asarray(a))[0])
    st, sl = crs.unrigid(m, 1 - 2/3*eps_a*0.5*(3*np.cos(th)**2 - 1), st, -sp_)
    return crs.fit(m, ur, ut, ul), crs.fit(m, sr, st, sl), eps_a


def row(name, P, Q):
    q = lambda Z, k, r: (Z[k]/Z[r]).real
    w = (q(P, 'wp', 'l0'), q(Q, 'wp', 'l0'))
    rel = abs(w[0]/w[1] - 1) if abs(w[1]) > 1e-9 else 0.0
    out(f'  {name:26s} w+/l0 {w[0]:+.6e} vs {w[1]:+.6e} ({rel:.1e})   l+/l0 {q(P, "lp", "l0"):+.3e} vs '
        f'{q(Q, "lp", "l0"):+.3e}   h+/h0 {q(P, "hp", "h0"):+.3e} vs {q(Q, "hp", "h0"):+.3e}   '
        f'l0/h0 {q(P, "l0", "h0"):.6f} vs {q(Q, "l0", "h0"):.6f}')
    return rel


def main():
    md = EarthModel('data/HOMOG.dat')
    a = md.a
    rho, lam, mu = (float(f(a/2)) for f in (md.rho[0], md.lam[0], md.mu[0]))
    out('Part 1: homogeneous body, Smith (1974) vs exact series (first value Smith, second exact)')
    for fl in (0.0, 1/298.26, 1/3000):
        out(f' flattening {fl:.6f}')
        for tide, m, fr in TIDES:
            for cor in (False, True):
                P = smith_terms(md, fl, m, fr, cor)
                G = ExactSpheroidGeometry(fl) if fl > 0 else None
                A, C = (G.A, G.C) if G else (1.0, 1.0)
                sig = fr*OMEGA_EARTH
                u, res = er.solve(m, lam/mu, rho*sig**2*a**2/mu, 2*rho*OMEGA_EARTH*sig*a**2/mu if cor else 0.0, A, C)
                xr, xt, xl, XR = er.surface(u, A, C, th)
                xt, xx = crs.unrigid(m, XR, xt, xl/1j)
                row(f'{tide} {"Coriolis" if cor else "no Coriolis"}', P, crs.fit(m, xr, xt, xx))
    md = EarthModel('data/SMOOTH.dat')
    out()
    out('Part 2: smooth radial profiles (data/SMOOTH.dat), Smith (1974) vs solver (first value Smith, second solver)')
    for fl in (1/298.26, 1/3000):
        out(f' flattening {fl:.6f}')
        for tide, m, fr in TIDES:
            for cor in (False, True):
                row(f'{tide} {"Coriolis" if cor else "no Coriolis"}',
                    smith_terms(md, fl, m, fr, cor), solver_terms(md, fl, m, fr, cor))
    md = EarthModel('data/HOMOG.dat')
    out()
    out('Part 3: self-gravitating hydrostatic homogeneous spheroid, Smith (1974) vs solver (first value Smith)')
    for spin in (1.0, 0.3):
        for tide, m, fr in TIDES:
            for cor in (False, True):
                P, Q, eps = grav_terms(md, spin, m, fr, cor)
                if tide == 'O1' and not cor:
                    out(f' reference rotation x {spin}: hydrostatic eps = {eps:.6f}')
                row(f'{tide} {"Coriolis" if cor else "no Coriolis"}', P, Q)
    md = EarthModel('data/TWOLAYER.dat')
    clairaut = ClairautGeometry(md)
    out()
    out('Part 4: self-gravitating two-layer body (data/TWOLAYER.dat: density jump 9.5 -> 5.0 at r = 3371 km, '
        'density gradient in both layers), Smith (1974) vs solver (first value Smith)')
    for spin in (1.0, 0.3):
        for tide, m, fr in TIDES:
            for cor in (False, True):
                P, Q, eps = layered_terms(md, clairaut, spin, m, fr, cor)
                if tide == 'O1' and not cor:
                    out(f' reference rotation x {spin}: Clairaut eps(a) = {eps:.6f}')
                row(f'{tide} {"Coriolis" if cor else "no Coriolis"}', P, Q)
    open('results/smith1974.txt', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
