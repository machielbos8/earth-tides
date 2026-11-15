# -*- coding: utf-8 -*-
#
# Is the (K, N) scatter of fluid-core runs caused by PREM's stratification?
# PREM's outer core has N^2 / Omega^2 from +0.9 to -0.3 and -5.6 just below
# the CMB (unstable). Replace the core density by a profile with constant N^2
# (0: neutral, Adams-Williamson; > 0: stable), keeping PREM's kappa and the
# core mass, and repeat the convergence scan with the production solver (no
# deflation). Also counts the near-null singular vectors (sigma < 1e-10, > 99
# % in the fluid core). -> results/convergence_stratification.txt
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
"""Is the (K, N) scatter of fluid-core runs caused by PREM's stratification? PREM's outer core has
N^2 / Omega^2 from +0.9 to -0.3 and -5.6 just below the CMB (unstable). Replace the core density by
a profile with constant N^2 (0: neutral, Adams-Williamson; > 0: stable), keeping PREM's kappa and the
core mass, and repeat the convergence scan with the production solver (no deflation). Also counts
the near-null singular vectors (sigma < 1e-10, > 99 % in the fluid core).
-> results/convergence_stratification.txt"""
import copy
import sys
import warnings
from multiprocessing import Pool
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
from numpy.polynomial import Chebyshev
from scipy.integrate import solve_ivp

from earthtides.solver import TidalSolver
from earthtides.model import EarthModel, G
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.iers import iers_fit

IC = 1          # outer-core layer


def stratified(N2_over_Om2, deg=12, layers=None):
    """PREM with the outer-core density replaced by a profile of constant N^2 / Omega^2 (0: neutral), PREM's
    kappa and the core mass. layers: [(thickness_m, N2_over_Om2), ...] of stable layers on top of that, the
    first next to the ICB, the second below the CMB (see layered); the core is split at their edges, so each
    piece keeps a smooth density polynomial."""
    md = copy.deepcopy(EarthModel('data/PREM.dat'))
    r0, r1 = md.r0[IC], md.r1[IC]
    kap = md.lam[IC]                                  # mu = 0
    Mtarget = md._mass_poly[IC](r1)
    (hb, nb), (ht, nt) = layers if layers else ((0.0, N2_over_Om2), (0.0, N2_over_Om2))
    cuts = [c for c in (r0 + hb, r1 - ht) if r0 < c < r1]
    N2 = lambda r: OMEGA_EARTH**2 * (nb if r < r0 + hb else nt if r > r1 - ht else N2_over_Om2)

    def profile(rho0):
        def f(r, y):
            rho, M = y
            g = G * M / r**2
            return [-rho**2 * g / kap(r) - rho * N2(r) / g, 4 * np.pi * rho * r * r]
        return solve_ivp(f, (r0, r1), [rho0, md.mass_below[IC]], rtol=1e-12, atol=1e-6, dense_output=True,
                         max_step=5e3 if cuts else np.inf)

    a, b = 9e3, 1.5e4
    for _ in range(60):                               # bisection on rho at the ICB for the mass
        c = 0.5 * (a + b)
        if profile(c).y[1, -1] - md.mass_below[IC] > Mtarget:
            b = c
        else:
            a = c
    sol = profile(0.5 * (a + b))
    rr = r0 + (r1 - r0) * (1 - np.cos(np.linspace(0, np.pi, 200))) / 2
    md.rho[IC] = Chebyshev.fit(rr, sol.sol(rr)[0], deg, domain=[r0, r1])
    md.lam[IC] = kap
    if cuts:
        md = md.split_layer(IC, cuts)
        for i in range(IC, IC + len(cuts) + 1):     # each piece fitted on its own interval
            a, b = md.r0[i], md.r1[i]
            rr = a + (b - a) * (1 - np.cos(np.linspace(0, np.pi, 200))) / 2
            md.rho[i] = Chebyshev.fit(rr, sol.sol(rr)[0], deg, domain=[a, b])
    # recompute the mass polynomials (EarthModel.__init__)
    m_below = 0.0
    for i in range(md.n_layers):
        r = Chebyshev.identity(domain=[md.r0[i], md.r1[i]])
        p = (4.0 * np.pi * md.rho[i] * r * r).integ(lbnd=md.r0[i])
        md.mass_below[i] = m_below
        md._mass_poly[i] = p
        m_below += p(md.r1[i])
    md.M = m_below
    md.g_surface = G * md.M / md.a**2
    return md


def layered(top_km=140.0, f_km=150.0, n2_top=1.0, n2_f=1.0):
    """A plausible core (P1 v2, figure of the stratified core): neutral (Adams-Williamson) with a stable layer
    of top_km below the CMB, N^2 / Omega^2 = n2_top (Buffett 2014: 140 km, N ~ Omega), and a stable F-layer of
    f_km above the ICB, n2_f; PREM's kappa and core mass. The core is three layers."""
    return stratified(0.0, layers=((f_km * 1e3, n2_f), (top_km * 1e3, n2_top)))


def n2_check(md):
    r = np.linspace(md.r0[IC] + 1e3, md.r1[IC] - 1e3, 7)
    rho, drho, kap = md.rho[IC](r), md.rho[IC].deriv()(r), md.lam[IC](r)
    g = G * (md.mass_below[IC] + md._mass_poly[IC](r)) / r**2
    return -g * (drho / rho + rho * g / kap) / OMEGA_EARTH**2


def run(args):
    case, m, sig, K, N = args
    md = EarthModel('data/PREM.dat') if case is None else stratified(case)
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=m, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve()
    f = iers_fit(S, g)
    return (str(case), m, sig, K, N) + tuple(f[k].real for k in ('h0', 'h2', 'l2', 'l1'))


if __name__ == '__main__':
    cases = (0.0, 0.5)
    for c in cases:
        md = stratified(c)
        print('N2/Om^2 = %.1f: check %s, mass change %.1e, rho ICB %.1f CMB %.1f (PREM %.1f %.1f)' % (
            c, np.array2string(n2_check(md), precision=3), md.M / EarthModel('data/PREM.dat').M - 1,
            md.rho[IC](md.r0[IC]), md.rho[IC](md.r1[IC]),
            EarthModel('data/PREM.dat').rho[IC](md.r0[IC]), EarthModel('data/PREM.dat').rho[IC](md.r1[IC])))
    res_ = [(3, 24), (4, 24), (4, 30), (5, 30), (6, 30), (7, 40)]
    waves = [(2, 1.9225), (2, 1.93227), (2, 1.9475), (1, 0.92954), (1, 0.905)]
    jobs = [(c, m, s, K, N) for c in cases for m, s in waves for K, N in res_]
    with Pool(8) as p:
        res = p.map(run, jobs, chunksize=1)
    with open('results/convergence_stratification.txt', 'w') as fo:
        fo.write('N2/Om2 m sigma K N h0 h2 l2 l1\n')
        for r in res:
            fo.write('%s %d %.5f %d %d %.10f %.10f %.10f %.10f\n' % r)
        for c in cases:
            for m, s in waves:
                rr = np.array([r[5:] for r in res if r[0] == str(c) and r[2] == s])
                line = 'N2/Om2 %s m=%d %.5f h0 %.5f h2 %.6f spread(1e-5) h0 %.4f h2 %.4f l2 %.4f l1 %.4f' % (
                    (c, m, s, rr[:, 0].mean(), rr[:, 1].mean()) + tuple(1e5 * (rr.max(0) - rr.min(0))))
                print(line)
                fo.write('# ' + line + '\n')
