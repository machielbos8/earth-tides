# -*- coding: utf-8 -*-
#
# Viscous limit against slipping core: does the core's stratification make them differ? (P1 v2 Sect. 5.3, review)
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
# 11/11/2026 Machiel Bos
#===============================================================================
"""For PREM without inner core the viscous core gives a resonant share of h(2) of +0.0015, converged in K (48, 72)
and in delta (250, 125 m; share_vs_delta.py), against +0.0040 for the inviscid slipping core
(fcn_share_inviscid_noic.py). On Hough's Earth (homogeneous core) the two agree, also with a compressible core
(wall_pressure_ingredients.py). PREM's core is stratified (N^2 / Omega^2 from +0.9 to -5.6). This script repeats
both computations, for the same Earth model, with the core made neutral (N^2 = 0: Adams-Williamson density with the
core's kappa and mass, integrated from the centre), next to PREM without inner core as the control. If the neutral
core brings the viscous share to the inviscid one, the stratification is the cause. (Result 11-11: it does not,
0.00146 / 0.00138 against 0.00395.) The 'uniform' variant replaces the core density by its mean (same core mass,
same kappa, same mantle): PREM's version of Hough's homogeneous core, where the viscous and slipping shares agree
(hough_share_test.py). If they agree here too, the density gradient of the core is what separates them.
  inviscid: core K 6 and 10, N 30; viscous: core K 48, N 30, band 12, delta 250 and 125 m
  Near the FCN the banded block solver stalls at residuals of 1-4e-10; run with EARTHTIDES_BAND_FALLBACK_MAX=1e9
  (SuperLU fallback; K 48: 4.8 GB per solve).
Usage: python scripts/strat_share_test.py NPROC [--variants PREM_noIC,neutral,uniform] [--deltas 250,125] [--K 48]
       ->  results/strat_share_test.txt (+ .npz)"""
import copy
import sys
import warnings
from multiprocessing import Pool
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np
from numpy.polynomial import Chebyshev
from scipy.integrate import solve_ivp

from earthtides.model import EarthModel, G
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit
import fcn_resonance_noic as fr


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


NPROC = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
VARIANTS = tuple(_opt('--variants', 'PREM_noIC,neutral').split(','))
DELTAS = tuple(float(x) for x in _opt('--deltas', '250,125').split(','))
KV = int(_opt('--K', 48))
INVISCID = ((6, 30), (10, 30))
MODEL = 'data/PREM_noIC.dat'
CORE = 0                                         # the liquid core reaches the centre


def neutral_noic(deg=12):
    """PREM without inner core, core density replaced by the neutral (Adams-Williamson) profile
    d rho / dr = -rho^2 g / kappa with the core's kappa and mass, integrated outward from the centre."""
    md = copy.deepcopy(EarthModel(MODEL))
    r1 = md.r1[CORE]
    kap = md.lam[CORE]
    Mtarget = md._mass_poly[CORE](r1)
    eps = 1.0

    def profile(rho0):
        def f(r, y):
            rho, M = y
            return [-rho ** 2 * (G * M / r ** 2) / kap(r), 4 * np.pi * rho * r * r]
        return solve_ivp(f, (eps, r1), [rho0, 4 / 3 * np.pi * rho0 * eps ** 3], rtol=1e-12, atol=1e-6,
                         dense_output=True)

    a, b = 9e3, 1.6e4
    for _ in range(60):                          # bisection on the central density for the core mass
        c = 0.5 * (a + b)
        if profile(c).y[1, -1] > Mtarget:
            b = c
        else:
            a = c
    sol = profile(0.5 * (a + b))
    rr = r1 * (1 - np.cos(np.linspace(0, np.pi, 200))) / 2
    rr[0] = eps
    md.rho[CORE] = Chebyshev.fit(rr, sol.sol(rr)[0], deg, domain=[0.0, r1])
    return _remass(md)


def uniform_noic():
    """PREM without inner core, core density replaced by its mean (same core mass, kappa and mantle)."""
    md = copy.deepcopy(EarthModel(MODEL))
    r1 = md.r1[CORE]
    md.rho[CORE] = Chebyshev([md._mass_poly[CORE](r1) / (4 / 3 * np.pi * r1 ** 3)], domain=[0.0, r1])
    return _remass(md)


def _remass(md):
    m_below = 0.0                                # mass polynomials as in EarthModel.__init__
    for i in range(md.n_layers):
        r = Chebyshev.identity(domain=[md.r0[i], md.r1[i]])
        p = (4.0 * np.pi * md.rho[i] * r * r).integ(lbnd=md.r0[i])
        md.mass_below[i] = m_below
        md._mass_poly[i] = p
        m_below += p(md.r1[i])
    md.M = m_below
    md.g_surface = G * md.M / md.a ** 2
    return md


def model(variant):
    return {'PREM_noIC': lambda: EarthModel(MODEL), 'neutral': neutral_noic, 'uniform': uniform_noic}[variant]()


def solve(job):
    variant, s, delta, Kc, Nc = job
    w = s * OMEGA_EARTH
    md = model(variant)
    if delta > 0:
        md = md.viscous_core(w, delta=delta)
    g = ClairautGeometry(md)
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': Kc, 'N': Nc}]})
    S = TidalSolver(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle', band=12 if delta > 0 else None)
    r = S.solve()
    if delta > 0:
        assert not any(S.true_fluid), 'a viscous layer was treated as inviscid'
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5
    print(f'{variant:9s} s {s:.5f} delta {delta:5.0f} K {Kc}  h0 {f["h0"].real:+.6f}  {S._solver_used}', flush=True)
    return job, np.array([complex(f[k]) for k in fr.KEYS])


def main():
    from provenance import Run
    with Run('viscous limit vs slipping core: PREM without inner core, stratified vs neutral vs uniform core',
             params=dict(variants=VARIANTS, deltas=DELTAS, Kv=KV, inviscid=INVISCID, sweep=fr.SWEEP),
             inputs=[MODEL]) as run:
        cases = [(v, 0.0, K, N) for v in VARIANTS for K, N in INVISCID] + \
                [(v, d, KV, 30) for v in VARIANTS for d in DELTAS]
        jobs = [(v, x, d, K, N) for v, d, K, N in cases for x in fr.SWEEP]
        with Pool(NPROC, maxtasksperchild=4) as pool:
            R = dict(pool.map(solve, jobs, chunksize=1))
        s = np.array(fr.SWEEP)
        md = neutral_noic()
        lines = ['PREM without inner core, stratified (PREM), neutral (Adams-Williamson) and uniform-density core; '
                 'one-pole fit (fcn_resonance_noic.fit); resonant parts relative to that of h0',
                 f'neutral core: central density {md.rho[CORE](0.0):.1f}, CMB {md.rho[CORE](md.r1[CORE]):.1f} kg/m^3 '
                 f'(PREM_noIC {EarthModel(MODEL).rho[CORE](0.0):.1f}, {EarthModel(MODEL).rho[CORE](md.r1[CORE]):.1f}; '
                 f'uniform {uniform_noic().rho[CORE](0.0):.1f})',
                 f'{"variant":10s} {"core":>8s} {"K":>3s} {"N":>3s} {"FCN sd":>8s} {"Q":>7s} '
                 + ' '.join(f'{k:>9s}' for k in fr.KEYS[1:]) + '  max |resid| h0']
        arrays = {}
        for v, d, K, N in cases:
            L = np.array([R[(v, x, d, K, N)] for x in fr.SWEEP])
            arrays[f'L_{v}_d{d:g}_K{K}_N{N}'] = L
            p, c, mx = fr.fit(s, L, d if d > 0 else 1.0)
            A = c[4]
            kind = f'{d:4.0f} m' if d > 0 else 'inviscid'
            lines.append(f'{v:10s} {kind:>8s} {K:3d} {N:3d} {1 / (p.real - 1):8.1f} '
                         f'{1 / (2 * p.imag) if p.imag else 0:7.0f} '
                         + ' '.join(f'{(A[i] / A[0]).real:+9.5f}' for i in range(1, len(fr.KEYS)))
                         + f'  {mx[0]:.1e}')
        run.write_text('results/strat_share_test.txt', '\n'.join(lines) + '\n')
        run.savez('results/strat_share_test.npz', sweep=s, **arrays)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
