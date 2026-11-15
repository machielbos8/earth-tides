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
# 28/10/2026 Machiel Bos
#===============================================================================
"""v3 (2026-10-28, plan item 1d): fcn_resonance_extrapolation.py (v2) with per-layer resolution, to check
the delta -> 0 values of the DDW table (tab:ddw) in the core's angular resolution: mantle and inner core
K 6 / N 24, core (and its Ekman sub-layers) K = KCORE, N 30, as Fig. 4. Same sweep, fit and quadratic
extrapolation (0.35, 0.5, 0.75 km; check 0.5, 0.75, 1 km) as v2; also the resonant parts of h(2), l(2)
relative to h(0) (residue ratios of the FCN pole) at each delta and in the limit.
Usage: python scripts/fcn_resonance_extrapolation_v3.py KCORE [WORKERS] [--fresh]
-> results/fcn_resonance_extrapolation_v3_K<KCORE>.{txt,npz}
Memory per solve: K12 1.75 GB, K16 ~3.6 GB (CLAUDE.md, 'Memory')."""
import os
import sys
import warnings
from multiprocessing import Pool
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, LevelGeometry, OMEGA_EARTH, equatorial_radius
from earthtides.solver import TidalSolver, resolution_per_layer, viscous_core_resolution
from earthtides.iers import iers_fit
from fcn_resonance_extrapolation import KEYS, DELTAS, SWEEP, TABLE, INVISCID, fit, evaluate


# --figure level: coordinate surfaces = level surfaces of the reference potential (geometry.LevelGeometry),
# output ..._K<K>_level; default: the first-order (Clairaut) figure of the paper
LEVEL = '--figure' in sys.argv and sys.argv[sys.argv.index('--figure') + 1] == 'level'


def solve(job):
    s, delta, Kc = job
    w = s * OMEGA_EARTH
    md = EarthModel('data/PREM.dat').viscous_core(w, delta=delta)
    g = LevelGeometry(md) if LEVEL else ClairautGeometry(md)
    if LEVEL:        # N per kind of sub-layer (sublayer_N_prem.py): 40 km layers 36 at every delta, 5-delta layers 16
        Kl, Nl = resolution_per_layer(md, viscous_core_resolution(Kc, delta, N_wall=36))
    else:            # the first-order results of the paper: N 30 in every fluid layer
        Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': Kc, 'N': 30}]})
    S = TidalSolver(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle')
    r = S.solve()
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * (md.a / equatorial_radius(g, md)) ** 5       # first-order: (1 + eps / 3)^-5 as before
    print('s %.5f delta %.2f km  h0 %+.6f %+.2e  %s' % (s, delta / 1e3, f['h0'].real, f['h0'].imag,
                                                     S._solver_used), flush=True)
    return (s, delta), np.array([complex(f[k]) for k in KEYS])


def lagrange0(P, ds):
    """Quadratic in delta through the three deltas, evaluated at delta = 0."""
    w = [np.prod([-dj / (di - dj) for dj in ds if dj != di]) for di in ds]
    return (sum(wi * P[d][0] for wi, d in zip(w, ds)), sum(wi * P[d][1] for wi, d in zip(w, ds)))


def shares(c):
    """Resonant parts of all six parameters relative to that of h(0): FCN residue ratios."""
    return c[4] / c[4][0]


if __name__ == '__main__':
    fresh = '--fresh' in sys.argv                       # recompute everything, ignore the stored solves
    argv = [a for a in sys.argv if a not in ('--fresh', '--figure', 'level', 'first-order')]   # (argv is left intact: spawned workers read --figure)
    Kc = int(argv[1])
    nproc = int(argv[2]) if len(argv) > 2 else 2
    out = 'results/fcn_resonance_extrapolation_v3_K%d' % Kc + ('_level' if LEVEL else '')
    R = {}
    if os.path.exists(out + '.npz') and not fresh:      # reuse earlier solves (same set-up)
        old = np.load(out + '.npz')
        for j, d in enumerate(old['deltas']):
            for i, x in enumerate(old['sweep']):
                R[(float(x), float(d))] = old['L'][j, i]
    jobs = [(s, d, Kc) for d in DELTAS for s in SWEEP if (s, d) not in R]
    with Pool(nproc, maxtasksperchild=4) as p:
        R.update(dict(p.map(solve, jobs, chunksize=1)))
    from provenance import Run
    run = Run('viscous limit of the Love numbers near the FCN: poles and residues extrapolated to delta -> 0',
              params=dict(K=Kc, figure='level' if LEVEL else 'first-order', fresh=fresh, nproc=nproc),
              inputs=['data/PREM.dat']).__enter__()
    run.savez(out + '.npz', sweep=np.array(SWEEP), deltas=np.array(DELTAS),
              L=np.array([[R[(x, d)] for x in SWEEP] for d in DELTAS]))
    s = np.array(SWEEP)
    lines, P = ['core K %d, N %s; mantle and inner core K 6, N 24' % (
        Kc, '30 (bulk) / 36 (40 km layers) / 16 (5-delta layers); level figure' if LEVEL else '30')], {}
    for d in DELTAS:
        L = np.array([R[(x, d)] for x in SWEEP])
        sf, c, mx = fit(s, L, d)
        P[d] = (np.array(sf), c)
        sh = shares(c)
        lines.append('delta %.2f km: FCN %.7f %+.2ei (%.1f sd)  FICN %.7f %+.2ei (%.0f sd)  max |resid| %s'
                     '  shares h2 %+.5f l2 %+.5f' % (
                         d / 1e3, sf[0].real, sf[0].imag, 1 / (sf[0].real - 1), sf[1].real, sf[1].imag,
                         1 / (sf[1].real - 1), np.array2string(mx, precision=1), sh[1].real, sh[4].real))
    sf0, c0 = lagrange0(P, (3.5e2, 5e2, 7.5e2))
    sfc, cc = lagrange0(P, (5e2, 7.5e2, 1e3))
    s0, sc = shares(c0), shares(cc)
    lines.append('delta -> 0: FCN %.7f %+.2ei (%.1f sd), FICN %.0f sd; check: FCN %.1f sd' % (
        sf0[0].real, sf0[0].imag, 1 / (sf0[0].real - 1), 1 / (sf0[1].real - 1), 1 / (sfc[0].real - 1)))
    lines.append('delta -> 0 resonant parts / h0: ' + ' '.join('%s %+.5f' % (k, z.real) for k, z in zip(KEYS, s0))
                 + '   check: ' + ' '.join('%s %+.5f' % (k, z.real) for k, z in zip(KEYS, sc)))
    lines.append('%-6s %-9s ' % ('', '') + ''.join('%11s' % k for k in KEYS))
    for name, x in TABLE.items():
        v, vc = evaluate(sf0, c0, x), evaluate(sfc, cc, x)
        lines.append('%-6s %-9s ' % (name, 'delta->0') + ''.join('%+11.5f' % z.real for z in v))
        lines.append('%-6s %-9s ' % ('', 'chk') + ''.join('%11.1e' % abs(a - b) for a, b in zip(v, vc)))
        lines.append('%-6s %-9s ' % ('', 'inviscid') + ''.join('%+11.5f' % z for z in INVISCID[name]))
    txt = '\n'.join(lines)
    print(txt)
    run.write_text(out + '.txt', txt + '\n')
    run.__exit__(None, None, None)
