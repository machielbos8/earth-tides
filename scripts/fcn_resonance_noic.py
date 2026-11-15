# -*- coding: utf-8 -*-
#
# Diurnal band with the viscous core for PREM WITHOUT an inner core (fluid to the centre): FCN resonance
# fitted at each Ekman thickness and extrapolated to delta -> 0, as fcn_resonance_extrapolation_v3.py.
# Plan 2026-10-19, task G: the Earth for which the two-layer torque model is complete.
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
# 29/10/2026 Machiel Bos
#===============================================================================
"""Diurnal band with the viscous core for PREM WITHOUT an inner core (data/PREM_noIC.dat: inner and outer
core merged into one fluid layer with the outer-core polynomials, scripts/convergence_fullsphere_core.py).
Same sweep, deltas and quadratic extrapolation as fcn_resonance_extrapolation_v3.py (mantle K 6 N 24,
core K = KCORE N 30), but the fit has the FCN pole only (no inner core, no FICN): cubic background +
FCN pole + 1/(s - 1). Task G (plan 2026-10-19): with no inner core the two-layer torque model is complete,
so its remaining differences from this solution are second-order ellipticity or core flow beyond a
uniform rotation.
Usage: python scripts/fcn_resonance_noic.py KCORE [WORKERS] -> results/fcn_resonance_noic_K<KCORE>.{txt,npz}"""
import os
import sys
import warnings
from multiprocessing import Pool
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import least_squares

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit
from fcn_resonance_extrapolation import KEYS, DELTAS, SWEEP, TABLE
from fcn_resonance_extrapolation import fit as fit2                  # FCN + second free pole

MODEL = 'data/PREM_noIC.dat'
# 2026-10-29: denser near the FCN, to identify the second damped pole near 1.0019 cpsd that the
# one-pole fit of the original sweep (v2 frequencies) left in its residuals
DENSE = (1.0005, 1.0008, 1.0012, 1.0016, 1.0018, 1.0019, 1.00205, 1.0023, 1.0026, 1.0030, 1.0040)
SWEEP = tuple(sorted(set(SWEEP) | set(DENSE)))


def solve(job):
    s, delta, Kc = job
    w = s * OMEGA_EARTH
    md = EarthModel(MODEL).viscous_core(w, delta=delta)
    g = ClairautGeometry(md)
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': Kc, 'N': 30}]})
    S = TidalSolver(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle')
    r = S.solve()
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5
    print('s %.5f delta %.2f km  h0 %+.6f %+.2e  %s' % (s, delta / 1e3, f['h0'].real, f['h0'].imag,
                                                     S._solver_used), flush=True)
    return (s, delta), np.array([complex(f[k]) for k in KEYS])


def basis(s, p):
    x = s - 1.0
    return np.stack([x ** k for k in range(4)] + [1.0 / (s - p), 1.0 / x], axis=1).astype(complex)


def fit(s, L, delta):
    """One FCN pole: s (n,), L (n, 6) -> pole, coefficients (6, 6), max |residual| (6,)."""
    def coefs(p):
        B = basis(s, p)
        return B, np.linalg.lstsq(B, L, rcond=None)[0]

    def res(q):
        B, c = coefs(q[0] + 1j * q[1])
        r = B @ c - L
        return np.concatenate([r.real.ravel(), r.imag.ravel()])
    sol = least_squares(res, [1.00225 + 1e-4 * delta / 1e3, 8e-4 * delta / 1e3], x_scale=1e-4,
                        xtol=1e-15, ftol=1e-15)
    p = sol.x[0] + 1j * sol.x[1]
    B, c = coefs(p)
    return p, c, np.abs(B @ c - L).max(axis=0)


def lagrange0(P, ds):
    w = [np.prod([-dj / (di - dj) for dj in ds if dj != di]) for di in ds]
    return sum(wi * P[d][0] for wi, d in zip(w, ds)), sum(wi * P[d][1] for wi, d in zip(w, ds))


def evaluate(p, c, s):
    return (basis(np.array([s - 1e-5, s + 1e-5]) if s == 1.0 else np.array([s]), p) @ c).mean(axis=0)


if __name__ == '__main__':
    sys.path.insert(0, 'scripts')
    from provenance import Run
    _run = Run('viscous core without inner core: FCN resonance at delta -> 0 (Sect. 5.3)', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
    Kc = int(sys.argv[1])
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    out = 'results/fcn_resonance_noic_K%d' % Kc
    R = {}
    if os.path.exists(out + '.npz'):                    # reuse earlier solves (same set-up)
        old = np.load(out + '.npz')
        for j, d in enumerate(old['deltas']):
            for i, x in enumerate(old['sweep']):
                R[(float(x), float(d))] = old['L'][j, i]
    jobs = [(s, d, Kc) for d in DELTAS for s in SWEEP if (s, d) not in R]
    with Pool(nproc, maxtasksperchild=4) as p:
        R.update(dict(p.map(solve, jobs, chunksize=1)))
    np.savez(out + '.npz', sweep=np.array(SWEEP), deltas=np.array(DELTAS),
             L=np.array([[R[(x, d)] for x in SWEEP] for d in DELTAS]))
    s = np.array(SWEEP)
    lines, P = ['PREM without inner core; core K %d, N 30; mantle K 6, N 24; one pole (FCN)' % Kc], {}
    for d in DELTAS:
        L = np.array([R[(x, d)] for x in SWEEP])
        pf, c, mx = fit(s, L, d)
        P[d] = (pf, c)
        lines.append('delta %.2f km: FCN %.7f %+.2ei (%.1f sd)  residue h0 %+.5e  max |resid| %s' % (
            d / 1e3, pf.real, pf.imag, 1 / (pf.real - 1), c[4][0].real, np.array2string(mx, precision=1)))
    lines.append('two poles (FCN + free second pole):')
    P2 = {}
    for d in DELTAS:
        L = np.array([R[(x, d)] for x in SWEEP])
        sf, c, mx = fit2(s, L, d)
        P2[d] = (np.array(sf), c)
        lines.append('delta %.2f km: FCN %.7f %+.2ei (%.1f sd) residue h0 %+.5e  2nd %.7f %+.2ei residue h0 %+.3e'
                     ' k %+.3e  max |resid| %s' % (d / 1e3, sf[0].real, sf[0].imag, 1 / (sf[0].real - 1),
                                                   c[4][0].real, sf[1].real, sf[1].imag, c[5][0].real,
                                                   c[5][5].real, np.array2string(mx, precision=1)))
    for ds in ((3.5e2, 5e2, 7.5e2), (5e2, 7.5e2, 1e3)):
        sf0, c20 = lagrange0(P2, ds)
        lines.append('two poles, delta -> 0 from %s km: FCN %.7f %+.2ei (%.1f sd) residue h0 %+.5e; 2nd %.7f %+.2ei'
                     ' residue h0 %+.2e' % ('/'.join('%.2f' % (x / 1e3) for x in ds), sf0[0].real, sf0[0].imag,
                                           1 / (sf0[0].real - 1), c20[4][0].real, sf0[1].real, sf0[1].imag,
                                           c20[5][0].real))
    lines.append('one pole:')
    p0, c0 = lagrange0(P, (3.5e2, 5e2, 7.5e2))
    pc, cc = lagrange0(P, (5e2, 7.5e2, 1e3))
    lines.append('delta -> 0: FCN %.7f %+.2ei (%.1f sd); check: FCN %.1f sd' % (
        p0.real, p0.imag, 1 / (p0.real - 1), 1 / (pc.real - 1)))
    lines.append('delta -> 0 FCN residues: ' + ' '.join('%s %+.5e' % (k, z.real) for k, z in zip(KEYS, c0[4]))
                 + '   check h0 %+.5e' % cc[4][0].real)
    lines.append('delta -> 0 resonant parts / h0: ' + ' '.join('%s %+.5f' % (k, (z / c0[4][0]).real)
                                                             for k, z in zip(KEYS, c0[4])))
    lines.append('%-6s %-9s ' % ('', '') + ''.join('%11s' % k for k in KEYS))
    for name, x in TABLE.items():
        v, vc = evaluate(p0, c0, x), evaluate(pc, cc, x)
        lines.append('%-6s %-9s ' % (name, 'delta->0') + ''.join('%+11.5f' % z.real for z in v))
        lines.append('%-6s %-9s ' % ('', 'chk') + ''.join('%11.1e' % abs(a - b) for a, b in zip(v, vc)))
    txt = '\n'.join(lines)
    print(txt)
    open(out + '.txt', 'w').write(txt + '\n')
    _run.output(out + '.txt')
    _run.output(out + '.npz')
    _run.__exit__(None, None, None)
