# -*- coding: utf-8 -*-
#
# Viscous limit against slipping core on PREM with exact ellipsoidal level surfaces (the decisive test, review of
# P1 v2 Sect. 5.3: does the slipping share move to the viscous one once the figure carries its O(eps^2) P4 term?)
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
# 12/11/2026 Machiel Bos
#===============================================================================
"""On Hough's Earth the slipping and the viscous core agree in the resonant share of h(2) on the exact spheroid
(+0.0020) and disagree under the first-order map r = s (1 - 2/3 eps P2) (slipping +0.0060, viscous +0.0020): the
map lacks the (12/35) eps^2 P4 term of the ellipse, and the slipping core amplifies it by ~1/eps
(hough_share_test.py). On PREM without inner core only the first-order (Clairaut) figure exists in the solver,
and the shares are +0.0040 (slipping) and +0.0015 (viscous). This script replaces each first-order level surface
by the exact ellipse of the same Clairaut flattening eps(s), R = s f(theta; eps(s)), f = (sin^2/A^2 + cos^2/C^2)^-1/2,
A = (1 - eps)^-1/3, C = A (1 - eps) (ExactLevelsGeometry, derivatives from sympy, checked against finite
differences at import), and computes the shares again: slipping core K 6 and 10 (N 30), viscous core delta 250 m
(K 48, N 30, band 12) as the control. If the slipping share moves to +0.0015 the whole difference of Sect. 5.3 is
the missing second-order term seen through the amplifier.
Usage: python scripts/exact_levels_share_test.py NPROC [--variants exact_levels,clairaut] [--deltas 250] [--K 48]
       [--slip-term] [--model data/PREM_noIC.dat] [--tag NAME]
       ->  results/exact_levels_share_test[_slipterm][_TAG].txt (+ .npz)"""
import sys
import warnings
from multiprocessing import Pool
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np
import sympy as sp

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit
import fcn_resonance_noic as fr
from slip_term import SlipTermSolver


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


NPROC = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
VARIANTS = tuple(_opt('--variants', 'exact_levels').split(','))
DELTAS = () if _opt('--deltas', '250') == 'none' else tuple(float(x) for x in _opt('--deltas', '250').split(','))
KV = int(_opt('--K', 48))
INVISCID = ((6, 30), (10, 30))
MODEL = _opt('--model', 'data/PREM_noIC.dat')
# --slip-term: inviscid runs with the slip condition for a wall that is not a level surface (slip_term.py);
# output then results/exact_levels_share_test_slipterm[_TAG].txt
SLIPTERM = '--slip-term' in sys.argv
OUT = 'results/exact_levels_share_test' + ('_slipterm' if SLIPTERM else '') + (f"_{_opt('--tag', '')}" if _opt('--tag', '') else '')


def _ellipse_derivatives():
    th, e = sp.symbols('theta epsilon', real=True)
    A = (1 - e) ** sp.Rational(-1, 3)
    C = A * (1 - e)
    f = (sp.sin(th) ** 2 / A ** 2 + sp.cos(th) ** 2 / C ** 2) ** sp.Rational(-1, 2)
    d = [f, sp.diff(f, th), sp.diff(f, th, 2), sp.diff(f, e), sp.diff(f, e, 2), sp.diff(f, th, e)]
    return sp.lambdify((th, e), d, 'numpy')


_F = _ellipse_derivatives()


class ExactLevelsGeometry(ClairautGeometry):
    """Clairaut's first-order flattening eps(s), but every level surface an exact ellipse: R = s f(theta; eps(s))."""

    def map_derivs(self, i, s, theta):
        s = np.asarray(s, float)
        e, de, dde = self.eps(i, s)
        f, ft, ftt, fe, fee, fte = _F(theta, e)
        z = 0 * (s + theta)
        R = s * f + z
        R_s = f + s * fe * de + z
        R_ss = 2 * fe * de + s * (fee * de ** 2 + fe * dde) + z
        R_t = s * ft + z
        R_st = ft + s * fte * de + z
        R_tt = s * ftt + z
        return R, R_s, R_ss, R_t, R_st, R_tt


def check_geometry(g, s=3.0e6, th=0.7, h=1e-5):
    """Finite differences of R against the analytic derivatives in the layer that holds s; raises above 1e-5."""
    md = g.model
    i = int(np.flatnonzero((md.r0 <= s) & (s <= md.r1))[0])
    R = lambda s_, t_: g.map_derivs(i, np.array(s_), t_)[0]
    _, R_s, R_ss, R_t, R_st, R_tt = g.map_derivs(i, np.array(s), th)
    fd = {'R_s': (R(s * (1 + h), th) - R(s * (1 - h), th)) / (2 * s * h),
          'R_ss': (R(s * (1 + h), th) - 2 * R(s, th) + R(s * (1 - h), th)) / (s * h) ** 2,
          'R_t': (R(s, th + h) - R(s, th - h)) / (2 * h),
          'R_tt': (R(s, th + h) - 2 * R(s, th) + R(s, th - h)) / h ** 2,
          'R_st': (R(s * (1 + h), th + h) - R(s * (1 + h), th - h) - R(s * (1 - h), th + h)
                   + R(s * (1 - h), th - h)) / (4 * s * h * h)}
    an = dict(R_s=R_s, R_ss=R_ss, R_t=R_t, R_st=R_st, R_tt=R_tt)
    scale = dict(R_s=1.0, R_ss=1.0 / s, R_t=s, R_st=1.0, R_tt=s)     # natural size of each derivative
    err = {k: float(abs(fd[k] - an[k]) / scale[k]) for k in fd}
    bad = {k: v for k, v in err.items() if v > 1e-5}
    assert not bad, f'ExactLevelsGeometry derivatives off: {bad}'
    return err


def geometry(variant, md):
    return ExactLevelsGeometry(md) if variant == 'exact_levels' else ClairautGeometry(md)


def solve(job):
    variant, s, delta, Kc, Nc = job
    w = s * OMEGA_EARTH
    md = EarthModel(MODEL)
    if delta > 0:
        md = md.viscous_core(w, delta=delta)
    g = geometry(variant, md)
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': Kc, 'N': Nc}]})
    cls = SlipTermSolver if (SLIPTERM and delta == 0) else TidalSolver
    S = cls(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
            omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
            frame='mantle', band=12 if delta > 0 else None, slip_nonlevel=SLIPTERM)
    r = S.solve()
    if delta > 0:
        assert not any(S.true_fluid), 'a viscous layer was treated as inviscid'
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5
    print(f'{variant:12s} s {s:.5f} delta {delta:5.0f} K {Kc}  h0 {f["h0"].real:+.6f}  {S._solver_used}', flush=True)
    return job, np.array([complex(f[k]) for k in fr.KEYS])


def main():
    from provenance import Run
    md = EarthModel(MODEL)
    err = check_geometry(ExactLevelsGeometry(md))
    print('ExactLevelsGeometry finite-difference check:', {k: f'{v:.1e}' for k, v in err.items()}, flush=True)
    with Run('viscous limit vs slipping core: PREM without inner core with exact ellipsoidal level surfaces',
             params=dict(variants=VARIANTS, deltas=DELTAS, Kv=KV, inviscid=INVISCID, sweep=fr.SWEEP, slip_term=SLIPTERM,
                         geometry_check=err), inputs=[MODEL]) as run:
        cases = [(v, 0.0, K, N) for v in VARIANTS for K, N in INVISCID] + \
                [(v, d, KV, 30) for v in VARIANTS for d in DELTAS]
        jobs = [(v, x, d, K, N) for v, d, K, N in cases for x in fr.SWEEP]
        with Pool(NPROC, maxtasksperchild=4) as pool:
            R = dict(pool.map(solve, jobs, chunksize=1))
        s = np.array(fr.SWEEP)
        g = ClairautGeometry(md)
        ec = g.eps(0, np.array([md.r1[0]]))[0][0]
        lines = ['PREM without inner core, level surfaces exact ellipses of the Clairaut flattening eps(s) '
                 '(exact_levels) or the first-order map (clairaut); one-pole fit (fcn_resonance_noic.fit); '
                 'resonant parts relative to that of h0',
                 f'eps at the CMB {ec:.6e}: (12/35) eps^2 R = {12 / 35 * ec ** 2 * md.r1[0]:.1f} m',
                 f'{"variant":12s} {"core":>8s} {"K":>3s} {"N":>3s} {"FCN sd":>8s} {"Q":>7s} '
                 + ' '.join(f'{k:>9s}' for k in fr.KEYS[1:]) + '  max |resid| h0']
        arrays = {}
        for v, d, K, N in cases:
            L = np.array([R[(v, x, d, K, N)] for x in fr.SWEEP])
            arrays[f'L_{v}_d{d:g}_K{K}_N{N}'] = L
            p, c, mx = fr.fit(s, L, d if d > 0 else 1.0)
            A = c[4]
            kind = f'{d:4.0f} m' if d > 0 else 'inviscid'
            lines.append(f'{v:12s} {kind:>8s} {K:3d} {N:3d} {1 / (p.real - 1):8.1f} '
                         f'{1 / (2 * p.imag) if p.imag else 0:7.0f} '
                         + ' '.join(f'{(A[i] / A[0]).real:+9.5f}' for i in range(1, len(fr.KEYS)))
                         + f'  {mx[0]:.1e}')
        run.write_text(OUT + '.txt', '\n'.join(lines) + '\n')
        run.savez(OUT + '.npz', sweep=s, **arrays)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
