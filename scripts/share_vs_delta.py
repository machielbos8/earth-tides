# -*- coding: utf-8 -*-
#
# Resonant shares of the viscous core at each delta and resolution: is the viscous limit converged?
# (P1 v2 Sect. 5.3, review)
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
# 10/11/2026 Machiel Bos
#===============================================================================
"""Generalizes noic_share_vs_delta.py (whose runs below delta ~ 130 m were inviscid in disguise before solver
tag solver-9-viscous-threshold). For each Ekman thickness delta and each core resolution (K, N; N in every core
layer and Ekman sub-layer) the diurnal band is swept and the free core nutation (and, with an inner core, the free
inner core nutation) is fitted as complex poles plus a smooth background, without any extrapolation in delta. Output
per (K, N, delta): FCN period, Q, and the resonant parts of h2, l0, l1, l2, k relative to that of h0, so that their
convergence in K, N and delta can be read directly.
  --model PREM_noIC (default; one pole, fcn_resonance_noic.fit) or PREM (two poles, fcn_resonance_extrapolation.fit)
Usage: python scripts/share_vs_delta.py NPROC [--model M] [--deltas d1,..(m)] [--K k1,..] [--N n1,..] [--band B] [--tag T]
       ->  results/share_vs_delta_<model>[_T].txt (+ .npz)"""
import sys
import warnings
from multiprocessing import Pool
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit
import fcn_resonance_noic as fr1
import fcn_resonance_extrapolation as fr2


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


NPROC = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
MODEL = _opt('--model', 'PREM_noIC')
DELTAS = tuple(float(x) for x in _opt('--deltas', '350,250,180,125,90').split(','))
KS = tuple(int(x) for x in _opt('--K', '12,16').split(','))
NS = tuple(int(x) for x in _opt('--N', '30').split(','))
TAG = _opt('--tag', '')
BAND = int(_opt('--band', 0)) or None           # banded angular coupling (solver-7) for high K
KEYS = fr2.KEYS
NOIC = MODEL.endswith('noIC')
SWEEP = fr1.SWEEP if NOIC else fr2.SWEEP


def solve(job):
    s, delta, Kc, Nc = job
    w = s * OMEGA_EARTH
    md = EarthModel(f'data/{MODEL}.dat').viscous_core(w, delta=delta)
    g = ClairautGeometry(md)
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': Kc, 'N': Nc}]})
    S = TidalSolver(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle', band=BAND)
    r = S.solve()
    assert not any(S.true_fluid), 'a viscous layer was treated as inviscid'
    f = iers_fit(S, g)
    f['k0'] = r['kk'] * (1 + g.flattening_surface / 3) ** -5
    print(f'  s {s:.5f} delta {delta:5.0f} K {Kc} N {Nc}  h0 {f["h0"].real:+.6f}  {S._solver_used}', flush=True)
    return job, np.array([complex(f[k]) for k in KEYS])


def main():
    from provenance import Run
    with Run(f'resonant shares of the viscous core per delta and resolution ({MODEL})',
             params=dict(model=MODEL, deltas=DELTAS, K=KS, N=NS, sweep=SWEEP), inputs=[f'data/{MODEL}.dat']) as run:
        jobs = [(x, d, K, N) for K in KS for N in NS for d in DELTAS for x in SWEEP]
        with Pool(NPROC, maxtasksperchild=4) as pool:
            R = dict(pool.map(solve, jobs, chunksize=1))
        s = np.array(SWEEP)
        lines = [f'{MODEL}, viscous core: complex-pole fit at each delta (no extrapolation in delta); resonant parts '
                 'relative to that of h0. Inviscid, no inner core: h2 +0.00395 l2 -0.00098 (fcn_share_inviscid_noic)',
                 f'{"K":>3s} {"N":>3s} {"delta m":>8s} {"FCN sd":>8s} {"Q":>7s} ' + ' '.join(f'{k:>9s}' for k in KEYS[1:])
                 + '  max |resid| h0']
        arrays = {}
        for K in KS:
            for N in NS:
                for d in DELTAS:
                    L = np.array([R[(x, d, K, N)] for x in SWEEP])
                    arrays[f'L_K{K}_N{N}_d{d:g}'] = L
                    if NOIC:
                        p, c, mx = fr1.fit(s, L, d)
                    else:
                        (p, _), c, mx = fr2.fit(s, L, d)
                    A = c[4]
                    lines.append(f'{K:3d} {N:3d} {d:8.0f} {1 / (p.real - 1):8.1f} {1 / (2 * p.imag):7.0f} '
                                 + ' '.join(f'{(A[i] / A[0]).real:+9.5f}' for i in range(1, len(KEYS)))
                                 + f'  {mx[0]:.1e}')
                    print(lines[-1], flush=True)
        out = f'results/share_vs_delta_{MODEL}' + (f'_{TAG}' if TAG else '')
        run.write_text(out + '.txt', '\n'.join(lines) + '\n')
        run.savez(out + '.npz', sweep=s, **arrays)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
