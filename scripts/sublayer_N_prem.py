# -*- coding: utf-8 -*-
#
# Radial resolution per sub-layer of the viscous core of PREM: which N does each sub-layer need?
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
"""Speed idea 1 (Machiel 2026-11-13; sublayer_N.py did this for Hough's Earth at delta 125 m: only the ~40 km
layer under the wall needs the high N). EarthModel.viscous_core splits the liquid core into the bulk, a 40 km
layer at each solid boundary and, inside it, a layer of 5 delta. All paper runs use one N (30) in all of them.
Here PREM on the level figure, elastic, m = 1, at O1 (0.92700), next to the FCN (1.00200) and psi1 (1.00273),
for delta = 1 km and 0.35 km: N of (bulk, 40 km layers, 5-delta layers) varied one at a time against a
reference with 44 everywhere; mantle and inner core K 6, N 24. Per configuration: max over the three
frequencies of |d h0|, |d h2|, |d l1|, |d k| from the reference, unknowns in the core, seconds per solve.
Usage: python scripts/sublayer_N_prem.py [--K 12] [--deltas 1000,350] [--solver block] [--band B]
       ->  results/sublayer_N_prem_K<K>.txt"""
import sys
import time
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import figure, equatorial_radius, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit
from provenance import Run

FREQS = (0.92700, 1.00200, 1.00273)
KEYS = ('h0', 'h2', 'l1', 'k')
CONFIGS = [('reference', (44, 44, 44)), ('paper (30 everywhere)', (30, 30, 30)), ('all 24', (24, 24, 24)),
           ('bulk 24', (24, 30, 30)), ('bulk 20', (20, 30, 30)), ('bulk 16', (16, 30, 30)),
           ('40 km 24', (30, 24, 30)), ('40 km 36', (30, 36, 30)),
           ('5 delta 24', (30, 30, 24)), ('5 delta 20', (30, 30, 20)), ('5 delta 16', (30, 30, 16)),
           ('20 / 30 / 20', (20, 30, 20)), ('16 / 30 / 16', (16, 30, 16)), ('20 / 36 / 20', (20, 36, 20))]


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def rules(K, Ns, delta):
    split = max(5.0, 10 * delta / 1e3)                 # km: the 5-delta layer is thinner, the 40 km layer thicker
    return {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': K, 'N': Ns[0]},
                                       {'fluid': True, 'thickness_km': [split, 100.0], 'N': Ns[1]},
                                       {'fluid': True, 'thickness_km': [0.0, split], 'N': Ns[2]}]}


def solve(md0, g0, s, delta, K, Ns, solver, band):
    w = s * OMEGA_EARTH
    md = md0.viscous_core(w, delta=delta)
    g = g0.restrict(md)
    Kl, Nl = resolution_per_layer(md, rules(K, Ns, delta))
    t = time.time()
    S = TidalSolver(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle', linear_solver=solver, band=band)
    r = S.solve()
    dt = time.time() - t
    f = iers_fit(S, g)
    f['k'] = r['kk'] * (md.a / equatorial_radius(g, md)) ** 5
    n = int(sum(S._nbl[i] for i, fl in enumerate(S.fluid_layers()) if fl)) if getattr(S, '_nbl', None) is not None else -1
    thick = sorted({round((md.r1[i] - md.r0[i]) / 1e3, 2) for i, fl in enumerate(S.fluid_layers()) if fl})
    return np.array([complex(f[k]) for k in KEYS]), dt, n, thick


def main():
    K = int(_opt('--K', 12))
    deltas = [float(x) for x in _opt('--deltas', '1000,350').split(',')]
    solver, band = _opt('--solver', 'block'), (int(_opt('--band', 0)) or None)
    with Run('N per sub-layer of the viscous core of PREM (bulk, 40 km, 5 delta), one at a time',
             params=dict(K=K, deltas=deltas, solver=solver, band=band, freqs=FREQS), inputs=['data/PREM.dat']) as run:
        md0 = EarthModel('data/PREM.dat')
        g0 = figure(md0, 'level')
        lines = [f'PREM, level figure, viscous core K {K}, solver {solver}, band {band}; frequencies {FREQS} cpsd; '
                 'N of (bulk, 40 km layers, 5-delta layers); differences from the reference: max over the frequencies']
        for delta in deltas:
            ref = None
            for name, Ns in CONFIGS:
                R = [solve(md0, g0, s, delta, K, Ns, solver, band) for s in FREQS]
                L = np.array([r[0] for r in R])
                if ref is None:
                    ref = L
                    lines += ['', f'delta {delta:g} m; fluid sub-layers (km): {R[0][3]}',
                              f'{"configuration":24s} {"N":>10s} {"core unk.":>9s} ' + ' '.join(f'{"|d " + k + "|":>9s}' for k in KEYS)
                              + f' {"s/solve":>8s}']
                d = np.abs(L - ref).max(axis=0)
                lines.append(f'{name:24s} {"/".join(str(n) for n in Ns):>10s} {R[0][2]:9d} '
                             + ' '.join(f'{v:9.1e}' for v in d) + f' {np.mean([r[1] for r in R]):8.2f}')
                print(lines[-1], flush=True)
        run.write_text(f'results/sublayer_N_prem_K{K}.txt', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
