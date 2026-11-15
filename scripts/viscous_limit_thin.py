# -*- coding: utf-8 -*-
#
# Viscous limit near the FCN with thinner Ekman layers (pressure unknown): does the extrapolation to delta = 0 settle?
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
"""The viscous limit of the Love numbers of paper 1 (fcn_resonance_extrapolation_v3.py) extrapolates poles and
residues from delta = 0.75, 0.5, 0.35 km; next to the resonance h0 and k are then uncertain by 1.6e-4 and 8e-5
(psi1), set by the extrapolation in delta, not by the resolution (lab notes 2026-11-15). Here thinner layers,
which need the pressure unknown (TidalSolver(fluid_pressure=True): below 250 m the displacement formulation
loses its digits). Elastic PREM on the level figure, m = 1, the 20 frequencies and the two-pole fit of
fcn_resonance_extrapolation.py; N per kind of fluid sub-layer: bulk 30, 40 km layers --N-wall (44),
5-delta layers 16; mantle and inner core K 6, N 24.
Per core K: pole (period, Q) and shares at every delta; then the extrapolation to delta = 0 (quadratic through
three successive deltas) for every window of three, from the thickest to the thinnest: the columns settle if
the limit is reached. Largest residual and peak memory of the solves. Every solve is cached (resume).
--check-fp: at the largest delta <= 500 m also without the pressure unknown (must agree).
2026-11-15, FINDING before the first run (lab notes): at delta = 1 km the Love numbers at psi1 scatter with the
core K by ~5e-4 in h0 (K 6 .. 24: 0.68827, 0.69145, 0.69026, 0.69111, 0.69069; the pressure-unknown form
scatters as much, and the two agree only as K grows), 5e-7 at O1. The uncertainty of h0 next to the
resonance is the angular truncation (K 12 in the paper), not delta. Hence the options:
--no-fp: the displacement formulation (fine down to 250 m, and it keeps the fast solver path);
--band B: banded angular coupling (needed above core K ~ 50); workers per K from --max-gb and the measured
peak of the previous K. The table then shows the limit per K: does it settle with K?
--solver layers: with the pressure unknown the default (block) path fails above core K ~ 8 and SuperLU takes over
(slow); the layer solver handles it at the speed of the displacement formulation (K 24, delta 250 m: 6 s both).
Usage: python scripts/viscous_limit_thin.py [--K 12,16] [--deltas 1000,750,500,350,250,175,125] [--N-wall 44]
       [--workers 6] [--max-gb 24] [--no-fp] [--band 12] [--solver layers] [--check-fp] [--tag NAME]
       ->  results/viscous_limit_thin[_TAG]_level.{txt,npz}"""
import hashlib
import json
import os
import resource
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
os.environ.setdefault('EARTHTIDES_FIGURE', 'level')        # outputs tagged _level by provenance
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import figure, equatorial_radius, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer, viscous_core_resolution
from earthtides.iers import iers_fit
from fcn_resonance_extrapolation import KEYS, SWEEP, TABLE, fit, evaluate
from provenance import Run

CACHE = Path('logs/viscous_limit_cache')
_G = {}


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def solve(job):
    s, delta, Kc, Nw, fp = job[:5]
    band = job[5] if len(job) > 5 else None
    ls = job[6] if len(job) > 6 else None
    f = CACHE / (hashlib.sha1(json.dumps(job).encode()).hexdigest()[:16] + '.json')
    if f.exists():
        d = json.loads(f.read_text())
        return job, np.array(d['re']) + 1j * np.array(d['im']), d['res'], d['gb'], d['sec']
    t = time.time()
    if 'md' not in _G:
        _G['md'] = EarthModel('data/PREM.dat')
        _G['g'] = figure(_G['md'], 'level')
    w = s * OMEGA_EARTH
    md = _G['md'].viscous_core(w, delta=delta)
    g = _G['g'].restrict(md)
    Kl, Nl = resolution_per_layer(md, viscous_core_resolution(Kc, delta, N_wall=Nw))
    S = TidalSolver(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle', fluid_pressure=bool(fp), band=band, **(dict(linear_solver=ls) if ls else {}))
    r = S.solve()
    fit_ = iers_fit(S, g)
    fit_['k0'] = r['kk'] * (md.a / equatorial_radius(g, md)) ** 5
    L = np.array([complex(fit_[k]) for k in KEYS])
    res = float((S.__dict__.get('_block_info') or S.__dict__.get('_layer_info') or {}).get('residual') or np.nan)
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == 'darwin' else 1e6)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(job=job, re=L.real.tolist(), im=L.imag.tolist(), res=res, gb=gb,
                                 sec=time.time() - t, solver=str(S._solver_used))))
    print('K %d delta %6.1f m s %.5f fp %d  h0 %+.6f %+.2e  %s  %.0f s  %.1f GB' % (
        Kc, delta, s, fp, L[0].real, L[0].imag, S._solver_used, time.time() - t, gb), flush=True)
    return job, L, res, gb, time.time() - t


def lagrange0(P, ds):
    w = [np.prod([-dj / (di - dj) for dj in ds if dj != di]) for di in ds]
    return (sum(wi * P[d][0] for wi, d in zip(w, ds)), sum(wi * P[d][1] for wi, d in zip(w, ds)))


def main():
    Ks = [int(x) for x in _opt('--K', '12,16').split(',')]
    deltas = sorted((float(x) for x in _opt('--deltas', '1000,750,500,350,250,175,125').split(',')), reverse=True)
    Nw, workers, tag = int(_opt('--N-wall', 44)), int(_opt('--workers', 6)), _opt('--tag', '')
    out = 'results/viscous_limit_thin' + (f'_{tag}' if tag else '')
    with Run('viscous limit near the FCN with thinner Ekman layers and the pressure unknown',
             params=dict(K=Ks, deltas=deltas, N_wall=Nw, workers=workers, check_fp='--check-fp' in sys.argv,
                         argv=sys.argv[1:]),
             inputs=['data/PREM.dat']) as run:
        FP = 0 if '--no-fp' in sys.argv else 1
        band = int(_opt('--band', 0)) or None
        max_gb = float(_opt('--max-gb', 24))
        ls = _opt('--solver', '')                  # 'layers': the direct layer solver (needed with the pressure unknown)
        key = lambda s_, d, K: (float(s_), d, K, Nw, FP) + ((band,) if band or ls else ()) + ((ls,) if ls else ())
        dchk = max(d for d in deltas if d <= 500.0)
        R, gb_prev, K_prev = {}, None, None
        for K in sorted(Ks):                                     # one pool per K: workers from the memory budget
            jobs = [key(s, d, K) for d in deltas for s in SWEEP]
            if '--check-fp' in sys.argv and K == Ks[0]:
                jobs += [(float(s), dchk, K, Nw, 0) for s in SWEEP]
            est = 1.2 * gb_prev * (K / K_prev) ** 1.2 if gb_prev else 0.1 * K     # GB per solve (Mac: K 48 4.5, K 72 6.3 GB peak)
            w = max(1, min(workers, int(max_gb // est)))
            print(f'core K {K}: {len(jobs)} solves, {w} workers (estimate {est:.1f} GB per solve)', flush=True)
            with Pool(w, maxtasksperchild=8) as p:
                for j, L, res, gb, sec in p.imap_unordered(solve, jobs, chunksize=1):
                    R[tuple(j)] = (L, res, gb, sec)
            gb_prev, K_prev = max(v[2] for k_, v in R.items() if k_[2] == K), K
        s = np.array(SWEEP)
        lines, save = [f'elastic PREM, level figure, {"pressure unknown" if FP else "displacement formulation"}, band {band}; N bulk 30 / 40 km layers {Nw} / 5-delta 16; '
                       'mantle and inner core K 6, N 24'], {}
        for K in Ks:
            P = {}
            lines += ['', f'== core K {K}']
            for d in deltas:
                L = np.array([R[key(x, d, K)][0] for x in SWEEP])
                sf, c, mx = fit(s, L, d)
                P[d] = (np.array(sf), c)
                sh = c[4] / c[4][0]
                q = [R[key(x, d, K)] for x in SWEEP]
                lines.append('delta %6.1f m: FCN %.7f %+.2ei (%.2f sd, Q %.0f)  shares h2 %+.5f l1 %+.5f l2 %+.5f  '
                             'fit misfit h0 %.1e  residual %.0e  %.0f s  %.1f GB' % (
                                 d, sf[0].real, sf[0].imag, 1 / (sf[0].real - 1), sf[0].real / (2 * sf[0].imag),
                                 sh[1].real, sh[3].real, sh[4].real, mx[0], np.nanmax([x[1] for x in q]),
                                 np.mean([x[3] for x in q]), max(x[2] for x in q)))
                save[f'L_K{K}_d{d:g}'] = L
            lines.append('extrapolation to delta = 0 through three successive deltas (m):')
            wins = [tuple(deltas[i:i + 3]) for i in range(len(deltas) - 2)]
            E = [lagrange0(P, w) for w in wins]
            lines.append('%-22s' % 'window' + ''.join('%22s' % '/'.join('%g' % d for d in w) for w in wins))
            lines.append('%-22s' % 'FCN period (sd)' + ''.join('%22.3f' % (1 / (e[0][0].real - 1)) for e in E))
            for name, x in TABLE.items():
                V = [evaluate(e[0], e[1], x) for e in E]
                for j, k in enumerate(KEYS):
                    lines.append('%-22s' % f'{name} {k}' + ''.join('%22.6f' % v[j].real for v in V))
            sh = [e[1][4] / e[1][4][0] for e in E]
            for j, k in enumerate(KEYS[1:], 1):
                lines.append('%-22s' % f'share {k}' + ''.join('%22.6f' % v[j].real for v in sh))
        if '--check-fp' in sys.argv:
            a = np.array([R[key(x, dchk, Ks[0])][0] for x in SWEEP])
            b = np.array([R[(float(x), dchk, Ks[0], Nw, 0)][0] for x in SWEEP])
            lines += ['', f'pressure unknown against the displacement formulation, K {Ks[0]}, delta {dchk:g} m: '
                          'max |difference| over the sweep ' + ' '.join('%s %.1e' % (k, v) for k, v in zip(KEYS, np.abs(a - b).max(axis=0)))]
        run.savez(out + '.npz', sweep=s, deltas=np.array(deltas), **save)
        run.write_text(out + '.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
