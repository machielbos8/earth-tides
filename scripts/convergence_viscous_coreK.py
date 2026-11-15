# -*- coding: utf-8 -*-
#
# Semidiurnal and diurnal sweeps of the viscous core at increasing core K (mantle and inner core
# at K 6 / N 24): do the bumps of Fig. 4 converge (physical resonances) or vanish (truncation)?
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
# 18/10/2026 Machiel Bos
#===============================================================================
"""Semidiurnal and diurnal sweeps of the viscous core at increasing core K (mantle and inner core
at K 6 / N 24): do the bumps of Fig. 4 converge (physical resonances) or vanish (truncation)?
The Ekman number E = (delta / r)^2 ~ 3e-7 at delta = 2 km; shear layers from the critical
latitudes have widths ~ E^(1/5)..E^(1/4) of the core radius, degrees ~60-120 to resolve.
Usage: python scripts/convergence_viscous_coreK.py KCORE NCORE DELTA_KM [WORKERS [STEP]] [--band B] [--window LO HI] [--split S [--interior-N N]] [--solid-P M] [--after PID]
       [--neutral]
DELTA_KM = 0: the inviscid core at the same per-layer resolution (Fig. 4 like-for-like).
STEP: frequency step in cpsd (default 0.0025; a finer step resolves the inviscid spikes, P1 v2 Fig. 4).
--band B: banded angular coupling (TidalSolver(band=B), plan_banded_coupling_2026-11-02); needed above core K ~50.
--window LO HI: diurnal frequencies LO..HI only (step STEP), plus the control points 0.93 (m 1) and 1.93 (m 2).
--split S [--interior-N N]: core interior (1261-3440 km) cut into S equal Chebyshev layers (N in them; the Ekman
  sub-layers keep NCORE), as core_flow_attractor.model (lab notes 2026-11-05: the beams need it; Love numbers ~1e-7).
--solid-P M: TidalSolver(solid_P=M), the potential in the solid layers at their own degrees + M (solver-8-solid-P).
--after PID: wait until process PID has ended (chain behind a running job).
A solve that fails is written with NaN and solver 'failed'; every solve is printed when it finishes.
--neutral: neutrally stratified core (N^2 = 0: Adams-Williamson density with PREM's kappa and core mass,
  convergence_stratification.stratified(0)); PREM's core has N^2/(2 Omega)^2 from -1.4 to +0.24.
-> results/convergence_viscous_coreK<K>_N<N>_d<delta>[_s<STEP>][_b<B>][_w<LO>-<HI>][_neutral].txt (columns of
   convergence_inertial_viscous)
Memory per solve (2026-10-18): K8 1.2 GB, K12 1.8, K16 3.6, K24 7.0 GB (diurnal); K32 10.2 GB peak (2026-11-01, delta 1 km, one worker: swap grew by 3 GB): WORKERS x peak
must stay below the free memory (CLAUDE.md, 'Memory')."""
import os
import sys
import warnings
from multiprocessing import Pool
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')    # Accelerate: one thread per worker
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH, figure

# --figure level: on the level figure (geometry.figure), output suffix _level; read from sys.argv at import,
# so that spawned workers see it too
FIGURE = sys.argv[sys.argv.index('--figure') + 1] if '--figure' in sys.argv else 'first-order'
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit


def run(args):
    m, sig, Kc, Nc, delta, band, neutral, split, Ni, solid_P = args
    if neutral:
        from convergence_stratification import stratified
        md = stratified(0.0)
    else:
        md = EarthModel('data/PREM.dat')
    if delta > 0:
        md = md.viscous_core(sig * OMEGA_EARTH, delta=delta)
    if split > 1:                                    # core interior: the thickest liquid layer
        i = max((j for j in range(md.n_layers) if abs(md.mu[j](0.5 * (md.r0[j] + md.r1[j]))) < 1e8),
                key=lambda j: md.r1[j] - md.r0[j])
        md = md.split_layer(i, list(np.linspace(md.r0[i], md.r1[i], split + 1)[1:-1]))
    g = figure(md, FIGURE)
    rules = [{'fluid': True, 'K': Kc, 'N': Nc}]
    if Ni and Ni != Nc:
        rules.append({'fluid': True, 'thickness_km': [100, 1e5], 'N': Ni})
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': rules})
    try:
        S = TidalSolver(md, g, m=m, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                        consistent_reference=True, frame='mantle', band=band, solid_P=solid_P)
        S.solve()
    except Exception as e:                           # one failed solve must not lose a night's sweep
        print('failed', m, sig, type(e).__name__, str(e)[:80], flush=True)
        return (m, sig, Kc, Nc, np.nan, np.nan, np.nan, np.nan, delta, 'failed')
    f = iers_fit(S, g)
    return (m, sig, Kc, Nc, f['h0'].real, f['h2'].real, f['l2'].real, f['l1'].real, delta, S._solver_used)


if __name__ == '__main__':
    argv = sys.argv[1:]
    band = window = None
    neutral = '--neutral' in argv
    argv = [a for a in argv if a != '--neutral']
    if '--figure' in argv:
        i = argv.index('--figure'); del argv[i:i + 2]
    if '--band' in argv:
        i = argv.index('--band'); band = int(argv[i + 1]); del argv[i:i + 2]
    if '--window' in argv:
        i = argv.index('--window'); window = (float(argv[i + 1]), float(argv[i + 2])); del argv[i:i + 3]
    split, Ni, solid_P, after = 1, None, None, None
    for name in ('--split', '--interior-N', '--solid-P', '--after'):
        if name in argv:
            i = argv.index(name); v = int(argv[i + 1]); del argv[i:i + 2]
            if name == '--split':
                split = v
            elif name == '--interior-N':
                Ni = v
            elif name == '--solid-P':
                solid_P = v
            else:
                after = v
    Kc, Nc, dkm = int(argv[0]), int(argv[1]), float(argv[2])
    nproc = int(argv[3]) if len(argv) > 3 else 1
    step = float(argv[4]) if len(argv) > 4 else 0.0025
    grid = lambda a, b: np.linspace(a, b, int(round((b - a) / step)) + 1)
    ex = (split, Ni, solid_P)
    if window is None:
        jobs = [(2, s, Kc, Nc, dkm * 1e3, band, neutral) + ex for s in grid(1.90, 1.96)]
        jobs += [(1, s, Kc, Nc, dkm * 1e3, band, neutral) + ex for s in grid(0.90, 0.95)]
    else:
        jobs = [(1, s, Kc, Nc, dkm * 1e3, band, neutral) + ex for s in grid(*window)]
        jobs += [(1, 0.93, Kc, Nc, dkm * 1e3, band, neutral) + ex, (2, 1.93, Kc, Nc, dkm * 1e3, band, neutral) + ex]
    out = 'results/convergence_viscous_coreK%d_N%d_d%g%s%s%s%s%s%s.txt' % (
        Kc, Nc, dkm, '' if step == 0.0025 else '_s%g' % step, '' if band is None else '_b%d' % band,
        '' if window is None else '_w%g-%g' % window, '_neutral' if neutral else '',
        '' if split == 1 else '_S%d%s' % (split, '' if not Ni else 'N%d' % Ni),
        '' if solid_P is None else '_sP%d' % solid_P)
    if FIGURE != 'first-order':
        out = out[:-4] + '_' + FIGURE + '.txt'
    if after:
        import time
        print('waiting for process', after, flush=True)
        while True:
            try:
                os.kill(after, 0)
            except ProcessLookupError:
                break
            except PermissionError:
                pass
            time.sleep(60)
    from provenance import Run
    with Run(f'core-K sweep, core K {Kc} N {Nc}, delta {dkm:g} km, step {step:g} cpsd'
             + (', neutrally stratified core' if neutral else ''),
             params=dict(K_core=Kc, N_core=Nc, delta_km=dkm, step=step, workers=nproc, band=band, window=window, figure=FIGURE,
                         neutral=neutral, split=split, interior_N=Ni, solid_P=solid_P),
             inputs=['data/PREM.dat']) as run_:
        res = []
        with Pool(nproc, maxtasksperchild=1) as p:          # fresh process per solve (memory)
            for r in p.imap(run, jobs, chunksize=1):        # in order; printed as they finish
                res.append(r)
                print('%d %.5f h0 %.10f h2 %.10f %s' % (r[0], r[1], r[4], r[5], r[9]), flush=True)
        lines = ['m sigma K N h0 h2 l2 l1 delta solver']
        lines += ['%d %.5f %d %d %.10f %.10f %.10f %.10f %.0f %s' % (r[:9] + (r[9].replace(' ', '_'),))
                  for r in res]
        run_.write_text(out, '\n'.join(lines) + '\n')
    print(out, len(res), 'solves;', sum(r[9] != 'block' for r in res), 'not on the block path')
