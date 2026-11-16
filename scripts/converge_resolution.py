# -*- coding: utf-8 -*-
#
# Convergence ladder: raise the band, N and K of the liquid core one at a time until nothing changes.
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
"""Boyd: a calculation is not finished before it is compared with the same calculation at another truncation.
Three truncations of the viscous core are varied here, one at a time: the angular band B (couplings
|n - n'| > B dropped), the number of Chebyshev polynomials N and the number of degrees K in the liquid layers
(Hough's Earth of hough_share_test.py). The mantle (K 6 / N 16 by default) has its own two rungs, mK and mN
(--mantle-K, --mantle-N), taken after the three of the core: the bump (degree 4) times the tide (degree 2) reaches
degree 6 at first order, so mantle K 6 is the least that holds it.

Tested: the Love numbers h0, h2, l1, l2 at a few frequencies, two away from the free core nutation and six
next to it (offsets from the FCN of the variant, located with a coarse inviscid scan); not the whole sweep.

Ladder: from the starting point raise the band one step with N and K fixed; if the Love numbers change by more
than the tolerance, keep the higher value and try the next; then N, then K (the expensive one last). A higher K
can ask for a wider band or more N again, so the three are repeated until a full pass changes nothing: that
last pass is the confirmation (every parameter raised once more, all changes below the tolerance).
Test: at the two far frequencies the Love numbers themselves (|dh0| <= TOL0, the others <= TOL2). Next to the
resonance a Love number at a fixed frequency mostly measures a shift of the FCN (h0 is steep there: 1e-3 in
h0 = 8e-7 cpsd = 0.05 sd of 243), so there the test is on what is quoted: a one-pole fit to all the test
frequencies gives the period, Q and the shares at each truncation; period to TOLP sd, Q to TOLQ, shares to TOLS.
Each step costs one new set of solves (the point before is the baseline); every solve is kept in
logs/converge_cache (key: variant, delta, frequency, K, N, band, solver), so a second pass or a restart
computes nothing twice. A step that would exceed --max-gb (peak of the last solve scaled with K^2 or N^2) is
not taken: the parameter is reported as 'budget', not as converged.
Agreement between truncations is convergence, not correctness: each row also gives the largest residual of the
linear solves, and the known answers stay in hough_share_test.py (slipping = viscous on the exact spheroid).

Usage: python scripts/converge_resolution.py VARIANT DELTA_M [--K 24,36,48,72,96] [--N 30,40,50]
              [--band 12,16,20,24] [--mantle-K 6] [--mantle-N 16] [--solver layers] [--fluid-pressure] [--workers 1] [--max-gb 8]
              [--tag NAME] [--cache DIR]
  (--cache: directory of the stored solves, default logs/converge_cache; solves made before 2026-11-13 07:50
   there used the dense panels of solver-11: give a new directory to repeat them with the banded panels)
  (a ladder of one value keeps that parameter fixed)
  (--workers W: at most W solves at a time, fewer when W x the expected peak of a solve exceeds --max-gb)
  ->  results/converge_resolution_VARIANT_dDELTA[_TAG].txt"""
import hashlib
import json
import os
import resource
import sys
import time
from multiprocessing import Pool
from pathlib import Path

_argv, sys.argv = sys.argv, sys.argv[:1]         # hough_share_test reads its own options from argv
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import numpy as np
import hough_share_test as hs
from earthtides.geometry import OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit
sys.argv = _argv

KEYS = ('h0', 'h2', 'l1', 'l2')
FAR = (0.96, 0.99)
OFFSETS = (-10e-4, -6e-4, -4e-4, -2e-4, 3e-4, 6e-4)   # from the FCN (half width 1e-4 .. 4e-4 at delta 125 .. 500 m)
TOLP, TOLQ, TOLS = 0.01, 0.005, 1e-5             # period (sidereal days), Q (relative), shares
TOL0, TOL2 = 1e-6, 1e-8
CACHE = Path('logs/converge_cache')                # --cache DIR: another one (the key does not hold the solver version)


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


CACHE = Path(_opt('--cache', str(CACHE)))


def solve(job):
    """One solve in its own process: (key, Love numbers, residual, seconds, peak GB)."""
    variant, delta, s, K, N, band, solver = job[:7]
    mK, mN = job[7:] if len(job) > 7 else (6, 16)    # the default mantle is not in the key: old cache stays valid
    key = hashlib.sha1(json.dumps(job).encode()).hexdigest()[:16]
    f = CACHE / f'{key}.json'
    if f.exists():
        d = json.loads(f.read_text())
        return job, np.array(d['re']) + 1j * np.array(d['im']), d['res'], d['s'], d['gb'], True
    t = time.time()
    w = s * OMEGA_EARTH
    md = hs.model(variant)
    if delta > 0:
        md = md.viscous_core(w, delta=delta)
    g, g_fit, fl = hs.geometry(variant, md)
    Kl, Nl = resolution_per_layer(md, {'K': mK, 'N': mN, 'rules': [{'fluid': True, 'K': K, 'N': N}]})
    # solver 'layers+Q': fluid_pressure=True (Pi = lambda div u as an unknown in the fluid layers); the name
    # is in the cache key
    S = TidalSolver(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle', band=band if delta > 0 else None, linear_solver=solver.split('+')[0],
                    fluid_pressure=solver.endswith('+Q') and delta > 0)
    S.solve()
    fit = iers_fit(S, g_fit)
    L = np.array([complex(fit[k]) for k in KEYS])
    info = getattr(S, '_layer_info', None) or getattr(S, '_block_info', None) or {}
    res = float(info.get('residual', np.nan))
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == 'darwin' else 1e6)
    dt = time.time() - t
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(job=job, re=L.real.tolist(), im=L.imag.tolist(), res=res, s=dt, gb=gb,
                                 used=S._solver_used)))
    return job, L, res, dt, gb, False


def main():
    from provenance import Run
    variant, delta = sys.argv[1], float(sys.argv[2])
    ladders = {'band': [int(x) for x in _opt('--band', '12,16,20,24').split(',')],
               'N': [int(x) for x in _opt('--N', '30,40,50').split(',')],
               'K': [int(x) for x in _opt('--K', '24,36,48,72,96').split(',')],
               'mK': [int(x) for x in _opt('--mantle-K', '6').split(',')],
               'mN': [int(x) for x in _opt('--mantle-N', '16').split(',')]}
    order = ('band', 'N', 'K', 'mK', 'mN')
    solver, workers = _opt('--solver', 'layers'), int(_opt('--workers', 1))
    if '--fluid-pressure' in sys.argv:
        solver += '+Q'
    max_gb, tag = float(_opt('--max-gb', 8)), _opt('--tag', '')
    panels = 'banded' if TidalSolver.LAYERS_BANDED else 'dense'
    out = f'results/converge_resolution_{variant.replace("+", "_")}_d{delta:g}' + (f'_{tag}' if tag else '')
    with Run('convergence ladder in band, N and K of the viscous core, one parameter at a time',
             params=dict(variant=variant, delta=delta, ladders=ladders, solver=solver, panels=panels,
                         cache=str(CACHE), far=FAR, offsets=OFFSETS,
                         tol0=TOL0, tol2=TOL2, max_gb=max_gb), inputs=[]) as run:
        hs.model(variant, write=True)
        with Pool(workers, maxtasksperchild=1) as pool:
            # the FCN of the variant: pole of the one-pole fit to a coarse inviscid scan (K 6), started at the
            # largest jump of h0 (as hough_share_test.locate)
            R = pool.map(solve, [(variant, 0.0, float(x), 6, 30, None, 'block') for x in hs.SCAN], chunksize=1)
            Ls = np.array([r[1] for r in R])
            j = int(np.argmax(np.abs(np.diff(Ls[:, 0].real))))
            pole, _, _ = hs.fit(np.array(hs.SCAN), Ls, complex(0.5 * (hs.SCAN[j] + hs.SCAN[j + 1]), 1e-9))
            fcn = round(float(pole.real), 5)
            freqs = [float(x) for x in FAR] + [round(fcn + o, 5) for o in OFFSETS]
            lines = [f'{variant}, delta {delta:g} m, solver {solver} ({panels} panels), cache {CACHE}; FCN (inviscid scan) {fcn:.5f} cpsd; '
                     f'frequencies {" ".join(f"{x:.5f}" for x in freqs)}',
                     f'tolerance: |dh0| <= {TOL0:g}, |dh2|, |dl1|, |dl2| <= {TOL2:g}, times max(1, |h0 - h0 far| / 0.01)',
                     f'{"step":>4s} {"what":>5s} {"band":>4s} {"N":>3s} {"K":>3s}   {"max |dh0|":>9s} {"at":>8s} '
                     f'{"max |dh2|":>9s} {"max |dl1|":>9s} {"max |dl2|":>9s} {"worst/tol":>9s}  verdict     '
                     f'{"residual":>8s} {"s/solve":>7s} {"GB":>5s} cached']
            print(lines[0], flush=True)

            def point(p, gb_est=0.0):
                # as many workers as fit: workers x (expected peak of one solve) <= max_gb
                w = max(1, min(workers, int(max_gb // gb_est) if gb_est > 0 else 1))
                mant = () if (p['mK'], p['mN']) == (6, 16) else (p['mK'], p['mN'])
                jobs = [(variant, delta, x, p['K'], p['N'], p['band'], solver) + mant for x in freqs]
                with Pool(w, maxtasksperchild=1) as pl:
                    r = pl.map(solve, jobs, chunksize=1)
                return (np.array([q[1] for q in r]), max(q[2] for q in r), float(np.mean([q[3] for q in r])),
                        max(q[4] for q in r), all(q[5] for q in r))

            def resonance(L):
                """One-pole fit (hough_share_test.fit: cubic + FCN + 1/(s-1)) to all test frequencies:
                (period sd, Q, shares of h2, l1, l2 relative to h0, pole, misfit of h0)."""
                pole, c, mx = hs.fit(np.array(freqs), L, complex(fcn, 4e-4 * delta / 1e3))
                A = c[4]
                return (1 / (pole.real - 1), 1 / (2 * pole.imag), (A[1:] / A[0]).real, pole, mx[0])

            def row(step, what, p, r, extra, verdict, res, ts, gb, cached):
                per, Q, sh, pole, mx = r
                return (f'{step:4d} {what:>5s} {p["band"]:4d} {p["N"]:3d} {p["K"]:3d} {p["mK"]:3d} {p["mN"]:3d}  '
                        f'{per:8.3f} {Q:7.0f} '
                        + ' '.join(f'{v:+9.5f}' for v in sh) + f' {mx:8.1e}  {extra}  {verdict:10s} {res:8.1e} '
                        f'{ts:7.1f} {gb:5.1f} {"yes" if cached else "no"}')

            lines[1] = (f'far frequencies ({", ".join(f"{x:g}" for x in FAR)}): |dh0| <= {TOL0:g}, |dh2|, |dl1|, |dl2| <= '
                        f'{TOL2:g}; resonance (one-pole fit to all {len(freqs)} frequencies): period to {TOLP:g} sd, '
                        f'Q to {100 * TOLQ:g} %, shares to {TOLS:g}')
            lines[2] = (f'{"step":>4s} {"what":>5s} {"band":>4s} {"N":>3s} {"K":>3s} {"mK":>3s} {"mN":>3s}  '
                        f'{"FCN sd":>8s} {"Q":>7s} '
                        f'{"share h2":>9s} {"share l1":>9s} {"share l2":>9s} {"misfit":>8s}  {"far dh0":>8s} '
                        f'{"dFCN cpsd":>9s} {"dPer sd":>8s} {"dQ %":>6s} {"dshare":>8s}  verdict    {"residual":>8s} '
                        f'{"s/solve":>7s} {"GB":>5s} cached')
            print('\n'.join(lines[1:]), flush=True)
            blank = f'{"":8s} {"":9s} {"":8s} {"":6s} {"":8s}'
            nf = len(FAR)
            idx = {k: 0 for k in ladders}
            cur = {k: ladders[k][0] for k in ladders}
            L, res, ts, gb, cached = point(cur)
            R = resonance(L)
            lines.append(row(0, 'start', cur, R, blank, '', res, ts, gb, cached))
            print(lines[-1], flush=True)
            status = {k: '' for k in ladders}
            step, changed = 0, True
            while changed:
                changed = False
                for k in order:
                    while True:
                        if idx[k] + 1 >= len(ladders[k]):
                            status[k] = 'ladder end (not confirmed)'
                            break
                        nxt = dict(cur)
                        nxt[k] = ladders[k][idx[k] + 1]
                        grow = (nxt[k] / cur[k]) ** 2 if k in ('K', 'N') else 1.0    # the mantle blocks are small
                        if gb * grow > max_gb:
                            status[k] = f'budget ({k} {nxt[k]} needs ~{gb * grow:.0f} GB > {max_gb:g})'
                            break
                        L2, res2, ts2, gb2, c2 = point(nxt, gb * grow)
                        R2 = resonance(L2)
                        d = np.abs(L2[:nf] - L[:nf])
                        dfar = max((d[:, 0] / TOL0).max(), (d[:, 1:] / TOL2).max())
                        dper, dq, dsh = R2[0] - R[0], R2[1] / R[1] - 1, np.abs(R2[2] - R[2]).max()
                        ok = dfar <= 1 and abs(dper) <= TOLP and abs(dq) <= TOLQ and dsh <= TOLS
                        step += 1
                        extra = (f'{d[:, 0].max():8.1e} {R2[3].real - R[3].real:+9.1e} {dper:+8.3f} {100 * dq:+6.2f} '
                                 f'{dsh:8.1e}')
                        lines.append(row(step, k, nxt, R2, extra, 'no change' if ok else 'KEEP ' + k, res2, ts2, gb2, c2))
                        print(lines[-1], flush=True)
                        if ok:
                            status[k] = 'converged'
                            break
                        cur, L, R, res, ts, gb = nxt, L2, R2, res2, ts2, max(gb2, 1e-3)
                        idx[k] += 1
                        changed = True
            lines += ['', f'final point: band {cur["band"]}, N {cur["N"]}, K {cur["K"]}, mantle K {cur["mK"]} N {cur["mN"]}: FCN {R[0]:.3f} sd, Q {R[1]:.0f}, '
                          f'shares h2 {R[2][0]:+.5f} l1 {R[2][1]:+.5f} l2 {R[2][2]:+.5f}'] + \
                     [f'  {k:5s} {status[k]}' for k in order] + \
                     ['  Love numbers at the final point (real parts): ' + ', '.join(KEYS)] + \
                     [f'  {x:.5f}  ' + '  '.join(f'{v.real:+.8f}' for v in row_) for x, row_ in zip(freqs, L)]
            print('\n'.join(lines[-(8 + len(freqs)):]), flush=True)
        run.write_text(out + '.txt', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
