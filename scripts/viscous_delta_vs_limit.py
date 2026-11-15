# -*- coding: utf-8 -*-
#
# Love numbers of the viscous core at finite delta against the limit delta -> 0 (P1 v2, Sect. 5.3:
# how far is a realistic boundary-layer thickness from the reference?).
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
# 02/11/2026 Machiel Bos
#===============================================================================
"""No new solves: reads the diurnal sweeps at delta = 2, 1, 0.75, 0.5, 0.35 km of
scripts/fcn_resonance_extrapolation_v3.py (core K 12 / N 30, mantle and inner core K 6 / N 24), fits the
FCN + FICN poles and a smooth background at each delta (same fit) and evaluates every tide of the DDW
table at that delta, against the quadratic extrapolation to delta -> 0. Also the FCN at each delta: period
(sidereal days) and the quality factor Q = 1 / (2 Im sigma) with respect to the nearly diurnal frequency in
the terrestrial frame, the convention of the VLBI value Q ~ 2e4 (Mathews et al. 2002).
Usage: python scripts/viscous_delta_vs_limit.py [KCORE]  ->  results/viscous_delta_vs_limit_K<KCORE>.txt"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np

from fcn_resonance_extrapolation import KEYS, TABLE, fit, evaluate
from fcn_resonance_extrapolation_v3 import lagrange0
from provenance import Run


def main_thin(tag, Kc=72, window=(500.0, 350.0, 250.0)):
    """--thin TAG: the K-converged sweeps of viscous_limit_thin.py (results/viscous_limit_thin_TAG_level.npz, deltas in
    m, core K 72); the limit from the window 500/350/250 m. The fit, evaluated far from the pole, is off by 3e-5 at O1
    (lab notes 2026-11-15): tides on the sweep grid also get the direct difference (Love number at delta minus its
    quadratic extrapolation in delta through the window)."""
    src = f'results/viscous_limit_thin_{tag}_level.npz'
    with Run(f'Love numbers at finite delta vs delta -> 0 (core K {Kc}, {src})',
             params=dict(K_core=Kc, window=window), inputs=[src]) as run_:
        p = np.load(src)
        s, deltas = p['sweep'], [float(d) for d in p['deltas'] if float(d) >= min(window)]
        L = {d: p[f'L_K{Kc}_d{d:g}'] for d in deltas}
        P = {}
        for d in deltas:
            sf, c, _ = fit(s, L[d], d)
            P[d] = (np.array(sf), c)
        sf0, c0 = lagrange0(P, window)
        lines = [f'viscous core, core K {Kc}, level figure; fitted sweeps of {src}; limit from {window} m',
                 'FCN at each delta: period (sidereal days), Q = 1 / (2 Im sigma), sigma in cycles per sidereal day']
        for d in deltas + [0.0]:
            sf = P[d][0] if d else sf0
            q = 1 / (2 * sf[0].imag) if d else np.inf
            lines.append(f'  delta {d / 1e3:5.3f} km: {1 / (sf[0].real - 1):7.2f} sd, Im sigma {sf[0].imag:+.2e}, '
                         f'Q {q:8.0f}')
        lines.append('Love numbers at delta minus the limit delta -> 0 (real parts; fit, and [direct] on the sweep grid)')
        lines.append(f'{"tide":6s} {"delta":>6s} ' + ''.join(f'{k:>24s}' for k in KEYS))
        for name, x in TABLE.items():
            v0 = evaluate(sf0, c0, x)
            i = int(np.argmin(abs(s - x)))
            on = abs(s[i] - x) < 1e-6
            if on:
                w = [np.prod([-b / (a - b) for b in window if b != a]) for a in window]
                d0 = sum(wi * L[a][i] for wi, a in zip(w, window))
            lines.append(f'{name:6s} {"->0":>6s} ' + ''.join(f'{z.real:+24.6f}' for z in v0))
            for d in deltas:
                v = evaluate(P[d][0], P[d][1], x)
                lines.append(f'{"":6s} {d / 1e3:6.3f} ' + ''.join(
                    f'{(a - b).real:+12.1e}' + (f' [{(L[d][i][j] - d0[j]).real:+.1e}]' if on else ' ' * 10)
                    for j, (a, b) in enumerate(zip(v, v0))))
        txt = '\n'.join(lines)
        print(txt)
        run_.write_text(f'results/viscous_delta_vs_limit_{tag}_level.txt', txt + '\n')


def main():
    if '--thin' in sys.argv:
        return main_thin(sys.argv[sys.argv.index('--thin') + 1])
    Kc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    src = f'results/fcn_resonance_extrapolation_v3_K{Kc}.npz'
    with Run(f'Love numbers at finite delta vs delta -> 0 (core K {Kc}), from the stored sweeps',
             params=dict(K_core=Kc), inputs=[src]) as run_:
        p = np.load(src)
        s, deltas, L = p['sweep'], [float(d) for d in p['deltas']], p['L']
        P = {}
        for j, d in enumerate(deltas):
            sf, c, _ = fit(s, L[j], d)
            P[d] = (np.array(sf), c)
        sf0, c0 = lagrange0(P, (3.5e2, 5e2, 7.5e2))
        lines = [f'viscous core, core K {Kc} N 30, mantle / inner core K 6 N 24; fitted sweeps of {src}',
                 'FCN at each delta: period (sidereal days), Q = 1 / (2 Im sigma), sigma in cycles per sidereal day']
        for d in deltas + [0.0]:
            sf = P[d][0] if d else sf0
            q = 1 / (2 * sf[0].imag) if d else np.inf
            lines.append(f'  delta {d / 1e3:5.2f} km: {1 / (sf[0].real - 1):6.1f} sd, Im sigma {sf[0].imag:+.2e}, '
                         f'Q {q:8.0f}')
        lines.append('Love numbers at delta minus the limit delta -> 0 (real parts; imaginary parts in brackets)')
        lines.append(f'{"tide":6s} {"delta":>6s} ' + ''.join(f'{k:>20s}' for k in KEYS))
        for name, x in TABLE.items():
            v0 = evaluate(sf0, c0, x)
            lines.append(f'{name:6s} {"->0":>6s} ' + ''.join(f'{z.real:+20.6f}' for z in v0))
            for d in deltas:
                v = evaluate(P[d][0], P[d][1], x)
                lines.append(f'{"":6s} {d / 1e3:6.2f} ' + ''.join(
                    f'{(a - b).real:+10.1e} ({(a - b).imag:+.0e})' for a, b in zip(v, v0)))
        txt = '\n'.join(lines)
        print(txt)
        run_.write_text(f'results/viscous_delta_vs_limit_K{Kc}.txt', txt + '\n')


if __name__ == '__main__':
    main()
