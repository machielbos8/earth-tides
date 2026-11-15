# -*- coding: utf-8 -*-
#
# Slipping core against the K-converged viscous limit on the level figure, at the table frequencies.
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
# 15/11/2026 Machiel Bos
#===============================================================================
"""Paper 1, Sect. 5.3 / Discussion / Online Resource 'The limit delta -> 0' (lab notes 2026-11-15):
the slipping core (earthtides.love, elastic PREM, level figure, slip_nonlevel default, core K 6 and 10) against
the viscous limit of viscous_limit_thin.py (results/viscous_limit_thin_Kthin_level.npz, core K 72, displacement
formulation, band 12). Away from the resonance (O1, P1, phi1, OO1) the viscous limit is the quadratic
extrapolation in delta of the Love numbers themselves (the pole fit of that script, evaluated far from the pole,
is off by 3e-5 in h0 at O1); K1 and psi1 come from the pole fit (window 500/350/250 m, in the .txt).

    .venv/bin/python scripts/slip_vs_viscous_level.py  ->  results/slip_vs_viscous_level.txt"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from earthtides import love
from provenance import Run

SRC = 'results/viscous_limit_thin_Kthin_level'
TIDES = (('O1', 0.927), ('P1', 0.99454), ('K1', 1.0), ('psi1', 1.00273), ('phi1', 1.00546), ('OO1', 1.073))
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2')            # k: love.parameters normalises it otherwise; not compared
WINDOW = (500.0, 350.0, 250.0)


def lagrange0(vals, ds):
    w = [np.prod([-b / (a - b) for b in ds if b != a]) for a in ds]
    return sum(wi * vals[a] for wi, a in zip(w, ds))


def pole_fit_rows(path, Kc=72):
    """'TIDE key' -> value of the window 500/350/250 for core K Kc from the viscous_limit_thin table."""
    out, inK = {}, False
    for line in Path(path).read_text().splitlines():
        if line.startswith('== core K'):
            inK = line.split()[-1] == str(Kc)
        elif inK and line.startswith('window'):
            col = line.split()[1:].index('/'.join('%g' % d for d in WINDOW))
        elif inK and line.split()[:1] and line.split()[0] in [t for t, _ in TIDES] + ['FCN']:
            p = line.split()
            name = 'FCN' if p[0] == 'FCN' else p[0] + ' ' + p[1]
            out[name] = float(p[-4:][col] if p[0] != 'FCN' else p[3:][col])
    return out


def main():
    with Run('slipping core against the K-converged viscous limit on the level figure (table frequencies)',
             params=dict(K_slip=[6, 10], window=WINDOW, K_viscous=72), inputs=[SRC + '.npz', SRC + '.txt']) as run:
        d = np.load(ROOT / (SRC + '.npz'))
        s = d['sweep']
        pf = pole_fit_rows(ROOT / (SRC + '.txt'))
        lines = ['elastic PREM, level figure; slipping core (love module, slip_nonlevel default) K 6 / K 10 against the '
                 'viscous limit (core K 72, %s); away from the resonance: direct extrapolation in delta, K1 and psi1: '
                 'pole fit' % '/'.join('%g' % x for x in WINDOW), '',
                 '%-6s %-4s %12s %12s %12s %10s %10s' % ('tide', 'key', 'slip K6', 'slip K10', 'viscous', 'K6-visc', 'K10-visc')]
        slip = {K: love.Options(anelastic=False, K=K) for K in (6, 10)}
        for name, f in TIDES:
            P = {K: love._job((o, 2, 1, f * love.F_TILT)) for K, o in slip.items()}
            i = int(np.argmin(abs(s - f)))
            for j, k in enumerate(KEYS):
                if name in ('K1', 'psi1'):
                    v = pf['%s %s' % (name, k)]
                else:
                    v = lagrange0({x: d['L_K72_d%g' % x][i, j].real for x in WINDOW}, WINDOW)
                a, b = complex(P[6][k]).real, complex(P[10][k]).real
                lines.append('%-6s %-4s %12.6f %12.6f %12.6f %+10.1e %+10.1e' % (name, k, a, b, v, a - v, b - v))
        lines += ['', 'viscous FCN period (pole fit, same window): %.3f sd' % pf['FCN']]
        run.write_text('results/slip_vs_viscous_level.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
