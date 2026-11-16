# -*- coding: utf-8 -*-
#
# Table 4 of paper 1 (PEPI): Love numbers of hydrostatic and non-hydrostatic PREM.
#
#
# This file is part of earth-tides 1.0.
#
# earth-tides is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 17/10/2026 Machiel Bos
#===============================================================================
"""Table 4 of paper 1: Love numbers of anelastic PREM (IERS conventions, earthtides.love) for the
hydrostatic Earth and two non-hydrostatic Earths with elastic support and the excess J2 of Chambat
et al. (2010): the excess e_f of Mathews et al. (2002) (FCN 438 sd here) and e_f scaled so that the FCN
lies at 431.4 sd (VLBI). Hydrostatic values and the differences non-hydrostatic minus hydrostatic.
Online Resource: the same for viscous support.

    .venv/bin/python paper_pepi/make_table4.py   ->  paper_pepi/table4.tex, tableS_viscous.tex,
                                                     results/paper1_table4.txt
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
# tex tables go to results/regen/ (for comparison) unless --paper: the paper's tables may carry
# values from other runs (lab notes 2026-11-08)
OUT = ROOT / 'paper_pepi' if '--paper' in sys.argv else ROOT / 'results' / 'regen'
# --figure level: on the level figure (love.Options(figure=...)); outputs get the suffix _level
FIGURE = sys.argv[sys.argv.index('--figure') + 1] if '--figure' in sys.argv else 'first-order'   # or 'level'
SUF = '' if FIGURE == 'first-order' else '_' + FIGURE
warnings.filterwarnings('ignore')
import numpy as np

from earthtides import love

WAVES = (('Mf', 0, 0.07320), ('O$_1$', 1, 0.92954), ('P$_1$', 1, 0.99726), ('K$_1$', 1, 1.00274),
         (r'$\psi_1$', 1, 1.005475688), (r'$\phi_1$', 1, 1.008213728), ('OO$_1$', 1, 1.07594),
         ('M$_2$', 2, 1.93227))
KEYS = (('h0', '$h^{(0)}$'), ('h2', '$h^{(2)}$'), ('l0', '$l^{(0)}$'), ('l1', '$l^{(1)}$'), ('k', '$k$'))
J2 = 1.14e-5
E_MHB = 9.73e-5


def fcn(opts, sig=(1.0012, 1.0016, 1.0030, 1.0036)):
    x = np.array(sig)
    k = np.array([love.solve_wave(opts, 1, s * love.F_TILT)[1]['kk'] for s in x]).real
    return np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0]


def table(opts):
    return {name: love._job((opts, 2, m, f)) for name, m, f in WAVES}


def latex(rows, cases, caption, label):
    """rows: hydrostatic table; cases: [(title, table)] shown as differences in 1e-4."""
    nc = len(KEYS)
    head = ' & '.join(k for _, k in KEYS)
    cols = 'l' + ('r' * nc) * (1 + len(cases))
    out = [r'\begin{table*}', r'\caption{' + caption + '}', r'\label{' + label + '}', r'\small',
           r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{@{}' + cols + '@{}}', r'\toprule',
           ' & ' + r'\multicolumn{%d}{c}{hydrostatic}' % nc
           + ''.join(r' & \multicolumn{%d}{c}{%s}' % (nc, t) for t, _ in cases) + r' \\',
           ''.join(r'\cmidrule(lr){%d-%d}' % (2 + j * nc, 1 + (j + 1) * nc) for j in range(1 + len(cases))),
           ' & ' + ' & '.join([head] * (1 + len(cases))) + r' \\', r'\midrule']
    for name, _, _ in WAVES:
        h = rows[name]
        vals = [f'{h[k].real:.5f}' if k in h else '' for k, _ in KEYS]
        for _, t in cases:
            vals += [f'{1e4 * (t[name][k].real - h[k].real):+.1f}' if k in h else '' for k, _ in KEYS]
        out.append(name + ' & ' + ' & '.join(vals) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}']
    return '\n'.join(out) + '\n'


def tex(x):
    m, e = f'{x:.2e}'.split('e')
    return rf'{m}\times10^{{{int(e)}}}'


def main(run):
    hyd = love.Options(figure=FIGURE)
    c0 = fcn(hyd)
    m1 = love.Options(figure=FIGURE, nonhydrostatic=dict(targets={'J2': J2, 'e_f': E_MHB}))
    c1 = fcn(m1)
    e431 = float(E_MHB * (1 / 431.4 - (c0 - 1)) / ((c1 - 1) - (c0 - 1)))
    m2 = love.Options(figure=FIGURE, nonhydrostatic=dict(targets={'J2': J2, 'e_f': e431}))
    c2 = fcn(m2)
    v1 = love.Options(figure=FIGURE, nonhydrostatic=dict(support='viscous', targets={'J2': J2, 'e_f': E_MHB}))
    c3 = fcn(v1)
    T0, T1, T2, T3 = table(hyd), table(m1), table(m2), table(v1)
    cmb = {n: love._model(o)[2].summary()['response'] for n, o in (('m1', m1), ('m2', m2), ('v1', v1))}
    p = lambda c: 1 / (c - 1)
    cap = (r'Love numbers of anelastic PREM (IERS conventions; real parts): hydrostatic, and the '
           r'differences (in $10^{-4}$) for two non-hydrostatic Earths with elastic support and the excess '
           r'$J_2$ of \citet{chambat2010}: the excess dynamical ellipticity of the core of '
           r'\citet[][MHB]{mathews2002} ($\Delta e_f=%.2e$, %.0f~m at the core--mantle boundary; FCN %.1f sidereal '
           r'days) and $\Delta e_f=%.2e$ (%.0f~m; FCN %.1f days, as observed by VLBI). Hydrostatic FCN %.1f days. '
           r'Inviscid core that slips at its boundaries, $K=4$, with the term of eq.~(\ref{M-eq:slipterm})'
           % (0, cmb['m1']['cmb'], p(c1), 0, cmb['m2']['cmb'], p(c2), p(c0)))
    cap = cap.replace('%.2e' % 0, tex(E_MHB), 1).replace('%.2e' % 0, tex(e431), 1)
    cap += '; level figure.' if FIGURE == 'level' else '; first-order figure.'
    run.write_text(str(OUT / f'table4{SUF}.tex'), 
        latex(T0, [('$\\Delta e_f$ of MHB', T1), ('FCN at 431 days', T2)], cap, 'tab:nonhydro_love'))
    capS = (r'As Table~\ref{tab:nonhydro_love} for viscous support (relative viscosities 50 and 1 in the lower and upper mantle, '
            r'80-km elastic lid; $\Delta e_f$ of \citet{mathews2002}; FCN %.1f sidereal days; %.0f~m of zonal '
            r'surface topography): differences from the hydrostatic Earth in $10^{-4}$, compared with elastic '
            r'support.' % (p(c3), cmb['v1']['surface']))
    run.write_text(str(OUT / f'tableS_viscous{SUF}.tex'), 
        latex(T0, [('elastic support', T1), ('viscous support', T3)], capS, 'tab:S_viscous_love'))
    lines = [f'FCN (sd): hydrostatic {p(c0):.1f}, MHB e_f {p(c1):.1f}, e_f {e431:.3e} {p(c2):.1f}, '
             f'viscous MHB e_f {p(c3):.1f}', f'targets reached: {cmb}', '']
    lines.append('wave   ' + '  '.join(f'{k:>9s}' for k, _ in KEYS) + '   | diffs 1e-4: MHB e_f | FCN 431 | viscous')
    for name, _, _ in WAVES:
        h = T0[name]
        r = f'{name:8s}' + ' '.join(f'{h[k].real:9.5f}' if k in h else ' ' * 9 for k, _ in KEYS)
        for T in (T1, T2, T3):
            r += ' |' + ' '.join(f'{1e4 * (T[name][k].real - h[k].real):+6.1f}' if k in h else ' ' * 6
                                 for k, _ in KEYS)
        lines.append(r)
    run.write_text(str(ROOT / 'results' / f'paper1_table4{SUF}.txt'), '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'scripts'))
    from provenance import Run
    with Run('Table of non-hydrostatic Love numbers (P1 v2 Table S12, S8)', params=dict(out=str(OUT)),
             inputs=['data/PREM.dat']) as run_:
        main(run_)
