# -*- coding: utf-8 -*-
#
# The Love-number differences of the comparison tables (Tables S2, S4) as surface displacement in mm.
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
# 17/11/2026 Machiel Bos
#===============================================================================
"""How large are the differences between Love-number tables in millimetres of station displacement?

A difference dL in one parameter of one tide moves a station by dL times the amplitude of that tide's
potential over gravity times the latitude shape of the parameter (TN36 eqs. 7.5-7.13, as in
earthtides.etide.displacement). We take the largest wave of the Tamura catalogue with the tide's Doodson
numbers and the largest value over latitude: radial for h(0), h(2); the horizontal vector for l(0), l(2),
l(1). Differences in k change the potential, not the displacement directly, and are not converted.

Read from the tables of the paper: paper_pepi/table_ddw.tex (elastic, hydrostatic PREM; this paper vs DDW)
and paper_pepi/table_anelastic.tex (anelastic; this paper, the same with the IERS dispersion, DDW, IERS).

    .venv/bin/python scripts/displacement_scale.py   ->  results/displacement_scale.txt"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from earthtides import etide

TABLES = dict(elastic=ROOT / 'paper_pepi' / 'table_ddw.tex', anelastic=ROOT / 'paper_pepi' / 'table_anelastic.tex')
PARAMS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k')
# tide: (m, Doodson multipliers of the catalogue, first six)
TIDES = {'Mf': (0, (0, 2, 0, 0, 0, 0)), 'O1': (1, (1, -1, 0, 0, 0, 0)), 'P1': (1, (1, 1, -2, 0, 0, 0)),
         'K1': (1, (1, 1, 0, 0, 0, 0)), 'psi1': (1, (1, 1, 1, 0, 0, -1)), 'phi1': (1, (1, 1, 2, 0, 0, 0)),
         'OO1': (1, (1, 3, 0, 0, 0, 0)), 'M2': (2, (2, 0, 0, 0, 0, 0))}
NAMES = {'Mf': 'Mf', 'O$_1$': 'O1', 'P$_1$': 'P1', 'K$_1$': 'K1', r'$\psi_1$': 'psi1', r'$\phi_1$': 'phi1',
         'OO$_1$': 'OO1', 'M$_2$': 'M2'}


def num(s):
    s = s.strip().strip('$').replace('\\', '').strip()
    return float(s) if s else np.nan


def read_table(path, first_table_only=True):
    """{tide: {row label: {param: value}}} from the first tabular of a table file."""
    t = path.read_text()
    t = t[:t.index(r'\end{table')] if first_table_only else t
    out, tide = {}, None
    for line in t.split('\n'):
        if '&' not in line or not line.rstrip().endswith(r'\\') or 'tide &' in line:
            continue
        cells = [c.strip() for c in line.rstrip()[:-2].split('&')]
        if len(cells) != 8:
            continue
        if cells[0]:
            tide = NAMES.get(cells[0])
        if tide is None:
            continue
        out.setdefault(tide, {})[cells[1]] = dict(zip(PARAMS, (num(c) for c in cells[2:])))
    return out


def scales():
    """mm of displacement per unit of each parameter, per tide (largest over latitude)."""
    cat = etide.Tamura()
    phi = np.radians(np.linspace(-89.5, 89.5, 1791))
    sp, cp = np.sin(phi), np.cos(phi)
    P2 = 1.5 * sp**2 - 0.5
    res = {}
    for name, (m, dood) in TIDES.items():
        k = np.flatnonzero((cat.n == 2) & (cat.m == m) & np.all(cat.dood[:, :6] == dood, axis=1))
        A = np.abs(cat.A[k]).max()
        W, dW = etide.W_functions(2, m, phi)
        hor = np.hypot(dW, m * W / cp)
        s = dict(h0=np.abs(W).max(), h2=np.abs(W * P2).max(), l0=hor.max(), l2=(np.abs(P2) * hor).max())
        if m > 0:
            kappa = {1: 2 / 3, 2: 1 / 3}[m]
            Wn = etide.CV * etide.RE / (etide.GE / etide.RE**2) * kappa
            N1 = {1: -3 * sp**2, 2: -6 * sp * cp}[m]
            E1 = {1: -3 * sp * np.cos(2 * phi), 2: 6 * cp * sp**2}[m]
            s['l1'] = (Wn * np.hypot(N1, E1)).max()
        res[name] = {p: 1e3 * A * v for p, v in s.items()}
    return res


def compare(lines, title, tab, sc, a, b):
    lines += ['', f'{title}: {a} minus {b}, in mm (largest over latitude); Love-number difference in brackets']
    lines.append(f'{"tide":6s}' + ''.join(f'{p:>20s}' for p in PARAMS[:5]))
    worst = {}
    for tide in TIDES:
        if tide not in tab or a not in tab[tide] or b not in tab[tide]:
            continue
        row = f'{tide:6s}'
        for p in PARAMS[:5]:
            d = tab[tide][a][p] - tab[tide][b][p]
            if np.isnan(d) or p not in sc[tide]:
                row += f'{"":>20s}'
                continue
            mm = abs(d) * sc[tide][p]
            row += f'{mm:9.3f} ({d:+.5f})'
            key = 'resonance (psi1)' if tide == 'psi1' else 'away from the resonance'
            worst[key] = max(worst.get(key, (0, '')), (mm, f'{p} {tide}'))
        lines.append(row)
    for key, (mm, where) in worst.items():
        lines.append(f'   largest {key}: {mm:.3f} mm ({where})')
    return worst


def main(run):
    sc = scales()
    lines = ['Displacement per unit Love number (mm), largest wave of the tide in the Tamura catalogue, '
             'largest over latitude',
             f'{"tide":6s}' + ''.join(f'{p:>10s}' for p in PARAMS[:5])]
    for tide, s in sc.items():
        lines.append(f'{tide:6s}' + ''.join(f'{s[p]:10.1f}' if p in s else f'{"":>10s}' for p in PARAMS[:5]))
    lines.append('  (so 1e-4 in h0 at M2 is ' f'{1e-4 * sc["M2"]["h0"]:.3f} mm, at K1 {1e-4 * sc["K1"]["h0"]:.3f} mm)')
    el = read_table(TABLES['elastic'])
    an = read_table(TABLES['anelastic'])
    compare(lines, 'Elastic, hydrostatic PREM (Table S2)', el, sc, 'this paper', 'DDW')
    compare(lines, 'Anelastic (Table S4)', an, sc, 'IERS dispersion', 'IERS')
    compare(lines, 'Anelastic (Table S4)', an, sc, 'this paper', 'IERS')
    compare(lines, 'Anelastic (Table S4)', an, sc, 'this paper', 'IERS dispersion')
    compare(lines, 'Anelastic (Table S4)', an, sc, 'this paper', 'DDW')
    run.write_text(str(ROOT / 'results' / 'displacement_scale.txt'), '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    from provenance import Run
    with Run('Love-number differences of Tables S2 and S4 as station displacement in mm',
             inputs=[str(p.relative_to(ROOT)) for p in TABLES.values()] + ['src/earthtides/data/Tamura.dat']) as run_:
        main(run_)
