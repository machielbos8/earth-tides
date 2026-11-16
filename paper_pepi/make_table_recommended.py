# -*- coding: utf-8 -*-
#
# Table of the recommended Love numbers (supplement S1): an excerpt of the stored all-waves table.
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
# 16/11/2026 Machiel Bos
#===============================================================================
"""The recommended degree-2 Love numbers of the paper for the main tides of each band, read from the stored
table of the package (src/earthtides/data/love_PREM_allwaves.json: anelastic PREM, level figure,
non-hydrostatic with the free core nutation at 431.4 sidereal days; tools/stored_tables/love_PREM.toml),
so the paper and the package give the same numbers.

    .venv/bin/python paper_pepi/make_table_recommended.py   ->  paper_pepi/table_recommended.tex"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from provenance import Run
import earthtides

# the stored table of the package in use (the installed one under run_all.py --installed, which has no src/)
TABLE = Path(earthtides.__file__).resolve().parent / 'data' / 'love_PREM_allwaves.json'
OUT = ROOT / 'paper_pepi' / 'table_recommended.tex'
# name, order m, frequency [cycles per day]
TIDES = (('Mm', 0, 0.0362916), ('Mf', 0, 0.0732022), ('Q$_1$', 1, 0.8932441), ('O$_1$', 1, 0.9295357),
         ('P$_1$', 1, 0.9972621), ('K$_1$', 1, 1.0027379), (r'$\psi_1$', 1, 1.0054758),
         (r'$\phi_1$', 1, 1.0082137), ('OO$_1$', 1, 1.0759401), ('N$_2$', 2, 1.8959820),
         ('M$_2$', 2, 1.9322736), ('S$_2$', 2, 2.0000000), ('K$_2$', 2, 2.0054758))
KEYS = (('h0', '$h^{(0)}$'), ('h2', '$h^{(2)}$'), ('l0', '$l^{(0)}$'), ('l1', '$l^{(1)}$'), ('l2', '$l^{(2)}$'),
        ('k', '$k$'), ('lp', "$l'$"), ('hp', "$h'$"))


def find(entries, m, f):
    hits = sorted((abs(e['f_cpd'] - f), i) for i, e in enumerate(entries) if e['n'] == 2 and e['m'] == m)
    if hits[0][0] > 2e-7:                    # the tide itself (nearest line; P1 has a sideband 2.6e-7 cpd away)
        raise ValueError(f'm={m} f={f}: nearest line {hits[0][0]:.1e} cpd away')
    return entries[hits[0][1]]


def fmt(x):
    if abs(x) < 5e-6:
        return '0.00000'
    s = f'{x:+.5f}'
    return '$-$' + s[1:] if s.startswith('-') else s[1:]


def main(run):
    t = json.loads(TABLE.read_text())
    md = t['metadata']
    rows = []
    for name, m, f in TIDES:
        e = find(t['entries'], m, f)
        re_ = [fmt(e[k][0]) if k in e else '' for k, _ in KEYS]
        im_ = [fmt(e[k][1]) if k in e else '' for k, _ in KEYS]
        rows.append(f'{name} & {e["f_cpd"]:.6f} & Re & ' + ' & '.join(re_) + r' \\')
        rows.append(r' & & Im & ' + ' & '.join(im_) + r' \\')
    cap = (r'Recommended degree-2 Love numbers (definition of \citet{mathews1995}, conventions of '
           r'Section~\ref{M-sec:love}; $h$, $l$ at the equatorial radius, $k$ at the mean radius): anelastic PREM '
           r'(Section~\ref{M-sec:anelastic}) on the level figure, non-hydrostatic with the excess $J_2$ of '
           r'\citet{chambat2010} and the excess dynamical ellipticity of the core that puts the free core nutation '
           r'at the observed 431.4 sidereal days ($\Delta e_f/e_f=4.86$ per cent), slipping core with '
           r'eq.~(\ref{M-eq:slipterm}). Frequencies in cycles per day. An excerpt of the table of every wave of '
           r'the Tamura (1987) catalogue stored with the software (data/love\_PREM\_allwaves.json, version '
           + md['version'] + r'). Next to the resonance the values depend steeply on the frequency: at $\psi_1$ '
           r'$h^{(0)}$ changes by $1\times10^{-4}$ per $10^{-7}$ cycles per day.')
    cols = 'lrl' + 'r' * len(KEYS)
    out = [r'\begin{table*}', r'\caption{' + cap + '}', r'\label{tab:recommended}', r'\small',
           r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}' + cols + '@{}}', r'\toprule',
           r'tide & frequency & & ' + ' & '.join(k for _, k in KEYS) + r' \\', r'\midrule']
    for i, r in enumerate(rows):
        out.append(r)
        if i % 2 and i < len(rows) - 1 and TIDES[i // 2][1] != TIDES[i // 2 + 1][1]:
            out.append(r'\midrule')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    run.write_text(str(OUT), '\n'.join(out) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    with Run('Supplement S1: table of the recommended Love numbers (excerpt of love_PREM_allwaves.json)',
             inputs=[str(TABLE.relative_to(ROOT)) if TABLE.is_relative_to(ROOT) else str(TABLE)]) as run_:
        main(run_)
