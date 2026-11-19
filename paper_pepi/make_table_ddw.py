# -*- coding: utf-8 -*-
#
# Table S3 of the PEPI supplement (tab:ddw): this paper against Dehant et al. (1999), from stored results.
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
# 19/11/2026 Machiel Bos
#===============================================================================
"""Table S3 of the supplement (tab:ddw): Love numbers of elastic, hydrostatic PREM on the first-order figure,
viscous core in the limit delta -> 0, against Dehant et al. (1999, tables 11 and 12). Assembles two stored results
(2026-11-19; the table was assembled by hand before):
  diurnal lines  results/fcn_resonance_extrapolation_v3_K12.txt  (poles and coefficients extrapolated to delta -> 0;
                 job fcn_extrapolation_fo, heavy)
  Mf and M2      results/paper1_tables_viscous.txt  (direct extrapolation from delta 1 and 0.5 km; job tables_viscous_fo)
The DDW values are those of paper_pepi/make_tables_viscous.py (DDW). Read by make_fig_ddw_iers.py and
displacement_scale.py.
Run from the project root:  .venv/bin/python paper_pepi/make_table_ddw.py [--paper]
-> results/regen/table_ddw.tex (with --paper also paper_pepi/table_ddw.tex)"""
import re
import sys
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'paper_pepi'))
from provenance import Run

POLE = 'results/fcn_resonance_extrapolation_v3_K12.txt'
DIRECT = 'results/paper1_tables_viscous.txt'
# DDW tables 11, 12: h0 h2 l0 l1 l2 k0 (Mf from table 11, no l1)
DDW = {'Mf': (0.59892, -0.00047, 0.08304, None, 0.00008, 0.29473),
       'O$_1$': (0.59553, -0.00048, 0.08298, 0.00070, 0.00008, 0.29335),
       'P$_1$': (0.57411, -0.00053, 0.08367, 0.00070, 0.00009, 0.28249),
       'K$_1$': (0.51400, -0.00065, 0.08558, 0.00060, 0.00011, 0.25205),
       '$\\psi_1$': (0.93394, 0.00024, 0.07227, 0.00130, -0.00004, 0.46477),
       '$\\phi_1$': (0.65422, -0.00036, 0.08113, 0.00084, 0.00006, 0.32308),
       'OO$_1$': (0.60092, -0.00047, 0.08283, 0.00077, 0.00008, 0.29607),
       'M$_2$': (0.60175, -0.00047, 0.08359, 0.00100, 0.00008, 0.29751)}
POLE_NAME = {'O$_1$': 'O1', 'P$_1$': 'P1', 'K$_1$': 'K1', '$\\psi_1$': 'psi1', '$\\phi_1$': 'phi1', 'OO$_1$': 'OO1'}
DIRECT_NAME = {'Mf': 'zonal', 'M$_2$': 'M$_2$'}
CAPTION = r"""\begin{table*}
\caption{Degree-2 Love numbers of elastic, hydrostatic PREM in the definition
of \citet{mathews1995}: this paper on the first-order figure and \citet[][DDW,
tables~11 and 12]{dehant1999}. Viscous core in the limit $\delta\to0$
(Section~\ref{M-sec:wayout}; extrapolation uncertainty in Sect.~\ref{sec:corenumbers}; cost of the
first-order figure near the resonance in Table~\ref{tab:levelfigure}). The small terms
$h'$ and $l'$, which DDW give in other normalizations and, for $l'$, another frame, are compared
below. Last block: the nominal latitude terms of the Conventions \citep[][Sect.~7.1.1]{petit2010}, used for
every line of a band (in the diurnal band their resonance formula makes them vary, Table~\ref{tab:ddw_anelastic}); their $h^{(0)}$, $l^{(0)}$ and $k$ include anelasticity and, in the diurnal band, ocean
loading, and are not comparable with the elastic values above.}
\label{tab:ddw}
\small\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}
\toprule
tide & & $h^{(0)}$ & $h^{(2)}$ & $l^{(0)}$ & $l^{(1)}$ & $l^{(2)}$ & $k$ \\
"""


def pole_rows(text):
    """'delta->0' rows of the pole fit: tide -> six values as printed (5 decimals)."""
    out = {}
    for line in text.splitlines():
        m = re.match(r'(\S+)\s+delta->0\s+(.*)$', line)
        if m:
            out[m.group(1)] = [Decimal(v) for v in m.group(2).split()]
    return out


def direct_rows(text):
    """'value' rows of the direct extrapolation (6 decimals), keyed by the tide label of that file."""
    out, tide = {}, None
    for line in text.splitlines():
        m = re.match(r'(\S+)\s+value\s+(.*)$', line)
        if m:
            out[m.group(1)] = [Decimal(v) for v in m.group(2).split()]
    return out


def fmt(v):
    if v is None:
        return ''
    v = Decimal(v).quantize(Decimal('0.00001'), rounding=ROUND_HALF_EVEN)
    if v == 0:
        v = abs(v)
    return f'${v}$'.replace('$-', '$-') if v < 0 else f'{v}'


def main():
    paper = '--paper' in sys.argv
    with Run('Table S3 (tab:ddw) assembled from stored results', params=dict(paper=paper),
             inputs=[POLE, DIRECT]) as run:
        pole, direct = pole_rows(open(ROOT / POLE).read()), direct_rows(open(ROOT / DIRECT).read())
        rows = []
        for tide, ddw in DDW.items():
            ours = pole[POLE_NAME[tide]] if tide in POLE_NAME else direct[DIRECT_NAME[tide]]
            rows.append(f'{tide} & this paper & ' + ' & '.join(fmt(v) for v in ours) + ' \\\\')
            rows.append((' & DDW & ' + ' & '.join(fmt(None if v is None else Decimal(str(v))) for v in ddw)).replace('&  &', '& &') + ' \\\\')
        tex = ('% Table S3 of the supplement (written by paper_pepi/make_table_ddw.py from stored results).\n'
               '% Read by make_fig_ddw_iers.py (Fig. S1) and scripts/displacement_scale.py.\n'
               + CAPTION + '\\midrule\n' + '\n'.join(rows) + '\n' + r"""\midrule
\multicolumn{8}{@{}l}{IERS Conventions, nominal values}\\
long period & & & $-0.0006$ & & & 0.0002 & \\
diurnal & & & $-0.0006$ & & 0.0012 & 0.0002 & \\
semidiurnal & & & $-0.0006$ & & 0.0024 & 0.0002 & \\
\bottomrule
\end{tabular*}
\end{table*}
""")
        run.write_text('results/regen/table_ddw.tex', tex)
        if paper:
            run.write_text('paper_pepi/table_ddw.tex', tex)
        print(tex)


if __name__ == '__main__':
    main()
