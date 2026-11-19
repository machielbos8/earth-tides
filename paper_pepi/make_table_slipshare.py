# -*- coding: utf-8 -*-
#
# Table S7 of the PEPI supplement (tab:slipshare): resonant shares of h(2), slipping and viscous core, from stored results.
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
"""Table S7 of the supplement (tab:slipshare): resonant share of h(2) (resonant part of h(2) relative to that of
h(0)) for a slipping core without and with the slip term of eq. (slipterm), and for the viscous core. Assembled by
READING stored results (2026-11-19; the table was assembled by hand before); no solver is run.
Every cell is located by a regular expression that must match exactly one line of its source file; the value is the
h(2) share printed on that line (full printed precision). Cells without a stored source are left empty and listed.
Inviscid rows: the highest core K of the file (K 10 where a file gives K 6 and K 10).
Run from the project root:  .venv/bin/python paper_pepi/make_table_slipshare.py
-> results/table_slipshare.txt (every cell: value, source file, source line, rounding check against the current
   table) and results/regen/table_slipshare.tex (tabular body, 5 decimals; the printed format is Machiel's call)"""
import re
import sys
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from provenance import Run

NUM = r'([+-]\d+\.\d+)'
SKIP = r'\s+\S+'
COLS = ('without', 'with', 'viscous')
ROWS = ('PREM', r'\quad level figure', r'\quad without inner core', "Hough's Earth, exact spheroid",
        r'\quad first-order map', r'\quad exact spheroid with bump')
# the values printed in supplement_v4.tex (tab:slipshare) on 2026-11-19, for the rounding check only
CURRENT = {
    'PREM': ('+0.0040', '+0.0014', '+0.0015'),
    r'\quad level figure': ('+0.0014', '+0.0014', '+0.0014'),
    r'\quad without inner core': ('+0.0040', '+0.0014', '+0.0015'),
    "Hough's Earth, exact spheroid": ('+0.0020', '+0.0020', '+0.0020'),
    r'\quad first-order map': ('+0.0060', '+0.0020', '+0.0020'),
    r'\quad exact spheroid with bump': ('-0.0324', '+0.0019', '+0.0019'),
}
F_PREM_INV = 'results/fcn_share_inviscid.txt'
F_PREM_TERM = 'results/exact_levels_share_test_slipterm_fullPREM.txt'
F_PREM_VISC = 'results/fcn_resonance_extrapolation_v3_K12.txt'
F_LEVEL_VISC = 'results/fcn_resonance_extrapolation_v3_K12_level.txt'
F_NOIC_INV = 'results/fcn_share_inviscid_noic.txt'
F_NOIC_TERM = 'results/exact_levels_share_test_slipterm.txt'
F_NOIC_VISC = 'results/fcn_resonance_noic_K12.txt'
F_HOUGH = 'results/hough_share_test.txt'
F_HOUGH_SOFT = 'results/hough_share_test_bump_d250_K72.txt'
F_HOUGH_TERM = 'results/hough_share_test_slipterm.txt'
F_HOUGH_BUMP = 'results/hough_share_test_bump.txt'
F_SOFT_VISC = 'results/converge_resolution_soft_d250_vp1000_mixed.txt'
F_BUMP_VISC = 'results/converge_resolution_bump_3_d250_vp1000_mixed.txt'
LADDER_K48 = r'^\s*4\s+K\s+16\s+60\s+48\s+6\s+16' + SKIP * 2 + r'\s+' + NUM   # step 4: band 16, N 60, K 48
# (row, column) -> (file, regex with the share of h2 as group 1, what the cell is) or None: no stored source
CELLS = {
    ('PREM', 'without'): (F_PREM_INV, r'^residue ratios: h2 ' + NUM,
                          'full PREM, first-order figure, slipping core without the term, core K 6 (only K in file)'),
    ('PREM', 'with'): (F_PREM_TERM, r'^clairaut\s+inviscid\s+10\s+30' + SKIP * 2 + r'\s+' + NUM,
                       "full PREM (--model data/PREM.dat; the file's header line says 'without inner core', copied "
                       'from the noIC script), first-order map, slip term, core K 10; one-pole fit without FICN, '
                       'max |resid| h0 1.2e-2'),
    ('PREM', 'viscous'): (F_PREM_VISC, r'^delta -> 0 resonant parts / h0: h0 \S+ h2 ' + NUM,
                          'full PREM, first-order figure, viscous core, delta -> 0 (core K 12)'),
    (r'\quad level figure', 'without'): ('results/fcn_share_inviscid_level.txt', r'^residue ratios: h2 ' + NUM,
                                         'full PREM, level figure, slipping core without the term, core K 6'),
    (r'\quad level figure', 'with'): ('results/fcn_share_inviscid_slipterm_level.txt', r'^residue ratios: h2 ' + NUM,
                                      'full PREM, level figure, slipping core with the term, core K 6'),
    (r'\quad level figure', 'viscous'): (F_LEVEL_VISC, r'^delta -> 0 resonant parts / h0: h0 \S+ h2 ' + NUM,
                                         'full PREM, level figure, viscous core, delta -> 0 (core K 12)'),
    (r'\quad without inner core', 'without'): (F_NOIC_INV, r'^\s+10\s+30' + SKIP + r'\s+' + NUM,
                                               'PREM_noIC, first-order figure, slipping core without the term, core K 10, N 30'),
    (r'\quad without inner core', 'with'): (F_NOIC_TERM, r'^clairaut\s+inviscid\s+10\s+30' + SKIP * 2 + r'\s+' + NUM,
                                            'PREM_noIC, first-order map, slip term, core K 10'),
    (r'\quad without inner core', 'viscous'): (F_NOIC_VISC, r'^delta -> 0 resonant parts / h0: h0 \S+ h2 ' + NUM,
                                               'PREM_noIC, first-order figure, viscous core, delta -> 0, one pole '
                                               '(core K 12)'),
    ("Hough's Earth, exact spheroid", 'without'): (F_HOUGH_SOFT, r'^soft\s+inviscid\s+10\s+30' + SKIP * 3 + r'\s+' + NUM,
                                                    "Hough's Earth, exact spheroid ('soft'), slipping core without the "
                                                    'term, core K 10 (run 11-12, before slip_nonlevel)'),
    ("Hough's Earth, exact spheroid", 'with'): (F_HOUGH_TERM, r'^soft\s+inviscid\s+10\s+30' + SKIP * 3 + r'\s+' + NUM,
                                                 "Hough's Earth, exact spheroid ('soft'), slip term, core K 10"),
    ("Hough's Earth, exact spheroid", 'viscous'): (F_SOFT_VISC, LADDER_K48,
                                                    "Hough's Earth, exact spheroid, viscous core, delta 250 m, core Vp "
                                                    '1000 (ladder), band 16, N 60, K 48 (step 4)'),
    (r'\quad first-order map', 'without'): (F_HOUGH, r'^cl_exacteps\s+inviscid\s+10\s+30' + SKIP * 3 + r'\s+' + NUM,
                                            "Hough's Earth, first-order map with the exact eps ('cl_exacteps'), "
                                            'slipping core without the term, core K 10'),
    (r'\quad first-order map', 'with'): (F_HOUGH_TERM, r'^cl_exacteps\s+inviscid\s+10\s+30' + SKIP * 3 + r'\s+' + NUM,
                                         "Hough's Earth, first-order map with the exact eps ('cl_exacteps'), slip "
                                         'term, core K 10'),
    (r'\quad first-order map', 'viscous'): (F_HOUGH, r'^cl_exacteps\s+250 m\s+48\s+30' + SKIP * 3 + r'\s+' + NUM,
                                            "Hough's Earth, first-order map with the exact eps ('cl_exacteps'), "
                                            'viscous core, delta 250 m, K 48 (default core, not the Vp 1000 ladder)'),
    (r'\quad exact spheroid with bump', 'without'): (F_HOUGH_BUMP,
                                                     r'^bump\+3\s+inviscid\s+10\s+30' + SKIP * 3 + r'\s+' + NUM,
                                                     "Hough's Earth, exact spheroid + 3 eps^2 P4, slipping core "
                                                     'without the term, core K 10'),
    (r'\quad exact spheroid with bump', 'with'): (F_HOUGH_TERM, r'^bump\+3\s+inviscid\s+10\s+30' + SKIP * 3 + r'\s+' + NUM,
                                                  "Hough's Earth, exact spheroid + 3 eps^2 P4, slip term, core K 10"),
    (r'\quad exact spheroid with bump', 'viscous'): (F_BUMP_VISC, LADDER_K48,
                                                     "Hough's Earth, exact spheroid + 3 eps^2 P4, viscous core, delta "
                                                     '250 m, core Vp 1000 (ladder), band 16, N 60, K 48 (step 4)'),
}
NO_SOURCE = {}
NOTES = [
    'PREM, with / viscous: both sources print +0.00145 (printed 0.0014 and 0.0015 before 2026-11-19; the table now has '
    '5 decimals).',
    'Level figure, slipping: scripts/fcn_share_inviscid.py --figure level [--slip-term] (2026-11-19): the term vanishes '
    'on a level wall, both +0.001423.',
    'Without inner core, viscous: fcn_share_inviscid_noic.txt prints "viscous limit (fcn_resonance_noic_K12.txt): '
    'h2 +0.00152", a literal hard-coded in scripts/fcn_share_inviscid_noic.py (line 74), not read from that file. '
    'The cited file prints +0.00154 (one-pole delta -> 0, residue h2 / residue h0 = 2.85194e-07 / 1.84732e-04 = '
    '0.001544); 0.00152 is h2 over its "check h0" 1.83770e-04 (0.001552). Both round to 0.0015. The table uses the '
    'cited file. Note also that this cell (like PREM) is the limit delta -> 0, whereas the caption says delta 250 m '
    'for every row except full PREM.',
    "Hough's Earth, exact spheroid, viscous: hough_share_test.txt held 'soft' 250 m K 48 +0.00205 on 11-11 "
    '(git 9fa669b) but was overwritten by the cl_exacteps / ex_cleps run; in results/ now: K 72 +0.00204 '
    '(hough_share_test_bump_d250_K72.txt, default core) and the Vp 1000 ladder +0.00201 used here (the same ladder '
    'family as the bump viscous cell). All round to 0.0020.',
    'First-order map: the cells use variant cl_exacteps (first-order map, exact eps) throughout, the only variant '
    'with all three cells in results/. The Clairaut-eps variant (clairaut) gives with the term +0.00200 '
    '(hough_share_test_slipterm.txt); its no-term +0.00595 and viscous +0.00204 (250 m K 48) exist only in git '
    '(b4780dc:results/hough_share_test.txt). Its viscous cell is the default core, not the Vp 1000 ladder of the '
    'exact-spheroid and bump cells.',
    'Bump, viscous: the ladder\'s "final point" is band 16, N 60, K 36 (+0.00192); the K 48 row (step 4) used here '
    'gives the same printed value. The K 48 band-12 value +0.00175 of hough_share_test_bump.txt is under-resolved '
    '(band 16 needed) and is not used.',
    'Level figure, viscous: viscous_limit_thin_Kthin_level.txt (K 72 / 96, windows down to 250 m) gives 0.001418 - '
    '0.001420, the same value at 4 decimals.',
]


CAPTION = r"""\begin{table}
\caption{Resonant share of $h^{(2)}$ (resonant part of $h^{(2)}$ relative to
that of $h^{(0)}$): slipping inviscid core without and with the term of
eq.~(\ref{M-eq:slipterm}) for a wall off its level surface, and viscous core
(PREM: the limit $\delta\to0$; Hough's Earth: $K=48$ in the core,
$\delta=250$~m). First-order figure of eq.~(\ref{M-eq:map}) unless stated;
bump: a degree-4 ripple of the wall of height $3\epsilon^2$ (210~m).}
\label{tab:slipshare}
"""


def find(path, pattern):
    """The single line of `path` matching `pattern`, and group 1 as a Decimal."""
    hits = [(line, m) for line in (ROOT / path).read_text().splitlines() for m in [re.search(pattern, line)] if m]
    if len(hits) != 1:
        raise RuntimeError(f'{path}: {len(hits)} lines match {pattern!r}')
    line, m = hits[0]
    return Decimal(m.group(1)), line


def q(v, places, rounding=ROUND_HALF_EVEN):
    return v.quantize(Decimal(1).scaleb(-places), rounding=rounding)


def sgn(v):
    return f'+{v}' if v >= 0 else f'{v}'


def main():
    inputs = sorted({c[0] for c in CELLS.values() if c})
    with Run('Table S7 (tab:slipshare) assembled from stored results', inputs=inputs) as run:
        rep = ['Table S7 (tab:slipshare): resonant share of h(2), read from stored results '
               '(paper_pepi/make_table_slipshare.py)', '']
        flags, tex_rows = [], []
        for row in ROWS:
            cells = []
            for i, col in enumerate(COLS):
                cur = CURRENT[row][i]
                spec = CELLS[(row, col)]
                rep.append(f'{row} | {col} | current table {cur}')
                if spec is None:
                    rep += [f'    value   : none', f'    source  : {NO_SOURCE[row]}', '']
                    flags.append(f'{row} | {col}: no stored source (table shows {cur})')
                    cells.append('--')
                    continue
                path, pattern, what = spec
                v, line = find(path, pattern)
                v5 = q(v, 5)
                even, up = q(v5, 4, ROUND_HALF_EVEN), q(v5, 4, ROUND_HALF_UP)
                rep += [f'    value   : {sgn(v)}   (5 decimals {sgn(v5)})', f'    what    : {what}',
                        f'    source  : {path}', f'    line    : {line.strip()}']
                if Decimal(cur) not in (even, up):
                    flags.append(f'{row} | {col}: source {sgn(v5)} -> {sgn(even)}, table shows {cur}  DIFFERENT')
                elif even != up:
                    flags.append(f'{row} | {col}: source {sgn(v5)} is a tie at 4 decimals (half-even {sgn(even)}, '
                                 f'half-up {sgn(up)}); table shows {cur}')
                rep.append('')
                cells.append(f'${sgn(v5)}$')
            tex_rows.append(f'{row} & ' + ' & '.join(cells) + r' \\')
        rep += ['Cells whose 5-decimal source value does not round unambiguously to the current table value, or that '
                'have no source:'] + ['  ' + f for f in flags] + ['', 'Notes:'] + ['  - ' + n for n in NOTES]
        tex = ('% Table S7 (tab:slipshare) of the supplement, written by paper_pepi/make_table_slipshare.py from stored\n'
               '% results. Sources per cell: results/table_slipshare.txt\n'
               + CAPTION + r'\small' '\n' r'\centering' '\n'
               r'\begin{tabular}{@{}lrrr@{}}' '\n' r'\toprule' '\n'
               r' & \multicolumn{2}{c}{slipping} & \\' '\n' r'Earth & without & with & viscous \\' '\n'
               r'\midrule' '\n' + '\n'.join(tex_rows) + '\n' r'\bottomrule' '\n' r'\end{tabular}' '\n'
               r'\end{table}' '\n')
        run.write_text('results/table_slipshare.txt', '\n'.join(rep) + '\n')
        run.write_text('results/regen/table_slipshare.tex', tex)
        if '--paper' in sys.argv:
            run.write_text('paper_pepi/table_slipshare.tex', tex)
        print('\n'.join(rep))
        print(tex)


if __name__ == '__main__':
    main()
