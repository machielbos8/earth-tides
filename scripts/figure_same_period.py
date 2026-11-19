# -*- coding: utf-8 -*-
#
# The level minus the first-order figure, both tuned to the observed FCN period (stored results).
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
"""Level minus first-order figure when both are tuned to the observed FCN period, 431.4 sidereal days (main Sect.
5.6, 8, Table 2; S7.4): anelastic PREM with the excess J2 of Chambat et al. (2010) and the Delta e_f that puts the FCN
at 431.4 days (486 m on the level, 504 m on the first-order figure). From the full-precision block 'FCN 431 state'
of paper_pepi/make_table4.py (jobs table_nonhydro_love and table_nonhydro_love_fo). (2026-11-19: taken by hand
from the rounded columns before.)
Run from the project root:  .venv/bin/python scripts/figure_same_period.py  ->  results/figure_same_period.txt"""
import sys
sys.path.insert(0, 'scripts')
from provenance import Run

LEVEL, FO = 'results/paper1_table4_level.txt', 'results/paper1_table4.txt'


def tuned(path):
    lines = open(path).read().splitlines()
    i = lines.index('FCN 431 state (real parts, 8 decimals)')
    keys = lines[i + 1].split()[1:]
    out = {}
    for line in lines[i + 2:]:
        if not line.strip():
            break
        name, *vals = line.split()
        out[name] = dict(zip(keys, map(float, vals)))
    return keys, out


def main():
    with Run('level minus first-order figure at the same FCN period (431.4 sd)', inputs=[LEVEL, FO]) as run:
        keys, lev = tuned(LEVEL)
        _, fo = tuned(FO)
        lines = ['level minus first-order figure, both tuned to the FCN period 431.4 sd; anelastic PREM, units 1e-5',
                 f'{"wave":10s} ' + ' '.join(f'{k:>9s}' for k in keys)]
        for w in lev:
            lines.append(f'{w:10s} ' + ' '.join(f'{1e5 * (lev[w][k] - fo[w][k]):+9.2f}' for k in keys))
        run.write_text('results/figure_same_period.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
