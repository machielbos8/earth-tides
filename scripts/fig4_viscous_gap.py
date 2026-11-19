# -*- coding: utf-8 -*-
#
# Gap between the viscous and inviscid h2 curves of P1 v2 Fig. 4 (Sect. 5.3, 'What the choice changes'),
# for the viscous curve at delta 4 km / K 24 (old Fig. 4) and delta 1 km / K 96 (Fig. 4 since 2026-11-03).
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
# 03/11/2026 Machiel Bos
#===============================================================================
"""h2(viscous) - h2(inviscid, core K 8, 16, 24) at O1 (0.927 cpsd) and 0.95 cpsd, and around M2 the
median and 90th percentile of |difference| (the inviscid spikes excluded by the percentile).
Usage: python scripts/fig4_viscous_gap.py  ->  results/fig4_viscous_gap.txt"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np

from provenance import Run

R = 'results/convergence_viscous_coreK'
# inviscid sweeps of the first-order figure WITHOUT the slip term of a non-level wall (eq. slipterm): the
# files of commit da66985 (2026-11-01), kept as *_noslipterm.txt when the sweeps were regenerated with the
# term (7d3a2f7, 2026-11-14); the paper quotes this gap ('without eq. (17) ... 7-8e-6 lower')
INV = {K: f'{R}{K}_N30_d0_s0.0005_noslipterm.txt' for K in (8, 16, 24)}
VIS = {'delta 4 km K 24': f'{R}24_N30_d4_s0.0005.txt', 'delta 1 km K 96': f'{R}96_N30_d1_s0.0005_b12.txt'}
# 2026-11-19: the level figure of the paper (with the slip term; Fig. 4 caption '3e-7 in (d)')
INV_L = {K: f'{R}{K}_N30_d0_s0.0005_level.txt' for K in (8, 16, 24)}
VIS_L = {'level, delta 1 km K 96': f'{R}96_N30_d1_s0.0005_b12_level.txt'}


def curve(path, m):
    a = np.genfromtxt(path, skip_header=1, usecols=range(9))
    a = a[a[:, 0] == m]
    return dict(zip(np.round(a[:, 1], 5), a[:, 5]))


def main():
    with Run('viscous - inviscid h2 gap of Fig. 4 at O1, 0.95 cpsd and around M2',
             inputs=list(INV.values()) + list(VIS.values()) + list(INV_L.values()) + list(VIS_L.values())) as run_:
        lines = ['h2(viscous) - h2(inviscid), units 1e-6']
        for name, pv in VIS.items():
            for K, pi in INV.items():
                v1, i1 = curve(pv, 1), curve(pi, 1)
                v2, i2 = curve(pv, 2), curve(pi, 2)
                d = np.abs([v2[s] - i2[s] for s in v2 if s in i2])
                lines.append(f'{name}, inviscid K {K:2d}: O1 {(v1[0.927] - i1[0.927]) * 1e6:+6.2f}, '
                             f'0.95 {(v1[0.95] - i1[0.95]) * 1e6:+6.2f}; M2 band |d| median '
                             f'{np.median(d) * 1e6:.2f}, 90 % {np.percentile(d, 90) * 1e6:.2f}')
        lines += ['', 'level figure, inviscid with the slip term (Fig. 4), h2(viscous) - h2(inviscid), units 1e-6;',
                  'O1 band 0.915-0.95 cpsd away from the spikes: median and 90 % of |d|']
        for name, pv in VIS_L.items():
            for K, pi in INV_L.items():
                v1, i1 = curve(pv, 1), curve(pi, 1)
                d1 = np.abs([v1[s] - i1[s] for s in v1 if s in i1 and 0.915 <= s <= 0.95])
                lines.append(f'{name}, inviscid K {K:2d}: O1 {(v1[0.927] - i1[0.927]) * 1e6:+6.2f}, '
                             f'0.95 {(v1[0.95] - i1[0.95]) * 1e6:+6.2f}; 0.915-0.95 |d| median '
                             f'{np.median(d1) * 1e6:.2f}, 90 % {np.percentile(d1, 90) * 1e6:.2f}')
        txt = '\n'.join(lines)
        run_.write_text('results/fig4_viscous_gap.txt', txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
