# -*- coding: utf-8 -*-
#
# Viscous core at delta = 1 km, K72: PREM's stratified core against a neutrally stratified one (N^2 = 0).
# Does the diurnal hump of Fig. 4 at 0.910 cpsd come from buoyancy (P1 v2, Sect. 3 and 5.3)?
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
"""Each h0(sigma), h2(sigma) band is split into a smooth background (cubic in sigma, least squares) and
the residual; the residual's largest excursion and its frequency measure the resonant structure.
The neutral model has an Adams-Williamson density, so a smooth offset from PREM is expected and is
taken out by the fit.
Usage: python scripts/compare_neutral_core.py  ->  results/compare_neutral_core_K72_d1.txt"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np

from provenance import Run

BASE = 'results/convergence_viscous_coreK72_N30_d1_s0.0005_b12'
FILES = dict(PREM=BASE + '.txt', neutral=BASE + '_neutral.txt')


def structure(s, y):
    res = y - np.polyval(np.polyfit(s - s.mean(), y, 3), s - s.mean())
    i = np.argmax(np.abs(res))
    return res, s[i], res[i], res.std()


def main():
    data = {k: np.loadtxt(v, skiprows=1, usecols=range(9)) for k, v in FILES.items()}
    with Run('viscous core K72 delta 1 km: PREM against a neutral core', params=dict(fit='cubic'),
             inputs=list(FILES.values())) as run_:
        lines = ['residual from a cubic background in sigma, units 1e-6; largest excursion (sigma) and rms']
        for m in (1, 2):
            for col, name in ((4, 'h0'), (5, 'h2')):
                for k, a in data.items():
                    a = a[a[:, 0] == m]
                    _, s0, r0, sd = structure(a[:, 1], a[:, col])
                    lines.append(f'm {m} {name} {k:8s} sigma {a[0, 1]:.4f}-{a[-1, 1]:.4f}: max {r0 * 1e6:+8.3f} '
                                 f'at {s0:.4f}, rms {sd * 1e6:6.3f}')
        lines += ['', 'diurnal h2 (1e-6): sigma, PREM, neutral, PREM residual, neutral residual']
        a, b = (d[d[:, 0] == 1] for d in (data['PREM'], data['neutral']))
        ra, rb = structure(a[:, 1], a[:, 5])[0], structure(b[:, 1], b[:, 5])[0]
        lines += [f'{s:.4f} {x * 1e6:9.3f} {y * 1e6:9.3f} {u * 1e6:+7.3f} {v * 1e6:+7.3f}'
                  for s, x, y, u, v in zip(a[::2, 1], a[::2, 5], b[::2, 5], ra[::2], rb[::2])]
        txt = '\n'.join(lines)
        run_.write_text('results/compare_neutral_core_K72_d1.txt', txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
