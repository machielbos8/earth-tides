# -*- coding: utf-8 -*-
#
# Uncertainty of the viscous limit delta -> 0, from the stored windows and core resolutions.
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
"""Uncertainty of the Love numbers of the viscous core in the limit delta -> 0 (main Sect. 4, 7, Table 2; S5.4),
from the stored extrapolations of scripts/viscous_limit_thin.py (job viscous_limit_Kthin, heavy): quadratic
extrapolation through three successive deltas, core K 72 and 96. The adopted values are K 72, window 500/350/250 m.
Two measures per Love number: (a) the window one step thicker (750/500/350 m) minus the adopted one, K 72;
(b) core K 96 minus K 72 in the window 750/500/350 m (at 250 m and below K 96 is not resolved: fit misfit 2e-5
and more). The uncertainty is the larger of the two. (2026-11-19: read by hand from the file before.)
Run from the project root:  .venv/bin/python scripts/viscous_limit_uncertainty.py
-> results/viscous_limit_uncertainty.txt"""
import re
import sys
sys.path.insert(0, 'scripts')
from provenance import Run

SRC = 'results/viscous_limit_thin_Kthin_level.txt'
ADOPT, THICK = '500/350/250', '750/500/350'


def blocks(path):
    """{K: {(wave, key): {window: value}}}"""
    out, K, windows = {}, None, None
    for line in open(path):
        m = re.match(r'== core K (\d+)', line)
        if m:
            K = int(m.group(1))
            out[K] = {}
            continue
        if line.startswith('window'):
            windows = line.split()[1:]
            continue
        m = re.match(r'(O1|P1|K1|psi1|phi1|OO1) (h0|h2|l0|l1|l2|k0)\s+(.*)$', line.strip())
        if m and K is not None:
            out[K][(m.group(1), m.group(2))] = dict(zip(windows, map(float, m.group(3).split())))
    return out


def main():
    with Run('uncertainty of the viscous limit delta -> 0 (windows and core K)', inputs=[SRC]) as run:
        b = blocks(SRC)
        lines = [f'adopted: core K 72, window {ADOPT} m; (a) window {THICK} minus adopted (K 72); '
                 f'(b) K 96 minus K 72, window {THICK}; units 1e-5',
                 f'{"wave":6s} {"key":4s} {"adopted":>10s} {"(a)":>8s} {"(b)":>8s} {"max":>8s}']
        worst = {}
        for (w, k), v in b[72].items():
            a = v[THICK] - v[ADOPT]
            bb = b[96][(w, k)][THICK] - v[THICK]
            u = max(abs(a), abs(bb))
            worst[(w, k)] = u
            lines.append(f'{w:6s} {k:4s} {v[ADOPT]:10.6f} {1e5 * a:+8.2f} {1e5 * bb:+8.2f} {1e5 * u:8.2f}')
        far = [u for (w, k), u in worst.items() if w != 'psi1']
        lines += ['', f'max uncertainty except psi1: {max(far):.1e} (median {sorted(far)[len(far) // 2]:.1e})',
                  f'psi1 h0 {worst[("psi1", "h0")]:.1e}, k {worst[("psi1", "k0")]:.1e}; other parameters at psi1: '
                  f'max {max(u for (w, k), u in worst.items() if w == "psi1" and k not in ("h0", "k0")):.1e}']
        run.write_text('results/viscous_limit_uncertainty.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
