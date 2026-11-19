# -*- coding: utf-8 -*-
#
# The deviatoric stress at the same FCN period: route B at 441 m minus route D at 410 m (stored results).
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
"""Effect of the deviatoric stress compared at the same FCN period, 431.4 sidereal days (main Sect. 6.3, 6.5,
Table 2, Conclusions; S7.5): the state with the stress reaches it with 441 m of excess CMB flattening (route B: tau
and slip terms), the state without it with 410 m (route D: slip terms, no tau). Difference B(441 m) - D(410 m) of
the Love numbers, from the stored results of scripts/nonhydro_stress_split.py (jobs stress_split_441, _410).
(2026-11-19: the difference was taken by hand before.)
Run from the project root:  .venv/bin/python scripts/stress_same_period.py  ->  results/stress_same_period.txt"""
import re
import sys
sys.path.insert(0, 'scripts')
from provenance import Run

F441, F410 = 'results/nonhydro_stress_split_441m_level.txt', 'results/nonhydro_stress_split_410m_level.txt'
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k')


def routes(path):
    """{wave: {route: [six values]}}: route B as printed, the other routes as B + their difference (1e-6)."""
    out, wave = {}, None
    for line in open(path):
        m = re.match(r'(\S+)?\s+([ABCD])\s+([-+\d.\s]+)$', line.rstrip())
        if not m or line.startswith(('route', 'A:', 'B:', 'C:', 'D:')):
            continue
        wave = m.group(1) or wave
        vals = [float(v) for v in m.group(3).split()]
        if len(vals) != 6:
            continue
        d = out.setdefault(wave, {})
        d[m.group(2)] = vals if m.group(2) == 'B' else [b + 1e-6 * v for b, v in zip(d['B'], vals)]
    return out


def main():
    with Run('deviatoric stress at the same FCN period: B(441 m) - D(410 m)', inputs=[F441, F410]) as run:
        b, d = routes(F441), routes(F410)
        lines = ['B(441 m, with the stress) - D(410 m, without it), both at the FCN period 431.4 sd; level figure, '
                 'elastic PREM; units 1e-5',
                 f'{"wave":6s} ' + ' '.join(f'{k:>9s}' for k in KEYS)]
        diff = {}
        for w in b:
            diff[w] = [x - y for x, y in zip(b[w]['B'], d[w]['D'])]
            lines.append(f'{w:6s} ' + ' '.join(f'{1e5 * v:+9.2f}' for v in diff[w]))
        far = [w for w in ('Mf', 'O1', 'OO1', 'M2') if w in diff]
        lines += ['', f'max |difference| over {", ".join(far)}, all parameters: '
                  f'{max(abs(v) for w in far for v in diff[w]):.1e}']
        lines += [f'h0 at {w}: {abs(diff[w][0]):.2e}' for w in ('P1', 'K1', 'psi1') if w in diff]
        run.write_text('results/stress_same_period.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
