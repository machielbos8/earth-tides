# -*- coding: utf-8 -*-
#
# Does PREM's M2 ray (Fig. 4b of P1 v2) have an attractor? A long Hamiltonian trace.
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
# 08/11/2026 Machiel Bos
#===============================================================================
"""The traced ray of Fig. 4b (scripts/rays_hamiltonian.py: RK4 step 1e-3 r_cmb, start P0 towards D0, PREM's
N^2 smoothed over 10 km) at M2, 50000 wall reflections, with PREM's buoyancy and without (N = 0). In windows
of 5000 reflections: the growth of log|k| per reflection (> 0: focusing) and the number of distinct CMB
colatitudes (0.1 deg) of the window's last 200 reflections (a few: an attractor, and their period).
Usage: python scripts/rays_m2_long.py [NREF]  ->  results/rays_m2_long.txt"""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
NREF = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
_argv, sys.argv = sys.argv, [sys.argv[0], '1e-3']    # the step of Fig. 4 for rays_hamiltonian's module H
import rays_hamiltonian as rh
sys.argv = _argv                                     # restored: Pool workers re-import this file with it
import numpy as np
from multiprocessing import Pool
from earthtides.model import EarthModel

F = 1.92700
WIN = 5000


def one(which):
    x, n2, dn2, rc = rh.buoyancy(EarthModel('data/PREM.dat'))
    if which == 'N = 0':
        n2, dn2 = 0 * n2, 0 * dn2
    R = rh.Rays(x, n2, dn2, F / 2, rh.ETA)
    P, logk, _, werr, dmin = R.trace(NREF, None)
    rows = []
    for end in range(WIN, len(logk) + 1, WIN):
        k = np.arange(end - WIN, end)
        g = np.polyfit(k, logk[end - WIN:end], 1)[0]
        q = P[max(0, end - 200):end + 1]
        q = q[np.abs(np.hypot(*q.T) - 1.0) < 1e-6]
        c = np.unique(np.round(np.degrees(np.arctan2(q[:, 0], q[:, 1])), 1))
        rows.append((end, g, len(c)))
    return which, rows, werr, dmin, R.stopped, len(P) - 1


if __name__ == '__main__':
    from provenance import Run
    with Run('long trace of the M2 ray, PREM N^2 and N = 0: is there an attractor?', params=dict(
            f=F, nref=NREF, window=WIN, step=rh.H, p0=rh.P0, d0=rh.D0), inputs=['data/PREM.dat']) as run_:
        with Pool(2) as p:
            res = p.map(one, ['PREM N^2', 'N = 0'])
        lines = [f'M2 ({F} cpsd), Hamiltonian rays as Fig. 4b, {NREF} wall reflections; per window of {WIN}: '
                 'growth of log|k| per reflection (> 0: focusing), distinct CMB colatitudes (0.1 deg) in its last 200']
        for which, rows, werr, dmin, stopped, n in res:
            lines.append(f'{which}: {n} reflections, W err {werr:.1e}, smallest discriminant {dmin:+.3f}'
                         + (f', stopped after {stopped[0]}' if stopped else ''))
            lines += [f'  reflections {e - WIN:6d}-{e:6d}: growth {g:+.5f}, distinct CMB points {c:4d}' for e, g, c in rows]
        run_.write_text('results/rays_m2_long.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))
