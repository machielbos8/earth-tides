# -*- coding: utf-8 -*-
#
# Rays of P1 v2 Fig. 3 with PREM's buoyancy: the paths for the paper figure, and where the waves cannot
# travel (the regions behind the turning surfaces, where the rays turn back before reaching a wall).
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
"""A parcel of the wave moves along a direction xi in the meridional plane (xi perpendicular to k):

    fh^2 = (xi . s^)^2 + n^2 (xi . r^)^2,     fh = omega/(2 Omega), n^2 = N^2/(2 Omega)^2,

the Coriolis force restoring the motion towards or away from the axis (s^), buoyancy the motion along
gravity (r^) (this is W = fh^2 of rays_hamiltonian.py with xi = (-q, p)/|k|). A wave travels only where some
xi gives fh^2, i.e. fh^2 lies between the eigenvalues of s^s^ + n^2 r^r^; the boundary (discriminant 0,
Rays.disc) is a turning surface, at which a ray turns back. On the axis (s^ perpendicular to r^) the
eigenvalues are 1 and n^2: no waves where n^2 > fh^2. In the equatorial plane (s^ = r^) they are 0 and
1 + n^2: no waves where n^2 < fh^2 - 1.
Paths: Hamiltonian tracer of rays_hamiltonian.py (step H 1e-3, segments 5941-6000, as its figure).
Usage: python scripts/rays_figure_paths.py  ->  results/rays_figure_paths.npz, .txt"""
import sys
import math
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from multiprocessing import Pool

import numpy as np

from earthtides.model import EarthModel
import rays_hamiltonian as rh

H = 1e-3
FIGF = rh.FIGF
NFIG = rh.NFIG


def job(args):
    f, which = args
    rh.H = H                    # the tracer reads its step from the module (spawned workers re-import it)
    x, n2, dn2, rc = rh.buoyancy(EarthModel('data/PREM.dat'))
    if which == 'straight':
        n2, dn2 = 0 * n2, 0 * dn2
    R = rh.Rays(x, n2, dn2, f / 2, rh.ETA)
    P, lk, path, werr, dmin = R.trace(NFIG, NFIG - 60)
    return f, which, path, rh.growth(lk), werr, R.stopped


def intervals(x, mask):
    """[x0, x1] ranges where mask holds."""
    out, i = [], 0
    while i < len(x):
        if mask[i]:
            j = i
            while j + 1 < len(x) and mask[j + 1]:
                j += 1
            out.append((x[i], x[j]))
            i = j + 1
        else:
            i += 1
    return out


def main():
    from provenance import Run
    x, n2, dn2, rc = rh.buoyancy(EarthModel('data/PREM.dat'))
    with Run('paths of the Fig. 3 rays with PREM N^2 for the paper figure; regions without waves',
             params=dict(H=H, nfig=NFIG, fig_freqs=FIGF, p0=rh.P0, d0=rh.D0, eta=rh.ETA),
             inputs=['data/PREM.dat']) as run_:
        lines = ['regions without waves (fh^2 not between the eigenvalues of s^s^ + n^2 r^r^), PREM outer core',
                 f'n^2 from {n2.min():+.3f} to {n2.max():+.3f}; radii in km']
        S_, Z_ = np.meshgrid(np.linspace(0, 1, 801), np.linspace(-1, 1, 1601))
        Rr = np.hypot(S_, Z_)
        inside = (Rr > rh.ETA) & (Rr < 1)
        out = dict(x=x, n2=n2, rc=rc, eta=rh.ETA, freqs=np.array(FIGF), S=S_[0], Z=Z_[:, 0])
        for f in FIGF:
            fh2 = (f / 2) ** 2
            R = rh.Rays(x, n2, dn2, f / 2, rh.ETA)
            D = np.where(inside, np.vectorize(lambda a, b: R.disc(a, b) if rh.ETA < math.hypot(a, b) < 1 else 1.0)(
                S_ + 1e-9, Z_), 1.0)
            out[f'disc_{f}'] = D
            vol = (S_ * inside * (D < 0)).sum() / (S_ * inside).sum()      # volume of revolution: weight s
            eq = intervals(x * rc / 1e3, n2 < fh2 - 1)
            ax = intervals(x * rc / 1e3, n2 > fh2)
            cm = np.linspace(0.05, 90, 1799)
            dc = np.array([R.disc(0.9999 * math.sin(math.radians(t)), 0.9999 * math.cos(math.radians(t)))
                           for t in cm])
            col = intervals(cm, dc < 0)
            lines += ['', f'{f} cpsd: fh^2 = {fh2:.4f}; no waves in {100 * vol:.1f} % of the core volume',
                      f'  equatorial plane, n^2 < fh^2 - 1 = {fh2 - 1:+.4f}: r ' +
                      (', '.join(f'{a:.0f}-{b:.0f}' for a, b in eq) or 'none'),
                      f'  rotation axis, n^2 > fh^2 = {fh2:+.4f}: r ' +
                      (', '.join(f'{a:.0f}-{b:.0f}' for a, b in ax) or 'none'),
                      f'  just below the CMB: colatitudes ' +
                      (', '.join(f'{a:.1f}-{b:.1f}' for a, b in col) or 'none') + ' deg',
                      f'  critical latitude arcsin(fh) = {math.degrees(math.asin(f / 2)):.1f} deg']
        jobs = [(f, w) for f in FIGF for w in ('straight', 'PREM')]
        with Pool(4) as pool:
            res = pool.map(job, jobs)
        lines += ['', f'paths, segments {NFIG - 59}-{NFIG}, step {H:g}: growth of log|k| per reflection, |W - fh^2|']
        for f, w, path, g, werr, st in res:
            out[f'path_{w}_{f}'] = path
            lines.append(f'  {f} {w:8s} growth {g:+.4f}, W err {werr:.1e}' + (f', stopped {st}' if st else ''))
        run_.savez('results/rays_figure_paths.npz', **out)
        txt = '\n'.join(lines)
        run_.write_text('results/rays_figure_paths.txt', txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
