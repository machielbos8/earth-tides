# -*- coding: utf-8 -*-
#
# Love numbers of elastic hydrostatic PREM on the first-order figure and on true level surfaces.
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
# 14/11/2026 Machiel Bos
#===============================================================================
"""What do the terms of second order in the figure change in the Love numbers? Elastic hydrostatic PREM,
slipping core with the slope term (equal to the viscous limit, Sect. 5.3 of paper 1), at the waves of the
supplement's comparison with Dehant et al. (1999): the first-order (Clairaut) map of the paper, and
coordinate surfaces that are level surfaces of the reference potential (level_geometry.py). K1 is the mean of
1 -+ 1e-5 cpsd. Since 2026-11-14 (evening) the potential is normalised at the equatorial radius of each figure
(geometry.equatorial_radius: level 6378103.7 m, first-order 6378081.3 m; before, the Clairaut value for both:
h0 of the level figure 8e-6 higher).
Also the Legendre content (m) of the core-mantle boundary: first-order map, exact ellipse and level surface.
Usage: python scripts/love_level_figure.py [--K 6] [--N 30]  ->  results/love_level_figure.txt"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np
from numpy.polynomial import legendre as L

_argv, sys.argv = sys.argv, sys.argv[:1]
import exact_levels_share_test as els
sys.argv = _argv
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, LevelGeometry, OMEGA_EARTH, mapping, equatorial_radius
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit
import level_geometry as lg
from provenance import Run

WAVES = (('Mf', 0, (0.07300,)), ('O1', 1, (0.92700,)), ('P1', 1, (0.99454,)), ('K1', 1, (1 - 1e-5, 1 + 1e-5)),
         ('psi1', 1, (1.00273,)), ('phi1', 1, (1.00548,)), ('OO1', 1, (1.07300,)), ('M2', 2, (1.92700,)))
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k')


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def love(md, g, m, sigs, K, N):
    out = []
    for s in sigs:
        S = TidalSolver(md, g, m=m, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=s * OMEGA_EARTH,
                        omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=True,
                        frame='mantle')
        r = S.solve()
        f = iers_fit(S, g)
        f['k'] = r['kk'] * (md.a / equatorial_radius(g, md)) ** 5 if 'kk' in r else np.nan
        out.append([complex(f.get(k, np.nan)).real for k in KEYS])
    return np.mean(out, axis=0)


def wall(md, g, s=3480e3):
    """Legendre coefficients 0, 2, 4, 6 (m) of R(theta) of the coordinate surface s."""
    i = int(np.flatnonzero(np.isclose(md.r1, s))[0])
    x, w = L.leggauss(40)
    th, s2 = np.arccos(x), np.array([[s]])
    R = (g.map_derivs(i, s2, th)[0] if hasattr(g, 'map_derivs') else mapping(s2, th, *g.eps(i, s2))['R'])[0]
    return [(2 * n + 1) / 2 * np.sum(w * R * L.Legendre.basis(n)(x)) for n in (0, 2, 4, 6)]


def main():
    K, N = int(_opt('--K', 6)), int(_opt('--N', 30))
    with Run('Love numbers of elastic hydrostatic PREM: first-order figure against true level surfaces',
             params=dict(K=K, N=N, waves=WAVES), inputs=['data/PREM.dat']) as run:
        md = EarthModel('data/PREM.dat')
        glev = LevelGeometry(md)
        figs = (('first-order map', ClairautGeometry(md)), ('exact ellipses', els.ExactLevelsGeometry(md)),
                ('level surfaces', glev))
        lines = [f'elastic hydrostatic PREM, slipping core with the slope term, K {K}, N {N}',
                 'core-mantle boundary, Legendre coefficients of R (m):       P0             P2         P4         P6']
        W = {n: wall(md, g) for n, g in figs}
        for n, _ in figs:
            lines.append(f'  {n:18s} ' + f'{W[n][0]:28.3f} ' + ' '.join(f'{v:+10.3f}' for v in W[n][1:]))
        lines.append('  minus first-order map:')
        for n in ('exact ellipses', 'level surfaces'):
            lines.append(f'  {n:18s} ' + ' ' * 29 + ' '.join(f'{a - b:+10.3f}' for a, b in zip(W[n][1:], W['first-order map'][1:])))
        lines += ['', f'{"wave":5s} {"figure":16s} ' + ' '.join(f'{k:>10s}' for k in KEYS)]
        for w, m, sigs in WAVES:
            a = love(md, figs[0][1], m, sigs, K, N)
            b = love(md, figs[2][1], m, sigs, K, N)
            lines.append(f'{w:5s} {"first-order":16s} ' + ' '.join(f'{v:10.5f}' for v in a))
            lines.append(f'{"":5s} {"level":16s} ' + ' '.join(f'{v:10.5f}' for v in b))
            lines.append(f'{"":5s} {"level - first":16s} ' + ' '.join(f'{v:+10.1e}' for v in b - a))
            print('\n'.join(lines[-3:]), flush=True)
        run.write_text('results/love_level_figure.txt', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
