# -*- coding: utf-8 -*-
#
# Homogeneous, non-gravitating elastic spheroid: solver vs exact polynomial
# solution.
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
# 5/10/2026 Machiel Bos
#===============================================================================
"""Homogeneous, non-gravitating elastic spheroid: solver vs exact polynomial solution.

Exact solution: src/earthtides/exact_spheroid.py (Chree-type cubic polynomial).
Solver: gravity=False, one homogeneous layer, either the exact spheroid map
(ExactSpheroidGeometry) or the first-order map r = s[1 - 2/3 eps P2] with
constant eps (as used for the Earth). Error = max |u_solver - u_exact| / max |u_exact|
over the surface (u_r and u_theta), at lambda = 0.
Results -> results/exact_spheroid.txt
"""
import sys
import warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ExactSpheroidGeometry
from earthtides.solver import TidalSolver
from earthtides.exact_spheroid import solve, surface_field

md = EarthModel('data/HOMOG.dat')
a, g = md.a, md.g_surface
rho, lam, mu = (float(f(a / 2)) for f in (md.rho[0], md.lam[0], md.mu[0]))
th = np.linspace(0.05, np.pi - 0.05, 241)
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


class FirstOrder:
    """Constant flattening with the first-order map of eq. (map)."""

    def __init__(self, fl):
        self.fl = fl

    def eps(self, i, s):
        s = np.asarray(s, float)
        return self.fl + 0 * s, 0 * s, 0 * s


out('Homogeneous non-gravitating spheroid (lambda = mu = 1e11 Pa, rho = 5500): '
    'max surface error / max |u|')
for m in (0, 2):
    for fl in (1 / 300, 1 / 30, 1 / 10):
        G = ExactSpheroidGeometry(fl)
        usol = solve(a * G.A, a * G.C, lam, mu, rho, g, a, m)
        ur_x, ut_x, _ = surface_field(usol, a * G.A, a * G.C, th)
        scale = max(np.abs(ur_x).max(), np.abs(ut_x).max())
        row = []
        for geom in (G, FirstOrder(fl)):
            errs = []
            for K in (2, 4, 6, 8, 10):
                S = TidalSolver(md, geom, m=m, K=K, N=12, omega_f=0.0, gravity=False)
                S.solve()
                ur, ut, _ = S.surface_field(th)
                errs.append(max(np.abs(ur - ur_x).max(), np.abs(ut - ut_x).max()) / scale)
            row.append(errs)
        out(f'm={m} 1/f={1 / fl:5.0f}  exact map   K=2..10: ' + ' '.join(f'{e:.1e}' for e in row[0]))
        out(f'               first-order map K=2..10: ' + ' '.join(f'{e:.1e}' for e in row[1])
            + f'   (f^2 = {fl**2:.1e})')
open('results/exact_spheroid.txt', 'w').write('\n'.join(lines) + '\n')
