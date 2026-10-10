# -*- coding: utf-8 -*-
#
# Spherical Earth on a distorted (bumped) mesh must give spherical Love
# numbers.
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
"""Spherical Earth on a distorted (bumped) mesh must give spherical Love numbers."""
import sys; sys.path.insert(0,'src')
from earthtides.model import EarthModel
from earthtides.geometry import BumpGeometry
from earthtides.solver import ZonalSolver
for name in ('PREMhard', 'PREMs'):
    md = EarthModel(f'data/{name}.dat')
    ref = ZonalSolver(md, K=2, N=30, omega_f=0).solve()
    for amp in (0.05, 0.2):
        for K, N in ((3, 24), (5, 30), (7, 30)):
            S = ZonalSolver(md, BumpGeometry(md, amp), K=K, N=N, omega_f=0, background='spherical')
            r = S.solve()
            print(f"{name:8s} amp={amp:.2f} K={K} N={N}: dh={r['h']-ref['h']:+.1e} dl={r['l']-ref['l']:+.1e}"
                  f" dk={r['kk']-ref['kk']:+.1e}  U0={r['U'][0]:+.1e} U4={r['U'][4]:+.1e} k0={r['k'][0]:+.1e}")
