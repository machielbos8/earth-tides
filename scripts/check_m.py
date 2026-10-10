# -*- coding: utf-8 -*-
#
# Sphere: Love numbers independent of m; bumped mesh invisible for every m.
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
"""Sphere: Love numbers independent of m; bumped mesh invisible for every m."""
import sys; sys.path.insert(0,'src')
import warnings; warnings.filterwarnings('ignore')
from earthtides.model import EarthModel
from earthtides.geometry import BumpGeometry
from earthtides.solver import TidalSolver
md = EarthModel('data/PREMhard.dat')
r0 = TidalSolver(md, m=0, K=2, N=24, omega_f=0).solve()
for m in (0, 1, 2):
    r = TidalSolver(md, m=m, K=2, N=24, omega_f=0).solve()
    print(f"sphere m={m}: dh={r['h']-r0['h']:+.1e} dl={r['l']-r0['l']:+.1e} dk={r['kk']-r0['kk']:+.1e}"
          f"  W={ {n: f'{v:.1e}' for n, v in r['W'].items()} }  alpha={r.get('alpha', 0)}")
    rb = TidalSolver(md, BumpGeometry(md, 0.2), m=m, K=5, N=30, omega_f=0, background='spherical').solve()
    print(f"bump   m={m}: dh={rb['h']-r0['h']:+.1e} dl={rb['l']-r0['l']:+.1e} dk={rb['kk']-r0['kk']:+.1e}"
          f"  maxW={max([abs(v) for v in rb['W'].values()] or [0]):.1e}"
          f"  max other U={max(abs(v) for n, v in rb['U'].items() if n != 2):.1e}")
