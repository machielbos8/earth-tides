# -*- coding: utf-8 -*-
#
# check_hard.
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
import sys; sys.path.insert(0,'src')
from earthtides.model import EarthModel
from earthtides.solver import ZonalSolver
REF = dict(h=0.41204134704, l=0.16953322651/2, k=0.40616483041/2)
for name, ref in (('PREMhard', REF),):
    md = EarthModel(f'data/{name}.dat')
    for N in (12, 16, 20, 24, 30, 40):
        r = ZonalSolver(md, K=2, N=N).solve()
        print(name, N, f"dh={r['h']-ref['h']:+.2e} dl={r['l']-ref['l']:+.2e} dk={r['kk']-ref['k']:+.2e}")
