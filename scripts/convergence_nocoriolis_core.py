# -*- coding: utf-8 -*-
#
# Attribution test: the production set-up with the Coriolis force switched off
# in the fluid core only (inertia kept). Without rotation the fluid core has
# no inertial modes; if the (K, N) scatter is due to them it must disappear.
# Scratch subclass; production code untouched. ->
# results/convergence_nocor_core.txt
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
# 17/10/2026 Machiel Bos
#===============================================================================
"""Attribution test: the production set-up with the Coriolis force switched off in the fluid core only
(inertia kept). Without rotation the fluid core has no inertial modes; if the (K, N) scatter is due to
them it must disappear. Scratch subclass; production code untouched. -> results/convergence_nocor_core.txt"""
import inspect
import re
import sys
import textwrap
import warnings
from multiprocessing import Pool
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides import solver as _sv
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.iers import iers_fit

src = textwrap.dedent(inspect.getsource(_sv.TidalSolver.operator))
ind = re.search(r"\n( *)cw = 2.0 \* wt \* self.coriolis", src).group(1)
src = src.replace("cw = 2.0 * wt * self.coriolis",
                  "cw = 0.0 if self.fluid_layers()[p['i']] else 2.0 * wt * self.coriolis")
ns = dict(vars(_sv))
exec(src, ns)


class NoCorCore(_sv.TidalSolver):
    operator = ns['operator']


def run(args):
    m, sig, K, N = args
    md = EarthModel('data/PREM.dat')
    g = ClairautGeometry(md)
    S = NoCorCore(md, g, m=m, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                  omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                  consistent_reference=True, frame='mantle')
    S.solve()
    f = iers_fit(S, g)
    return (m, sig, K, N) + tuple(f[k].real for k in ('h0', 'h2', 'l2', 'l1'))


if __name__ == '__main__':
    sys.path.insert(0, 'scripts')
    from provenance import Run
    _run = Run('inviscid core without the Coriolis force in the core (S9)', params=dict(argv=sys.argv[1:]), inputs=['data/PREM.dat'])
    jobs = [(2, s, K, N) for s in (1.9225, 1.92700, 1.9475) for K, N in
            ((3, 24), (4, 24), (4, 30), (5, 30), (6, 30), (7, 40))]
    with Pool(4) as p:          # Mac memory budget (2026-11-10)
        res = p.map(run, jobs, chunksize=1)
    with open('results/convergence_nocor_core.txt', 'w') as fo:
        fo.write('m sigma K N h0 h2 l2 l1\n')
        for r in res:
            fo.write('%d %.5f %d %d %.10f %.10f %.10f %.10f\n' % r)
    print('done', len(res))
    _run.output('results/convergence_nocor_core.txt')
    _run.__exit__(None, None, None)
