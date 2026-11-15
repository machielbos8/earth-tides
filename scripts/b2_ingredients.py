# -*- coding: utf-8 -*-
#
# B2: which ingredient of PREM makes the inviscid core near-singular? Wahr
# truncation, N refinement, solver spread in h(0) and k.
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
# 20/10/2026 Machiel Bos
#===============================================================================
"""B2: which ingredient of PREM makes the inviscid core near-singular? Wahr truncation, N refinement,
solver spread in h(0) and k."""
import sys, copy, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, SphereGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
import convergence_stratification as CS
DEG = {1: ([2], [1, 3]), 2: ([2], [3])}
def prem(): return EarthModel('data/PREM.dat')
def stiff_core(f=100.0):
    md = prem(); ic = [i for i in range(md.n_layers) if abs(md.mu[i](0.5*(md.r0[i]+md.r1[i]))) < 1][0]
    md = copy.copy(md); md.lam = list(md.lam); md.lam[ic] = md.lam[ic] * f; return md
variants = [('PREM', prem, {}), ('PREM, no inner core', lambda: EarthModel('data/PREM_noIC.dat'), {}),
            ('PREM, neutral core (N2 = 0)', lambda: CS.stratified(0.0), {}),
            ('PREM, core kappa x 100', stiff_core, {}),
            ('PREM, no self-gravitation', prem, {'gravity': False})]
def run(md, m, s, N, ls, flat, extra):
    g = ClairautGeometry(md) if flat else SphereGeometry()
    S = TidalSolver(md, g, m=m, N=N, degrees=DEG[m], omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=s * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=flat and extra.get('gravity', True), frame='mantle', linear_solver=ls, **extra)
    r = S.solve(); return np.array([r['h'], r['kk']], complex)
flat = len(sys.argv) > 1 and sys.argv[1] == 'flat'
for name, mk, extra in variants:
    md = mk()
    for m, s in ((2, 1.9275), (1, 0.927)):
        row = []
        for N in (16, 32, 60):
            v = np.array([run(md, m, s, N, ls, flat, extra) for ls in ('sparse', 'block', 'dense')])
            row.append(f'N{N}: spread {np.abs(v - v[0]).max():.0e}')
        print(f'{"flat  " if flat else "sphere"} {name:30s} m={m}: ' + ' | '.join(row), flush=True)
