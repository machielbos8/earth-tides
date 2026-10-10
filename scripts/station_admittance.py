# -*- coding: utf-8 -*-
#
# Section 6: body-tide surface field of anelastic, hydrostatic PREM at every
# line of the HARDISP catalog (342 degree-2 lines), for harmonic synthesis of
# station displacements.
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
# 8/10/2026 Machiel Bos
#===============================================================================
"""Section 6: body-tide surface field of anelastic, hydrostatic PREM at every line of the HARDISP
catalog (342 degree-2 lines), for harmonic synthesis of station displacements.

One solve per line: fluid core, Coriolis and inertia, consistent reference potential, mantle frame
(as in convention_table.py), absorption-band anelasticity evaluated at the line frequency.
Stored: u_r, u_theta, u_lambda on the flattened surface (rigid mantle rotation removed),
normalized as the IERS Love numbers (potential at the equatorial radius), on a colatitude grid.
Usage: python scripts/station_admittance.py  ->  results/station_admittance.npz  (~25 min)
"""
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np

sys.path.insert(0, 'src')
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.synthesis import catalog, doodson_freq

md0 = EarthModel('data/PREM.dat')
geo = ClairautGeometry(md0)
a = md0.a
e, _, _ = geo.eps(md0.n_layers - 1, np.array([a]))
norm = (a * (1 + e[0] / 3.0))**4 / a**3
th = np.radians(np.linspace(0.25, 179.75, 719))
idd, tamp = catalog()
f_cpd = doodson_freq(idd)
U = np.zeros((len(idd), 3, len(th)), complex)
t0 = time.time()
for k, (d, f) in enumerate(zip(idd, f_cpd)):
    m = int(d[0])
    w = f * 2 * np.pi / 86400.0
    S = TidalSolver(md0.anelastic(w), geo, m=m, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=w, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve()
    U[k] = np.array(S.surface_field(th)) / norm
    if k % 20 == 0:
        print(f'{k:3d}/{len(idd)}  m={m}  f={f:.6f} cpd  ({time.time() - t0:.0f} s)', flush=True)
np.savez('results/station_admittance.npz', theta=th, U=U, idd=idd, f_cpd=f_cpd)
print('done', time.time() - t0)
