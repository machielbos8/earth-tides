# -*- coding: utf-8 -*-
#
# Is the 8e-4 angular-momentum floor the first-order figure? Homogeneous rotating spheroid (P1 v2 Sect. 6.3, S12)
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
# 10/11/2026 Machiel Bos
#===============================================================================
"""The balance dH/dt + Omega x H = N holds for PREM to 8e-4 of the tidal torque, which Sect. S12 attributes to the
first-order description of the figure. Test on a homogeneous, self-gravitating, rotating elastic body
(data/HOMOG.dat), O1 frequency 0.927 cycles per rotation, the spin scaled by sqrt(scale) (flattening ~ scale):
(1) the first-order map of eq. (4) with the Clairaut flattening, as for the Earth; (2) the exact spheroid
(ExactSpheroidGeometry) with that flattening; (3) the exact spheroid with the exact equilibrium flattening of a
homogeneous body (Maclaurin). The torque is itself proportional to the flattening, so a relative residual ~ f is
an absolute error ~ f^2.
Usage: python scripts/am_floor_spheroid.py  ->  results/am_floor_spheroid.txt"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import brentq

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, ExactSpheroidGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
import check_angular_momentum as cam

G = 6.674e-11
RHO = 5500.0                       # data/HOMOG.dat
SCALES = (0.25, 0.5, 1.0, 2.0)
SIG = 0.927


def maclaurin(om):
    q = om ** 2 / (np.pi * G * RHO)
    F = lambda e: 2 * np.sqrt(1 - e * e) * (3 - 2 * e * e) * np.arcsin(e) / e ** 3 - 6 * (1 - e * e) / e ** 2 - q
    e = brentq(F, 1e-4, 0.9)
    return 1 - np.sqrt(1 - e * e)


def residual(md, g, om, K=4, N=20):
    w = SIG * om
    S = TidalSolver(md, g, m=1, K=K, N=N, omega_f=0.0, coriolis=om, omega_t=w, omega_rot=om, wahr_frame=True,
                    consistent_reference=True)
    S.solve()
    Hc, Nc, lhs = cam.balance(S, w)
    return float(np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc))


def main():
    from provenance import Run
    with Run('angular-momentum floor on a homogeneous rotating spheroid: first-order map vs exact figure',
             params=dict(scales=SCALES, sig=SIG), inputs=['data/HOMOG.dat']) as run:
        md = EarthModel('data/HOMOG.dat')
        f1 = ClairautGeometry(md).flattening_surface          # at scale 1
        lines = ['homogeneous rotating elastic body (HOMOG), O1-like frequency 0.927 cycles per rotation, K 4 N 20; '
                 '|dH/dt + Omega x H - N| / |N|',
                 f'{"scale":>5s} {"f Clairaut":>10s} {"f Maclaurin":>11s} {"first-order map":>16s} '
                 f'{"exact, f Cl.":>13s} {"exact, f Mac.":>14s}']
        for sc in SCALES:
            om = OMEGA_EARTH * np.sqrt(sc)
            fc, fm = f1 * sc, maclaurin(om)
            r1 = residual(md, ClairautGeometry(md, scale=sc), om)
            r2 = residual(md, ExactSpheroidGeometry(fc), om)
            r3 = residual(md, ExactSpheroidGeometry(fm), om)
            lines.append(f'{sc:5.2f} {fc:10.6f} {fm:11.6f} {r1:9.2e} ={r1 / fc:5.3f}f {r2:8.2e} ={r2 / fc:5.3f}f '
                         f'{r3:9.2e}')
            print(lines[-1], flush=True)
        om = OMEGA_EARTH
        lines.append('exact figure (Maclaurin), scale 1, resolution: ' + ', '.join(
            f'K {K} N {N} {residual(md, ExactSpheroidGeometry(maclaurin(om)), om, K, N):.2e}'
            for K, N in ((4, 20), (6, 24), (8, 30))))
        run.write_text('results/am_floor_spheroid.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
