# -*- coding: utf-8 -*-
#
# Validation of the level figure (geometry.LevelGeometry): step 1 of docs/plan_level_figure_2026-11-14.md.
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
"""(a) Published second-order hydrostatic figure of PREM: Chambat, Ricard & Valette (2010, GJI 183, 727), their
    eqs (26)-(27) with the rotational factor of Denis (1989), m = Omega^2 R^3 / GM = 0.00345039:
    f2 = -2.228947e-3, f4 = 4.445e-6 (Denis: -2.228946e-3, 4.465e-6), J2 = 1.0721e-3 (a = 6378137 m), where the
    surface is s = r (1 + f2 P2 + f4 P4), r its mean radius. Our rotation rate is set to give the same m.
    J2, J4 from the exterior coefficients of the reference potential of the figure.
(b) Is degree 6 enough? Correction of degrees 2-8 with the reference potential at K 5 (degrees 0-8).
(c) Chebyshev nodes per layer M 10 -> 14.
(d) The offset of the FCN from 1 cpsd must grow as Omega^2 (both e_f and the compliance do): elastic PREM
    spun 0, 2 and 3.27 % faster, slipping core, K 4, N 24, period in days of that Earth times (1 + x)^2.
Usage: python scripts/validate_level_figure.py  ->  results/validate_level_figure.txt"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel, G
from earthtides.geometry import figure, LevelGeometry, OMEGA_EARTH
from earthtides.refpot import ReferencePotential
from earthtides.solver import TidalSolver
from provenance import Run

A_REF, M_DENIS = 6378137.0, 0.00345039


def shape(g, s=None):
    c = g.surface_legendre(s=s)
    return c[0], c[2] / c[0], c[4] / c[0], c[6] / c[0], c[8] / c[0]


def fcn(md, geo, Om, K=4, N=24, sig=(1.0012, 1.0016, 1.0030, 1.0036)):
    x = np.array(sig)
    k = []
    for s in x:
        T = TidalSolver(md, geo, m=1, K=K, N=N, omega_f=0.0, coriolis=Om, omega_t=s * Om, omega_rot=Om,
                        wahr_frame=True, dyn_fluid=True, consistent_reference=True, frame='mantle')
        T.solve()
        k.append(T.results()['kk'].real)
    k = np.array(k)
    return 1 / (np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0] - 1)


def main():
    with Run('validation of the level figure: published second-order figure of PREM, degrees, nodes, Omega^2',
             params=dict(m_denis=M_DENIS), inputs=['data/PREM.dat']) as run:
        md = EarthModel('data/PREM.dat')
        m0 = OMEGA_EARTH ** 2 * md.a ** 3 / (G * md.M)
        Om = OMEGA_EARTH * np.sqrt(M_DENIS / m0)
        g = LevelGeometry(md, omega=Om)
        r0, f2, f4, f6, f8 = shape(g)
        rp = ReferencePotential(md, g, Om)
        c = rp.coef[md.n_layers * rp.nb:]
        J2, J4 = c[1] * md.a ** 3 / (G * md.M * A_REF ** 2), c[2] * md.a ** 5 / (G * md.M * A_REF ** 4)
        g1 = figure(md, 'first-order', omega=Om)
        lines = [f'(a) PREM (data/PREM.dat: ocean merged with the crust), m = {M_DENIS} (ours at Omega_Earth: {m0:.8f})',
                 f'    {"":28s} {"f2":>14s} {"f4":>11s} {"J2":>12s} {"J4":>11s}',
                 f'    {"level figure":28s} {f2:14.6e} {f4:11.3e} {J2:12.4e} {J4:11.2e}   (f6 {f6:.1e}, mean radius {r0:.1f} m)',
                 f'    {"Chambat et al. (2010)":28s} {-2.228947e-3:14.6e} {4.445e-6:11.3e} {1.0721e-3:12.4e} {"":>11s}',
                 f'    {"Denis (1989)":28s} {-2.228946e-3:14.6e} {4.465e-6:11.3e}',
                 f'    {"first-order map":28s} {-2 / 3 * g1.flattening_surface:14.6e} {0.0:11.3e}',
                 f'    iterations {len(g.history)}, max height left {g.history[-1][1]:.1e} m']
        print('\n'.join(lines), flush=True)
        g = LevelGeometry(md)
        b = shape(g)
        cm = shape(g, 3480e3)
        lines += ['', '(b), (c) at Omega_Earth: surface f2, f4, f6, f8 and core-mantle boundary f2, f4; change from the default']
        fmt = lambda s_, c_: ' '.join(f'{v:13.6e}' for v in s_[1:]) + '  | ' + ' '.join(f'{v:13.6e}' for v in c_[1:3])
        lines.append(f'    {"default (2-6, K 4, M 10)":28s} ' + fmt(b, cm))
        for name, kw in (('degrees 2-8, K 5', dict(degrees=(2, 4, 6, 8), ref_K=5)), ('M 14', dict(M=14)),
                         ('degrees 2-8, K 5, M 14', dict(degrees=(2, 4, 6, 8), ref_K=5, M=14))):
            gg = LevelGeometry(md, **kw)
            s_, c_ = shape(gg), shape(gg, 3480e3)
            lines.append(f'    {name:28s} ' + fmt(s_, c_))
            lines.append(f'    {"  change (m of radius)":28s} ' + ' '.join(f'{(v - w) * md.a:13.2e}' for v, w in zip(s_[1:], b[1:]))
                         + '  | ' + ' '.join(f'{(v - w) * 3480e3:13.2e}' for v, w in zip(c_[1:3], cm[1:3]))
                         + f'   ({len(gg.history)} it., left {gg.history[-1][1]:.1e} m, outside {gg.history[-1][2]:.1e} m)')
            print('\n'.join(lines[-2:]), flush=True)
        lines += ['', '(d) FCN (days of that Earth) x (1 + x)^2, elastic PREM, slipping core, K 4, N 24',
                  f'    {"spin-up x":12s} {"first-order":>12s} {"level":>12s}']
        for x in (0.0, 0.02, 0.0327):
            O = OMEGA_EARTH * (1 + x)
            P = [fcn(md, figure(md, kind, omega=O), O) * (1 + x) ** 2 for kind in ('first-order', 'level')]
            lines.append(f'    {100 * x:10.2f} % {P[0]:12.3f} {P[1]:12.3f}')
            print(lines[-1], flush=True)
        run.write_text('results/validate_level_figure.txt', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
