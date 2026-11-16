# -*- coding: utf-8 -*-
#
# Is the core-mantle boundary a level surface, and does the angular-momentum residual follow that?
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
"""With slip_nonlevel on, the residual of the angular-momentum balance (check_angular_momentum.balance: volume
integrals of the whole Earth, no wall terms) of hydrostatic PREM with a slipping core rises at psi1 from
9.3e-4 to 1.3e-3 (lab notes 2026-11-14). Test: is the wall a level surface of the reference potential, and
does the rise disappear where it is one? Three figures of the same Earth: the first-order (Clairaut) map and
exact ellipses of the same flattening (exact_levels_share_test.ExactLevelsGeometry), and a figure whose
coordinate surfaces are level surfaces of the solver's reference potential (level_geometry.py). Per figure:
- the height of each fluid-solid wall above the level surface through its mean, from the reference potential
  the solver uses: h(theta) = int (dPhi0/dr R_theta + dPhi0/dtheta) dtheta / g (peak to peak, and its
  Legendre content P2, P4, P6);
- the residual |i omega H + Omega x H - N| / |N| with the term off and on, and the change of h0 and h2.
If the wall is level the term vanishes and off = on.
Usage: python scripts/am_wall_level.py [--K 4] [--N 24]  ->  results/am_wall_level_K<K>_N<N>.txt"""
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
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit
import check_angular_momentum as cam
import level_geometry as lg
from provenance import Run

WAVES = (('O1', 0.92700), ('P1', 0.99454), ('psi1', 1.002730421), ('phi1', 1.005460985), ('OO1', 1.07300))


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def solver(md, g, s, K, N, on):
    S = TidalSolver(md, g, m=1, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=s * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle', slip_nonlevel=on)
    S.solve()
    return S


def walls(S):
    """Height (m) of every fluid-solid wall above the level surface through its mean."""
    md, out = S.model, []
    th = np.asarray(S.theta[0]).ravel()
    o = np.argsort(th)
    for i in range(md.n_layers):
        if not S.true_fluid[i]:
            continue
        for s, other in ((md.r0[i], i - 1), (md.r1[i], i + 1)):
            if other < 0 or other >= md.n_layers or S.true_fluid[other]:
                continue
            p = S.props(i, np.array([s]))
            d = (p.ph0['r'] * p.R_t + p.ph0['t'])[0][o]                  # d Phi0 / d theta along the wall
            g = float(np.mean(np.abs(p.ph0['r'][0])))
            t = th[o]
            h = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(t))]) / g
            x = np.cos(t)
            c = L.legfit(x, h, [0, 2, 4, 6])
            out.append((s / 1e3, np.ptp(h), c[2], c[4], c[6], np.ptp((p.r[0][o] - np.mean(p.r[0])))))
    return out


def main():
    K, N = int(_opt('--K', 4)), int(_opt('--N', 24))
    with Run('is the core wall a level surface, and does the angular-momentum residual follow that?',
             params=dict(K=K, N=N, waves=WAVES), inputs=['data/PREM.dat']) as run:
        md = EarthModel('data/PREM.dat')
        lines = [f'hydrostatic elastic PREM, slipping core, K {K}, N {N}; residual of the angular-momentum balance '
                 'relative to the tidal torque']
        glev, hist = lg.build(md, OMEGA_EARTH, log=lambda t: lines.append('  ' + t))
        for name, g in (('first-order map', ClairautGeometry(md)), ('exact ellipses', els.ExactLevelsGeometry(md)),
                        ('level surfaces (level_geometry.py)', glev)):
            lines += ['', f'== {name}']
            first = True
            for w, s in WAVES:
                row = []
                for on in (False, True):
                    S = solver(md, g, s, K, N, on)
                    if first:
                        lines += ['  wall (km)   peak-to-peak of the wall (m)   height above its level surface: '
                                  'peak-to-peak, P2, P4, P6 (m)']
                        lines += [f'  {a:9.1f}   {pr:26.0f}   {pp:10.3f} {c2:+9.3f} {c4:+9.3f} {c6:+9.3f}'
                                  for a, pp, c2, c4, c6, pr in walls(S)]
                        lines += [f'  {"wave":5s} {"AM off":>9s} {"AM on":>9s} {"d h0":>10s} {"d h2":>10s} {"h0 (on)":>11s} {"h2 (on)":>11s}']
                        first = False
                    Hc, Nc, lhs = cam.balance(S, s * OMEGA_EARTH)
                    f = iers_fit(S, g)
                    row.append((np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc), f['h0'].real, f['h2'].real))
                lines.append(f'  {w:5s} {row[0][0]:9.2e} {row[1][0]:9.2e} {row[1][1] - row[0][1]:+10.2e} '
                             f'{row[1][2] - row[0][2]:+10.2e} {row[1][1]:+11.7f} {row[1][2]:+11.7f}')
                print(lines[-1], flush=True)
        run.write_text(f'results/am_wall_level_K{K}_N{N}.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
