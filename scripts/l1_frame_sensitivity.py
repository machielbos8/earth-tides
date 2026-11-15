# -*- coding: utf-8 -*-
#
# Sensitivity of the diurnal l(1) to how the rigid rotation of the mantle is removed (P1 review, M3).
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
# 15/11/2026 Machiel Bos
#===============================================================================
"""Referee (M3): DDW's diurnal l(1) exceeds ours by 2-3e-4; could it come from how the rigid rotation xi of
the mantle is defined or removed? The rotation removed is metres (3 m at O1, 17 m at P1); on the flattened
surface xi x x has a degree-3 toroidal part of order eps times the tilt.

Elastic, hydrostatic PREM on the first-order figure (like for like with DDW, Table tab:ddw), inviscid
slipping core, K 4, N 24, set-up of scripts/ddw_fingerprint.py. For each tide the parameters are fitted to
the surface displacement after removing the rotation in five ways:
  tisserand       Tisserand rotation of the mantle, xi x x at the flattened surface R(a, theta) (the paper)
  sphere          the same xi, subtracted as xi x x at r = a (a pure degree-1 field on the sphere)
  xi +1 %, -1 %   the Tisserand xi scaled by 1.01 / 0.99 (the nutations are reproduced to 1-1.4 %)
  surface         the rotation that best fits the surface displacement (Wahr 1981, eqs 4.3, 4.6)
  none            nothing removed (the size of the effect)

    .venv/bin/python scripts/l1_frame_sensitivity.py   ->  results/l1_frame_sensitivity.txt"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH, equatorial_radius
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit_fields
from provenance import Run

TIDES = (('O1', 1, 0.92700), ('P1', 1, 0.99454), ('OO1', 1, 1.07300), ('M2', 2, 1.4053e-4 / OMEGA_EARTH))
DDW_L1 = {'O1': 0.00070, 'P1': 0.00070, 'OO1': 0.00077, 'M2': 0.00100}
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2')


def main(run):
    md = EarthModel(str(ROOT / 'data' / 'PREM.dat'))
    g = ClairautGeometry(md)
    a = md.a
    norm = equatorial_radius(g, md)**4 / a**3
    th = np.linspace(0.03, np.pi - 0.03, 600)
    lines = ['l(1) and the other parameters for five ways of removing the rigid rotation of the mantle;',
             'elastic hydrostatic PREM, first-order figure, inviscid slipping core, K 4, N 24',
             '', f'{"tide":5s} {"removal":12s} ' + ' '.join(f'{k:>9s}' for k in KEYS) + '   xi*a (m per m of tide)']
    for name, m, sig in TIDES:
        S = TidalSolver(md, g, m=m, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                        omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=True,
                        frame='mantle')
        S.solve()
        ur, ut0, up0 = S.surface_field(th, remove_rigid=False)
        variants = {}
        if m == 1:
            ut, up = S._remove_mantle_rotation(th, ut0, up0)
            xi = S._xi
            bt, bp = S._rigid_pattern(md.n_layers - 1, a, th)                 # at R(a, theta)
            st, sp = a + 0 * th, -a * np.cos(th)                               # at r = a
            variants = {'tisserand': (ut, up), 'sphere': (ut0 - xi * st, up0 - xi * sp),
                        'xi +1 %': (ut0 - 1.01 * xi * bt, up0 - 1.01 * xi * bp),
                        'xi -1 %': (ut0 - 0.99 * xi * bt, up0 - 0.99 * xi * bp)}
            S.frame = 'surface'
            variants['surface'] = S.surface_field(th)[1:]
            variants['none'] = (ut0, up0)
        else:
            xi = 0.0
            variants = {'none (m=2)': (ut0, up0)}
        for lab, (ut, up) in variants.items():
            f = iers_fit_fields(m, th, ur / norm, ut / norm, up / norm)
            lines.append(f'{name:5s} {lab:12s} ' + ' '.join(f'{complex(f[k]).real:9.5f}' for k in KEYS)
                         + (f'   {abs(xi) * a / norm:.3f}' if lab == 'tisserand' else ''))
        lines.append(f'{name:5s} {"DDW l(1)":12s} ' + ' ' * 30 + f'{DDW_L1[name]:9.5f}')
        lines.append('')
    print('\n'.join(lines))
    run.write_text(str(ROOT / 'results' / 'l1_frame_sensitivity.txt'), '\n'.join(lines) + '\n')


if __name__ == '__main__':
    with Run('P1 review M3: diurnal l(1) against the way the rigid rotation of the mantle is removed',
             params=dict(figure='first-order', K=4, N=24), inputs=['data/PREM.dat']) as run_:
        main(run_)
