# -*- coding: utf-8 -*-
#
# The spurious normal stress on a wall off its level surface, Hough's Earth (stored wall fields).
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
# 19/11/2026 Machiel Bos
#===============================================================================
"""S5.5, '74 against 70 Pa': on a bumped wall (Hough's Earth, bump +3 eps^2 in P4, 0.96 cpsd) the conditions of a
level boundary hand the mantle a spurious normal stress rho g times the jump of the displacement across the level
surfaces, u . grad Phi0 (un), degree 4. Compared with the whole difference of the degree-4 normal stress on the
mantle side of the wall between the slipping and the viscous core. Both per unit U2 on the mantle side, as
scripts/wall_fields.py prints them:
  un4 (slipping core)   results/wall_fields_bump_3_un_check.txt  (job wall_fields_un_check)
  trr4 slip and visc    results/wall_fields_soft_bump_3.txt      (wall_fields.py soft,bump+3; heavy, stored)
(2026-11-19: lab notes 2026-11-14 only before.)
Run from the project root:  .venv/bin/python scripts/wall_pressure_check.py  ->  results/wall_pressure_check.txt"""
import importlib.util
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from provenance import Run

UN, TRR = 'results/wall_fields_bump_3_un_check.txt', 'results/wall_fields_soft_bump_3.txt'
COLS_NEW = ['U2', 'U4', 'U6', 'U8', 'W1', 'W3', 'trr2', 'trr4', 'trr6', 'trr8', 'un4', 'un6', 'un8', 'pE2', 'pE4',
            'pE6', 'pE8', 'phi4']
COLS_OLD = ['U2', 'U4', 'U6', 'V2', 'V4', 'V6', 'W1', 'W3', 'W5', 'trr2', 'trr4', 'trr6', 'phi2', 'phi4']


def rows(path, cols, case, kind):
    """{height above the wall (m): {column: real part}} of one block (case at 0.96 cpsd, kind slip / visc)."""
    out, inside, mine = {}, False, False
    for line in open(path):
        if line.startswith(case + ', 0.96000'):
            inside = True
            continue
        if inside and line.strip().startswith(('slip:', 'visc:')):
            mine = line.strip().startswith(kind + ':')
            continue
        if inside and not line.strip():
            if out:
                break
            continue
        t = line.split()
        if inside and mine and t and len(t) == 1 + 2 * len(cols):
            out[float(t[0])] = {c: float(t[1 + 2 * i]) for i, c in enumerate(cols)}
    return out


def main():
    with Run("spurious normal stress on the bumped wall of Hough's Earth: rho g x jump of u.grad Phi0", inputs=[UN, TRR]) as run:
        spec = importlib.util.spec_from_file_location('hough1895', 'examples/ex5/hough1895.py')
        H = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(H)
        md = H.make(vs_m=6.0, vp_c=1000.0, path=Path(tempfile.gettempdir()) / 'HOUGH_wall_check.dat',
                    rho_c=H.RHO, rho_m=H.RHO)
        rc = H.RC * 1e3
        rho, g = float(md.rho[0](rc / 2)), float(md.gravity(0, rc))
        new = rows(UN, COLS_NEW, 'bump+3', 'slip')
        slip, visc = rows(TRR, COLS_OLD, 'bump+3', 'slip'), rows(TRR, COLS_OLD, 'bump+3', 'visc')
        jump = new[-1.0]['un4'] - new[1.0]['un4']
        lines = ["Hough's Earth, bump +3, 0.96 cpsd, degree 4, per unit U2 on the mantle side of the wall",
                 f'rho {rho:.0f} kg/m3, g at the wall {g:.3f} m/s2, rho g {rho * g:.0f}',
                 f'un4 fluid side {new[-1.0]["un4"]:+.3e}, mantle side {new[1.0]["un4"]:+.3e}, jump {jump:+.3e}',
                 f'rho g x jump                                   {rho * g * jump:+7.1f}',
                 f'trr4 mantle side: slipping {slip[1.0]["trr4"]:+.2f}, viscous {visc[1.0]["trr4"]:+.2f}; '
                 f'difference {slip[1.0]["trr4"] - visc[1.0]["trr4"]:+7.1f}']
        run.write_text('results/wall_pressure_check.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
