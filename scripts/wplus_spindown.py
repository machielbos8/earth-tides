# -*- coding: utf-8 -*-
#
# Second-order part of the degree-3 toroidal term w+ and of l(1) in PREM (P1 review, M3).
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
"""Does a first-order theory miss part of the diurnal w+ (and so of l(1)) of PREM?

DDW's diurnal l(1) exceeds ours by 2.2-2.8e-4 (their w+ 0.00024 against our 0.00016 at O1), while at M2 the
two agree to 1e-5. In the Smith test (smith1974_sign_test.txt, two-layer solid body) the first-order w+
differs from the solver's by 15-22 per cent at the Earth's flattening, falling in proportion to the
flattening: a large second-order part. Here the same spin-down on PREM, with the solver only: the
first-order (Clairaut) ellipticity scaled by k = spin^2, the reference rotation (centrifugal potential,
consistent reference) by spin, the Coriolis force kept at the Earth's rate, as in check_smith1974.py.
Each quantity q(k) is fitted by c + a k + b k^2 over k = 1, 0.5625, 0.25, 0.1225, 0.0625: a first-order
theory has c + a k, so b is the second-order part at the Earth's flattening (k = 1). If DDW's offset were a
second-order effect that first-order theory misses, b would be about -8e-5 in w+ and -2.2e-4 in l(1) at O1,
and about 0 at M2.

Quantities: w+/l0 and h+/h0, l+/l0 after removing the best-fitting rigid rotation (Wahr 1981; as the Smith
test, harmonic form of DDW eq. 20), and l(1) of the IERS fit (mantle frame, as in the paper). Elastic PREM,
inviscid slipping core, K 4, N 24.

    .venv/bin/python scripts/wplus_spindown.py   ->  results/wplus_spindown.txt"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit_fields
import check_rotating_spheroid as crs
from check_smith1974 import Scaled
from provenance import Run

SPINS = (1.0, 0.75, 0.5, 0.35, 0.25)
TIDES = (('O1', 1, 0.92700), ('OO1', 1, 1.07300), ('M2', 2, 1.4053e-4 / OMEGA_EARTH))


def one(md, clairaut, spin, m, fr):
    a = md.a
    geom = Scaled(clairaut, spin**2)
    S = TidalSolver(md, geom, m=m, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=fr * OMEGA_EARTH,
                    omega_rot=spin * OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve()
    th = crs.th
    eps_a = float(geom.eps(md.n_layers - 1, np.asarray(a))[0])
    ur, ut, up = S.surface_field(th, remove_rigid=False)
    st, sl = crs.unrigid(m, 1 - 2 / 3 * eps_a * 0.5 * (3 * np.cos(th)**2 - 1), ut, -up)
    f = crs.fit(m, ur, st, sl)
    q = lambda k, r: complex(f[k] / f[r]).real
    ur2, ut2, up2 = S.surface_field(th)                      # mantle (Tisserand) frame, as the paper
    g = iers_fit_fields(m, th, ur2, ut2, up2)
    return dict(wp=q('wp', 'l0'), hp=q('hp', 'h0'), lp=q('lp', 'l0'),
                l1=complex(g['l1'] / g['l0']).real, eps=eps_a)


def main(run):
    md = EarthModel(str(ROOT / 'data' / 'PREM.dat'))
    clairaut = ClairautGeometry(md)
    k = np.array([s**2 for s in SPINS])
    lines = ['Second-order part b of q(k) = c + a k + b k^2, k = (spin)^2 scales the first-order ellipticity;',
             'elastic PREM, inviscid slipping core, K 4 N 24, Coriolis at the Earth rate.',
             'q: w+/l0, h+/h0, l+/l0 (best-fit rigid rotation removed), l(1)/l(0) (mantle frame).',
             'b/q(1): second-order share at the Earth\'s flattening; rms: fit residual.', '']
    for name, m, fr in TIDES:
        R = [one(md, clairaut, s, m, fr) for s in SPINS]
        lines.append(f'{name}: eps at the surface ' + ', '.join(f'{r["eps"]:.2e}' for r in R))
        for key in ('wp', 'hp', 'lp', 'l1'):
            y = np.array([r[key] for r in R])
            M = np.column_stack([1 + 0 * k, k, k**2])
            c = np.linalg.lstsq(M, y, rcond=None)[0]
            rms = np.sqrt(np.mean((M @ c - y)**2))
            lines.append(f'  {key:3s}/ref  q(k) = ' + ' '.join(f'{v:+.6e}' for v in y)
                         + f'   c {c[0]:+.3e}  a {c[1]:+.3e}  b {c[2]:+.3e}  b/q(1) {c[2] / y[0]:+.3f}'
                           f'  rms {rms:.1e}')
        lines.append('')
    print('\n'.join(lines))
    run.write_text(str(ROOT / 'results' / 'wplus_spindown.txt'), '\n'.join(lines) + '\n')


if __name__ == '__main__':
    with Run('P1 review M3: second-order part of the diurnal w+ and l(1) of PREM (spin-down)',
             params=dict(spins=list(SPINS), K=4, N=24), inputs=['data/PREM.dat']) as run_:
        main(run_)
