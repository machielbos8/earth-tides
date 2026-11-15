# -*- coding: utf-8 -*-
#
# Supplement figures S1 (convergence at M2) and S2 (latitude dependence at M2) on the level figure.
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
# 15/11/2026 Machiel Bos
#===============================================================================
"""Figures fig3_convergence.pdf and fig4_latitude_M2.pdf of the supplement, recomputed on the figure of
EARTHTIDES_FIGURE (default 'level'), with the computation and the plots of paper/make_figures.py (compute(),
fig3(), fig4(); first-order figure, 2026-10-10/17): hydrostatic PREM with a weak-solid core (PREMs), M2,
rotating, mantle frame. Convergence: K = 2..6 at N = 40 and N = 10..34 at K = 7 against K = 7, N = 40.
Latitude: flattening only (no Coriolis force), Coriolis force only (sphere) and both; K 5, N 30. The
displacement is normalised with the potential at the equatorial radius of the figure.

    .venv/bin/python paper_pepi/make_figS_level.py [--replot]
    ->  paper_pepi/figures/fig3_convergence.pdf, fig4_latitude_M2.pdf, results/figdata_S_level.npz"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'paper'))
os.environ.setdefault('EARTHTIDES_FIGURE', 'level')
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import OMEGA_EARTH, figure, polar_equatorial_radius
from earthtides.solver import TidalSolver, ybar
from earthtides.iers import iers_fit
from figure_choice import FIGURE, SUF
from provenance import Run
import make_figures as mf                     # paper/make_figures.py: the plots (fig3, fig4) and the style

mf.FIG = ROOT / 'paper_pepi' / 'figures'
CACHE = ROOT / 'results' / f'figdata_S{SUF}.npz'
W_M2 = mf.W_M2


def compute(run):
    md = EarthModel(ROOT / 'data/PREMs.dat')
    g = figure(md, FIGURE)
    d = {}

    def m2(K, N):
        S = TidalSolver(md, g, m=2, K=int(K), N=int(N), omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=W_M2, omega_rot=OMEGA_EARTH)
        S.solve()
        f = iers_fit(S, g)
        return [f['h0'], f['h2'], f['l2'], f['l1']]
    ref = np.array(m2(7, 40))
    d['f3_ref'] = ref
    d['f3_K'] = np.arange(2, 7)
    d['f3_errK'] = np.array([np.abs(np.array(m2(K, 40)) - ref) for K in d['f3_K']])
    d['f3_N'] = np.array([10, 14, 18, 22, 26, 30, 34])
    d['f3_errN'] = np.array([np.abs(np.array(m2(7, N)) - ref) for N in d['f3_N']])
    th = np.linspace(np.radians(10), np.radians(170), 321)
    d['f4_th'] = th
    for tag, scale, cor in (('ell', 1.0, 0.0), ('cor', 0.0, OMEGA_EARTH), ('both', 1.0, OMEGA_EARTH)):
        gg = figure(md, FIGURE, scale=scale)
        S = TidalSolver(md, gg, m=2, K=5, N=30, omega_f=0.0, coriolis=cor, omega_t=W_M2,
                        omega_rot=OMEGA_EARTH * np.sqrt(scale))
        S.solve()
        re = polar_equatorial_radius(gg, md)[1]
        norm = re ** 4 / md.a ** 3
        ur, ut, up = (v / norm for v in S.surface_field(th))
        phi = np.pi / 2 - th
        Y = ybar(2, 2, th, 1)
        d[f'f4_h_{tag}'] = ur / Y[0]
        d[f'f4_lN_{tag}'] = (-ut) / (-Y[1])
        d[f'f4_lE_{tag}'] = up / (-2 * Y[0] / np.cos(phi))
    run.savez(str(CACHE.relative_to(ROOT)), **d)
    return d


def main():
    os.chdir(ROOT)
    with Run(f'supplement figures S1/S2 (convergence and latitude at M2) on the {FIGURE} figure',
             params=dict(figure=FIGURE), inputs=['data/PREMs.dat']) as run:
        d = dict(np.load(CACHE)) if ('--replot' in sys.argv and CACHE.exists()) else compute(run)
        mf.fig3(d)
        mf.fig4(d)
        for f in ('fig3_convergence.pdf', 'fig4_latitude_M2.pdf'):
            run.output(f'paper_pepi/figures/{f}')
        print('reference K 7 N 40 (h0 h2 l2 l1):', np.real(d['f3_ref']))
        print('max |change| K 3 -> 7:', np.abs(d['f3_errK'][1]).max(), ' K 4:', np.abs(d['f3_errK'][2]).max())
        print('N 26 / 30 at K 7:', np.abs(d['f3_errN'][4]).max(), np.abs(d['f3_errN'][5]).max())


if __name__ == '__main__':
    main()
