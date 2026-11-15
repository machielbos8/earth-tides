# -*- coding: utf-8 -*-
#
# Fig. 1 of P1 v2: the level-surface coordinates, drawn for the quarter of the meridional plane that the
# solver computes.
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
# 09/11/2026 Machiel Bos
#===============================================================================
"""As Fig. 1 of the J Geod paper (paper/make_figures.py fig_grid, from results/figdata.npz), but panel (a)
draws in full only the quarter of the meridional plane that the solver works on (one order m, one parity
about the equator), with its angular nodes. Run from the project root:
python paper_pepi/make_fig_grid.py  ->  paper_pepi/figures/fig_grid.pdf"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'paper'))
import make_figures as mf

if __name__ == '__main__':
    mf.fig_grid(dict(np.load(mf.CACHE)), quarter=True, out=ROOT / 'paper_pepi' / 'figures' / 'fig_grid.pdf')
    print('paper_pepi/figures/fig_grid.pdf')
