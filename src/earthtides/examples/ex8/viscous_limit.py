# -*- coding: utf-8 -*-
#
# The Love numbers of a viscous liquid core in the limit of vanishing viscosity.
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
"""An inviscid liquid core inside a shell with an inner core has an ill-posed forced response; a viscous
core is well posed, and its limit of vanishing viscosity is the reference (manual, Sect. 'The viscous
core'). Elastic PREM on the level figure, O1: the core gets a viscosity with Ekman-layer thickness
delta = 1000, 500 and 250 m (EarthModel.viscous_core: mu = i omega eta, sub-layers of 40 km and 5 delta
at every wall), core K = 12, and the Love numbers are extrapolated to delta -> 0 with a quadratic in delta.
The limit is compared with the coarse inviscid core that slips along its boundaries (K = 4, N = 24),
which on the level figure gives the same Love numbers away from the free core nutation.
Run:  python viscous_limit.py   ->  viscous_limit.out   (about 30 seconds, 1 GB)
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.love import model_path
from earthtides.geometry import figure, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer, viscous_core_resolution
from earthtides.iers import iers_fit

SIG = 0.92700                       # O1 [cycles per sidereal day]
DELTAS = (1000.0, 500.0, 250.0)     # Ekman-layer thickness [m]
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2')


def solve(md, layer_K=None, layer_N=None, K=4, N=24, band=None):
    g = figure(md, 'level')
    S = TidalSolver(md, g, m=1, K=K, N=N, layer_K=layer_K, layer_N=layer_N, omega_f=0.0,
                    coriolis=OMEGA_EARTH, omega_t=SIG * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True,
                    dyn_fluid=True, consistent_reference=True, frame='mantle', band=band)
    S.solve()
    f = iers_fit(S, g)
    return np.array([complex(f[k]).real for k in KEYS])


def main():
    md = EarthModel(str(model_path('PREM')))
    lines = [f'O1, elastic PREM, level figure; {"":14s}' + ''.join(f'{k:>10s}' for k in KEYS)]
    rows = []
    for d in DELTAS:
        mv = md.viscous_core(SIG * OMEGA_EARTH, delta=d)
        Kl, Nl = resolution_per_layer(mv, viscous_core_resolution(12, d))
        rows.append(solve(mv, Kl, Nl, band=12))
        lines.append(f'viscous core, delta {d:6.0f} m   ' + ''.join(f'{x:10.6f}' for x in rows[-1]))
    x = np.array(DELTAS) / 1e3
    limit = np.array([np.polyfit(x, np.array(rows)[:, j], 2)[-1] for j in range(len(KEYS))])
    lines.append('viscous core, delta -> 0      ' + ''.join(f'{v:10.6f}' for v in limit))
    slip = solve(md)
    lines.append('inviscid core, slipping       ' + ''.join(f'{v:10.6f}' for v in slip))
    lines.append('difference (limit - slipping) ' + ''.join(f'{v:+10.1e}' for v in limit - slip))
    print('\n'.join(lines))
    open('viscous_limit.out', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
