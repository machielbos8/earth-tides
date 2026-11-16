# -*- coding: utf-8 -*-
#
# The level figure against the first-order (Clairaut) figure of PREM.
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
"""The first-order figure (Clairaut) describes the flattened level surfaces to first order in the
flattening; the level figure makes every coordinate surface a level surface of the model's own
reference potential (manual, Sect. 'The level figure'). For elastic, hydrostatic PREM the script prints,
on both figures, the inverse flattening of the surface, the equatorial minus polar radius of the
core-mantle boundary, the period of the free core nutation (pole fit of k, as in ex7) and h0 at O1 and
psi1. Away from the resonance the two figures agree to a few 1e-5; next to it the period differs by
about a day, and h0 at psi1 by 5e-3.
Run:  python figures.py   ->  figures.out   (about 10 seconds)
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from earthtides import love
from earthtides.model import EarthModel
from earthtides.geometry import figure, polar_equatorial_radius

WAVES = (('O1', 1, 0.92954), ('psi1', 1, 1.005475688))


def fcn_period(opts, sig=(1.0012, 1.0016, 1.0030, 1.0036)):
    x = np.array(sig)
    k = np.array([love.solve_wave(opts, 1, s * love.F_TILT)[1]['kk'] for s in x]).real
    s_fcn = np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0]
    return 1.0 / (s_fcn - 1.0)


def main():
    md = EarthModel(str(love.model_path('PREM')))
    i_cmb = [i for i in range(md.n_layers) if float(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) == 0.0][-1]
    lines = []
    for kind in ('first-order', 'level'):
        g = figure(md, kind)
        c, a = polar_equatorial_radius(g, md)
        c_cmb, a_cmb = polar_equatorial_radius(g, md, i_cmb, md.r1[i_cmb])
        opts = love.Options(anelastic=False, figure=kind)
        h0 = {w: complex(love._job((opts, 2, m, f))['h0']).real for w, m, f in WAVES}
        lines.append(f'{kind:12s} surface 1/f {a / (a - c):.2f}  CMB a - c {a_cmb - c_cmb:.1f} m  '
                     f'FCN {fcn_period(opts):.2f} sd  ' + '  '.join(f'h0 {w} {v:.5f}' for w, v in h0.items()))
    print('\n'.join(lines))
    open('figures.out', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
