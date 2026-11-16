# -*- coding: utf-8 -*-
#
# A non-hydrostatic Earth: mantle loads, their supporting stress, and the free core nutation.
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
"""Hydrostatic PREM puts the free core nutation (FCN) at 459 sidereal days (anelastic, level figure);
VLBI observes about 431. Here two zonal density loads in the mantle (lower and upper mantle) are fitted
to the excess J2 of Chambat et al. (2010) and to an excess dynamical ellipticity of the core, supported
elastically, and the deformed Earth, with its deviatoric stress, becomes the reference state of the
tidal calculation (manual, Sect. 'A non-hydrostatic Earth').

For two states, the excess e_f of Mathews et al. (2002) and the excess that gives the observed period,
the script prints the excess flattening of the core-mantle boundary, the FCN period and the Love numbers
at O1, psi1 and M2. The FCN is found from k at four frequencies just above the resonance, where
k = a + b / (sigma_FCN - sigma): a pole fit, linear in (a, sigma_FCN, a sigma_FCN - b).
Run:  python nonhydrostatic.py   ->  nonhydrostatic.out   (about 30 seconds)
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from earthtides import love

J2 = 1.14e-5                      # excess J2 (Chambat et al. 2010)
STATES = (('hydrostatic', None), ('e_f of Mathews et al. (2002)', 9.73e-5), ('FCN as observed', 1.2383e-4))
WAVES = (('O1', 1, 0.92954), ('psi1', 1, 1.005475688), ('M2', 2, 1.93227))   # cycles per day


def fcn_period(opts, sig=(1.0012, 1.0016, 1.0030, 1.0036)):
    """FCN period [sidereal days, in space] from a pole fit of k at frequencies sig [cycles per sidereal day]."""
    x = np.array(sig)
    k = np.array([love.solve_wave(opts, 1, s * love.F_TILT)[1]['kk'] for s in x]).real
    s_fcn = np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0]
    return 1.0 / (s_fcn - 1.0)


def main():
    lines = []
    for name, e_f in STATES:
        nh = None if e_f is None else dict(targets={'J2': J2, 'e_f': e_f})
        opts = love.Options(nonhydrostatic=nh)
        cmb = 0.0 if e_f is None else love._model(opts)[2].summary()['response']['cmb']
        lines.append(f'{name}: excess CMB flattening {cmb:.0f} m, FCN {fcn_period(opts):.1f} sidereal days')
        for w, m, f in WAVES:
            p = love._job((opts, 2, m, f))
            lines.append(f'  {w:5s} ' + '  '.join(f'{k} {complex(p[k]).real:+.5f}' for k in ('h0', 'h2', 'l0', 'l1', 'k')))
    print('\n'.join(lines))
    open('nonhydrostatic.out', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
