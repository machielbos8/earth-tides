# -*- coding: utf-8 -*-
#
# Sensitivity of the K1/O1 gravimetric ratio to mantle structure: an upper bound on the effect of
# lateral heterogeneity on the tidal-gravity test of the ocean-loading resonance (paper Sect. 7).
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
# 9/10/2026 Machiel Bos
#===============================================================================
"""K1/O1 gravimetric ratio of 1-D models perturbed in the mantle.

Lateral structure changes the K1/O1 ratio at a station only through the free core nutation
resonance, which is what makes K1 differ from O1. A spherically symmetric change of the mantle
applied everywhere bounds the effect of a lateral anomaly of the same size from above: for a
degree-2 forcing the response integrates over the whole mantle, and a regional anomaly reaches it
only in part. Typical long-wavelength tomographic anomalies are dlnVs ~ 1 per cent, i.e. dlnmu ~ 2
per cent, and dlnrho ~ 0.3 per cent.
Models: PREM; mantle mu x 1.02 and x 0.98; upper mantle (above 670 km) or lower mantle only x 1.02;
mantle density x 1.01 (mass and gravity recomputed); STW105. Anelastic (cot), hydrostatic,
gravimetric factors at 50 N; K1 = mean of sigma = 1 -+ 2e-4 (gravimetric_factor_spinup.py).
Usage: python scripts/k1o1_sensitivity.py -> results/k1o1_sensitivity.txt
"""
import copy
import sys
import warnings
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np
from numpy.polynomial import Chebyshev
from earthtides.model import EarthModel, G
from earthtides.love import model_path
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides import etide
from gravimetric_factor_spinup import delta

R670 = 6371.0e3 - 670.0e3
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def mantle_layers(md):
    fluid = [abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) < 1.0 for i in range(md.n_layers)]
    cmb = max(md.r1[i] for i in range(md.n_layers) if fluid[i])
    return [i for i in range(md.n_layers) if md.r0[i] >= cmb - 1.0 and md.r1[i] <= 6346.6e3 + 1.0]


def scaled(md, which, f, region='mantle'):
    new = copy.deepcopy(md)
    for i in mantle_layers(md):
        mid = 0.5 * (md.r0[i] + md.r1[i])
        if region == 'upper' and mid < R670 or region == 'lower' and mid > R670:
            continue
        if which == 'mu':
            new.mu[i] = md.mu[i] * f
            new.lam[i] = md.lam[i] - 2.0 / 3.0 * md.mu[i] * (f - 1)   # bulk modulus fixed
        else:
            new.rho[i] = md.rho[i] * f
    if which == 'rho':                        # mass and surface gravity
        m_below = 0.0
        new._mass_poly = []
        for i in range(new.n_layers):
            r = Chebyshev.identity(domain=[new.r0[i], new.r1[i]])
            p = (4.0 * np.pi * new.rho[i] * r * r).integ(lbnd=new.r0[i])
            new.mass_below[i] = m_below
            new._mass_poly.append(p)
            m_below += p(new.r1[i])
        new.M = m_below
        new.g_surface = G * new.M / new.a**2
    return new


def ratio(md):
    colat = [np.pi / 2 - etide.geocentric(50.0)[0]]
    geo = ClairautGeometry(md, omega=OMEGA_EARTH)
    dO = delta(md, geo, OMEGA_EARTH, 0.92700, colat)[0][0]
    dK = np.mean([delta(md, geo, OMEGA_EARTH, s, colat)[0][0] for s in (1 - 2e-4, 1 + 2e-4)])
    return dO.real, dK.real


def main():
    prem = EarthModel(str(model_path('PREM')))
    cases = [('PREM', prem),
             ('mantle mu x1.02', scaled(prem, 'mu', 1.02)),
             ('mantle mu x0.98', scaled(prem, 'mu', 0.98)),
             ('upper mantle mu x1.02', scaled(prem, 'mu', 1.02, 'upper')),
             ('lower mantle mu x1.02', scaled(prem, 'mu', 1.02, 'lower')),
             ('mantle rho x1.01', scaled(prem, 'rho', 1.01)),
             ('STW105', EarthModel('data/STW105.dat'))]
    out('K1/O1 gravimetric ratio at 50 N, anelastic hydrostatic models')
    out(f"{'model':24s} {'delta O1':>9s} {'delta K1':>9s} {'K1/O1':>9s} {'change of K1/O1':>16s} {'change of O1':>13s}")
    ref = None
    for name, md in cases:
        dO, dK = ratio(md)
        r = dK / dO
        if ref is None:
            ref = (dO, r)
        out(f'{name:24s} {dO:9.5f} {dK:9.5f} {r:9.6f} {100*(r/ref[1]-1):+15.4f} % {100*(dO/ref[0]-1):+12.4f} %')
    Path('results/k1o1_sensitivity.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
