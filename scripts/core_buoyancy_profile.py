# -*- coding: utf-8 -*-
#
# Buoyancy of PREM's liquid core: N^2, g^2/c^2 and the Bullen parameter (P1 v2, Sect. 3, the rays of
# Fig. 3 hold for a neutral core).
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
# 03/11/2026 Machiel Bos
#===============================================================================
"""N^2 = -g rho'/rho - g^2/c^2 (c^2 = lambda/rho, mu = 0) = (g^2/c^2)(eta_B - 1), with the Bullen parameter
eta_B = -(c^2/(rho g)) rho' (1: neutral, adiabatic); all over (2 Omega)^2.
Usage: python scripts/core_buoyancy_profile.py  ->  results/core_buoyancy_profile.txt"""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
import numpy as np

from earthtides.model import EarthModel
from provenance import Run

OMEGA = 7.292115e-5


def main():
    md = EarthModel('data/PREM.dat')
    oc = [i for i in range(md.n_layers) if abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) < 1.0][0]
    r = np.linspace(md.r0[oc], md.r1[oc], 2001)
    rho, drho, g = md.rho[oc](r), md.rho[oc].deriv()(r), md.gravity(oc, r)
    c2 = md.lam[oc](r) / rho
    w2 = (2 * OMEGA) ** 2
    gc = g ** 2 / c2 / w2
    eta = -c2 * drho / (rho * g)
    n2 = (-g * drho / rho - g ** 2 / c2) / w2
    with Run('buoyancy of PREM\'s liquid core: N^2, g^2/c^2, Bullen parameter', params=dict(omega=OMEGA),
             inputs=['data/PREM.dat']) as run_:
        lines = ['PREM outer core: N^2/(2 Omega)^2 = (g^2/c^2)/(2 Omega)^2 x (eta_B - 1)',
                 f'{"r [km]":>8s} {"N2/(2Om)^2":>11s} {"g2/c2/(2Om)^2":>14s} {"eta_B - 1":>10s}']
        lines += [f'{a / 1e3:8.0f} {b:+11.3f} {c:14.1f} {d - 1:+10.4f}'
                  for a, b, c, d in zip(r[::100], n2[::100], gc[::100], eta[::100])]
        lines += ['', f'N2/(2Om)^2 from {n2.min():+.3f} (r {r[n2.argmin()] / 1e3:.0f} km) to {n2.max():+.3f} '
                  f'(r {r[n2.argmax()] / 1e3:.0f} km); g2/c2/(2Om)^2 from {gc.min():.1f} to {gc.max():.1f}; '
                  f'eta_B - 1 from {(eta - 1).min():+.4f} to {(eta - 1).max():+.4f}']
        txt = '\n'.join(lines)
        run_.write_text('results/core_buoyancy_profile.txt', txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
