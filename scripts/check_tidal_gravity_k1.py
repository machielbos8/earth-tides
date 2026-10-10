# -*- coding: utf-8 -*-
#
# K1/O1 tidal gravity ratio against DDW99 after ocean tide loading: a test of the sign of the
# in-phase resonant ocean-loading correction (paper Sect. 7).
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
"""K1/O1 gravimetric-factor ratio after ocean tide loading, relative to DDW99 (inelastic, NH).

rho = [dc(K1)/dc(O1)] / [DDW K1/O1] - 1. The ratio removes the calibration. Green's-function
loading (CARGA, PREM, component 3 = gravity incl. Newtonian; results/tidal_gravity/*.csv) has no
FCN resonance, so the oceanless DDW leaves the resonant load term in rho:
(3 rho_w/5 rho_bar) A21 (dh'-1.5dk')/delta(K1) = +0.064 % with our admittance, -0.064 % with the
IERS f_FCN f_OD (imaginary part +0.038 deg in both).
CARGA phase p -> local phase lead -p - lon (checked against Baker & Bos 2003, Tables 2 and 4: O1
loading magnitudes to 5 %, phases to a uniform 6 deg = GOT99.2b vs GOT5.5).
Observations: K1 from Ducarme (2012, BIM 147, Table 4, VAV04, GGP 1997-2010, not corrected for
loading); O1 at Membach from the same paper (Table 1, ETERNA) -- the only same-series ratio; O1
elsewhere from Baker & Bos (2003), a different series, so their calibrations need not cancel.
DDW99: Table 9 (delta0, delta+), latitude dependence after Agnew (2015, eq. 20).
Usage: python scripts/check_tidal_gravity_k1.py
"""
import csv
from pathlib import Path

import numpy as np

D = Path('results/tidal_gravity')
R, G, A1 = 6.371e6, 9.81, 0.1992                 # A1*R of TN36 eq. 6.8 (Hbar = A1 Hf, P21 normalized)
HF = {'o1': 0.26221, 'k1': 0.36870}
K1 = {'Brussels': (1.13712, 0.248), 'Membach': (1.13716, 0.280), 'Metsahovi': (1.13998, 0.083),
      'Strasbourg': (1.13695, 0.269), 'Wettzell': (1.13673, 0.204), 'Medicina': (1.13484, 0.355),
      'Moxa': (1.13631, 0.227), 'Vienna': (1.13392, 0.204)}
O1 = {'Membach': (1.14935, 0.1072, 'Ducarme 2012'), 'Strasbourg': (1.1490, 0.08, 'BB03'),
      'Wettzell': (1.1485, 0.08, 'BB03 lower'), 'Vienna': (1.1479, 0.10, 'BB03'),
      'Medicina': (1.1493, 0.17, 'BB03'), 'Brussels': (1.1533, 0.06, 'BB03'),
      'Metsahovi': (1.1532, 0.25, 'BB03')}
MODELS = ['FES2022b', 'GOT5.5', 'TPXO10']
# this paper: K1/O1 of anelastic hydrostatic PREM with the FCN at 431 sd (gravimetric_factor_spinup.py)
OURS = {'Membach': 0.983004, 'Strasbourg': 0.983009, 'Wettzell': 0.983008, 'Vienna': 0.983010,
        'Medicina': 0.983020, 'Brussels': 0.983003, 'Metsahovi': 0.982980}


def main():
    st = {l.split()[3]: (float(l.split()[0]), float(l.split()[1]))
          for l in (D / 'Stations.dat').read_text().splitlines() if l.strip() and not l.startswith('#')}
    load = {}
    for m in MODELS:
        for r in csv.reader((D / f'CARGA_{m}.csv').open()):
            r = [x.strip() for x in r]
            load[(m, r[0], r[3])] = float(r[4]), float(r[5])

    def ath(s, w):
        lat = np.radians(st[s][1])
        return 2 * G * A1 * HF[w] * np.sqrt(15) * np.sin(lat) * np.cos(lat) / R * 1e8   # microGal

    def lvec(m, s, w):
        a, p = load[(m, s, w)]
        return a / ath(s, w) * np.exp(1j * np.radians(-p - st[s][0]))

    def ddw(s, w):
        th = np.radians(90 - st[s][1])
        d0, dp = {'o1': (1.15424, 0.00008), 'k1': (1.13489, -0.00057)}[w]
        return d0 + dp * np.sqrt(3) / (2 * np.sqrt(2)) * (7 * np.cos(th)**2 - 3)

    obs = lambda d, a: d * np.exp(1j * np.radians(a))
    print('rho [%] = [dc(K1)/dc(O1)] / [theory K1/O1] - 1;  predicted +0.065 (our sign), -0.065 (IERS sign)')
    print('theory: DDW99 NH (first three columns) and this paper, FCN at 431 sd (last three)')
    print(f'{"station":11s} {"O1 from":12s} ' + ' '.join(f'{m:>9s}' for m in MODELS * 2) + '   K1 corr. phase')
    for s in K1:
        if s not in O1:
            continue
        rho = []
        for m in MODELS:
            k = obs(*K1[s]) - lvec(m, s, 'k1')
            o = obs(*O1[s][:2]) - lvec(m, s, 'o1')
            rho.append(100 * ((k.real / o.real) / (ddw(s, 'k1') / ddw(s, 'o1')) - 1))
        for m in MODELS:
            k = obs(*K1[s]) - lvec(m, s, 'k1')
            o = obs(*O1[s][:2]) - lvec(m, s, 'o1')
            rho.append(100 * ((k.real / o.real) / OURS[s] - 1))
        kc = obs(*K1[s]) - lvec('FES2022b', s, 'k1')
        print(f'{s:11s} {O1[s][2]:12s} ' + ' '.join(f'{x:+9.3f}' for x in rho)
              + f'   {np.degrees(np.angle(kc)):+.3f} deg')


if __name__ == '__main__':
    main()
