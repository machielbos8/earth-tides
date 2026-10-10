# -*- coding: utf-8 -*-
#
# Decisive test of the phase conventions behind the ocean admittance A21: a synthetic ocean.
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
# 10/10/2026 Machiel Bos
#===============================================================================
"""Synthetic ocean tides with a known admittance, written in the convention of BLQ files and
ocean tide models, passed through the code that gives our A21 (otl_fcn_resonance.admittance_grid).

The convention is not assumed but taken from earthtides.synthesis: the potential of a degree-2,
order-1 line in HARDISP's form, V/g = TAMP Re[c_1(theta) e^{i lambda} e^{i(arg + 90 deg)}], with
the Cartwright-Tayler-Edden amplitude TAMP (negative for Q1, O1, P1, positive for K1) and the
Doodson argument. This form reproduces DEHANTTIDEINEL (computed from the ephemerides) to 0.06 mm,
and BLQ files written from it with synthesis.blq (amplitude |Z| |TAMP|, phase lag -arg Z) are
reproduced by the real HARDISP to 0.05 mm (scripts/body_tide_check.py). In that convention the
sign of TAMP stays in TAMP: an ocean tide zeta = A21 x (V/g) has amplitude |A21 c_1 TAMP| and
phase lag -arg(A21 c_1 e^{i lambda}) (convention H). The alternative, in which the phase lag
absorbs a negative TAMP (+180 deg; convention S), is tested as well: the two coincide for K1
(TAMP > 0) and differ by 180 deg for Q1, O1, P1 (TAMP < 0).

Expected in convention H: A21 = +1 for an equilibrium ocean and e^{-i delta} for an ocean that
lags it by delta, for every line; a 180-degree error in the chain would give -1. Which convention
the ocean tide models use follows from FES2022b itself: the ocean response varies smoothly across
the diurnal band, so P1 and K1 (0.0055 cpd apart) cannot differ in phase by 180 deg.
Usage: python scripts/check_admittance_convention.py
"""
import sys

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from earthtides import etide, synthesis
from otl_fcn_resonance import admittance_grid

WAVES = {'Q1': ([1, -2, 0, 1, 0, 0], 0.893244), 'O1': ([1, -1, 0, 0, 0, 0], 0.929536),
         'P1': ([1, 1, -2, 0, 0, 0], 0.997262), 'K1': ([1, 1, 0, 0, 0, 0], 1.002738)}


def ocean_grids(A21, tamp, conv='H', nlat=360, nlon=720):
    """Amplitude [m] and Greenwich phase lag [deg] grids of the ocean tide zeta = A21 (V/g)."""
    lat = np.radians(90.0 - 180.0 * (np.arange(nlat) + 0.5) / nlat)        # row 0 = north
    lon = np.radians(360.0 * np.arange(nlon) / nlon)
    th = np.pi / 2 - lat
    Z = A21 * synthesis.ct(1, th)[:, None] * np.exp(1j * lon)[None, :]       # admittance per point
    amp = np.abs(Z) * abs(tamp)
    pha = -np.degrees(np.angle(Z)) + (180.0 if (conv == 'S' and tamp < 0) else 0.0)
    return amp, np.mod(pha, 360.0), lat, lon


def main():
    idd, tamp_cte = synthesis.catalog()
    cat = etide.Tamura()
    fabs = np.abs(cat.f_cpd)
    di = (cat.n == 2) & (cat.m == 1)
    for conv in ('H', 'S'):
        print(f'Synthetic oceans, convention {conv}, through otl_fcn_resonance.admittance_grid')
        print(f"{'wave':5s} {'TAMP (CTE)':>11s} {'equilibrium ocean':>22s} {'lag 30 deg':>22s} "
              f"{'lag 133 deg':>22s}")
        for name, (d, fc) in WAVES.items():
            tamp = tamp_cte[np.where((idd == d).all(1))[0][0]]
            a_tam = cat.A[np.argmin(np.abs(fabs - fc) + 10 * ~di)]
            res = []
            for lag in (0.0, 30.0, 133.0):
                A = admittance_grid(*ocean_grids(np.exp(-1j * np.radians(lag)), tamp, conv), a_tam)
                res.append(f'{A.real:+.4f}{A.imag:+.4f}i ({np.degrees(np.angle(A)):+6.1f})')
            print(f'{name:5s} {tamp:+11.6f} ' + ' '.join(f'{r:>22s}' for r in res))
        print()
    print('FES2022b (results/otl_fcn_resonance.txt) read in either convention: phase [deg] per line')
    fes = {}
    for ln in open('results/otl_fcn_resonance.txt'):
        w = ln.split()
        if len(w) > 4 and w[0] in WAVES and w[2] == 'A21':
            fes[w[0]] = complex(w[4].replace('j', 'j'))
    for conv in ('H', 'S'):
        ph = []
        for name, (d, fc) in WAVES.items():
            tamp = tamp_cte[np.where((idd == d).all(1))[0][0]]
            A = fes[name] * (-1 if (conv == 'S' and tamp < 0) else 1)
            ph.append(f'{name} {np.degrees(np.angle(A)):+7.1f}')
        print(f'  convention {conv}: ' + '   '.join(ph))
    print('IERS f_FCN f_OD:       ' + '   '.join(f'{n} {np.degrees(np.angle(v)):+7.1f}' for n, v in
          (('Q1', 0.312 - 0.273j), ('O1', 0.271 - 0.251j), ('P1', 0.198 - 0.214j), ('K1', 0.199 - 0.218j))))


if __name__ == '__main__':
    main()
