# -*- coding: utf-8 -*-
#
# Harmonic analysis of the radial displacement difference PREM - IERS (paper Sect. 7.1).
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
"""Radial body-tide difference PREM table minus IERS model (etide, 2026 hourly, means removed) at
the nine stations of Sect. 7.2, its rms and maximum, the horizontal rms, and the amplitudes (mm) of the main tidal
lines from a least-squares fit of cosines and sines at their frequencies.
Usage: python scripts/prem_iers_harmonics.py -> results/prem_iers_harmonics.txt
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
from earthtides.etide import Tamura, displacement
from displacement_budget import variants
from station_error_budget import STATIONS

LINES = {'M2': 1.932274, 'S2': 2.0, 'N2': 1.895982, 'K2': 2.005476, 'K1': 1.002738,
         'O1': 0.929536, 'P1': 0.997262, 'Q1': 0.893244, 'psi1': 1.005476, 'Mf': 0.073202, 'Mm': 0.036292,
         'Ssa': 0.005476}
MJD0 = 61041.0                                     # 2026-01-01
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def main():
    cat = Tamura()
    V = variants(cat)
    mjd = MJD0 + np.arange(365 * 24) / 24.0
    t = mjd - mjd[0]
    cols = [np.ones_like(t)]
    for f in LINES.values():
        cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    A = np.column_stack(cols)
    out('Radial difference PREM - IERS, 2026 hourly, mm: rms, max |.|, and line amplitudes')
    out(f"{'station':8s} {'lat':>7s} {'rms':>6s} {'max':>6s}" + ''.join(f'{k:>6s}' for k in LINES))
    for st, (lat, lon) in STATIONS.items():
        a = displacement(mjd, lat, lon, love=V['prem'], cat=cat)
        b = displacement(mjd, lat, lon, love=V['iers'], cat=cat)
        d, dn, de = (1e3 * (x - y) for x, y in zip(a, b))
        d, dn, de = d - d.mean(), dn - dn.mean(), de - de.mean()
        hor = np.sqrt(np.mean(dn**2 + de**2))
        c, *_ = np.linalg.lstsq(A, d, rcond=None)
        amp = np.hypot(c[1::2], c[2::2])
        out(f'{st:8s} {lat:7.2f} {np.sqrt(np.mean(d**2)):6.2f} {np.abs(d).max():6.2f}'
            + ''.join(f'{x:6.2f}' for x in amp) + f'   hor rms {hor:.2f}')
    Path('results/prem_iers_harmonics.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
