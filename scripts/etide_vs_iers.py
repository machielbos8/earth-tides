# -*- coding: utf-8 -*-
#
# Validation of earthtides.etide against the IERS routine DEHANTTIDEINEL, piece
# by piece.
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
# 8/10/2026 Machiel Bos
#===============================================================================
"""Validation of earthtides.etide against the IERS routine DEHANTTIDEINEL, piece by piece.

Same set-up as scripts/body_tide_check.py (4 stations, 2026 hourly, Sun and Moon from DE421). The
IERS routine is run whole and in variants that keep one term at a time (degree 2 nominal only;
+ l(1); + out-of-phase; no degree 3), each compiled from the TN36 sources with XCORSTA zeroed after
every accumulation loop. Columns: radial, south, west; rms/max in mm, means removed.
Usage: python scripts/etide_vs_iers.py <scratch dir: de421.bsp, dehant, dehant_d2, dehant_n3,
       dehant_l1, dehant_op>  ->  results/etide_vs_iers.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, 'scripts')
import body_tide_check as B
from earthtides import etide as E

lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


t = B.epochs(days=365)
xs, xm = B.sun_moon(t)
mjd = B.utc_jd(t) - 2400000.5
cat = E.Tamura()
d2 = cat.n == 2


def run(L, deg, lat, lon):
    ur, un, ue = E.displacement(mjd, lat, lon, 0.0, L, degrees=deg, cat=cat)
    return np.column_stack([ur, -un, -ue])


def nominal():
    L = E.LoveModel(cat, 'iers')
    L.h0[d2], L.l0[d2], L.l1[:] = 0.6078, 0.0847, 0.0
    return L


out('etide (Tamura 1987, 1200 waves) vs IERS DEHANTTIDEINEL, 2026 hourly; radial south west, '
    'rms/max mm')
for name, (lat, lon) in B.STATIONS.items():
    x = B.ecef(lat, lon)
    D = {k: B.to_local(B.dehant(k, x, t, xs, xm), x)[0]
         for k in ('dehant', 'dehant_d2', 'dehant_n3', 'dehant_l1', 'dehant_op')}
    out(f'\n{name} (lat {lat:+.3f}, lon {lon:+.3f})')
    b = run(nominal(), (2,), lat, lon)
    dd = b - D['dehant_d2']
    out(f'  degree 2, nominal h, l           {B.stats(dd)}  mean ' +
        ' '.join(f'{1e3 * v:+.3f}' for v in dd.mean(0)))
    Li = E.LoveModel(cat, 'iers')
    L = nominal()
    L.l1[:] = Li.l1
    L.l1[d2 & (cat.m == 1)] = 0.0012                # TN36 step-1 value
    out(f'  l(1) terms                       {B.stats(run(L, (2,), lat, lon) - b - (D["dehant_l1"] - D["dehant_d2"]))}')
    L = nominal()
    for mm, (hI, lI) in {1: (-0.0025, -0.0007), 2: (-0.0022, -0.0007)}.items():
        s = d2 & (cat.m == mm)
        L.h0[s], L.l0[s] = 0.6078 + 1j * hI, 0.0847 + 1j * lI
    out(f'  out-of-phase terms               {B.stats(run(L, (2,), lat, lon) - b - (D["dehant_op"] - D["dehant_d2"]))}')
    out(f'  degree 3                         {B.stats(run(Li, (3,), lat, lon) - (D["dehant"] - D["dehant_n3"]))}')
    for src in ('iers', 'mhb', 'ddw'):
        out(f'  full model, --love {src:5s}        {B.stats(run(src, (2, 3), lat, lon) - D["dehant"])}')
    out(f'  PREM table - IERS                {B.stats(run("prem-allwaves", (2, 3), lat, lon) - D["dehant"])}')
    L4 = E.LoveModel(cat, 'iers')
    out(f'  degree 4 (not in IERS), size     {B.stats(run(L4, (4,), lat, lon))}')
open('results/etide_vs_iers.txt', 'w').write('\n'.join(lines) + '\n')
