#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Verify a wave-group set: the displacement with one Love number per group
# against the per-wave reference, for two Earth models, on a grid of stations,
# over one year hourly.
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
"""Verify a wave-group set: the displacement with one Love number per group against the per-wave
reference, for two Earth models, on a grid of stations, over one year hourly.

  PREM : group table computed by the solver (love.group_table) vs the per-wave PREM table
  IERS : the IERS model per wave (etide 'iers') vs the same with every degree-2 wave given the
         values of its group's main wave (grouping emulated, no solver)
Reports the largest radial and horizontal error over all stations and epochs [mm].
Usage: python tools/verify_groups.py [GROUPFILE] [--jobs 8]
"""
import argparse
from pathlib import Path

import numpy as np

from earthtides import etide, love

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2')


def grouped_copy(cat, L, groups):
    """L with every degree-2 wave (f > 0) given the values of its group's largest wave."""
    G = etide.LoveModel(cat, 'iers')
    for k in KEYS:
        getattr(G, k)[:] = getattr(L, k)
    for name, f1, f2 in groups:
        s = np.where((cat.n == 2) & (np.abs(cat.f_cpd) > 1e-9) & (cat.f_cpd >= f1 - 1e-9)
                     & (cat.f_cpd <= f2 + 1e-9))[0]
        if len(s):
            r = s[np.argmax(np.abs(cat.A[s]))]
            for k in KEYS:
                getattr(G, k)[s] = getattr(L, k)[r]
    return G


def max_error(cat, A, B, mjd):
    worst = np.zeros(2)
    for lat in np.arange(-85, 86, 10):
        for lon in (0.0, 90.0, 180.0, 270.0):
            a = np.column_stack(etide.displacement(mjd, lat, lon, 0.0, A, cat=cat))
            b = np.column_stack(etide.displacement(mjd, lat, lon, 0.0, B, cat=cat))
            d = np.abs(a - b)
            worst = np.maximum(worst, [d[:, 0].max(), np.hypot(d[:, 1], d[:, 2]).max()])
    return 1e3 * worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument('groups', nargs='?', default=str(ROOT / 'src/earthtides/data/groups_default.txt'))
    p.add_argument('--jobs', type=int, default=8)
    a = p.parse_args()
    groups = love.read_groups(a.groups)
    cat = etide.Tamura()
    mjd = 61041.0 + np.arange(0, 365, 1 / 24)
    ref = etide.LoveModel(cat, str(ROOT / 'src/earthtides/data/love_PREM_allwaves.json'))
    t = love.group_table(love.Options(), groups, jobs=a.jobs, progress=False)
    tmp = Path(a.groups).with_suffix('.verify.json')
    love.write(t, tmp)
    grp = etide.LoveModel(cat, str(tmp))
    tmp.unlink()
    e1 = max_error(cat, grp, ref, mjd)
    iers = etide.LoveModel(cat, 'iers')
    e2 = max_error(cat, grouped_copy(cat, iers, groups), iers, mjd)
    print(f'{len(groups)} groups ({a.groups})')
    print(f'  PREM, solver groups vs per wave : max radial {e1[0]:.3f} mm, horizontal {e1[1]:.3f} mm')
    print(f'  IERS, grouped vs per wave       : max radial {e2[0]:.3f} mm, horizontal {e2[1]:.3f} mm')


if __name__ == '__main__':
    main()
