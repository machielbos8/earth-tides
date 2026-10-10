#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Design the default wave groups of earthtides-love from the admittance
# itself.
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
"""Design the default wave groups of earthtides-love from the admittance itself.

For every degree-2 wave k of the Tamura catalogue the error of giving it the Love numbers of its
group's main wave r is bounded by

    e_k = |h_k - h_r| * R_k + |l_k - l_r| * T_k      (and the same for l1)

with R_k, T_k the largest radial and horizontal displacement per unit Love number of that wave
over all latitudes (the potential and its gradient from etide). Consecutive waves of one order m
are put in one group while the sum of e_k over all groups stays within the budget; the main wave
of a group is its largest. The admittance is taken from two models with different free core
nutation frequencies, so that the groups suit both: the per-wave PREM table of earth-tides
(FCN ~457 d) and the IERS model (resonance formula, FCN ~430 d); e_k uses the larger of the two.
The sum of the bounds is a worst case (all errors in phase at the worst latitude); the real
error, checked by tools/verify_groups.py, is several times smaller.
Usage: python tools/design_groups.py [--budget 0.15] [--allwaves TABLE.json]
       -> src/earthtides/data/groups_default.txt
"""
import argparse
from pathlib import Path

import numpy as np

from earthtides import etide

ROOT = Path(__file__).resolve().parents[1]


def amplitudes(cat):
    """Largest radial (W) and horizontal (dW/dphi, m W / cos phi) displacement per unit Love number
    of every degree-2 wave over latitude [mm]."""
    phi = np.radians(np.linspace(-89.5, 89.5, 359))
    R, T = np.zeros(len(cat.n)), np.zeros(len(cat.n))
    for m in (0, 1, 2):
        W, dW = etide.W_functions(2, m, phi)
        hor = np.maximum(np.abs(dW), np.abs(m * W / np.cos(phi)))
        s = (cat.n == 2) & (cat.m == m)
        R[s] = np.abs(cat.A[s]) * np.abs(W).max() * 1e3
        T[s] = np.abs(cat.A[s]) * hor.max() * 1e3
    return R, T


def design(cat, models, budget):
    R, T = amplitudes(cat)
    keys = ('h0', 'h2', 'l0', 'l1', 'l2')

    def err(k, r):
        e = 0.0
        for L in models:
            e = max(e, abs(L.h0[k] - L.h0[r]) * R[k] + abs(L.h2[k] - L.h2[r]) * R[k]
                    + sum(abs(getattr(L, q)[k] - getattr(L, q)[r]) * T[k] for q in ('l0', 'l1', 'l2')))
        return e

    def cost(members):
        r = max(members, key=lambda k: abs(cat.A[k]))
        return sum(err(k, r) for k in members), r

    # scan with a tolerance per group, lowered until the total is within the budget
    waves = [np.where((cat.n == 2) & (cat.m == m) & (np.abs(cat.f_cpd) > 1e-9))[0] for m in (0, 1, 2)]
    waves = [w[np.argsort(cat.f_cpd[w])] for w in waves]
    tol = budget
    while True:
        groups, total = [], 0.0
        for w in waves:
            cur = [w[0]]
            for k in w[1:]:
                c, _ = cost(cur + [k])
                if c <= tol:
                    cur.append(k)
                else:
                    groups.append(cur)
                    cur = [k]
            groups.append(cur)
        total = sum(cost(g)[0] for g in groups)
        if total <= budget:
            return groups, total
        tol *= 0.7


def boundaries(cat, groups):
    """[name, f1, f2] per group, the limits halfway between neighbouring groups (cpd)."""
    out = []
    by_band = {}
    for g in groups:
        by_band.setdefault(int(cat.m[g[0]]), []).append(g)
    edges = {0: (0.0, 0.501369), 1: (0.501370, 1.470243), 2: (1.470244, 2.451943)}
    for m, gs in sorted(by_band.items()):
        gs = sorted(gs, key=lambda g: cat.f_cpd[g].min())
        lo = edges[m][0]
        for i, g in enumerate(gs):
            hi = edges[m][1] if i == len(gs) - 1 else 0.5 * (cat.f_cpd[g].max() + cat.f_cpd[gs[i + 1]].min())
            r = max(g, key=lambda k: abs(cat.A[k]))
            out.append((f'{"LDS"[m]}{i + 1:02d}', lo, hi, cat.f_cpd[r], len(g)))
            lo = hi + 1e-9
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--budget', type=float, default=0.15, help='sum of the error bounds [mm]')
    p.add_argument('--allwaves', default=str(ROOT / 'src/earthtides/data/love_PREM_allwaves.json'))
    p.add_argument('-o', default=str(ROOT / 'src/earthtides/data/groups_default.txt'))
    a = p.parse_args()
    cat = etide.Tamura()
    models = [etide.LoveModel(cat, a.allwaves), etide.LoveModel(cat, 'iers')]
    groups, total = design(cat, models, a.budget)
    rows = boundaries(cat, groups)
    with open(a.o, 'w') as fp:
        fp.write('# Default wave groups of earthtides-love (tools/design_groups.py): limits in cycles\n'
                 f'# per day; designed from the per-wave admittance of PREM (earth-tides) and of the IERS\n'
                 f'# model, sum of the error bounds {total:.3f} mm (budget {a.budget} mm); main wave in\n'
                 '# the comment. Columns: f1 f2 name\n')
        for name, f1, f2, fr, n in rows:
            fp.write(f'{f1:12.9f} {f2:12.9f} {name:5s}   # main wave {fr:.6f} cpd, {n} waves\n')
        fp.write(f'{2.451944:12.9f} {7.0:12.9f} HIGH    # degrees 3 and 4 only (spherical values), no solve\n')
    print(f'{len(rows)} groups, sum of error bounds {total:.3f} mm -> {a.o}')


if __name__ == '__main__':
    main()
