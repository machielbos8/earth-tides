# -*- coding: utf-8 -*-
#
# Does the free core nutation follow the excess dynamical ellipticity of the core or the CMB flattening?
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
# 16/11/2026 Machiel Bos
#===============================================================================
"""Review, Table S9: 450 m of excess CMB flattening gives Delta e_f / e_f = 5.87 % with three load shells fitted
exactly to (J2, CMB, surface) and 4.49 % with two shells fitted to (J2, CMB). Elastic PREM, level figure,
elastic support, inviscid slipping core (K 4, N 24). FCN period (pole fit of k, as make_table4.py) for:
  A  three shells, exact fit to J2, CMB 450 m, surface 100 m           (Table S9 'exact')
  B  two shells, J2 and CMB 450 m                                      (Table 1, 4.49 %)
  C  two shells, J2 and the Delta e_f of A                             (same e_f as A, other CMB)
  R  three shells, regularised (sigmas, prior 0.5 %)                    (Table S9 'reg.')
If the period follows Delta e_f, A and C agree and B differs.

    .venv/bin/python scripts/ef_vs_cmb_fcn.py   ->  results/ef_vs_cmb_fcn_level.txt"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
warnings.filterwarnings('ignore')
import numpy as np

from earthtides import love

J2 = 1.14e-5
SHELLS3 = [[3480, 5701], [5701, 5971], [5971, 6346.6]]          # km, as scripts/nonhydro_support.py
TARGET3 = {'J2': J2, 'cmb': 450.0, 'surface': 100.0}
SIGMA3 = {'J2': 0.1e-6, 'cmb': 50.0, 'surface': 250.0}
SIG = (1.0012, 1.0016, 1.0030, 1.0036)


def options(nh):
    return love.Options(anelastic=False, figure='level', nonhydrostatic=nh)


def k_at(a):
    nh, s = a
    return love.solve_wave(options(nh), 1, s * love.F_TILT)[1]['kk'].real


def state(nh):
    r = love._model(options(nh))[2].summary()['response']
    return r['cmb'], r['e_f']


def fcn(pool, nh):
    k = np.array(pool.map(k_at, [(nh, s) for s in SIG]))
    x = np.array(SIG)
    c = np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0]
    return 1.0 / (c - 1.0)


def main(run):
    cases = {'A': dict(shells_km=SHELLS3, targets=TARGET3),
             'B': dict(targets={'J2': J2, 'cmb': 450.0}),
             'R': dict(shells_km=SHELLS3, targets=TARGET3, sigma=SIGMA3, prior=5e-3)}
    lines = ['Elastic PREM, level figure, elastic support, slipping core K 4 N 24; FCN in sidereal days (pole fit)', '',
             f'{"case":4s} {"CMB (m)":>8s} {"Delta e_f":>11s} {"FCN":>8s}  loads']
    with Pool(4) as pool:
        res = {}
        for name in ('A', 'B', 'R'):
            cmb, ef = state(cases[name])
            res[name] = (cmb, ef, fcn(pool, cases[name]))
        cases['C'] = dict(targets={'J2': J2, 'e_f': res['A'][1]})
        cmb, ef = state(cases['C'])
        res['C'] = (cmb, ef, fcn(pool, cases['C']))
    for name in ('A', 'B', 'C', 'R'):
        cmb, ef, p = res[name]
        lines.append(f'{name:4s} {cmb:8.1f} {ef:11.4e} {p:8.2f}  {cases[name]}')
    lines += ['', f'A - C (same e_f, CMB {res["A"][0] - res["C"][0]:+.0f} m): FCN {res["A"][2] - res["C"][2]:+.2f} d',
              f'A - B (same CMB, e_f {res["A"][1] / res["B"][1] - 1:+.1%}): FCN {res["A"][2] - res["B"][2]:+.2f} d']
    run.write_text(str(ROOT / 'results' / 'ef_vs_cmb_fcn_level.txt'), '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    from provenance import Run
    with Run('Review Table S9: FCN follows Delta e_f or the CMB flattening', inputs=['data/PREM.dat']) as run_:
        main(run_)
