# -*- coding: utf-8 -*-
#
# Golden-value cases: the numbers the package must keep reproducing through
# the restructure.
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
"""Golden-value cases: the numbers the package must keep reproducing through the restructure.

Each case returns a flat dict {name: float or complex or list of them}. tools/make_golden.py runs
them on the code of tag round-1 and writes tests/golden/round1.json; tests/test_golden.py runs them
on the current code and compares. Keep the cases fast (whole set about one minute).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
M2 = 1.4053e-4
TIDES = {'M2': (2, 2 * 0.9661368), 'O1': (1, 0.92700), 'Mf': (0, 0.0730)}   # cycles per sidereal day


def _imports():
    """The package (earthtides; elliptical_earth in the code of tag round-1)."""
    try:
        from earthtides.model import EarthModel
        from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
        from earthtides.solver import TidalSolver
        from earthtides.iers import iers_fit
    except ImportError:
        from elliptical_earth.model import EarthModel
        from elliptical_earth.geometry import ClairautGeometry, OMEGA_EARTH
        from elliptical_earth.solver import TidalSolver
        from elliptical_earth.iers import iers_fit
    return EarthModel, ClairautGeometry, OMEGA_EARTH, TidalSolver, iers_fit


def _etide():
    try:
        from earthtides import etide
    except ImportError:                       # tag round-1: etide/etide.py
        sys.path.insert(0, str(ROOT / 'etide'))
        import etide
    return etide


def sphere_body():
    """Spherical elastic PREM, degree-2 body tide at M2."""
    EarthModel, _, _, TidalSolver, _ = _imports()
    S = TidalSolver(EarthModel(DATA / 'PREM.dat'), m=2, N=24, n_force=2, degrees=([2], []), omega_f=M2)
    r = S.solve()
    return {'h': r['h'], 'l': r['l'], 'k': r['kk']}


def sphere_load():
    """Spherical elastic PREM, load Love numbers at M2, degrees 2, 10, 100."""
    EarthModel, _, _, TidalSolver, _ = _imports()
    md = EarthModel(DATA / 'PREM.dat')
    out = {}
    for n in (2, 10, 100):
        S = TidalSolver(md, m=n, K=1, N=24 if n < 20 else 60, omega_f=M2, n_force=n, forcing='load')
        S.solve()
        L = S.load_love()
        out.update({f"h'{n}": L['h'][n], f"l'{n}": L['l'][n], f"k'{n}": L['k'][n]})
    return out


def _flattened(tide, anelastic=False, model='PREM.dat'):
    EarthModel, ClairautGeometry, OMEGA, TidalSolver, iers_fit = _imports()
    md0 = EarthModel(DATA / model)
    geo = ClairautGeometry(md0)
    m, fr = TIDES[tide]
    w = fr * OMEGA
    md = md0.anelastic(w) if anelastic else md0
    S = TidalSolver(md, geo, m=m, K=4, N=24, omega_f=0.0, coriolis=OMEGA, omega_t=w,
                    omega_rot=OMEGA, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    r = S.solve()
    f = iers_fit(S, geo)
    out = {k: f[k] for k in ('h0', 'h2', 'l0', 'l1', 'l2', 'lp', 'hp') if k in f}
    out['k'] = r['kk']
    th = np.radians([10.0, 32.6, 57.0, 90.0, 141.0])
    ur, ut, up = S.surface_field(th)
    out.update({'u_r': list(ur), 'u_theta': list(ut), 'u_lambda': list(up)})
    return out


def flattened_elastic():
    """Hydrostatic elastic PREM, fluid core, Coriolis, mantle frame: IERS parameters, k and the
    surface field at five colatitudes, for M2, O1 and Mf (the settings of the DDW comparison)."""
    out = {}
    for tide in TIDES:
        out.update({f'{tide}.{k}': v for k, v in _flattened(tide).items()})
    return out


def flattened_anelastic():
    """As flattened_elastic, anelastic (absorption band), M2 and O1."""
    out = {}
    for tide in ('M2', 'O1'):
        out.update({f'{tide}.{k}': v for k, v in _flattened(tide, anelastic=True).items()})
    return out


def flattened_load():
    """Hydrostatic elastic PREM: degree-2, order-2 load Love numbers at M2 on the flattened Earth."""
    EarthModel, ClairautGeometry, _, TidalSolver, _ = _imports()
    md = EarthModel(DATA / 'PREM.dat')
    S = TidalSolver(md, ClairautGeometry(md), m=2, K=4, N=24, omega_f=M2, n_force=2, forcing='load')
    S.solve()
    L = S.load_love()
    return {f"{q}'{n}": L[q][n] for q in ('h', 'l', 'k') for n in sorted(L[q])}


def etide_series():
    """etide at Onsala, 2026-01-01, 72 h hourly, for the iers, mhb and ddw Love models."""
    E = _etide()
    cat = E.Tamura()
    mjd = 61041.0 + np.arange(72) / 24.0
    out = {}
    for src in ('iers', 'mhb', 'ddw'):
        ur, un, ue = E.displacement(mjd, 57.395, 11.926, 0.0, src, cat=cat)
        out.update({f'{src}.up': list(ur), f'{src}.north': list(un), f'{src}.east': list(ue)})
    return out


CASES = [sphere_body, sphere_load, flattened_elastic, flattened_anelastic, flattened_load,
         etide_series]


def encode(v):
    """JSON form: complex -> [re, im], arrays -> lists."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [encode(x) for x in v]
    v = complex(v)
    return [v.real, v.imag]


def decode(v):
    if isinstance(v[0], list):
        return np.array([complex(*x) for x in v])
    return complex(*v)
