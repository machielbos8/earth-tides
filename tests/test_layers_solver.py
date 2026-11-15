# -*- coding: utf-8 -*-
#
# linear_solver='layers': direct LU by layers with pivoting across the interfaces (_LayerLU).
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
# 12/11/2026 Machiel Bos
#===============================================================================
"""The direct layer solver against SuperLU on the kinds of system the package solves: viscous and slipping
cores in the semidiurnal and diurnal bands, static runs with and without rotation regions (border rows), a
banded run. No GMRES and no fallback: S._solver_used == 'layers' and the residual of the scaled system is
at the level of SuperLU's."""
import numpy as np
import pytest

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

PREM = 'data/PREM.dat'
KEYS = ('h0', 'h2', 'l2', 'l1')


def _pair(build):
    out = {}
    for ls in ('layers', 'sparse'):
        S, fit = build(ls)
        r = S.solve()
        out[ls] = (np.array([fit(S)[k] for k in KEYS]) if fit else np.array([r[k] for k in ('h', 'l', 'kk')]), S)
    S = out['layers'][1]
    assert S._solver_used == 'layers'
    return out['layers'][0], out['sparse'][0], S._layer_info


@pytest.mark.parametrize('m, sig', [(2, 1.93), (1, 0.93)])
def test_viscous_core(m, sig):
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=2.0e3)
    g = ClairautGeometry(md)
    a, b, info = _pair(lambda ls: (TidalSolver(md, g, m=m, K=4, N=24, linear_solver=ls, omega_f=0.0,
                                               coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                                               omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                                               consistent_reference=True, frame='mantle'),
                                   lambda S: iers_fit(S, g)))
    assert info['residual'] < 1e-11 and info['growth'] < 1e3
    assert np.max(np.abs(a - b)) < 1e-10


@pytest.mark.parametrize('m, sig', [(2, 1.93), (1, 0.93)])
def test_slipping_core(m, sig):
    """True fluid core (slip at the CMB and ICB), K 4 / N 24: determinate (lab notes, near-null pair)."""
    md = EarthModel(PREM)
    g = ClairautGeometry(md)
    a, b, info = _pair(lambda ls: (TidalSolver(md, g, m=m, K=4, N=24, linear_solver=ls, omega_f=0.0,
                                               coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                                               omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                                               consistent_reference=True, frame='mantle'),
                                   lambda S: iers_fit(S, g)))
    assert info['residual'] < 1e-10
    assert np.max(np.abs(a - b)) < 1e-8


@pytest.mark.parametrize('m', [0, 1, 2])
def test_static_weak_core(m):
    """Static PREMs; m = 1 has rotation regions (Tisserand rows and rigid-rotation forces in the border)."""
    md = EarthModel('data/PREMs.dat')
    a, b, info = _pair(lambda ls: (TidalSolver(md, m=m, K=3, N=24, omega_f=0.0, linear_solver=ls), None))
    assert info['residual'] < 1e-10
    assert np.max(np.abs(a - b)) < 1e-10


def test_banded_viscous_core():
    sig = 0.93
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=2.0e3)
    g = ClairautGeometry(md)
    a, b, info = _pair(lambda ls: (TidalSolver(md, g, m=1, K=4, N=24, fluid_K=16, band=12, linear_solver=ls,
                                               omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                                               omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                                               consistent_reference=True, frame='mantle'),
                                   lambda S: iers_fit(S, g)))
    assert info['residual'] < 1e-10
    assert np.max(np.abs(a - b)) < 1e-9
    assert info['banded']                            # with a band the panels are banded (_BandLayerLU)


def _viscous(ls, sig=0.93, banded=True, **kw):
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=2.0e3)
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=1, K=4, N=24, linear_solver=ls, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                    consistent_reference=True, frame='mantle', **kw)
    S.LAYERS_BANDED = banded
    return S, lambda S: iers_fit(S, g)


@pytest.mark.parametrize('sig', [0.93, 1.0022])
def test_banded_panels_narrow_band(sig):
    """_BandLayerLU with a band much narrower than the core (K 16, B 4: the degree windows move), away from
    and next to the FCN, against SuperLU on the same banded matrix; and against the dense panels."""
    a, b, info = _pair(lambda ls: _viscous(ls, sig, fluid_K=16, band=4))
    # next to the FCN the residual of every solver is 1e-10 .. 7e-10 (lab notes 2026-11-13)
    assert info['banded'] and info['residual'] < (1e-10 if sig < 1 else 1e-9) and info['growth'] < 1e3
    assert np.max(np.abs(a - b)) < (1e-9 if sig < 1 else 1e-7)
    S, fit = _viscous('layers', sig, banded=False, fluid_K=16, band=4)
    S.solve()
    assert not S._layer_info['banded']
    assert np.max(np.abs(a - np.array([fit(S)[k] for k in KEYS]))) < (1e-9 if sig < 1 else 1e-7)
    assert info['factors_gb'] < S._layer_info['factors_gb']


@pytest.mark.parametrize('case', ['viscous', 'slipping', 'static0', 'static1'])
def test_banded_panels_without_band(case):
    """The same elimination (auxiliary unknowns at the interfaces, dense stage on them, border rows) without
    a band: every layer is one degree group. Covers the border rows of the rotation regions (static m = 1)."""
    def build(ls):
        if case == 'viscous':
            return _viscous(ls, banded='always')
        if case == 'slipping':
            md = EarthModel(PREM)
            g = ClairautGeometry(md)
            S = TidalSolver(md, g, m=1, K=4, N=24, linear_solver=ls, omega_f=0.0, coriolis=OMEGA_EARTH,
                            omega_t=0.93 * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                            consistent_reference=True, frame='mantle')
            S.LAYERS_BANDED = 'always'
            return S, lambda S: iers_fit(S, g)
        S = TidalSolver(EarthModel('data/PREMs.dat'), m=int(case[-1]), K=3, N=24, omega_f=0.0, linear_solver=ls)
        S.LAYERS_BANDED = 'always'
        return S, None
    a, b, info = _pair(build)
    # static m = 1: the border rows (Tisserand) are never pivots, in _LayerLU as here; the first solve has a
    # residual of order 10 - 100 and the refinement does the work (dense panels: 1.5e-12 after 3 steps, here
    # 2e-8 after 2; Love numbers equal to SuperLU's to 3e-14 in both). The default solver sends this case to SuperLU.
    assert info['banded'] and info['residual'] < (1e-7 if case == 'static1' else 1e-10)
    assert np.max(np.abs(a - b)) < (1e-8 if case == 'slipping' else 1e-10)
