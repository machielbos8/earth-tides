# -*- coding: utf-8 -*-
#
# fluid_pressure=True: Pi = lambda div u as an unknown of its own in the (weak-)fluid layers.
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
# 13/11/2026 Machiel Bos
#===============================================================================
"""The mixed form (stress Pi I + 2 mu eps, rows div u - Pi / lambda = 0; docs/plan_fluid_pressure_2026-11-13.md)
against the displacement form on cases where the latter is well conditioned: the same equations, so the Love
numbers must agree up to the discretisation (Pi has two Chebyshev coefficients less and its own angular
truncation), and the difference must fall with N and K."""
import numpy as np
import pytest

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit

PREM = 'data/PREM.dat'
KEYS = ('h0', 'h2', 'l2', 'l1')


def _sphere(md, N, fp, ls='sparse'):
    S = TidalSolver(md, m=2, K=1, N=N, omega_f=1.93 * OMEGA_EARTH, linear_solver=ls, fluid_pressure=fp)
    r = S.solve()
    return np.array([r[k] for k in ('h', 'l', 'kk')]), S


def _spheroid(md, g, K, N, fp, ls='layers', sig=0.93):
    S = TidalSolver(md, g, m=1, K=K, N=N, linear_solver=ls, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                    consistent_reference=True, frame='mantle', fluid_pressure=fp)
    S.solve()
    f = iers_fit(S, g)
    return np.array([f[k] for k in KEYS]), S


def test_sphere_viscous_core():
    """Sphere, no rotation, degree 2, viscous PREM with its Ekman sub-layers (measured 2e-11 .. 6e-11)."""
    md = EarthModel(PREM).viscous_core(1.93 * OMEGA_EARTH, delta=2.0e3)
    a, S0 = _sphere(md, 24, False)
    b, S1 = _sphere(md, 24, True)
    fl = S1.fluid_layers()
    assert all(('Q', 2) in S1.B[i].fields for i in range(md.n_layers) if fl[i])
    assert not any(t == 'Q' for i in range(md.n_layers) if not fl[i] for t, _ in S1.B[i].fields)
    assert S1.Hsp.shape[0] == S0.Hsp.shape[0] + 24 * sum(fl)
    assert np.max(np.abs(a - b)) < 1e-9


def test_sphere_converges_in_N():
    """One thick viscous layer (delta 50 km, no sub-layers): the two forms approach each other with N
    (measured 1.4e-5, 6e-7, 2e-10 at N 16, 24, 36)."""
    md = EarthModel(PREM).viscous_core(1.93 * OMEGA_EARTH, delta=5.0e4, boundary_layers=False)
    d = [np.max(np.abs(_sphere(md, N, False)[0] - _sphere(md, N, True)[0])) for N in (16, 24, 36)]
    assert d[0] > d[1] > d[2] and d[2] < 2e-9


def test_rotating_spheroid():
    """Rotating spheroid, diurnal, thick viscous layer (converged in angle at K 8): measured 5e-8 in h0,
    2e-8 in h2 at K 8 / N 36 (5e-7 at N 24)."""
    sig = 0.93
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=5.0e4, boundary_layers=False)
    g = ClairautGeometry(md)
    a, _ = _spheroid(md, g, 8, 36, False)
    b, S = _spheroid(md, g, 8, 36, True)
    assert S._solver_used == 'layers' and S._layer_info['residual'] < 1e-11
    assert np.max(np.abs(a - b)) < 2e-7


@pytest.mark.parametrize('ls', ['layers', 'block'])
def test_solvers_on_the_saddle_point_system(ls):
    """The Q rows have a tiny diagonal (1 / lambda): the direct solvers against SuperLU."""
    sig = 0.93
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=2.0e3)
    g = ClairautGeometry(md)
    a, _ = _spheroid(md, g, 4, 24, True, ls='sparse')
    b, S = _spheroid(md, g, 4, 24, True, ls=ls)
    assert np.max(np.abs(a - b)) < 2e-8


def test_true_fluid_refused():
    with pytest.raises(ValueError):
        TidalSolver(EarthModel(PREM), m=2, K=1, N=16, fluid_pressure=True)
