# -*- coding: utf-8 -*-
#
# Known-answer tests of the gravimetric factor.
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
import numpy as np
import pytest

from earthtides.model import EarthModel
from earthtides.love import model_path
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.gravimetry import gravimetric_factor, surface_gravity


@pytest.mark.parametrize('m', [0, 1, 2])
def test_sphere_static(m):
    """Static, non-rotating sphere: delta = 1 + h - 3/2 k exactly, no latitude dependence."""
    S = TidalSolver(EarthModel(str(model_path('PREMs'))), m=m, K=3, N=24, omega_f=0.0)
    S.solve()
    R = S.results()
    d0, dp, mis = gravimetric_factor(S)
    assert abs(d0 - (1 + R['h'] - 1.5 * R['kk'])) < 1e-12
    assert abs(dp) < 1e-12 and mis < 1e-12


def test_rigid_tilt_changes_no_gravity():
    """Just above the tilt-over frequency the solution is dominated by a rigid tilt xi of the whole
    Earth, which a gravimeter cannot see: dg per unit xi must be ~0 (it was 5e-5 m/s^2 with the
    normal of the mapped surface instead of the plumb line)."""
    md = EarthModel(str(model_path('PREM')))
    om = OMEGA_EARTH * (1 + 1e-7)
    S = TidalSolver(md, ClairautGeometry(md), m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=om, omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve()
    S.surface_field(np.array([1.0]))
    th, dg, dgT = surface_gravity(S)
    assert np.sqrt(np.mean(np.abs(dg)**2)) / abs(S._xi) < 2e-6
