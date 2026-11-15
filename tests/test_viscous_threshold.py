# -*- coding: utf-8 -*-
#
# A viscous core stays viscous (welded) at thin Ekman layers
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
# 10/11/2026 Machiel Bos
#===============================================================================
"""Until 2026-11-10 TidalSolver treated a layer with |mu| < 1 Pa as a true fluid; a viscous core with delta below
~130 m (PREM) then became a slipping inviscid core without warning. Now only mu identically zero is a true fluid."""
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.love import model_path


def _solver(md):
    w = 0.927 * OMEGA_EARTH
    return TidalSolver(md, ClairautGeometry(md), m=1, K=2, N=8, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                       omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                       frame='mantle')


def test_prem_liquid_core_is_true_fluid():
    md = EarthModel(str(model_path('PREM')))
    S = _solver(md)
    assert sum(S.true_fluid) == 1


def test_thin_viscous_core_is_welded():
    md0 = EarthModel(str(model_path('PREM')))
    for delta in (2e3, 125.0, 30.0):
        md = md0.viscous_core(0.927 * OMEGA_EARTH, delta=delta)
        mu_core = max(abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) for i in range(md.n_layers)
                      if md.r0[i] >= 1221e3 and md.r1[i] <= 3480e3 + 1)
        assert mu_core > 0
        assert not any(_solver(md).true_fluid), delta
