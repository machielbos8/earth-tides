# -*- coding: utf-8 -*-
#
# solid_P: the potential in the solid layers at their own degrees, with harmonic fields beyond the CMB / ICB,
# reproduces the full potential (plan step 4, lab notes 2026-11-05).
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
# 05/11/2026 Machiel Bos
#===============================================================================
import numpy as np
import pytest

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit

KEYS = ('h0', 'h2', 'l0', 'l1', 'l2')


def love(m, f, solid_P, Kc=16):
    md = EarthModel('data/PREM.dat').viscous_core(f * OMEGA_EARTH, delta=1e3)
    g = ClairautGeometry(md)
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': Kc, 'N': 30}]})
    S = TidalSolver(md, g, m=m, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=f * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=True,
                    frame='mantle', band=12, solid_P=solid_P)
    r = S.solve()
    fit = iers_fit(S, g)
    return np.array([complex(fit[k]) for k in KEYS] + [r['kk']]), S


@pytest.mark.parametrize('m, f', [(1, 0.927), (2, 1.927)])
def test_solid_P_matches_full_potential(m, f):
    full, S0 = love(m, f, None)
    cut, S1 = love(m, f, 0)
    assert S1._ntot < S0._ntot                       # fewer unknowns: the mantle lost its high-degree P
    assert len(S1.degP) < len(S0.degP)               # exterior coefficients: the surface layer's degrees
    assert np.max(np.abs(cut - full)) < 1e-11        # measured 2026-11-05: <= 8e-14 (O1), 2e-14 (M2)


def test_solid_P_refuses_inviscid_core():
    md = EarthModel('data/PREM.dat')
    with pytest.raises(ValueError):
        TidalSolver(md, ClairautGeometry(md), m=2, K=4, N=12, solid_P=0)
