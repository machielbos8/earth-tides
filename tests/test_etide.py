# -*- coding: utf-8 -*-
#
# etide at special latitudes: finite and continuous at the equator (the l(1) term once divided by
# F_21, which vanishes there).
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
# 9/10/2026 Machiel Bos
#===============================================================================
import numpy as np

from earthtides.etide import LoveModel, Tamura, displacement, geocentric


def test_equator_finite_and_continuous():
    mjd = 61041.0 + np.arange(0, 3, 1 / 24)
    u0 = np.array(displacement(mjd, 0.0, 10.0))
    u1 = np.array(displacement(mjd, 1e-7, 10.0))
    assert np.isfinite(u0).all()
    assert np.abs(u0 - u1).max() < 1e-9


def _one_term(cat, key, i):
    L = LoveModel(cat, 'iers')
    for k in ('h0', 'h2', 'l0', 'l2', 'l1', 'hp', 'lp'):
        getattr(L, k)[:] = 0
    getattr(L, key)[i] = 1e-3
    return L


def test_lprime_hprime_follow_online_resource():
    """l' (diurnal) is a rigid rotation: north = H l' at every latitude, east = H l' sin(phi);
    h' (zonal) is a uniform radial H h'. H: Cartwright-Tayler amplitude (OR 1, Sect. S8)."""
    cat = Tamura()
    mjd = np.arange(60676, 60706, 1 / 24)
    d1 = np.flatnonzero((cat.n == 2) & (cat.m == 1))
    k1 = d1[np.argmin(abs(cat.f_cpd[d1] - 1.0027379))]
    for lat in (0.0, 30.0, 60.0):
        phi, _ = geocentric(lat)
        ur, un, ue = displacement(mjd, lat, 0.0, love=_one_term(cat, 'lp', k1), degrees=(2,))
        assert abs(np.abs(un).max() - 0.36878e-3) < 2e-7
        assert abs(np.abs(ue).max() - 0.36878e-3 * abs(np.sin(phi))) < 2e-7
        assert np.abs(ur).max() == 0
    z = np.flatnonzero((cat.n == 2) & (cat.m == 0))
    mf = z[np.argmin(abs(cat.f_cpd[z] - 0.0732022))]
    for lat in (0.0, 45.0):
        ur, un, ue = displacement(mjd, lat, 0.0, love=_one_term(cat, 'hp', mf), degrees=(2,))
        assert abs(np.abs(ur).max() - 0.06663e-3) < 2e-7
