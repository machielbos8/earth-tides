# -*- coding: utf-8 -*-
#
# etide at special latitudes: finite and continuous at the equator (the l(1) term once divided by
# F_21, which vanishes there).
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

from earthtides.etide import displacement


def test_equator_finite_and_continuous():
    mjd = 61041.0 + np.arange(0, 3, 1 / 24)
    u0 = np.array(displacement(mjd, 0.0, 10.0))
    u1 = np.array(displacement(mjd, 1e-7, 10.0))
    assert np.isfinite(u0).all()
    assert np.abs(u0 - u1).max() < 1e-9
