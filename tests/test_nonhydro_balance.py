# -*- coding: utf-8 -*-
#
# The core's static pressure of a non-hydrostatic state balances the solids' normal traction at the CMB and
# the ICB (task C, 2026-10-31).
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
# 31/10/2026 Machiel Bos
#===============================================================================
"""Hough's Earth with 450 m excess CMB flattening (scripts/hough_nonhydro_fcn.py). FlatState.core_dp must
satisfy tau_nn = -dp at both fluid-solid boundaries in degree 0 and P2 (to 1e-5 of max |dp|); the earlier gauge form p0' s_r q
was off by 2.4e5 Pa in degree 0, which made the welded viscous FCN overshoot the e_f formula by 15-20 %."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


def test_core_dp_balances_cmb_and_icb(monkeypatch):
    monkeypatch.chdir(ROOT)
    h = pytest.importorskip('hough_nonhydro_fcn')                # scripts of the paper (reproduction package)
    nb = pytest.importorskip('nonhydro_boundary_balance')
    md, geo, st, ef, AA, r = h.setup(450.0)
    x, w = np.polynomial.legendre.leggauss(32)
    th = np.arccos(x)
    nb.CORE_DP = 'static'
    for solid, rb in ((st.ioc + 1, md.r1[st.ioc]), (st.ioc - 1, md.r0[st.ioc])):
        b = nb.balance(md, st, solid, rb, th, w)
        assert abs(b['imb0']) < 1e-5 * b['scale']
        assert abs(b['imb2']) < 1e-5 * b['scale']
