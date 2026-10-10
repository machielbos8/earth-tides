# -*- coding: utf-8 -*-
#
# Brute-force check of the inverse-map derivatives and the chain rule.
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
# 5/10/2026 Machiel Bos
#===============================================================================
"""Brute-force check of the inverse-map derivatives and the chain rule.

F(s,theta) is an arbitrary smooth function; G(r,theta) = F(S(r,theta),theta)
is evaluated by Newton inversion of R(s,theta) and differentiated with
central finite differences.
"""
import numpy as np

from earthtides.geometry import mapping, chain

A, K = 0.05, 3.0e-7          # eps(s) = A (1 + sin(K s)), deliberately large


def eps(s):
    return A * (1 + np.sin(K * s)), A * K * np.cos(K * s), -A * K * K * np.sin(K * s)


def R_of(s, t):
    return mapping(s, t, *eps(s))['R']


def S_of(r, t):
    s = r
    for _ in range(60):
        e, de, dde = eps(s)
        m = mapping(s, t, e, de, dde)
        s = s - (m['R'] - r) * m['s_r']
    return s


def F(s, t):
    return np.cos(2e-6 * s) * np.sin(t) ** 2 + 1e-6 * s * np.cos(3 * t)


def F_derivs(s, t):
    a = 2e-6
    Fv = F(s, t)
    Fs = -a * np.sin(a * s) * np.sin(t) ** 2 + 1e-6 * np.cos(3 * t)
    Fss = -a * a * np.cos(a * s) * np.sin(t) ** 2
    Ft = np.cos(a * s) * np.sin(2 * t) - 3e-6 * s * np.sin(3 * t)
    Fst = -a * np.sin(a * s) * np.sin(2 * t) - 3e-6 * np.sin(3 * t)
    Ftt = 2 * np.cos(a * s) * np.cos(2 * t) - 9e-6 * s * np.cos(3 * t)
    return Fv, Fs, Fss, Ft, Fst, Ftt


def test_chain_rule():
    s0, t0 = 4.2e6, 0.7
    r0 = R_of(s0, t0)
    m = mapping(s0, t0, *eps(s0))
    d = chain(*F_derivs(s0, t0), m)

    Gf = lambda r, t: F(S_of(r, t), t)
    hr, ht = 50.0, 1e-4
    fd = dict(
        r=(Gf(r0 + hr, t0) - Gf(r0 - hr, t0)) / (2 * hr),
        t=(Gf(r0, t0 + ht) - Gf(r0, t0 - ht)) / (2 * ht),
        rr=(Gf(r0 + hr, t0) - 2 * Gf(r0, t0) + Gf(r0 - hr, t0)) / hr**2,
        tt=(Gf(r0, t0 + ht) - 2 * Gf(r0, t0) + Gf(r0, t0 - ht)) / ht**2,
        rt=(Gf(r0 + hr, t0 + ht) - Gf(r0 + hr, t0 - ht)
            - Gf(r0 - hr, t0 + ht) + Gf(r0 - hr, t0 - ht)) / (4 * hr * ht),
    )
    for k, v in fd.items():
        assert abs(d[k] - v) <= 1e-5 * (abs(v) + 1e-12 / (hr if 'r' in k else 1)), (k, d[k], v)
