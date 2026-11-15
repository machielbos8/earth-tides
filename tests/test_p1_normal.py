# -*- coding: utf-8 -*-
#
# The initial-stress traction P1(tau).n uses the true boundary normal where tau = -q I (fluid side) and the
# background normal elsewhere (route B, 1/delta term, 2026-10-31).
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
"""For tau = -q I, P1 = tau D - tau G^T + omega tau - tau omega = -q (D I - G^T), so P1.n = -q (D n - G^T n)
with the normal passed in. With the background normal n_bg the normal part keeps q (n - n_bg)_t d_r u_theta,
which grows as 1/delta in the Ekman layer of a welded viscous core (the AM residual of route B next to the
FCN: 3.5e-3 -> 8.0e-3 from delta 0.5 to 0.25 km; with the true normal on the core side 5.5e-4 / 4.7e-4)."""
import numpy as np
import pytest

from earthtides.jets import J
from earthtides.model import EarthModel
from earthtides.solver import TidalSolver


def _jets(rr, tt, pp, rt):
    z = np.zeros_like(rr)
    return {'rr': J(rr, z, z), 'tt': J(tt, z, z), 'pp': J(pp, z, z), 'rt': J(rt, z, z)}


def test_isotropic_detects_fluid_side():
    q = np.array([-3.1e5, 2.0e5, 7.7e4])
    assert TidalSolver._isotropic(_jets(q, q, q, 0 * q))
    assert not TidalSolver._isotropic(_jets(q, q * (1 + 1e-6), q, 0 * q))     # deviatoric (mantle side)
    assert not TidalSolver._isotropic(_jets(q, q, q, 1e-3 * q))


def test_p1_isotropic_known_answer():
    """_p1_traction for tau = -q I equals -q (D n - G^T n) for an arbitrary displacement gradient."""
    T = TidalSolver.__new__(TidalSolver)
    T.m = m = 1
    rng = np.random.default_rng(1)
    n = 5
    r, th = np.full(n, 3.48e6), rng.uniform(0.2, 2.9, n)
    q = rng.normal(size=n) * 1e5

    class Props(dict):
        pass
    p = Props(tau=_jets(-q, -q, -q, 0 * q))
    p.r, p.isn, p.cot = r, 1 / np.sin(th), 1 / np.tan(th)
    u = [J(*(rng.normal(size=n) for _ in range(3))) for _ in range(3)]
    a = rng.uniform(-0.1, 0.1, n)
    nr, nt = np.cos(a), np.sin(a)
    tr, tt, tp = T._p1_traction(*u, p, nr, nt)
    A, B, C = (x.v for x in u)
    G = {'rr': u[0].r, 'rt': (u[0].t - B) / r, 'rl': (-m * A * p.isn - C) / r,
         'tr': u[1].r, 'tt': (u[1].t + A) / r, 'tl': (-m * B * p.isn - C * p.cot) / r,
         'lr': u[2].r, 'lt': u[2].t / r, 'll': (m * C * p.isn + A + B * p.cot) / r}
    D = G['rr'] + G['tt'] + G['ll']
    # (G^T n)_a = sum_c G[c a] n_c with n = (nr, nt, 0)
    want = (-q * (D * nr - (G['rr'] * nr + G['tr'] * nt)),
            -q * (D * nt - (G['rt'] * nr + G['tt'] * nt)),
            -q * (0 - (G['rl'] * nr + G['tl'] * nt)))
    for g, w in zip((tr, tt, tp), want):
        assert np.allclose(g, w, rtol=1e-9, atol=1e-9 * np.max(np.abs(w)))


def test_p1_normal_option():
    md = EarthModel('data/PREM.dat')
    with pytest.raises(ValueError):
        TidalSolver(md, p1_normal='true')
