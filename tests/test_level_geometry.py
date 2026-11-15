# -*- coding: utf-8 -*-
#
# LevelGeometry: coordinate surfaces as level surfaces of the reference potential.
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
# 14/11/2026 Machiel Bos
#===============================================================================
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from earthtides.model import EarthModel, G
from earthtides.geometry import (ClairautGeometry, LevelGeometry, OMEGA_EARTH, mapping_from_R, figure,
                                 equatorial_radius, surface_eps)
from earthtides.iers import iers_fit
from earthtides.solver import TidalSolver


def _flattening(g, md):
    R = g.map_derivs(md.n_layers - 1, np.array([[md.r1[-1]]]), np.array([1e-9, np.pi / 2]))[0][0]
    return (R[1] - R[0]) / R[1]


def test_derivatives():
    md = EarthModel(ROOT / 'data/PREM.dat')
    g = LevelGeometry(md)
    for s, th, h in ((3.0e6, 0.7, 1e-5), (5.9e6, 1.1, 1e-5)):
        i = int(np.flatnonzero((md.r0 <= s) & (s <= md.r1))[0])
        R = lambda s_, t_: g.map_derivs(i, np.array(s_), t_)[0]
        _, R_s, R_ss, R_t, R_st, R_tt = g.map_derivs(i, np.array(s), th)
        assert abs((R(s * (1 + h), th) - R(s * (1 - h), th)) / (2 * s * h) - R_s) < 1e-8
        assert abs((R(s, th + h) - R(s, th - h)) / (2 * h) - R_t) / s < 1e-8
        assert abs((R(s, th + h) - 2 * R(s, th) + R(s, th - h)) / h ** 2 - R_tt) / s < 1e-4
        assert abs((R(s * (1 + h), th) - 2 * R(s, th) + R(s * (1 - h), th)) / (s * h) ** 2 - R_ss) * s < 1e-4


def test_maclaurin():
    """Homogeneous body: the level figure is the Maclaurin spheroid."""
    md = EarthModel(ROOT / 'data/HOMOG.dat')
    rho = float(md.rho[0](md.a / 2))
    rhs = OMEGA_EARTH ** 2 / (2 * np.pi * G * rho)
    mac = lambda e: np.sqrt(1 - e * e) / e ** 3 * (3 - 2 * e * e) * np.arcsin(e) - 3 * (1 - e * e) / e ** 2 - rhs
    e = brentq(mac, 1e-3, 0.5)
    f = 1 - np.sqrt(1 - e * e)
    g = LevelGeometry(md)
    assert g.history[-1][1] < 1e-3                                  # m
    assert abs(_flattening(g, md) / f - 1) < 2e-6, (_flattening(g, md), f)
    assert abs(g.flattening_surface / f - 1) > 1e-4                 # the first-order figure is not it


def test_prem_level_and_angular_momentum():
    """PREM: walls level to < 1 mm, the slope term of the slipping wall vanishes and the angular-momentum
    balance holds to 1e-7 (first-order figure: 8e-4)."""
    import check_angular_momentum as cam
    md = EarthModel(ROOT / 'data/PREM.dat')
    g = LevelGeometry(md)
    assert g.history[-1][1] < 5e-4
    assert abs(1 / _flattening(g, md) - 299.72) < 0.01
    res, h0 = [], []
    for on in (False, True):
        S = TidalSolver(md, g, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=1.00273 * OMEGA_EARTH,
                        omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                        frame='mantle', slip_nonlevel=on)
        S.solve()
        Hc, Nc, lhs = cam.balance(S, 1.00273 * OMEGA_EARTH)
        res.append(np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc))
        h0.append(S.results()['h'])
    assert max(res) < 1e-7, res
    assert abs(h0[1] - h0[0]) < 1e-7


def test_chambat2010():
    """Second-order hydrostatic figure of PREM, Chambat et al. (2010) eq. (26) at m = 0.00345039:
    f2 = -2.228947e-3, f4 = 4.445e-6 (Denis 1989: 4.465e-6)."""
    md = EarthModel(ROOT / 'data/PREM.dat')
    m0 = OMEGA_EARTH ** 2 * md.a ** 3 / (G * md.M)
    g = LevelGeometry(md, omega=OMEGA_EARTH * np.sqrt(0.00345039 / m0))
    c = g.surface_legendre()
    assert abs(c[2] / c[0] / -2.228947e-3 - 1) < 2e-5
    assert 4.44e-6 < c[4] / c[0] < 4.48e-6


def test_figure_switch_and_conventions():
    md = EarthModel(ROOT / 'data/PREM.dat')
    g1, g2 = figure(md, 'first-order'), figure(md, 'level')
    assert type(g1) is ClairautGeometry and type(g2) is LevelGeometry
    assert type(figure(md, 'level', scale=0.0)) is ClairautGeometry          # the sphere
    # first-order: the conventions' numbers are the old formulas
    assert equatorial_radius(g1, md) == md.a * (1 + g1.flattening_surface / 3.0)
    assert surface_eps(g1, md)[0] == g1.flattening_surface
    # level: from its own map
    assert abs(equatorial_radius(g2, md) - 6378103.7) < 0.1
    assert abs(1 / surface_eps(g2, md)[0] - 299.14) < 0.01
    S = TidalSolver(md, g2, m=2, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=1.927 * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, dyn_fluid=True, consistent_reference=True, frame='mantle')
    S.solve()
    f = iers_fit(S, g2)
    S.geom = g1                      # the same field with the first-order normalisation: (a_e ratio)^4
    r = (equatorial_radius(g2, md) / equatorial_radius(g1, md)) ** 4
    assert abs(iers_fit(S, g1)['h0'] / f['h0'] - r) < 1e-12


def test_love_figure_option():
    """love.Options() (figure='level', the default) = the direct solver call on the level figure; 'first-order'
    is the earlier figure."""
    from earthtides import love
    o = love.Options(model='PREM', anelastic=False)                 # the level figure is the default
    S, r = love.solve_wave(o, 1, 0.92700 * OMEGA_EARTH * 86400 / (2 * np.pi))
    assert type(S.geom) is LevelGeometry
    p = love.parameters(S, r, o)
    assert abs(p['h0'] - iers_fit(S, S.geom)['h0']) < 1e-14
    assert o.figure == 'level'
    o1 = love.Options(model='PREM', anelastic=False, figure='first-order')
    S1, r1 = love.solve_wave(o1, 1, 0.92700 * OMEGA_EARTH * 86400 / (2 * np.pi))
    assert type(S1.geom) is ClairautGeometry
    d = (p['h0'] - love.parameters(S1, r1, o1)['h0']).real
    assert -3.5e-5 < d < -1.5e-5                                    # -2.5e-5 at O1 (love_level_figure.py)
    assert 'first' in love.default_name(o1) and 'first' not in love.default_name(o)


def test_nonhydro_on_level_figure():
    """The non-hydrostatic machinery on the level figure (plan step 4): zero load = hydrostatic exactly; the
    450 m state balances angular momentum to < 5e-5 (first-order figure: 8e-4 .. 1.5e-3); its FCN lies 25.3 sd
    below the hydrostatic one (first-order figure: 457.33 -> 431.90, 25.4 sd); a state built on one figure is
    refused on another."""
    import pytest
    import check_angular_momentum as cam
    from earthtides.nonhydro import FlatSupport, FlatState, FlatMap
    md, mds = EarthModel(ROOT / 'data/PREM.dat'), EarthModel(ROOT / 'data/PREMs.dat')
    shells = [(3480e3, 5701e3), (5701e3, 6346.6e3)]
    base = figure(md, 'level')

    def run(sig, geo, nh, K=4, N=24):
        T = TidalSolver(md, geo, m=1, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                        omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                        frame='mantle', nonhydro=nh)
        T.solve()
        return T
    S = FlatSupport(md, mds, shells, base, OMEGA_EARTH)
    st0 = FlatState(S, [0.0, 0.0], base, OMEGA_EARTH)
    a, b = run(0.927, base, None, 3, 16).results(), run(0.927, FlatMap(base, st0), st0, 3, 16).results()
    assert a['h'] == b['h'] and a['kk'] == b['kk']
    st = FlatState(S, S.fit({'J2': 1.14e-5, 'cmb': 450.0}), base, OMEGA_EARTH)
    geo = FlatMap(base, st)
    for sig in (0.927, 1.00273):
        T = run(sig, geo, st)
        Hc, Nc, lhs = cam.balance(T, sig * OMEGA_EARTH)
        assert np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc) < 5e-5
    x = np.array([1.0012, 1.0016, 1.0030, 1.0036])
    P = []
    for g_, nh in ((base, None), (geo, st)):
        k = np.array([run(s, g_, nh).results()['kk'] for s in x]).real
        P.append(1 / (np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0] - 1))
    assert abs(P[0] - 456.24) < 0.05 and abs(P[0] - P[1] - 25.3) < 0.2
    with pytest.raises(ValueError):
        FlatMap(figure(md, 'first-order'), st)


def _viscous_h0(md, g, sig=1.00273, K=6, N=24):
    S = TidalSolver(md, g, m=1, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                    frame='mantle')
    S.solve()
    return iers_fit(S, g)


def test_viscous_core_figure_from_the_unsplit_model():
    """The level figure of a viscous model (core split into Ekman sub-layers, different for every frequency
    and delta) started from the figure of the unsplit model: one potential solve confirms it (no iteration),
    it equals the figure built from scratch on the split model to < 1 mm, and the Love numbers agree."""
    md = EarthModel(ROOT / 'data/PREM.dat')
    w = 1.00273 * OMEGA_EARTH
    mv = md.viscous_core(w, delta=500.0)
    assert mv.n_layers > md.n_layers
    LevelGeometry._lcache.clear(); LevelGeometry._parents.clear()
    g_scratch = LevelGeometry(mv)                                  # nothing cached: the full iteration
    assert len(g_scratch.history) > 10
    LevelGeometry._lcache.clear(); LevelGeometry._parents.clear()
    g0 = LevelGeometry(md)
    g_auto = LevelGeometry(mv)                                     # finds g0 in the cache of this process
    g_expl = g0.restrict(md.viscous_core(w, delta=250.0))          # explicit (worker processes)
    assert len(g_auto.history) == 1 and g_auto.history[0][1] < 5e-4
    assert len(g_expl.history) == 1 and g_expl.history[0][1] < 5e-4
    for s in (mv.a, 3480e3, 1221.5e3):
        a, b = g_scratch.surface_legendre(s=s), g_auto.surface_legendre(s=s)
        assert max(abs(a[n] - b[n]) for n in a) < 1e-3             # m
    fa, fb = _viscous_h0(mv, g_scratch), _viscous_h0(mv, g_auto)
    assert abs(fa['h0'] - fb['h0']) < 2e-8 and abs(fa['h2'] - fb['h2']) < 2e-9
    # a model that is not a split of the cached one starts from scratch
    assert g0._restricted_to(EarthModel(ROOT / 'data/HOMOG.dat')) is None


def test_orders_0_and_2_on_level_figure():
    """Mf (m = 0) and M2 (m = 2) on the level figure: the Love module equals the direct solver call, and the
    difference from the first-order figure is the measured one (love_level_figure.py: h0 -1.0e-5 and
    -0.8e-5, latitude terms below 4e-6)."""
    from earthtides import love
    cpd = OMEGA_EARTH * 86400 / (2 * np.pi)
    for m, sig, dh0 in ((0, 0.07300, -1.0e-5), (2, 1.92700, -0.8e-5)):
        p = {}
        for fig in ('level', 'first-order'):
            o = love.Options(model='PREM', anelastic=False, figure=fig, K=6, N=30)
            S, r = love.solve_wave(o, m, sig * cpd)
            p[fig] = love.parameters(S, r, o)
            if fig == 'level':
                assert abs(p[fig]['h0'] - iers_fit(S, S.geom)['h0']) < 1e-14
        d = {k: (p['level'][k] - p['first-order'][k]).real for k in ('h0', 'h2', 'l0', 'l2')}
        assert abs(d['h0'] - dh0) < 3e-6, (m, d)
        assert abs(d['h2']) < 4e-6 and abs(d['l2']) < 4e-6 and abs(d['l0']) < 4e-6, (m, d)


def test_viscous_support_on_level_figure():
    """Viscous support of the non-hydrostatic state (nonhydro.viscous_model) on the level figure: the fit reaches
    its targets, the FCN moves as on the first-order figure (459.0 -> 437.1 sd, anelastic, e_f of Mathews et
    al.; results/paper1_table4_level.txt) and the kilometres of surface topography are there."""
    from earthtides import love
    o = love.Options(nonhydrostatic=dict(support='viscous', targets={'J2': 1.14e-5, 'e_f': 9.73e-5}))
    assert o.figure == 'level'
    resp = love._model(o)[2].summary()['response']
    assert abs(resp['e_f'] / 9.73e-5 - 1) < 1e-9 and abs(resp['J2'] / 1.14e-5 - 1) < 1e-9
    assert 3600 < resp['surface'] < 4000 and 340 < resp['cmb'] < 385
    x = np.array([1.0012, 1.0016, 1.0030, 1.0036])
    k = np.array([love.solve_wave(o, 1, s * love.F_TILT)[1]['kk'] for s in x]).real
    c = np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0]
    assert abs(1 / (c - 1) - 437.1) < 0.3
