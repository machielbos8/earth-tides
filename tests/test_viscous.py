# -*- coding: utf-8 -*-
#
# EarthModel.split_layer / viscous_core: the split keeps the model;
# viscous core = purely imaginary mu, kappa kept, Ekman boundary layers.
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
# 17/10/2026 Machiel Bos
#===============================================================================
"""EarthModel.split_layer / viscous_core (manual / lab notes: the inviscid fluid core)."""
import numpy as np
import pytest

from earthtides.model import EarthModel, OMEGA_ROT

PREM = 'data/PREM.dat'


def test_split_layer_keeps_model():
    md = EarthModel(PREM)
    sp = md.split_layer(1, [1.5e6, 3.0e6])
    assert sp.n_layers == md.n_layers + 2
    assert sp.M == pytest.approx(md.M, rel=1e-14)
    r = np.linspace(md.r0[1], md.r1[1], 7)
    for k, rr in ((1, r[r <= 1.5e6]), (2, r[(r >= 1.5e6) & (r <= 3.0e6)]), (3, r[r >= 3.0e6])):
        assert np.allclose(sp.rho[k](rr), md.rho[1](rr), rtol=1e-13)
        assert np.allclose(sp.lam[k](rr), md.lam[1](rr), rtol=1e-13)
    assert np.allclose(sp.gravity(4, sp.r1[4]), md.gravity(2, md.r1[2]), rtol=1e-13)
    with pytest.raises(ValueError):
        md.split_layer(1, [3.0e6, 1.5e6])


def test_viscous_core():
    md = EarthModel(PREM)
    w = 1.4e-4
    v = md.viscous_core(w, delta=5e3)
    assert v.n_layers == md.n_layers + 2 and v.is_complex
    # boundary layers of d_min = 40 km at the ICB and the CMB (5 delta = 25 km > d_min / 2: no inner layer)
    assert v.r1[0] == md.r1[0] and v.r1[1] == pytest.approx(md.r0[1] + 4e4)
    assert v.r1[2] == pytest.approx(md.r1[1] - 4e4) and v.r1[3] == md.r1[1]
    # delta = 0.5 km: inner layers of 5 delta = 2.5 km inside the 40 km layers
    v5 = md.viscous_core(w, delta=5e2)
    assert v5.n_layers == md.n_layers + 4
    assert np.allclose(v5.r1[1:5] - md.r0[1], [2.5e3, 4e4, md.r1[1] - md.r0[1] - 4e4,
                                                md.r1[1] - md.r0[1] - 2.5e3], rtol=0.02)
    for k in (1, 2, 3):
        mu = v.mu[k](0.5 * (v.r0[k] + v.r1[k]))
        assert mu.real == 0 and mu.imag > 0
        # kappa unchanged
        r = 0.5 * (v.r0[k] + v.r1[k])
        assert v.lam[k](r) + 2 * mu / 3 == pytest.approx(md.lam[1](r), rel=1e-14)
    # delta <-> eta with the mean core density
    eta = v.mu[2].coef[0].imag / w
    assert np.sqrt(eta / (1.0e4 * OMEGA_ROT)) == pytest.approx(5e3, rel=0.1)
    # solid layers untouched
    assert v.mu[0] is md.mu[0] or np.allclose(v.mu[0].coef, md.mu[0].coef)
    assert md.viscous_core(w, delta=5e3, boundary_layers=False).n_layers == md.n_layers
    with pytest.raises(ValueError):
        md.viscous_core(w)


def test_ekman_spin_over_coefficient():
    """The Ekman-layer stress integral of examples/ex6 gives Greenspan's (1968) spin-over correction."""
    ek = _ex6()
    assert ek.ekman_coefficient(1.0) == pytest.approx(0.2585 + 2.6205j, abs=2e-4)


def test_hough_fcn_ekman_damping():
    """Hough's Earth (ex5) with a no-slip viscous core, delta = 1 km: the complex FCN shift equals
    (A/A_m) K, K = Ekman (Greenspan) correction, to 4 % (K = 6; examples/ex6: 1.6 % at K = 12,
    delta = 0.25 km; the residual decreases slowly with delta: critical-latitude structure)."""
    from earthtides.geometry import OMEGA_EARTH
    ek = _ex6()
    s0 = ek.job((0.0, 6, 24))[1]
    s = ek.job((1e3, 6, 24))[1]
    assert abs(s0.imag) < 1e-9
    k = (s - s0) * ek.RC / (ek.A_RATIO * 1e3)
    ke = ek.ekman_coefficient(s.real)
    assert abs(k - ke) < 0.04 * abs(ke), (k, ke)
    assert abs(k.imag / ke.imag - 1) < 0.03


def _ex6():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / 'examples' / 'ex6' / 'ekman.py'
    spec = importlib.util.spec_from_file_location('ekman_ex6', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_viscous_core_resolution_rule():
    """solver.viscous_core_resolution: N per kind of fluid sub-layer. The 5-delta layers need no more than
    N 16 (same Love numbers as N 30 to 1e-8 next to the FCN), the 40 km layers get 36 below delta 500 m."""
    from pathlib import Path
    from earthtides.geometry import figure, OMEGA_EARTH
    from earthtides.solver import TidalSolver, resolution_per_layer, viscous_core_resolution
    from earthtides.iers import iers_fit
    DATA_ = Path(__file__).resolve().parents[1] / 'data'
    md0 = EarthModel(DATA_ / 'PREM.dat')
    w = 1.00273 * OMEGA_EARTH
    md = md0.viscous_core(w, delta=350.0)
    g = figure(md0, 'level').restrict(md)
    spec = viscous_core_resolution(12, 350.0)
    Kl, Nl = resolution_per_layer(md, spec)
    thick = [(md.r1[i] - md.r0[i]) / 1e3 for i in range(md.n_layers)]
    fluid = [abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) < 1e8 for i in range(md.n_layers)]
    assert sorted({Nl[i] for i in range(md.n_layers) if fluid[i]}) == [16, 30, 36]
    assert all(Nl[i] == 16 for i in range(md.n_layers) if fluid[i] and thick[i] < 3.5)
    assert all(Nl[i] == 36 for i in range(md.n_layers) if fluid[i] and 3.5 < thick[i] < 100)
    assert all(Kl[i] == (12 if fluid[i] else 6) for i in range(md.n_layers))
    assert viscous_core_resolution(12, 1000.0)['rules'][1]['N'] == 30
    h = []
    for ne in (16, 30):
        Kl, Nl = resolution_per_layer(md, viscous_core_resolution(12, 350.0, N_ekman=ne))
        S = TidalSolver(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                        omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                        frame='mantle')
        S.solve()
        h.append(iers_fit(S, g)['h0'])
    assert abs(h[0] - h[1]) < 1e-8
