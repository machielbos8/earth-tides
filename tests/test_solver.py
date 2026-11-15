# -*- coding: utf-8 -*-
#
# Regression tests for the zonal solver.
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
# 5/10/2026 Machiel Bos
#===============================================================================
"""Regression tests for the zonal solver.

* sphere: rigid-core PREM vs astyanax computelove (static, omega = 1e-9)
* distorted mesh: a spherical Earth on a bumped mesh gives the sphere result
* hydrostatic ellipsoid: golden values of the first prototype (2026-10-05)
"""
from pathlib import Path

import numpy as np
import pytest

from earthtides.model import EarthModel
from earthtides.geometry import BumpGeometry, ClairautGeometry, OMEGA_EARTH
from earthtides.solver import ZonalSolver

DATA = Path(__file__).resolve().parents[1] / 'data'


@pytest.fixture(scope='module')
def hard():
    return EarthModel(DATA / 'PREMhard.dat')


@pytest.fixture(scope='module')
def prems():
    return EarthModel(DATA / 'PREMs.dat')


def test_sphere_vs_astyanax(hard):
    # astyanax computelove --bodytide --n=2, PREMhard, omega = 1e-9, elastic
    ref = dict(h=0.41076339800, l=0.16927332394 / 2, k=0.40495107685 / 2)
    r = ZonalSolver(hard, K=2, N=24, omega_f=0.0).solve()
    assert abs(r['h'] - ref['h']) < 5e-6
    assert abs(r['l'] - ref['l']) < 5e-6
    assert abs(r['kk'] - ref['k']) < 5e-6


def test_distorted_mesh_is_invisible(hard):
    r0 = ZonalSolver(hard, K=2, N=30, omega_f=0.0).solve()
    r = ZonalSolver(hard, BumpGeometry(hard, 0.2), K=5, N=30, omega_f=0.0,
                    background='spherical').solve()
    for key in ('h', 'l', 'kk'):
        assert abs(r[key] - r0[key]) < 1e-8
    assert abs(r['U'][0]) < 1e-9 and abs(r['U'][4]) < 1e-7


def test_hydrostatic_prem_golden(prems):
    g = ClairautGeometry(prems)
    assert abs(1 / g.flattening_surface - 299.90) < 0.01
    r = ZonalSolver(prems, g, K=5, N=30, omega_f=0.0, omega_rot=OMEGA_EARTH).solve()
    assert np.isclose(r['U'][0], -8.64082e-05, rtol=1e-4)      # before the Vp fix: -8.63774e-05
    # values after the correction of the inner-core Vp of PREM (2026-10-09; before: 0.60305786,
    # 0.08353429, 0.29728042)
    assert np.isclose(r['U'][2], 0.60305790, atol=2e-8)
    assert np.isclose(r['U'][4], -2.53074e-04, rtol=1e-4)
    assert np.isclose(r['V'][2], 0.08353430, atol=2e-8)
    assert np.isclose(r['k'][2], 0.29728047, atol=2e-8)


@pytest.mark.parametrize('m', [1, 2])
def test_sphere_independent_of_order(hard, m):
    from earthtides.solver import TidalSolver
    r0 = TidalSolver(hard, m=0, K=2, N=24, omega_f=0.0).solve()
    r = TidalSolver(hard, m=m, K=2, N=24, omega_f=0.0).solve()
    for key in ('h', 'l', 'kk'):
        assert abs(r[key] - r0[key]) < 1e-11
    assert max(abs(v) for v in r['W'].values()) < 1e-11


@pytest.mark.parametrize('m', [1, 2])
def test_distorted_mesh_is_invisible_tesseral(hard, m):
    from earthtides.solver import TidalSolver
    r0 = TidalSolver(hard, m=0, K=2, N=30, omega_f=0.0).solve()
    r = TidalSolver(hard, BumpGeometry(hard, 0.2), m=m, K=5, N=30, omega_f=0.0,
                    background='spherical').solve()
    for key in ('h', 'l', 'kk'):
        assert abs(r[key] - r0[key]) < 1e-8
    assert max(abs(v) for v in r['W'].values()) < 1e-7


@pytest.mark.parametrize('m, gold', [
    (1, dict(U2=6.0351e-01, U4=-2.3076e-04, W1=-9.5778e-04, W3=-1.3568e-04, k2=0.297822)),
    (2, dict(U2=6.0486e-01, U4=-1.6260e-04, W3=-2.1731e-04, k2=0.299459)),
])
def test_hydrostatic_prem_tesseral_golden(prems, m, gold):
    from earthtides.solver import TidalSolver
    r = TidalSolver(prems, ClairautGeometry(prems), m=m, K=4, N=30, omega_f=0.0,
                    omega_rot=OMEGA_EARTH).solve()
    got = dict(U2=r['U'][2], U4=r['U'][4], k2=r['kk'],
               **{f'W{n}': v for n, v in r['W'].items()})
    for k, v in gold.items():
        assert np.isclose(got[k], v, rtol=2e-4), (k, got[k], v)


def test_anelastic_sphere_vs_astyanax(hard):
    # astyanax computelove --bodytide --n=2, PREMhard, M2, dissipation Tm=200 s, alpha=0.15
    from earthtides.solver import TidalSolver
    w = 1.4053e-4
    ref = dict(h=0.41802544078 - 1.5958626226e-03j, l=(0.17485207100 - 1.4327460387e-03j) / 2,
               k=(0.41283110220 - 1.7765143008e-03j) / 2)
    r = TidalSolver(hard.anelastic(w, dispersion='astyanax'), m=0, K=2, N=24, omega_f=w).solve()
    assert abs(r['h'] - ref['h']) < 5e-6 and abs(r['l'] - ref['l']) < 5e-6
    assert abs(r['kk'] - ref['k']) < 5e-6
    assert abs((r['h'] - ref['h']).imag) < 1e-7


def test_coriolis_distorted_mesh_and_symmetry(hard):
    """Rotating spherical Earth: bumped mesh invisible; Coriolis response odd in Omega."""
    from earthtides.solver import TidalSolver
    kw = dict(m=2, K=4, N=30, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=1.4053e-4)
    r0 = TidalSolver(hard, **kw).solve()
    rb = TidalSolver(hard, BumpGeometry(hard, 0.2), background='spherical', **kw).solve()
    for key in ('U', 'V', 'W', 'k'):
        for n in r0[key]:
            assert abs(rb[key][n] - r0[key][n]) < 1e-7
    rp = TidalSolver(hard, **{**kw, 'coriolis': 0.01 * OMEGA_EARTH}).solve()
    rm = TidalSolver(hard, **{**kw, 'coriolis': -0.01 * OMEGA_EARTH}).solve()
    assert abs(rp['W'][3] + rm['W'][3]) < 1e-3 * abs(rp['W'][3])
    assert np.isclose(r0['W'][3], -5.212e-05, rtol=1e-3)


def test_coriolis_prem_golden(prems):
    from earthtides.solver import TidalSolver
    from earthtides.iers import iers_fit
    g = ClairautGeometry(prems)
    S = TidalSolver(prems, g, m=2, K=5, N=30, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=1.4053e-4, omega_rot=OMEGA_EARTH)
    S.solve()
    f = iers_fit(S, g)
    assert np.isclose(f['l1'], 0.000990, atol=2e-6)
    assert np.isclose(f['h2'], -0.000489, atol=2e-6)


def test_zonal_wahr_frame_vs_ddw1999(prems):
    """Rotation-rate feedback kept (Wahr 1981 eq. 4.6): zonal l0 = DDW1999 Table 11."""
    from earthtides.solver import TidalSolver
    from earthtides.iers import iers_fit
    g = ClairautGeometry(prems)
    S = TidalSolver(prems, g, m=0, K=5, N=30, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=1e-6,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True)
    S.solve()
    f = iers_fit(S, g)
    assert abs(f['l0'].real - 0.08304) < 5e-5           # DDW 0.08304
    assert abs(f['h2'].real + 0.00047) < 5e-5           # DDW -0.00047
    assert 0.00008 < f['hp'].real < 0.00012             # DDW 0.00014


@pytest.mark.parametrize('m', [0, 2])
def test_exact_homogeneous_spheroid(m):
    """Non-gravitating homogeneous spheroid: exact polynomial solution is reproduced;
    the first-order shape leaves an error of order f^2."""
    from earthtides.solver import TidalSolver
    from earthtides.geometry import ExactSpheroidGeometry
    from earthtides.exact_spheroid import solve, surface_field
    md = EarthModel(DATA / 'HOMOG.dat')
    a, g = md.a, md.g_surface
    rho, lam, mu = (float(f(a / 2)) for f in (md.rho[0], md.lam[0], md.mu[0]))
    fl = 1 / 30
    G_ = ExactSpheroidGeometry(fl)
    th = np.linspace(0.05, np.pi - 0.05, 121)
    urx, utx, _ = surface_field(solve(a * G_.A, a * G_.C, lam, mu, rho, g, a, m),
                                a * G_.A, a * G_.C, th)
    sc = max(np.abs(urx).max(), np.abs(utx).max())
    S = TidalSolver(md, G_, m=m, K=10, N=12, omega_f=0.0, gravity=False)
    S.solve()
    ur, ut, _ = S.surface_field(th)
    assert max(np.abs(ur - urx).max(), np.abs(ut - utx).max()) / sc < 1e-11


def test_fluid_core_sphere_vs_astyanax():
    """True fluid outer core with slip at ICB/CMB: PREM at M2 vs astyanax computelove."""
    from earthtides.solver import TidalSolver
    md = EarthModel(DATA / 'PREM.dat')
    r = TidalSolver(md, m=0, K=2, N=24, omega_f=1.4053e-4).solve()
    assert abs(r['h'] - 0.6061405951) < 1e-5
    assert abs(r['l'] - 0.16800390091 / 2) < 5e-6
    assert abs(r['kk'] - 0.59876288861 / 2) < 1e-5


def test_free_core_nutation_emerges():
    """Diurnal band with fluid core and rotation: FCN resonance near 1 + 1/458 cpsd (DDW1999)."""
    from earthtides.solver import TidalSolver
    from earthtides.iers import iers_fit
    md = EarthModel(DATA / 'PREM.dat')
    g = ClairautGeometry(md)

    def h0(fr):
        S = TidalSolver(md, g, m=1, K=3, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=fr * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True)
        S.solve()
        return iers_fit(S, g)['h0'].real
    # sign change of the resonant part between 1 + 1/450 and 1 + 1/465
    assert h0(1 + 1 / 465) < 0.3 and h0(1 + 1 / 450) > 1.0
    assert abs(h0(0.92700) - 0.59553) < 2e-3          # O1, DDW1999 Table 11
    assert abs(h0(1.00273) - 0.93394) < 5e-3          # psi1


def test_diurnal_no_tilt_over_pole():
    """With the Poisson-consistent reference potential the rigid tilt is force-free:
    h(2) has no pole at K1 and matches DDW1999 at O1 and P1."""
    from earthtides.solver import TidalSolver
    from earthtides.iers import iers_fit
    md = EarthModel(DATA / 'PREM.dat')
    g = ClairautGeometry(md)

    def fit(fr):
        S = TidalSolver(md, g, m=1, K=3, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=fr * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True,
                        dyn_fluid=True, consistent_reference=True, frame='mantle')
        S.solve()
        return iers_fit(S, g)
    o1, p1, k1m = fit(0.92700), fit(0.99454), fit(0.99900)
    assert abs(o1['h2'].real + 0.00048) < 4e-5                 # DDW O1 -0.00048
    assert abs(p1['h2'].real + 0.00053) < 1e-4                 # DDW P1 -0.00053
    assert abs(k1m['h2'].real - o1['h2'].real) < 4e-4          # no 1/(omega - Omega) pole
    assert abs(o1['l0'].real - 0.08298) < 2e-4 and abs(o1['l2'].real - 0.00008) < 2e-5


def test_diurnal_k2_tilt_correction():
    """k2 with the tilt potential of the flattened Earth removed: DDW1999 Table 12."""
    from earthtides.solver import TidalSolver
    md = EarthModel(DATA / 'PREM.dat')
    g = ClairautGeometry(md)
    conv = (1 + g.flattening_surface / 3) ** -5
    for fr, ref in ((0.92700, 0.29335), (1.00546, 0.32308)):
        S = TidalSolver(md, g, m=1, K=3, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=fr * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True,
                        dyn_fluid=True, consistent_reference=True, frame='mantle')
        assert abs(S.solve()['kk'].real * conv - ref) < 1e-3


def _example(name, script):
    import importlib.util
    path = DATA.parent / 'examples' / name / script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_greff_lefftz_2005_analytical():
    """Homogeneous incompressible self-gravitating flattened Earth (Greff-Lefftz et al. 2005);
    examples/ex4."""
    cg = _example('ex4', 'greff2005.py')
    md = cg.make(100000)
    A_ = cg.compare(md, 0.0)
    for k, v in dict(hn=0.603140, hp=-0.000332, ln=0.181544, lp=-0.000064, ls=-0.000224,
                     kn=0.363824, kp=-0.000633).items():
        assert abs(A_[k] - v) < 1e-5, (k, A_[k], v)


def test_hough_fcn_rigid_mantle():
    """Free core nutation of a Maclaurin spheroid with a rigid mantle and a fluid
    core (Hough 1895): sigma = Omega [1 + (A/A_m) e_f] to first order in e_f; examples/ex5."""
    ch = _example('ex5', 'hough1895.py')
    md = ch.make()
    fl = ch.maclaurin_flattening(OMEGA_EARTH)
    p = ch.prediction(fl)
    roots = ch.fcn(md, fl, OMEGA_EARTH, K=4, N=16, lo=0.8, hi=1.1, n=4)
    assert len(roots) == 1
    assert abs((roots[0] - p) / (p - 1)) < 1e-3


def test_swing_example():
    """Pumped swing (examples/ex1): Galerkin converges to the integrated periodic solution."""
    sw = _example('ex1', 'swing.py')
    ex, _ = sw.galerkin(30)
    assert abs(ex.sum() - sw.periodic_solution()) < 1e-9
    assert abs(sw.galerkin(2)[0][0] - ex[0]) < 2e-7


def test_drum_example():
    """Loaded elliptical drum (examples/ex2): Chebyshev collocation in s + Galerkin in theta.
    Uniform load exact with one harmonic; quadrupole load converges exponentially (f = 0.1)."""
    dr = _example('ex2', 'drum.py')
    assert dr.max_error('uniform', 0.1, *dr.solve('uniform', 0.1, 1, 24)) < 1e-12
    assert dr.max_error('quadrupole', 0.1, *dr.solve('quadrupole', 0.1, 8, 24)) < 1e-8
    assert dr.max_error('quadrupole', 1 / 300, *dr.solve('quadrupole', 1 / 300, 6, 24)) < 1e-12


def test_homogeneous_spheres_example():
    """Bos & Scherneck (2013) model b (exact Bessel solution, table 1.3) at degrees 2 and 100, and
    model a (fluid core) vs astyanax computelove with exact gravity (examples/ex3)."""
    ex = _example('ex3', 'homogeneous.py')
    mb = ex.write_model('modelb', [(6371.0, 5517.0, 146e9, 347e9)])
    for n, N in ((2, 24), (100, 96)):
        v = ex.love(mb, n, ex.W_M2, N)
        v = (v[0], n * v[1], n * v[2], v[3], n * v[4], n * v[5])
        assert max(abs(x - y) for x, y in zip(v, ex.BS_B[n])) < 6e-6
    ma = ex.write_model('modela', [(6371.0, 11020.0, 0.0, 950e9), (6371.0 * 0.45, 4460.0, 174e9, 231e9)])
    v = ex.love(ma, 2, 2 * np.pi / (12 * 3600))
    v = (v[0], 2 * v[1], 2 * v[2], v[3], 2 * v[4], 2 * v[5])
    assert max(abs(x - y) for x, y in zip(v, ex.AST_A['12h'])) < 1e-8


def test_load_love_sphere_vs_astyanax():
    """Load Love numbers (forcing='load') of spherical PREM at M2 vs astyanax computelove
    (columns h', n l', n k'). Static runs with a fluid core differ (Longman ambiguity)."""
    from earthtides.solver import TidalSolver
    md = EarthModel(DATA / 'PREM.dat')
    ref = {2: (-0.99435692054, 0.046520554685, -0.61351783915),
           10: (-1.4228458699, 0.28308371756, -0.68952213202),
           100: (-2.9649801864, 0.89338037042, -1.431139683)}
    for n, (h, nl, nk) in ref.items():
        S = TidalSolver(md, m=n, K=1, N=24 if n < 20 else 60, omega_f=1.4053e-4,
                        n_force=n, forcing='load')
        S.solve()
        L_ = S.load_love()
        assert abs(L_['h'][n] - h) < 1e-5
        assert abs(L_['l'][n] - nl / n) < 1e-5
        assert abs(L_['k'][n] - nk / n) < 1e-5


def test_deg_window_equals_full():
    """deg_window = 2 (degrees n' +- 4) reproduces the full degree set for a load
    on the flattened Earth (coupling to n' +- 6 is ~1e-9)."""
    from earthtides.solver import TidalSolver
    md = EarthModel(DATA / 'PREM.dat')
    g = ClairautGeometry(md)
    kw = dict(m=2, N=24, omega_f=1.4053e-4, omega_rot=OMEGA_EARTH, n_force=10, forcing='load')
    A = TidalSolver(md, g, K=7, **kw)
    B = TidalSolver(md, g, deg_window=2, **kw)
    A.solve(); B.solve()
    a, b = A.load_love(), B.load_love()
    for n in (8, 10, 12):
        assert abs(a['h'][n] - b['h'][n]) < 1e-9
        assert abs(a['k'][n] - b['k'][n]) < 1e-9


def test_loadgreen_sphere_is_farrell():
    """LoadGreen on the sphere (diagonal Love numbers, a^2 area element) reproduces the
    Legendre sums (G / a g) sum h'_n P_n(cos psi) and the horizontal l'_n dP_n/dpsi."""
    from numpy.polynomial import legendre as Lg
    from earthtides.loadgreen import LoadGreen
    from earthtides.model import G
    lg = LoadGreen(DATA.parent / 'results' / 'load_love_table_PREM.npz', DATA / 'PREM.dat', nmax=12)
    ths = np.radians(45.0)
    thl, dl = np.radians([30., 60., 100.]), np.radians([10., -40., 120.])
    up, no, ea = lg.green(ths, thl, dl, model='sph0')
    cpsi = np.cos(ths) * np.cos(thl) + np.sin(ths) * np.sin(thl) * np.cos(dl)
    h, l = lg.sph[0]
    ch, cl = np.zeros(13), np.zeros(13)
    ch[2:], cl[2:] = h[2:13], l[2:13]
    assert np.allclose(up, G / (lg.a * lg.g) * Lg.legval(cpsi, ch), rtol=1e-10)
    hor = G / (lg.a * lg.g) * np.abs(Lg.legval(cpsi, Lg.legder(cl)) * np.sqrt(1 - cpsi**2))
    assert np.allclose(np.hypot(no, ea), hor, rtol=1e-10)


def test_synthesis_catalog():
    """HARDISP catalog and Cartwright-Tayler functions (Sect. 6): 342 lines, M2 equilibrium
    amplitude at the equator, BLQ text in the fixed HARDISP format (6 rows of 11 fields)."""
    from earthtides.synthesis import catalog, ct, blq, doodson_freq
    idd, tamp = catalog()
    assert idd.shape == (342, 6) and abs(tamp[0] - 0.632208) < 1e-9
    assert abs(tamp[0] * ct(2, np.pi / 2) - 0.2442) < 1e-4
    assert abs(doodson_freq(idd[:1])[0] - 1.9322736) < 1e-6
    rows = blq(*(0.1 * np.ones(11, complex) for _ in range(3))).splitlines()
    assert len(rows) == 6 and all(len(r) == 1 + 77 for r in rows)


def test_etide_vs_hardisp_catalog():
    """etide (Tamura 1987, Sect. 6) and the HARDISP-catalogue synthesis give the same degree-2 body
    tide for the same nominal Love numbers to the truncation of the smaller catalogue (0.3 mm),
    apart from the permanent tide, which only Tamura contains."""
    from earthtides import etide as E
    from earthtides.synthesis import catalog, synthesize, body_admittance
    mjd = 61041.0 + np.arange(0, 10, 1 / 24)
    lat, lon = 57.395, 11.926
    cat = E.Tamura()
    L = E.LoveModel(cat, 'iers')
    d2 = cat.n == 2
    L.h0[d2], L.l0[d2], L.l1[:], L.h2[:], L.l2[:] = 0.6078, 0.0847, 0, 0, 0
    ur, un, ue = E.displacement(mjd, lat, lon, 0.0, L, degrees=(2,), cat=cat)
    phi, _ = E.geocentric(lat)
    idd, _ = catalog()
    Z = np.zeros((len(idd), 3), complex)
    for m in (0, 1, 2):
        Z[idd[:, 0] == m] = np.array(body_admittance(np.pi / 2 - phi, np.radians(lon), m, 0.6078, 0.0847))
    u = synthesize(mjd + 2400000.5, Z, tt_utc=69.184)
    d = np.column_stack([ur, -un, -ue]) - u
    d -= d.mean(0)
    assert np.abs(d).max() < 4e-4


@pytest.mark.parametrize('m,fr', [(1, 0.92700), (2, 1.9323)])
def test_rotating_spheroid_series(m, fr):
    """Flattening and Coriolis parts of w+ (Dehant et al. 1999, eq. 20) on a homogeneous
    non-gravitating spheroid: solver vs the polynomial series of exact_rotating.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('crs', DATA.parent / 'scripts' / 'check_rotating_spheroid.py')
    crs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crs)
    md = EarthModel(DATA / 'HOMOG.dat')
    P, Q, res = crs.compare(md, 1 / 30, 'x', m, fr, True)
    assert res < 1e-11
    for k, r in (('wp', 'l0'), ('lp', 'l0'), ('hp', 'h0'), ('l0', 'h0')):
        assert abs(Q[k] / Q[r] / (P[k] / P[r]) - 1) < 1e-6


def test_smith1974_layered_diurnal():
    """Smith's (1974) equations for a smooth, flattened, rotating, non-gravitating body agree with the
    solver for O1 to first order in the flattening (w+, the degree-3 toroidal term, included)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('cs', DATA.parent / 'scripts' / 'check_smith1974.py')
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    md = EarthModel(DATA / 'SMOOTH.dat')
    P = cs.smith_terms(md, 1 / 3000, 1, 0.927, True)
    Q = cs.solver_terms(md, 1 / 3000, 1, 0.927, True)
    for k, r, tol in (('wp', 'l0', 1e-3), ('lp', 'l0', 3e-3), ('hp', 'h0', 3e-3), ('l0', 'h0', 1e-5)):
        assert abs(P[k] / P[r] / (Q[k] / Q[r]) - 1) < tol


def test_smith1974_selfgravitating_diurnal():
    """Smith's (1974) equations with self-gravitation and the hydrostatic initial state agree with the
    solver for O1 (w+ to first order in the flattening); reference rotation 0.3 x Earth's, eps = 3.9e-4."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('cs', DATA.parent / 'scripts' / 'check_smith1974.py')
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    P, Q, eps = cs.grav_terms(EarthModel(DATA / 'HOMOG.dat'), 0.3, 1, 0.927, True)
    for k, r, tol in (('wp', 'l0', 1e-3), ('lp', 'l0', 1e-2), ('hp', 'h0', 1e-2), ('l0', 'h0', 3e-4)):
        assert abs(P[k] / P[r] / (Q[k] / Q[r]) - 1) < tol


def test_smith1974_twolayer_diurnal():
    """Smith's (1974) equations for a self-gravitating body of two welded solid layers with a density jump
    and density gradients agree with the solver for O1 to first order (reference rotation 0.3 x Earth's);
    needs the corrected sign of the radial rho_2 term of (5.26) (l+ is 7% off with the printed sign)."""
    import importlib.util
    from earthtides.geometry import ClairautGeometry
    spec = importlib.util.spec_from_file_location('cs', DATA.parent / 'scripts' / 'check_smith1974.py')
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    md = EarthModel(DATA / 'TWOLAYER.dat')
    P, Q, eps = cs.layered_terms(md, ClairautGeometry(md), 0.3, 1, 0.927, True)
    for k, r, tol in (('wp', 'l0', 1e-3), ('lp', 'l0', 5e-3), ('hp', 'h0', 2e-3), ('l0', 'h0', 1e-4)):
        assert abs(P[k] / P[r] / (Q[k] / Q[r]) - 1) < tol


def test_nonhydro_operator_gauge():
    """The non-hydrostatic correction C(u) vanishes for tau = -q I (pressure moved into tau): the
    assembled diurnal matrix (PREM, Clairaut, rotating, fluid core) is unchanged to round-off, while
    a deviatoric tau changes it."""
    from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
    from earthtides.jets import J as Jet
    from earthtides.solver import TidalSolver
    md = EarthModel(DATA / 'PREM.dat')
    a, Q0 = md.a, 3e8

    class Tau:
        def __init__(self, iso):
            self.iso = iso

        def drho(self, *args):
            return None

        def tau(self, i, s2, th, mp):
            r, c, sn = mp['R'], np.cos(th), np.sin(th)
            f, f_t, f_tt = 1 + 0.15 * (3 * c**2 - 1), -0.9 * c * sn, -0.9 * (c**2 - sn**2)
            x = r / a
            q, q_r, q_t = Q0 * x**2 * f, 2 * Q0 * x / a * f, Q0 * x**2 * f_t
            q_rr, q_rt, q_tt = 2 * Q0 / a**2 * f, 2 * Q0 * x / a * f_t, Q0 * x**2 * f_tt
            z = Jet(0 * r)
            if not self.iso:
                return dict(rr=Jet(q, q_r, q_t), tt=z, pp=z, rt=z, br=Jet(q_r, q_rr, q_rt), bt=z)
            iso = Jet(-q, -q_r, -q_t)
            return dict(rr=iso, tt=iso, pp=iso, rt=z, br=Jet(-q_r, -q_rr, -q_rt),
                        bt=Jet(-q_t / r, -(q_rt / r - q_t / r**2), -q_tt / r))

    def H(nh):
        S = TidalSolver(md, ClairautGeometry(md), m=1, N=16, K=3, coriolis=OMEGA_EARTH,
                        omega_t=0.93 * OMEGA_EARTH, consistent_reference=True, omega_rot=OMEGA_EARTH,
                        dyn_fluid=True, frame='mantle', nonhydro=nh)
        S.solve()
        H = S.H.toarray() if hasattr(S.H, 'toarray') else S.H
        nb = S.nfld * S.N               # interior collocation rows only (the boundary rows carry
        rows = [i * nb + k for i in range(md.n_layers)        # the traction P1(tau).n, which is not
                for k in range(S.nfld, nb - S.nfld)]          # a gauge where q != 0)
        return H[rows]

    H0, Hg, Hd = H(None), H(Tau(True)), H(Tau(False))
    scale = np.abs(H0).max(axis=1)
    assert (np.abs(Hg - H0).max(axis=1) / scale).max() < 1e-12
    assert (np.abs(Hd - H0).max(axis=1) / scale).max() > 1e-6


def test_nonhydro_zero_load_is_hydrostatic():
    """The full non-hydrostatic machinery (FlatMap geometry, extra density in props and refpot, tau,
    slip terms) with zero load reproduces the hydrostatic diurnal run exactly."""
    from earthtides.nonhydro import FlatSupport, FlatState, FlatMap
    from earthtides.solver import TidalSolver
    md = EarthModel(DATA / 'PREM.dat')
    base = ClairautGeometry(md)
    S = FlatSupport(md, EarthModel(DATA / 'PREMs.dat'), [(3480e3, 5701e3), (5701e3, 6346.6e3)], base, OMEGA_EARTH)
    st = FlatState(S, [0.0, 0.0], base, OMEGA_EARTH)
    out = []
    for geo, nh in ((base, None), (FlatMap(base, st), st)):
        T = TidalSolver(md, geo, m=1, K=3, N=16, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=0.927 * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True,
                        dyn_fluid=True, consistent_reference=True, frame='mantle', nonhydro=nh)
        out.append(T.solve())
    for k in ('h', 'l', 'kk'):
        assert out[0][k] == out[1][k]


def test_flat_support_three_targets():
    """Three load shells fitted to J2, CMB and surface on the supported state on the level figure (Table S9 of
    the PEPI paper): the exact fit reaches the targets; the regularised fit (prior 0.5 %) keeps the loads below
    0.2 % and reaches J2, CMB 443 m and surface 297 m."""
    from earthtides.geometry import figure
    from earthtides.nonhydro import FlatSupport
    md, mds = EarthModel(DATA / 'PREM.dat'), EarthModel(DATA / 'PREMs.dat')
    shells = [(3480e3, 5701e3), (5701e3, 5971e3), (5971e3, 6346.6e3)]
    S = FlatSupport(md, mds, shells, figure(md, 'level'), OMEGA_EARTH)
    target = {'J2': 1.14e-5, 'cmb': 450.0, 'surface': 100.0}
    r = S.response(S.fit(target))
    assert abs(r['J2'] - 1.14e-5) < 1e-9 and abs(r['cmb'] - 450.0) < 0.5 and abs(r['surface'] - 100.0) < 0.5
    a = S.fit(target, {'J2': 0.1e-6, 'cmb': 50.0, 'surface': 250.0}, 5e-3)
    r = S.response(a)
    assert max(abs(x) for x in a) < 2e-3
    assert abs(r['J2'] - 1.14e-5) < 1e-8 and abs(r['cmb'] - 443.0) < 1.0 and abs(r['surface'] - 297.0) < 2.0

def test_love_setup_file_nonhydrostatic(tmp_path):
    """A TOML set-up file with a [nonhydrostatic] table: the fit reaches its targets, the table
    records them, and the result equals a direct TidalSolver run on the same state."""
    from earthtides import love
    f = tmp_path / 'run.toml'
    f.write_text('model = "PREM"\nanelastic = false\n[nonhydrostatic]\n'
                 'targets = {J2 = 1.14e-5, cmb = 450.0}\n[run]\nfreq = [0.92954]\norder = 1\n')
    opts, run = love.read_setup(f)
    assert run['freq'] == [0.92954] and opts.nonhydrostatic['targets']['cmb'] == 450.0
    out = tmp_path / 'love.json'
    love.main(['--setup', str(f), '--jobs', '1', '-o', str(out)])
    t = love.read(out)
    meta = t['metadata']['nonhydrostatic']['response']
    assert abs(meta['cmb'] - 450.0) < 1e-6 and abs(meta['J2'] / 1.14e-5 - 1) < 1e-9
    S, r = love.solve_wave(opts, 1, 0.92954)
    assert S.nonhydro is love._model(opts)[2].state
    p = love.parameters(S, r, opts)
    assert abs(t['entries'][0]['h0'] - p['h0']) < 1e-10      # round-off differs between platforms (1.3e-12 on Ubuntu, Python 3.14)
    with pytest.raises(ValueError):
        love.Options(nonhydrostatic={'targets': {'J2': 1e-5}}, flattening=False)


def test_nonhydro_flat_state_consistent():
    """Non-hydrostatic final state supported on the flattened rotating background (nonhydro.FlatState):
    the angular-momentum balance dH/dt + Omega x H = N holds as well as for the hydrostatic model
    (O1, psi1), and the FCN moves as (A/A_m)(e_f - beta) predicts for 450 m of excess CMB flattening."""
    import importlib.util
    from earthtides.nonhydro import FlatSupport, FlatState, FlatMap
    from earthtides.solver import TidalSolver
    spec = importlib.util.spec_from_file_location('cam', DATA.parent / 'scripts' / 'check_angular_momentum.py')
    cam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cam)
    md = EarthModel(DATA / 'PREM.dat')
    base = ClairautGeometry(md)
    S = FlatSupport(md, EarthModel(DATA / 'PREMs.dat'), [(3480e3, 5701e3), (5701e3, 6346.6e3)], base,
                    OMEGA_EARTH)
    st = FlatState(S, S.fit({'J2': 1.14e-5, 'cmb': 450.0}), base, OMEGA_EARTH)
    geo = FlatMap(base, st)

    def run(sig):
        T = TidalSolver(md, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True,
                        dyn_fluid=True, consistent_reference=True, frame='mantle', nonhydro=st)
        T.solve()
        return T
    for sig in (0.927, 1.00546):
        T = run(sig)
        Hc, Nc, lhs = cam.balance(T, sig * OMEGA_EARTH)
        assert np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc) < 2e-3
    x = np.array([1.0012, 1.0016, 1.0030, 1.0036])
    k = np.array([run(s).results()['kk'] for s in x]).real
    c = np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0]
    assert abs(1 / (c - 1) - 431.6) < 2.0


def test_nonhydro_viscous_support():
    """Viscous end member (mantle mu = s eta_rel, map interpolated between anchors): the
    angular-momentum balance holds (O1, psi1) and the FCN follows (A/A_m)(e_f - beta)."""
    import importlib.util
    from earthtides.nonhydro import NonHydrostatic
    from earthtides.solver import TidalSolver
    spec = importlib.util.spec_from_file_location('cam', DATA.parent / 'scripts' / 'check_angular_momentum.py')
    cam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cam)
    md = EarthModel(DATA / 'PREM.dat')
    base = ClairautGeometry(md)
    nh = NonHydrostatic(md, dict(support='viscous', targets={'J2': 1.14e-5, 'cmb': 450.0}), base,
                        OMEGA_EARTH, EarthModel(DATA / 'PREMs.dat'))
    assert nh.state.anchors[0] == 3480e3 and len(nh.state.anchors) == 5

    def run(sig):
        T = TidalSolver(md, nh.geometry, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True,
                        dyn_fluid=True, consistent_reference=True, frame='mantle', nonhydro=nh.state)
        T.solve()
        return T
    for sig in (0.927, 1.00546):
        T = run(sig)
        Hc, Nc, lhs = cam.balance(T, sig * OMEGA_EARTH)
        assert np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc) < 3e-3
    x = np.array([1.0012, 1.0016, 1.0030, 1.0036])
    k = np.array([run(s).results()['kk'] for s in x]).real
    c = np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0]
    assert abs(1 / (c - 1) - 430.3) < 2.0


