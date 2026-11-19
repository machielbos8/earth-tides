# -*- coding: utf-8 -*-
#
# Numbers quoted in the text of P1 v2 (PEPI: main_v2.tex and supplement_v2.tex) that no other script prints.
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
# 08/11/2026 Machiel Bos
#===============================================================================
"""Numbers in the text of P1 v2 that are not printed by another script (docs/paper_v2_numbers_inventory.md
lists them all and where the others come from). Each line: location in the paper, quantity, the value
computed here, the value as printed.

Sections (run all, or name some on the command line):
  analytic   one-line formulas: Coriolis/inertia ratio, cot(alpha pi/2) vs 2/(pi alpha), Ekman thickness and
             viscosity, Ekman overshoot, centrifugal reduction of gravity, internal shear-layer widths
  gravity    S1: rho g a / mu of the Earth, the radius where it is 1, and how much self-gravitation
             reduces h of a homogeneous body (static, spherical)
  angmom     S5 (verification): remainder of the angular-momentum balance in units of f, for the model
             spun at 0.7, 1 and 1.3 times the Earth's rate, and with half the hydrostatic flattening
  l1mm       S9: the semidiurnal and diurnal l(1) of the IERS Conventions minus ours, as horizontal
             displacement of M2 and K1 (Tamura catalog, TN36 eqs 7.12-7.13 as in etide)
  voigt      S3: h2 of the Voigt average of transversely isotropic PREM (data/PREMtransS.dat) minus that of
             isotropic PREM (data/PREMs.dat), spherical, M2 (the solver's treatment of a TI model)
  rock       S1: ocean replaced by rock of crustal density: change of h2 (sphere) and of the latitude
             parameters h(2), l(2), l(1) at O1 and M2 (flattened, rotating, viscous core delta 1 km, core K 24)
  poisson    S12: jumps of the reference potential phi0 and of its normal flux at every interface (and the
             exterior match at the surface), hydrostatic PREM and the elastically supported 450 m state
  rosat      5.1: FCN resonance strength of the gravimetric factor (residue from
             scripts/gravimetric_factor_fcn430.py, stored) in deg/h against Rosat et al. (2009, J. Geodyn.
             48, 331: a_R = 0.667e-3 deg/h from superconducting gravimeters)
  refpot     3.4: first-order level-surface gravity (eq. W) against the gravity of the model's own density
             plus the exact centrifugal force (refpot.ReferencePotential), hydrostatic PREM
Usage: python scripts/paper_numbers_v2.py [section ...]  ->  results/paper_numbers_v2.txt (all sections)
"""
import copy
import sys
import time
import warnings

import numpy as np
from numpy.polynomial import Chebyshev

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
from earthtides.model import EarthModel, G
from earthtides.geometry import OMEGA_EARTH
from earthtides.solver import TidalSolver

RC = 3480.0e3                                       # core radius (m)
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def item(where, what, ours, paper):
    out(f'{where:12s} {what:64s} {ours:>24s}   paper: {paper}')


# ------------------------------------------------------------------------------ analytic
def sec_analytic():
    Om = OMEGA_EARTH
    for name, f in (('O1', 0.92700), ('M2', 1.93227)):            # cycles per sidereal day
        item('S2.x', f'Coriolis / inertial force 2 Omega / omega, {name}', f'{2 / f:.2f}',
             'about twice (diurnal), as large (semidiurnal)')
    a = 0.15
    item('3.5', 'small-alpha 2/(pi alpha) over cot(alpha pi/2), alpha 0.15',
         f'{2 / (np.pi * a) / (1 / np.tan(a * np.pi / 2)) - 1:+.1%}', '2 per cent larger')
    d = lambda nu: np.sqrt(nu / Om)
    item('5.3', 'Ekman thickness for nu = 1e-6 m2/s (molecular, Alfe 2000)', f'{d(1e-6):.3f} m', 'about 0.1 m')
    item('5.3', 'Ekman thickness for nu = 10-30 m2/s (Koot 2010, ICB)', f'{d(10) / 1e3:.2f}-{d(30) / 1e3:.2f} km',
         '0.4-0.6 km')
    item('5.3', 'Ekman thickness for nu = 0.02-0.04 m2/s (Koot 2010, CMB)', f'{d(0.02):.0f}-{d(0.04):.0f} m',
         'about 20 m')
    item('5.3', 'Ekman thickness for magnetic diffusivity 2 m2/s', f'{d(2.0):.0f} m', 'about 170 m')
    item('5.3', 'kinematic viscosity Omega delta^2 for delta = 0.25-4 km',
         f'{Om * 250**2:.1f}-{Om * 4000**2:.0f} m2/s', '5 to 1200 m2/s')
    item('5.3', 'kinematic viscosity Omega delta^2 for delta = 1 km', f'{Om * 1000**2:.1f} m2/s', '70 m2/s')
    z = 3 * np.pi / 4                                               # max of 1 - exp(-z) cos z
    item('5.3', 'Ekman spiral: overshoot of the real part, max 1 - e^-z cos z',
         f'{-np.exp(-z) * np.cos(z):.1%}', 'up to 7 per cent')
    md = EarthModel('data/PREM.dat')
    g = G * md.M / md.a**2
    item('S8', 'centrifugal reduction of surface gravity 2/3 Omega^2 a / g', f'{2 / 3 * Om**2 * md.a / g:.2e}',
         '2.3e-3')
    for dl, paper in ((1.0e3, '190 to 720'), (0.25e3, '370 to 1800'), (20.0, '1300 to 10000')):
        E = (dl / RC)**2
        w4, w3 = E**0.25 * RC, E**(1 / 3) * RC
        item('5.3', f'beam width E^1/4..E^1/3 r_c and degree pi r_c / width, delta {dl:g} m',
             f'{w4 / 1e3:.0f}-{w3 / 1e3:.0f} km, {np.pi * RC / w4:.0f}-{np.pi * RC / w3:.0f}', paper)
    E = (2.0e3 / RC)**2
    w4, w3 = E**0.25 * RC / 1e3, E**(1 / 3) * RC / 1e3
    item('S11', 'shear layers E^1/4, E^1/3 of the core radius, delta = 2 km', f'{w4:.0f}-{w3:.0f} km', '80 to 25 km')
    item('S11', 'degree that resolves them, pi r_c / width', f'{np.pi * RC / 1e3 / w4:.0f}-{np.pi * RC / 1e3 / w3:.0f}',
         'of order 100 to 450')


# ------------------------------------------------------------------------------ gravity
def mantle_means(md):
    """Volume averages of mu and lambda over the PREM mantle (CMB to Moho, 24.4 km depth)."""
    v = mu = lam = 0.0
    for i in range(md.n_layers):
        r0, r1 = md.r0[i], md.r1[i]
        if r0 < RC - 1 or r1 > md.a - 24.0e3 + 1:
            continue
        r = np.linspace(r0, r1, 2001)
        w = 4 * np.pi * r**2
        v += np.trapezoid(w, r)
        mu += np.trapezoid(w * md.mu[i](r).real, r)
        lam += np.trapezoid(w * md.lam[i](r).real, r)
    return mu / v, lam / v


def homogeneous(rho, mu, lam):
    """data/HOMOG.dat with constant rho, mu, lambda (mass polynomial recomputed)."""
    md = copy.deepcopy(EarthModel('data/HOMOG.dat'))
    dom = [md.r0[0], md.r1[0]]
    md.rho[0], md.mu[0], md.lam[0] = (Chebyshev([x], domain=dom) for x in (rho, mu, lam))
    r = Chebyshev.identity(domain=dom)
    md._mass_poly[0] = (4.0 * np.pi * md.rho[0] * r * r).integ(lbnd=dom[0])
    md.mass_below[0] = 0.0
    md.M = md._mass_poly[0](dom[1])
    md.g_surface = G * md.M / md.a**2
    return md


def h_static(md, gravity):
    S = TidalSolver(md, None, m=2, K=1, N=16, omega_f=0.0, gravity=gravity)
    return complex(S.solve()['h']).real


def sec_gravity():
    prem = EarthModel('data/PREM.dat')
    rho = prem.M / (4 / 3 * np.pi * prem.a**3)
    mu, lam = mantle_means(prem)
    g = G * prem.M / prem.a**2
    ratio = rho * g * prem.a / mu
    item('S1.1', f'rho g a / mu, mean density {rho:.0f}, mantle mu {mu / 1e9:.0f} GPa', f'{ratio:.2f}', 'about 2')
    item('S1.1', 'radius where rho g a / mu = 1 (uniform body, ratio ~ a^2)', f'{prem.a / np.sqrt(ratio) / 1e3:.0f} km',
         'about 4500 km')
    for target, paper in ((ratio, '14 per cent (Earth size)'), (0.3, 'less than 3 per cent below 0.3')):
        s = ratio / target                              # same rho g a / mu by stiffening mu and lambda together
        md = homogeneous(rho, mu * s, lam * s)
        hg, h0 = h_static(md, True), h_static(md, False)
        item('S1.1', f'self-gravitation reduces h, rho g a / mu = {target:.2f} (lambda/mu {lam / mu:.2f})',
             f'{1 - hg / h0:.1%}', paper)


# ------------------------------------------------------------------------------ refpot
def sec_refpot():
    from earthtides.geometry import ClairautGeometry, mapping, chain
    from earthtides.refpot import ReferencePotential
    md = EarthModel('data/PREM.dat')
    geo = ClairautGeometry(md)
    rp = ReferencePotential(md, geo, OMEGA_EARTH)
    th = np.linspace(0.01, np.pi - 0.01, 181)
    worst = surf = 0.0
    for i in range(md.n_layers):
        s = np.linspace(md.r0[i], md.r1[i], 41)[1:-1] if i == 0 else np.linspace(md.r0[i], md.r1[i], 41)
        s2 = s[:, None]
        m = mapping(s2, th, *geo.eps(i, s2))
        gs = md.gravity(i, s2)
        W1 = gs - 2.0 / 3.0 * OMEGA_EARTH**2 * s2
        W2 = 4 * np.pi * G * md.rho[i](s2) - 2 * gs / s2 - 2.0 / 3.0 * OMEGA_EARTH**2
        lev = chain(0.0, W1, W2, 0.0, 0.0, 0.0, m)
        con = rp.phys_derivs(i, s, th, m)
        R = m['R']
        d = np.hypot(lev['r'] - con['r'], (lev['t'] - con['t']) / R) / np.hypot(con['r'], con['t'] / R)
        worst = max(worst, float(np.nanmax(d)))
        if i == md.n_layers - 1:
            surf = float(np.nanmax(d[-1]))
    item('3.4', 'first-order level-surface gravity vs exact: max |diff| / g, whole Earth', f'{worst:.1e}',
         'about 1e-5 g (of order eps^2)')
    item('3.4', 'the same at the surface', f'{surf:.1e}', '')
    f = geo.flattening_surface if hasattr(geo, 'flattening_surface') else 1 / 298.26
    item('3.4', 'eps^2 at the surface for comparison', f'{f**2:.1e}', '')


# ------------------------------------------------------------------------------ angmom
def sec_angmom():
    from earthtides.geometry import ClairautGeometry
    from check_angular_momentum import balance
    md = EarthModel('data/PREM.dat')
    sig = 0.92700
    res = {}
    for spin, scale, label in ((0.7, 0.49, 'spin x 0.7'), (1.0, 1.0, 'spin x 1'), (1.3, 1.69, 'spin x 1.3'),
                               (1.0, 0.5, 'half the hydrostatic flattening, spin x 1')):
        geo = ClairautGeometry(md, scale=scale)
        Om = spin * OMEGA_EARTH
        w = sig * Om
        S = TidalSolver(md, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=Om, omega_t=w, omega_rot=Om,
                        wahr_frame=True, dyn_fluid=True, consistent_reference=True, frame='mantle')
        S.solve()
        Hc, Nc, lhs = balance(S, w)
        rem = np.linalg.norm(lhs - Nc) / np.linalg.norm(Nc)
        res[label] = rem
        f = scale * geo.flattening_surface              # flattening_surface is that of scale 1
        hyd = abs(scale - spin**2) < 1e-9
        item('S5', f'angular-momentum remainder / |N| at O1, {label}',
             f'{rem / f:.3f} f' if hyd else f'{rem:.0%} of N',
             ('0.25f; same within 2 % at 0.7, 1.3' if hyd else 'breaks the balance by 35 per cent'))


# ------------------------------------------------------------------------------ l1mm
def sec_l1mm():
    from earthtides import etide
    cat = etide.Tamura()
    phi = np.radians(np.linspace(-89.5, 89.5, 1791))
    sp, cp = np.sin(phi), np.cos(phi)
    # IERS l(1): 0.0024 semidiurnal, 0.0012 diurnal (TN36 Table 7.1 / eq. 7.13); ours from Table S3 (tab:ddw)
    for name, dood, m, l1_iers, l1_ours, paper in (('M2', (2, 0, 0, 0, 0, 0), 2, 0.0024, 0.00099, '0.43 mm'),
                                                    ('K1', (1, 1, 0, 0, 0, 0), 1, 0.0012, 0.00036, '0.35 mm')):
        k = np.flatnonzero((cat.n == 2) & np.all(cat.dood[:, :6] == dood, axis=1))
        A = np.abs(cat.A[k]).max()
        kappa = {1: 2 / 3, 2: 1 / 3}[m]
        Wn = etide.CV * etide.RE / (etide.GE / etide.RE**2) * kappa
        N1 = {1: -3 * sp**2, 2: -6 * sp * cp}[m]
        E1 = {1: -3 * sp * np.cos(2 * phi), 2: 6 * cp * sp**2}[m]
        d = (l1_iers - l1_ours) * A * Wn
        hv = d * np.hypot(N1, E1)
        item('S9', f'{name}: (l1 IERS - ours) = {l1_iers - l1_ours:.5f}; max horizontal (vector)',
             f'{1e3 * hv.max():.2f} mm', paper)
        item('S9', f'{name}: max north, max east component', f'{1e3 * np.abs(d * N1).max():.2f}, '
             f'{1e3 * np.abs(d * E1).max():.2f} mm', '')


def sec_latmm():
    """h(2) and l(2) of the IERS Conventions (-0.0006, 0.0002) minus ours (elastic, Table tab:ddw, as l1mm):
    largest radial and horizontal displacement over latitude, M2 and K1, shapes of TN36 eqs 7.? (MDG 1997
    eqs 3-4): radial 3 h(phi) sin cos (m=1) / 3 h(phi) cos^2 (m=2); horizontal from l(phi) = l0 + l2 P2."""
    from earthtides import etide
    cat = etide.Tamura()
    phi = np.radians(np.linspace(-89.5, 89.5, 1791))
    sp, cp = np.sin(phi), np.cos(phi)
    P2 = 0.5 * (3 * sp**2 - 1)
    for name, dood, m, h2o, l2o in (('M2', (2, 0, 0, 0, 0, 0), 2, -0.00049, 0.00009),
                                     ('K1', (1, 1, 0, 0, 0, 0), 1, -0.00061, 0.00010)):
        k = np.flatnonzero((cat.n == 2) & np.all(cat.dood[:, :6] == dood, axis=1))
        A = np.abs(cat.A[k]).max()
        kappa = {1: 2 / 3, 2: 1 / 3}[m]
        Wn = etide.CV * etide.RE / (etide.GE / etide.RE**2) * kappa
        R = {1: 3 * sp * cp, 2: 3 * cp**2}[m]                     # radial shape of h(phi)
        N = {1: 3 * np.cos(2 * phi), 2: -6 * sp * cp}[m]          # north shape of l(phi)
        E = {1: 3 * sp, 2: 6 * cp}[m]                             # east shape of l(phi)
        dh, dl = (-0.0006 - h2o) * A * Wn, (0.0002 - l2o) * A * Wn
        item('S3', f'{name}: h(2) IERS - ours = {-0.0006 - h2o:+.5f}; max radial',
             f'{1e3 * np.abs(dh * P2 * R).max():.3f} mm', '')
        item('S3', f'{name}: l(2) IERS - ours = {0.0002 - l2o:+.5f}; max horizontal (vector)',
             f'{1e3 * (np.abs(dl * P2) * np.hypot(N, E)).max():.3f} mm', '')


# ------------------------------------------------------------------------------ voigt
def h2_sphere(md, f_cpd=1.9322736):
    w = f_cpd * 2 * np.pi / 86400.0
    S = TidalSolver(md, m=0, N=24, n_force=2, degrees=([2], []), omega_f=w)
    return complex(S.solve()['h']).real


def sec_voigt():
    hv, hi = h2_sphere(EarthModel('data/PREMtransS.dat')), h2_sphere(EarthModel('data/PREMs.dat'))
    item('S3', 'h2, Voigt average of TI PREM minus isotropic PREM (sphere, M2)', f'{hv - hi:+.1e}', '6e-6')


# ------------------------------------------------------------------------------ rock
def latitude_terms(md, m, sig):
    from earthtides.geometry import ClairautGeometry
    from earthtides.solver import resolution_per_layer
    from earthtides.iers import iers_fit
    mv = md.viscous_core(sig * OMEGA_EARTH, delta=1.0e3)
    g = ClairautGeometry(mv)
    Kl, Nl = resolution_per_layer(mv, {'K': 6, 'N': 24, 'rules': [{'fluid': True, 'K': 24, 'N': 30}]})
    S = TidalSolver(mv, g, m=m, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle', band=12)
    S.solve()
    f = iers_fit(S, g)
    return np.array([complex(f[k]).real for k in ('h2', 'l2', 'l1')])


def sec_rock():
    import os
    src = open('data/PREM.dat').read()
    assert src.count('15.0    2.28400') == 1
    tmp = 'results/_PREM_rock.dat'
    open(tmp, 'w').write(src.replace('15.0    2.28400', '15.0    2.60000', 1))
    try:
        prem, rock = EarthModel('data/PREM.dat'), EarthModel(tmp)
        item('S1', 'ocean -> rock (2600 kg/m3, top 15 km layer): change of h2, sphere',
             f'{h2_sphere(rock) - h2_sphere(prem):+.1e}', '+5e-4')
        for name, m, sig in (('O1', 1, 0.92700), ('M2', 2, 1.93227)):
            d = latitude_terms(rock, m, sig) - latitude_terms(prem, m, sig)
            item('S1', f'ocean -> rock: change of h(2), l(2), l(1) at {name} (max)', f'{np.abs(d).max():.1e}',
                 'less than 1e-6')
    finally:
        os.remove(tmp)


# ------------------------------------------------------------------------------ poisson
def refpot_jumps(md, geo, drho=None, projected=False):
    """Largest jump of phi0 and of its normal flux across the interfaces and at the surface (exterior
    field), relative to the largest phi0 and flux: pointwise at 157 colatitudes, or projected on the kept
    degrees P0..P6 with the quadrature of refpot (what the solve imposes)."""
    from earthtides.refpot import ReferencePotential
    from earthtides.basis import cheb_basis, ybar
    rp = ReferencePotential(md, geo, OMEGA_EARTH, drho=drho)
    th = rp.th if projected else np.linspace(0.02, np.pi - 0.02, 157)
    Y = {n: ybar(n, 0, th, 2) for n in rp.deg}
    proj = (lambda q: np.array([np.sum(q * Y[n][0] * rp.w) for n in rp.deg])) if projected else (lambda q: q)

    def side(i, xx):
        s, mp, _ = rp.derivs(i, np.array([xx]))
        mp = rp._map(i, np.array([[s[0]]]), th[None, :])
        T, T1, _ = cheb_basis(np.array([xx]), rp.N)
        sc = 2.0 / (md.r1[i] - md.r0[i])
        val = fr = ft = 0.0
        for j, n in enumerate(rp.deg):
            c = rp.coef[i * rp.nb + j * rp.N: i * rp.nb + (j + 1) * rp.N]
            a0, a1 = float(c @ T[:, 0]), float(c @ T1[:, 0]) * sc
            # d/dr, d/dtheta at fixed r through the map: phi(s, theta) with s = S(r, theta)
            val = val + a0 * Y[n][0]
            fr = fr + a1 * Y[n][0] * mp['s_r'][0]
            ft = ft + a1 * Y[n][0] * mp['s_t'][0] + a0 * Y[n][1]
        R, Rt = mp['R'][0], mp['R_t'][0]
        h = np.hypot(R, Rt)
        return val, (fr * R + ft / R * (-Rt)) / h, R, Rt, h

    jv = jf = 0.0
    scale_v = scale_f = 0.0
    for i in range(md.n_layers - 1):
        v0, f0, *_ = side(i, 1.0)
        v1, f1, *_ = side(i + 1, -1.0)
        jv, jf = max(jv, np.abs(proj(v0 - v1)).max()), max(jf, np.abs(proj(f0 - f1)).max())
        scale_v, scale_f = max(scale_v, np.abs(proj(v0)).max()), max(scale_f, np.abs(proj(f0)).max())
    v, f, R, Rt, h = side(md.n_layers - 1, 1.0)
    a = md.a
    ext = sum(rp.coef[md.n_layers * rp.nb + j] * (a / R)**(n + 1) * Y[n][0] for j, n in enumerate(rp.deg))
    dext = sum(rp.coef[md.n_layers * rp.nb + j] * ((-(n + 1) / R) * (a / R)**(n + 1) * Y[n][0] * R
                                                    + (a / R)**(n + 1) * Y[n][1] / R * (-Rt)) / h
               for j, n in enumerate(rp.deg))
    jv, jf = max(jv, np.abs(proj(v - ext)).max()), max(jf, np.abs(proj(f - dext)).max())
    return jv / scale_v, jf / scale_f


def sec_poisson():
    from earthtides.geometry import ClairautGeometry
    from earthtides.nonhydro import NonHydrostatic
    md = EarthModel('data/PREM.dat')
    base = ClairautGeometry(md)
    nh = NonHydrostatic(md, dict(support='elastic', targets={'J2': 1.14e-5, 'cmb': 450.0}), base, OMEGA_EARTH,
                        EarthModel('data/PREMs.dat'))
    for label, geo, drho in (('hydrostatic PREM', base, None), ('450 m elastic state', nh.geometry, nh.state.drho)):
        for projected in (True, False):
            jv, jf = refpot_jumps(md, geo, drho, projected)
            item('S12', f'phi0 jumps, {label}, {"projected on P0-P6" if projected else "pointwise"}: phi0, flux',
                 f'{jv:.0e}, {jf:.0e}', '1e-13')


# ------------------------------------------------------------------------------ rosat
def sec_rosat():
    import re
    txt = open('results/gravimetric_factor_fcn430.txt').read()
    res = float(re.search(r'FCN residue ([+-][\d.]+e[+-]\d+)', txt).group(1))     # cycles per sidereal day
    deg_h = 360.0 / (86164.0905 / 3600.0)                    # 1 cpsd in degrees per (solar) hour
    a = abs(res) * deg_h
    item('5.1', 'FCN resonance strength of the gravimetric factor (deg/h)', f'{a:.3e} ({a / 0.667e-3 - 1:+.1%})',
         'agrees to 1 % with Rosat et al. 2009 (a_R 0.667e-3)')


# ------------------------------------------------------------------------------ nutation
def iau_term(path, F, D, Om):
    """In-phase coefficient (microarcsec) of the luni-solar term with arguments F, D, Om only (j = 0 block of
    IERS Conventions 2010 Table 5.3a/b)."""
    for ln in open(path):
        w = ln.split()
        if ln.startswith('j = 1'):
            break
        if len(w) == 17 and w[0].isdigit():
            arg = [int(x) for x in w[3:]]
            if arg[:5] == [0, 0, F, D, Om] and not any(arg[5:]):
                return float(w[1]) if 'tab5.3a' in path else float(w[2])
    raise ValueError(f'no term F {F} D {D} Om {Om} in {path}')


def sec_nutation():
    import re
    eps0 = np.radians(84381.406 / 3600)
    a = 6378.137e3
    removed = {m.group(1): float(m.group(2)) for m in
               re.finditer(r'^(O1|P1)\s+\S+\s+\S+\s+\S+\s+\S+\s+([\d.]+)', open('results/frame_size.txt').read(), re.M)}
    for tide, name, (F, D, Om) in (('O1', 'fortnightly (13.66 d)', (2, 0, 2)), ('P1', 'semi-annual (182.6 d)', (2, -2, 2))):
        dpsi = iau_term('data/iers/tab5.3a.txt', F, D, Om) * 1e-6
        deps = iau_term('data/iers/tab5.3b.txt', F, D, Om) * 1e-6
        amp = max(abs(deps + dpsi * np.sin(eps0)), abs(deps - dpsi * np.sin(eps0))) / 2     # arcsec
        m = np.radians(amp / 3600) * a
        item('3.3', f'{name} nutation (IAU 2000A), circular amplitude, at the equator', f'{amp:.3f}" = {m:.2f} m',
             f'{tide}: rotation removed {removed[tide] / 1e3:.2f} m (frame_size.txt)')


# ------------------------------------------------------------------------------ nhamp
def sec_nhamp():
    """Load amplitudes of the elastic non-hydrostatic states of Sect. 6 (excess J2 of Chambat et al. 2010): the
    largest |delta rho / rho0| = |a_k| (P2 = 1 at the poles), and its mean over a sphere (zero for P2)."""
    from earthtides import love
    J2 = 1.14e-5
    big = 0.0
    for name, tg in (('e_f of MHB (383 m)', {'J2': J2, 'e_f': 9.73e-5}), ('CMB 450 m', {'J2': J2, 'cmb': 450.0}),
                     ('CMB 500 m', {'J2': J2, 'cmb': 500.0}), ('FCN 431.4 sd (503 m)', {'J2': J2, 'e_f': 1.280e-4})):
        st = love._model(love.Options(nonhydrostatic=dict(targets=tg)))[2]
        a = np.array(st.summary()['amplitudes'])
        big = max(big, np.abs(a).max())
        item('6.2', f'load amplitudes lower, upper mantle, {name}', f'{a[0]:+.2e} {a[1]:+.2e}', '')
    x, w = np.polynomial.legendre.leggauss(16)
    item('6.2', 'largest anomaly |delta rho / rho0| of these states; mean of P2 over a sphere',
         f'{big:.2%}; {0.5 * np.sum(w * 0.5 * (3 * x ** 2 - 1)):.0e}', 'at most 0.14 per cent; average to zero on every sphere')


# ------------------------------------------------------------------------------ wall (2026-11-19)
def sec_wall():
    """Main 5.3, S5.5: gravity against the centrifugal acceleration at the core-mantle boundary, and the factor
    2 g / (Omega^2 r) of the slip term; S2: distance between the flattened surface and the mean radius."""
    import importlib.util
    import tempfile
    from pathlib import Path
    from earthtides.geometry import ClairautGeometry
    spec = importlib.util.spec_from_file_location('hough1895', 'examples/ex5/hough1895.py')
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    hough = H.make(vs_m=6.0, vp_c=1000.0, path=Path(tempfile.gettempdir()) / 'HOUGH_paper_numbers.dat',
                   rho_c=H.RHO, rho_m=H.RHO)
    prem = EarthModel('data/PREM.dat')
    for name, md, paper in (('PREM', prem, '577; factor about 1150'), ("Hough's Earth", hough, '289; factor about 580')):
        ic = [i for i in range(md.n_layers) if abs(float(md.mu[i](0.5 * (md.r0[i] + md.r1[i])))) < 1.0][-1]
        rc = md.r1[ic]
        ratio = float(md.gravity(ic, rc)) / (OMEGA_EARTH ** 2 * rc)
        item('5.3/S5.5', f'{name}: g / (Omega^2 r) at the CMB; factor 2 g / (Omega^2 r)', f'{ratio:.1f}; {2 * ratio:.0f}', paper)
    f = ClairautGeometry(prem).flattening_surface
    item('S2', 'flattened surface to the mean radius: 2 f a / 3 (pole), f a / 3 (equator)',
         f'{2 * f * prem.a / 3e3:.1f} / {f * prem.a / 3e3:.1f} km', '14 / 7 km')


SECTIONS = dict(wall=sec_wall, analytic=sec_analytic, gravity=sec_gravity, refpot=sec_refpot, angmom=sec_angmom, l1mm=sec_l1mm,
                voigt=sec_voigt, rock=sec_rock, poisson=sec_poisson, rosat=sec_rosat, latmm=sec_latmm, nutation=sec_nutation, nhamp=sec_nhamp)


def main():
    names = sys.argv[1:] or list(SECTIONS)
    out(f"{'where':12s} {'quantity':64s} {'this script':>24s}   value in the paper")
    for n in names:
        t = time.time()
        SECTIONS[n]()
        print(f'   [{n}: {time.time() - t:.0f} s]', flush=True)


if __name__ == '__main__':
    from provenance import Run
    if sys.argv[1:]:
        main()
    else:
        with Run('numbers in the text of P1 v2 not printed by other scripts', params=dict(sections=list(SECTIONS)),
                 inputs=['data/PREM.dat', 'data/HOMOG.dat', 'data/iers/tab5.3a.txt', 'data/iers/tab5.3b.txt', 'results/frame_size.txt']) as run_:
            main()
            run_.write_text('results/paper_numbers_v2.txt', '\n'.join(lines) + '\n')
