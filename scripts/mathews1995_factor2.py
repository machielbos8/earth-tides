# -*- coding: utf-8 -*-
#
# Review round 2, R1: the defining equation of Mathews et al. (1995a) and the factor of two in their l(1).
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
# 16/11/2026 Machiel Bos
#===============================================================================
"""Two checks.

(a) Symbolic: the displacement of eq. (6)-(7) of Mathews, Buffett & Shapiro (1995a, GRL 22, 579),
      u = (Phi/g) [ h(theta) Y r^ + l(theta) grad_1 Y + l(1) P1(cos theta) i r^ x grad_1 Y ],
    h(theta) = h0 + h2 P2(cos theta), l(theta) = l0 + l2 P2(cos theta), grad_1 the gradient on the unit sphere,
    Y = P_2^m(cos theta) exp(i chi), chi = tide argument + m lambda, real part taken, against the site
    displacement of the IERS Conventions (McCarthy 1996, Ch. 7, eqs 5a-c; Petit & Luzum 2010, eqs 7.12-7.13;
    Mathews et al. 1997, eqs 2-4) for m = 0, 1, 2. Per component the ratio of each Love-number term to the
    same term of the Conventions must be one constant (the normalization of the tidal amplitude).
(b) Numerical: l(1) of elastic PREM (level figure, slipping core K 4 N 24) with the free core nutation tuned
    to the 430 sidereal days of the 'modified PREM' of Mathews et al. (1995a) (excess dynamical ellipticity of
    the core, two load shells), at the lines of their Table 1, against their values (4 decimals). A factor of
    the definitions would be the same at every line; a difference of the computation changes near the FCN.

    .venv/bin/python scripts/mathews1995_factor2.py   ->  results/mathews1995_factor2.txt"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
warnings.filterwarnings('ignore')
import numpy as np
import sympy as sp

from earthtides import love

J2 = 1.14e-5
TARGET_FCN = 430.0                                   # sidereal days, Mathews et al. (1995a) Table 1 caption
# Table 1 of Mathews et al. (1995a): Doodson number, cycles per day, l(1)
TABLE1 = (('O1 145.555', 1, 0.92954, 0.0010), ('P1 163.555', 1, 0.99726, 0.0010),
          ('165.545', 1, 1.00259, 0.0009), ('K1 165.555', 1, 1.00274, 0.0009), ('165.565', 1, 1.00288, 0.0009),
          ('psi1 166.554', 1, 1.005475688, 0.0017), ('semidiurnal (M2)', 2, 1.93227, 0.0020))
SIG = (1.0012, 1.0016, 1.0030, 1.0036)
CATALOGUE = ROOT / 'src' / 'earthtides' / 'data' / 'love_PREM_allwaves.json'
ANEL = ROOT / 'results' / 'paper1_table_anelastic_level.txt'      # Table S5 data (make_table_anelastic.py)


def exact(m, f):
    """Frequency [cpd] of the Tamura (1987) line nearest to f (next to the FCN, 1e-7 cpd is 1e-4 in h0)."""
    import json
    e = json.loads(CATALOGUE.read_text())['entries']
    return min((x['f_cpd'] for x in e if x['n'] == 2 and x['m'] == m), key=lambda x: abs(x - f))


def pole_fit(f, y, period):
    """y = L0 + L1 / (f_FCN - f), f_FCN [cpd] from the FCN period [sidereal days]; returns L0, L1, residuals."""
    ff = love.F_TILT * (1 + 1 / period)
    X = np.column_stack([np.ones_like(f), 1 / (ff - f)])
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return c[0], c[1], X @ c - y, ff


def symbolic():
    """Ratios (eq. 6 term) / (Conventions term) per component and parameter, for m = 0, 1, 2."""
    phi, chi = sp.symbols('phi chi', real=True)       # latitude, tide argument + m lambda
    h0, h2, l0, l2, l1 = sp.symbols('h0 h2 l0 l2 l1', real=True)
    s, c = sp.sin(phi), sp.cos(phi)
    P2 = sp.Rational(3, 2) * s**2 - sp.Rational(1, 2)  # P2(cos theta), cos theta = sin phi
    hf, lf = h0 + h2 * P2, l0 + l2 * P2
    P = {0: P2, 1: 3 * s * c, 2: 3 * c**2}             # P_2^m(cos theta) up to constants (Ferrers sign absorbed)
    out = []
    for m in (0, 1, 2):
        Y = P[m] * sp.exp(sp.I * chi)
        dY_dphi = sp.diff(Y, phi)
        dY_dlam = sp.I * m * Y                          # d/d lambda of exp(i m lambda)
        # grad_1 Y = dY/dphi n^ + (1/cos phi) dY/dlambda e^ ;  r^ x n^ = -e^,  r^ x e^ = n^
        g_n, g_e = dY_dphi, dY_dlam / c
        t_n, t_e = g_e, -g_n                             # r^ x grad_1 Y
        u_r = sp.re(sp.expand(hf * Y))
        u_n = sp.re(sp.expand(lf * g_n + l1 * s * sp.I * t_n))
        u_e = sp.re(sp.expand(lf * g_e + l1 * s * sp.I * t_e))
        # Conventions (real tide: potential proportional to P_2^m cos chi; m = 1 written with sin(chi), which is
        # cos(chi - pi/2): compare at the same phase by shifting chi)
        if m == 0:                                       # 5a: r^ h(phi)(3/2 sin^2 - 1/2) cos chi; n^ 3 l(phi) s c cos chi;
            ref_r = hf * P2 * sp.cos(chi)                # e^ l(1) only through cos(phi) 3 l1 sin^2 phi sin chi (zero
            ref_n = 3 * lf * s * c * sp.cos(chi)         # in the zonal band: no lambda dependence, so eq. 6 gives
            ref_e = 3 * l1 * c * s**2 * sp.sin(chi)      # it only through the time phase; listed for completeness)
        elif m == 1:                                     # 5b with sin(chi) -> cos(chi) (phase shift pi/2)
            ref_r = hf * 3 * s * c * sp.cos(chi)
            ref_n = (3 * lf * sp.cos(2 * phi) - 3 * l1 * s**2) * sp.cos(chi)
            ref_e = -(3 * lf * s - 3 * l1 * s * sp.cos(2 * phi)) * sp.sin(chi)
        else:                                            # 5c
            ref_r = hf * 3 * c**2 * sp.cos(chi)
            ref_n = -6 * s * c * (lf + l1) * sp.cos(chi)
            ref_e = -6 * c * (lf + l1 * s**2) * sp.sin(chi)
        for comp, u, ref in (('r', u_r, ref_r), ('n', u_n, ref_n), ('e', u_e, ref_e)):
            for par in (h0, h2, l0, l2, l1):
                a = sp.simplify(sp.diff(u, par))
                b = sp.simplify(sp.diff(ref, par))
                if a == 0 and b == 0:
                    continue
                ratio = sp.simplify(a / b) if b != 0 else sp.oo
                out.append((m, comp, str(par), str(sp.simplify(a)), str(sp.simplify(b)), str(ratio)))
    return out


def options(e_f):
    nh = None if e_f is None else dict(targets={'J2': J2, 'e_f': e_f})
    return love.Options(anelastic=False, figure='level', nonhydrostatic=nh)


def k_at(a):
    e_f, s = a
    return love.solve_wave(options(e_f), 1, s * love.F_TILT)[1]['kk'].real


def fcn(pool, e_f):
    k = np.array(pool.map(k_at, [(e_f, s) for s in SIG]))
    x = np.array(SIG)
    c = np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0][0]
    return 1.0 / (c - 1.0)


def l1_at(a):
    e_f, m, f = a
    return complex(love._job((options(e_f), 2, m, f))['l1']).real


def main(run):
    lines = ['(a) eq. (6) of Mathews et al. (1995a) against the Conventions (1996 eqs 5a-c = 2010 eqs 7.12-7.13):',
             '    term of eq. 6 / term of the Conventions, per order m, component and parameter', '']
    rows = symbolic()
    for m, comp, par, a, b, r in rows:
        lines.append(f'    m={m} {comp} {par:3s} ratio {r:>12s}    eq6: {a}   |   Conv: {b}')
    for m in (0, 1, 2):
        rs = {r for mm, comp, par, a, b, r in rows if mm == m}
        lines.append(f'    m={m}: ratios {sorted(rs)}')
    with Pool(4) as pool:
        p = {}
        e_lo, e_hi = 1.10e-4, 1.25e-4                   # bracket of the elastic e_f for 430 sd (431.0 at 1.145e-4)
        p[e_lo], p[e_hi] = fcn(pool, e_lo), fcn(pool, e_hi)
        e_f = e_lo + (1 / TARGET_FCN - 1 / p[e_lo]) / (1 / p[e_hi] - 1 / p[e_lo]) * (e_hi - e_lo)
        P = fcn(pool, e_f)
        fx = [exact(m, f) for _, m, f, _ in TABLE1]
        l1 = pool.map(l1_at, [(e_f, m, f) for (_, m, _, _), f in zip(TABLE1, fx)])
        l1_h = pool.map(l1_at, [(None, m, f) for (_, m, _, _), f in zip(TABLE1, fx)])
    lines += ['', f'(b) elastic PREM, level figure, slipping core; FCN tuned to {TARGET_FCN} sd: Delta e_f = {e_f:.4e}, '
              f'FCN {P:.2f} sd (hydrostatic: see last column)', '',
              f'    {"line":18s} {"MBS 1995a":>10s} {"ours":>9s} {"ratio":>7s} {"ratio range (4-decimal rounding)":>34s} {"ours hydr.":>11s}']
    for (name, m, f, v), x, xh in zip(TABLE1, l1, l1_h):
        lines.append(f'    {name:18s} {v:10.4f} {x:9.6f} {v / x:7.2f}      {(v - 5e-5) / x:5.2f} .. {(v + 5e-5) / x:5.2f}'
                     f'{"":14s}{xh:9.6f}')
    # (c) constant part and resonance: L0 + L1 / (f_FCN - f) through the six diurnal lines
    d = [i for i, (_, m, _, _) in enumerate(TABLE1) if m == 1]
    f = np.array([fx[i] for i in d])
    ours, theirs = np.array([l1[i] for i in d]), np.array([TABLE1[i][3] for i in d])
    a0, a1, ra, _ = pole_fit(f, ours, P)
    b0, b1, rb, ff = pole_fit(f, theirs, P)
    pred = 2 * a0 + a1 / (ff - f)
    lines += ['', '(c) l(1) = L0 + L1 / (f_FCN - f) through the six diurnal lines (FCN of the row above):',
              f'    ours     L0 {a0:.6f}  L1 {a1:+.4e}  (fit residual max {np.abs(ra).max():.1e})',
              f'    MBS 1995a L0 {b0:.6f}  L1 {b1:+.4e}  (fit residual max {np.abs(rb).max():.1e})',
              f'    ratio L0 {b0 / a0:.2f}   ratio L1 {b1 / a1:.2f}',
              f'    2 L0(ours) + L1(ours) against MBS 1995a: max |diff| {np.abs(pred - theirs).max():.1e} '
              f'(rounding of their table 5e-5); 2 x ours: max |diff| {np.abs(2 * ours - theirs).max():.1e}']
    # (d) the Conventions against our anelastic model at 431.4 sd (data of Table S5)
    rows = {}
    for ln in ANEL.read_text().splitlines():
        for who in ('this paper', 'IERS'):
            if ln[10:26].strip() == who and ln[:10].strip() in ('O$_1$', 'P$_1$', 'K$_1$', r'$\psi_1$', r'$\phi_1$', 'OO$_1$', 'M$_2$'):
                rows.setdefault(ln[:10].strip(), {})[who] = float(ln[28:][30:40])        # l(1): fourth column
    names = ('O$_1$', 'P$_1$', 'K$_1$', r'$\psi_1$', r'$\phi_1$', 'OO$_1$')
    fd = np.array([exact(1, v) for v in (0.92954, 0.99726, 1.00274, 1.00548, 1.00821, 1.07594)])
    o2, i2 = np.array([rows[n]['this paper'] for n in names]), np.array([rows[n]['IERS'] for n in names])
    c0, c1, rc, _ = pole_fit(fd, o2, 431.4)
    e0, e1, re_, _ = pole_fit(fd, i2, 431.4)
    lines += ['', '(d) the Conventions against our anelastic model (FCN 431.4 sd; Table S5 data), same fit:',
              f'    ours L0 {c0:.6f} L1 {c1:+.4e} (resid {np.abs(rc).max():.1e});  IERS L0 {e0:.6f} L1 {e1:+.4e} '
              f'(resid {np.abs(re_).max():.1e})',
              f'    ratio L0 {e0 / c0:.2f}   ratio L1 {e1 / c1:.2f}   semidiurnal ratio (M2) '
              f'{rows["M$_2$"]["IERS"] / rows["M$_2$"]["this paper"]:.2f}']
    run.write_text(str(ROOT / 'results' / 'mathews1995_factor2.txt'), '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    from provenance import Run
    with Run('Review round 2 R1: Mathews 1995a eq. 6 for m = 0,1,2 and the factor 2 in l(1)',
             inputs=['data/PREM.dat', 'src/earthtides/data/love_PREM_allwaves.json',
                     'results/paper1_table_anelastic_level.txt']) as run_:
        main(run_)
