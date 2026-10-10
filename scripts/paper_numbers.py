# -*- coding: utf-8 -*-
#
# The numbers of the paper that no other script prints, computed and set against the printed value.
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
# 10/10/2026 Machiel Bos
#===============================================================================
"""Numbers quoted in the text of the paper (main text and Online Resource 1) that are not printed by
another script. Each line gives the section, the quantity, our value and the value in the paper.

Sections (run all, or name some on the command line):
  model       Sect. 2.4  ocean replaced by rock of crustal density: change of h2 (spherical PREM)
  coeffs      Sect. 3.4  number of coefficients for PREM, K = 5, N = 30
  refpot      Sect. 3.5  consistent reference potential: change of the Love numbers (M2, Mf)
  misfit      Sect. 3.7  misfit of the IERS parametrization (Mf, O1, K1, M2)
  coriolis_m  Sect. 5.2  Coriolis part of l(1) at a fixed frequency for m = 1 and m = 2
  fcn         Sect. 6.1, S3, S4  FCN period: elastic, anelastic, stiffened mantle, spun-up
              models; Hough's formula; dynamical ellipticity of the core; compliance beta
  residues    Table 4    FCN residues of h^L, k^L (load) and h(0), k (body tide), elastic PREM
  load345     Sect. 6.3  load Love numbers of degrees 3-5: rotating flattened / spherical
  ficn        Sect. 6.1  the weak feature near 0.998 cpsd (from results/figdata_fcn.npz)
  mesh        S3         distorted mesh with the Coriolis force and with a fluid core
  convergence S3         diurnal convergence between K = 3 and K = 8; halving the flattening
  mm          Sects 5.3, 6.3, 8.2, 8.3  conversions of Love-number changes to millimetres
Usage: python scripts/paper_numbers.py [section ...] -> results/paper_numbers.txt (all sections)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
from earthtides.model import EarthModel, G
from earthtides.love import model_path, Options, solve_wave, parameters
from earthtides.geometry import ClairautGeometry, BumpGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver

F = {'Mf': 0.0732022, 'O1': 0.9295357, 'K1': 1.0027379, 'M2': 1.9322736}   # cycles per day
M_OF = {'Mf': 0, 'O1': 1, 'K1': 1, 'M2': 2}
SID = 1.002737909                                   # sidereal days per solar day
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def item(sect, what, ours, paper):
    out(f'{sect:10s} {what:62s} {ours:>22s}   paper: {paper}')


def prem():
    return EarthModel(str(model_path('PREM')))


def opts(**kw):
    o = dict(model='PREM', anelastic=False, rotation=True, flattening=True, centrifugal=True,
             frame='mantle', latitude='geocentric', direction='radial', surface='deformed',
             norm='equatorial', K=4, N=24, dispersion='cot')
    o.update(kw)
    return Options(**o)


def h2_sphere(md, f_cpd=1.9322736):
    w = f_cpd * 2 * np.pi / 86400.0
    S = TidalSolver(md, m=0, N=24, n_force=2, degrees=([2], []), omega_f=w)
    return complex(S.solve()['h']).real


# ------------------------------------------------------------------------------ sections
def sec_model():
    md = prem()
    src = Path(model_path('PREM')).read_text()
    assert '15.0    2.28400' in src
    tmp = Path('results/_PREM_rock.dat')
    tmp.write_text(src.replace('15.0    2.28400', '15.0    2.60000', 1))
    try:
        d = h2_sphere(EarthModel(str(tmp))) - h2_sphere(md)
    finally:
        tmp.unlink()
    item('2.4', 'ocean -> rock (2600 kg/m3, top 15 km layer): change of h2', f'{d:+.1e}', '+5e-4')
    item('2.4', 'transversely isotropic PREM: change of h2', 'not computed', '-3e-4 (astyanax, Bos & Scherneck 2013)')


def sec_coeffs():
    md = prem()
    K, N = 5, 30
    S = TidalSolver(md, ClairautGeometry(md), m=2, K=K, N=N, omega_f=1.4053e-4)
    n = S.nfld * N * md.n_layers
    item('3.4', 'unknown coefficients, PREM, K = 5, N = 30 (m = 2)', f'{n}', 'about 7000')


def sec_refpot():
    for name in ('Mf', 'M2'):
        a = parameters(*solve_wave(opts(), M_OF[name], F[name]), opts())
        o = opts()
        md, geo = prem(), ClairautGeometry(prem())
        w = F[name] * 2 * np.pi / 86400.0
        S = TidalSolver(md, geo, m=M_OF[name], K=4, N=24, omega_rot=OMEGA_EARTH, omega_f=0.0,
                        coriolis=OMEGA_EARTH, omega_t=w, wahr_frame=(M_OF[name] != 2),
                        dyn_fluid=True, consistent_reference=False, frame='mantle')
        b = parameters(S, S.solve(), o)
        d = max(abs(a[k] - b[k]) for k in a if k != 'misfit' and k in b)
        item('3.5', f'consistent minus level-surface reference potential, {name}: max change',
             f'{d:.1e}', 'about 2e-6')


def sec_misfit():
    for name in F:
        p = parameters(*solve_wave(opts(anelastic=True), M_OF[name], F[name]), opts(anelastic=True))
        item('3.7', f'misfit of the IERS parametrization, {name} (relative)', f'{p["misfit"]:.1e}',
             '< 1.5e-6 (Mf, M2); <= 1.3e-5 (diurnal)')


def sec_coriolis_m():
    o = opts(flattening=False)
    for name in ('M2', 'O1'):
        v = []
        for m in (1, 2):
            p = parameters(*solve_wave(o, m, F[name]), o)
            v.append(p['l1'].real)
        item('5.2', f'Coriolis part of l(1) at the {name} frequency, m = 1 / m = 2',
             f'{v[0]:.5f} / {v[1]:.5f}', '0.00033 (M2), 0.00016 (O1); barely depends on m')


def _fcn(md, geo, Om, anelastic, sig=(1.0012, 1.0016, 1.0030, 1.0036)):
    """sigma_FCN (cpsd of that Earth's day) from k at four frequencies, as scripts/fcn_period.py."""
    ks = []
    for s in sig:
        w = s * Om
        m_ = md.anelastic(w, dispersion='cot') if anelastic else md
        S = TidalSolver(m_, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=Om, omega_t=w,
                        omega_rot=Om, wahr_frame=True, dyn_fluid=True,
                        consistent_reference=True, frame='mantle')
        S.solve()
        ks.append(S.results()['kk'])
    x, k = np.array(sig), np.array(ks).real
    c, a, d = np.linalg.lstsq(np.column_stack([k, x, -np.ones_like(x)]), k * x, rcond=None)[0]
    return c


def fluid(md, i):
    s = np.linspace(md.r0[i], md.r1[i], 5)
    return bool(np.all(np.abs(md.mu[i](s)) == 0.0))


def moments(md, geo):
    """Mean equatorial moments A (Earth), A_f (fluid outer core), A_s (inner core) and the
    dynamical ellipticity e_f = (C_f - A_f)/A_f of the outer core, first order in the flattening:
    I = (8 pi / 3) int rho s^4 ds, C - A = (8 pi / 15) int rho d(s^5 eps)/ds ds, A = I - (C - A)/3."""
    tot = {}
    for i in range(md.n_layers):
        s = np.linspace(max(md.r0[i], 1.0), md.r1[i], 4001)
        rho = md.rho[i](s)
        e, de, _ = geo.eps(i, s)
        I = 8 * np.pi / 3 * np.trapezoid(rho * s**4, s)
        CA = 8 * np.pi / 15 * np.trapezoid(rho * (5 * s**4 * e + s**5 * de), s)
        tot[i] = (I - CA / 3, CA)
    fl = [i for i in range(md.n_layers) if fluid(md, i)]
    ic = list(range(min(fl)))
    A = sum(v[0] for v in tot.values())
    Af = sum(tot[i][0] for i in fl)
    ef = sum(tot[i][1] for i in fl) / Af
    As = sum(tot[i][0] for i in ic)
    return A, Af, As, ef


def sec_fcn():
    md = prem()
    geo = ClairautGeometry(md)
    A, Af, As, ef = moments(md, geo)
    Am = A - Af - As
    item('6.1', 'dynamical ellipticity of the fluid core e_f', f'1/{1 / ef:.1f}', '1/392.5')
    item('6.1', 'A_m / A', f'{Am / A:.3f}', '0.886')
    item('6.1', "Hough's formula: period A_m/(A e_f) [sidereal days]", f'{Am / (A * ef):.0f}', '348')
    stiff = EarthModel(str(model_path('PREM')))
    for i in range(stiff.n_layers):
        if not fluid(stiff, i):
            stiff.mu[i] = stiff.mu[i] * 1e4
            stiff.lam[i] = stiff.lam[i] * 1e4
    c = _fcn(stiff, ClairautGeometry(stiff), OMEGA_EARTH, False)
    item('6.1', 'FCN, mantle and inner core stiffened 1e4 times [sidereal days]', f'{1 / (c - 1):.1f}', '356.5')
    ce = _fcn(md, geo, OMEGA_EARTH, False)
    item('6.1', 'FCN, elastic hydrostatic PREM [sidereal days]', f'{1 / (ce - 1):.1f}', '457.2')
    beta = ef - (ce - 1) * Am / A
    item('S3', 'compliance beta = e_f - (sigma_FCN - 1) A_m / A', f'{beta:.2e}', '6.1e-4')
    base = None
    for x in (0.0, 0.020, 0.025, 0.0327):
        Om = OMEGA_EARTH * (1 + x)
        c = _fcn(md, ClairautGeometry(md, omega=Om), Om, True)
        P = 1 / (c - 1)
        base = base or P
        item('S3/S4', f'FCN, anelastic PREM spun {100 * x:.2f} % faster [days of that Earth]',
             f'{P:.1f} (Omega^2: {base / (1 + x)**2:.1f})',
             {0.0: '460.0', 0.020: '442.1', 0.025: '437.8', 0.0327: '431.3'}[x])


def _resfit(sig, val, poles):
    x = np.asarray(sig)
    M = np.column_stack([np.ones_like(x), x - 1] + [1 / (x - p) for p in poles])
    c, *_ = np.linalg.lstsq(M, val, rcond=None)
    return c


def sec_residues():
    md = prem()
    geo = ClairautGeometry(md)
    sF = _fcn(md, geo, OMEGA_EARTH, False)
    poles = [sF, 0.99840]
    SIG = [0.893, 0.927, 0.960, 0.980, 0.990, 0.9945, 0.9975, 0.9990, 1.0006, 1.0012, 1.0035,
           1.0050, 1.0080, 1.0200]
    hL, kL, h0, kk = [], [], [], []
    o = opts()
    for s in SIG:
        w = s * OMEGA_EARTH
        S = TidalSolver(md, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                        omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                        consistent_reference=True, frame='mantle', forcing='load')
        S.solve()
        R = TidalSolver(md, m=0, n_force=2, degrees=([2], []), omega_f=w, dyn_fluid=True, forcing='load')
        R.solve()
        a, b = S.load_love(), R.load_love()
        hL.append(a['h'][2] - b['h'][2])
        kL.append(a['k'][2] - b['k'][2])
        p = parameters(*solve_wave(o, 1, s * SID), o)
        h0.append(p['h0'])
        kk.append(p['k'])
    for name, v, paper in (('h^L', hL, '2.78e-4'), ('k^L', kL, '1.39e-4'), ('h(0)', h0, '1.83e-4'),
                           ('k', kk, '0.92e-4')):
        c = _resfit(SIG, np.array(v, complex), poles)
        item('Table 4', f'FCN residue of {name}, elastic PREM [cpsd]', f'{abs(c[2].real):.3e}', paper)


def sec_load345():
    md = prem()
    geo = ClairautGeometry(md)
    for name, s in (('O1', 0.92700), ('K1', 1.0), ('near FCN', 1.0019)):
        w = s * OMEGA_EARTH
        r = []
        for n in (3, 4, 5):
            S = TidalSolver(md, geo, m=1, K=5, N=24, n_force=n, omega_f=0.0, coriolis=OMEGA_EARTH,
                            omega_t=w, omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                            consistent_reference=True, frame='mantle', forcing='load')
            S.solve()
            R = TidalSolver(md, m=0, n_force=n, degrees=([n], []), omega_f=w, dyn_fluid=True, forcing='load')
            R.solve()
            r.append(complex(S.load_love()['h'][n] / R.load_love()['h'][n]).real - 1)
        item('6.3', f'h^L_n rotating flattened / spherical - 1, n = 3, 4, 5, {name}',
             ' '.join(f'{100 * x:+.2f}%' for x in r), '0.2 % (constant)')


def sec_ficn():
    d = np.load('results/figdata_fcn.npz')             # frequency sweep of scripts/run_diurnal.py
    sig, k = d['sweep'], np.real(d['k0'])
    w = (sig > 0.990) & (sig < 1.0)
    st = np.diff(k[w])
    j = np.argmax(np.abs(st - np.median(st)))           # step that departs most from the trend
    mid = 0.5 * (sig[w][j] + sig[w][j + 1])
    item('6.1', 'weak feature in k below the tilt-over mode, at [cpsd]', f'{mid:.4f}', '0.998')


def sec_mesh():
    """Love numbers h, l, k of a spherical Earth on a distorted (bumped) mesh minus those on the
    spherical mesh (M2, K = 4, N = 30, bump 20 per cent of the layer thickness)."""
    def diff(md, **kw):
        r0 = TidalSolver(md, **kw).solve()
        rb = TidalSolver(md, BumpGeometry(md, 0.2), background='spherical', **kw).solve()
        return max(abs(complex(rb[k]) - complex(r0[k])) for k in ('h', 'l', 'kk'))
    d = diff(EarthModel('data/PREMhard.dat'), m=2, K=4, N=30, omega_f=0.0, coriolis=OMEGA_EARTH,
             omega_t=1.4053e-4)
    item('S3', 'distorted mesh with the Coriolis force: max |dh|, |dl|, |dk|', f'{d:.1e}', '1e-10')
    d = diff(prem(), m=2, K=4, N=30, omega_f=1.4053e-4, dyn_fluid=True)
    item('S3', 'distorted mesh with the fluid core: max |dh|, |dl|, |dk|', f'{d:.1e}', '2e-7')


def sec_convergence():
    md = prem()
    geo = ClairautGeometry(md)
    o = opts(anelastic=True)
    for name in ('O1', 'K1'):
        res = {}
        for K in (3, 8):
            o.K = K
            res[K] = parameters(*solve_wave(o, 1, F[name]), o)
        d = max(abs(res[3][k] - res[8][k]) for k in res[3] if k != 'misfit')
        item('S3', f'diurnal convergence {name}: max change K = 3 -> 8', f'{d:.1e}', '6e-6')
    o = opts()
    h2 = {}
    for scale in (1.0, 0.5):
        g = ClairautGeometry(md, scale=scale)
        w = F['M2'] * 2 * np.pi / 86400.0
        S = TidalSolver(md, g, m=2, K=4, N=24, omega_rot=OMEGA_EARTH * np.sqrt(scale), omega_f=0.0,
                        coriolis=OMEGA_EARTH, omega_t=w, dyn_fluid=True, consistent_reference=True,
                        frame='mantle')
        h2[scale] = parameters(S, S.solve(), o)['h2'].real
    item('S3', 'halving the flattening: h(2)(f/2) / h(2)(f)', f'{h2[0.5] / h2[1.0]:.3f}', '0.50 within 1 %')


def sec_mm():
    from earthtides.etide import Tamura, W_functions
    cat = Tamura()
    phi = np.radians(45.0)
    i = np.argmin(np.abs(np.abs(cat.f_cpd) - 1.002738) + 10 * ((cat.n != 2) | (cat.m != 1)))
    W, _ = W_functions(2, 1, np.array([phi]))
    eq = abs(cat.A[i]) * abs(W[0])
    item('6.3', 'K1 equilibrium tide at 45 deg [m]', f'{eq:.3f}', '0.14')
    item('6.3', 'Re dh(0) = 0.005 at K1, 45 deg: amplitude / rms [mm]',
         f'{0.005 * eq * 1e3:.2f} / {0.005 * eq * 1e3 / np.sqrt(2):.2f}', '0.7 / 0.5')
    rms = [88, 117]
    item('8.3', 'normalization 0.45 % of the degree-2 tide (88-117 mm rms) [mm]',
         f'{0.0045 * rms[0]:.2f}-{0.0045 * rms[1]:.2f}', '0.4-0.5')
    item('8.3', 'centrifugal reference, dh(0) = 8e-4 of h(0) = 0.6 (88-117 mm rms) [mm]',
         f'{8e-4 / 0.6 * rms[0]:.2f}-{8e-4 / 0.6 * rms[1]:.2f}', '0.12-0.16')


SECTIONS = {'model': sec_model, 'coeffs': sec_coeffs, 'refpot': sec_refpot, 'misfit': sec_misfit,
            'coriolis_m': sec_coriolis_m, 'fcn': sec_fcn, 'residues': sec_residues,
            'load345': sec_load345, 'ficn': sec_ficn, 'mesh': sec_mesh,
            'convergence': sec_convergence, 'mm': sec_mm}


def main():
    names = sys.argv[1:] or list(SECTIONS)
    out(f"{'section':10s} {'quantity':62s} {'this script':>22s}   value in the paper")
    for n in names:
        t = time.time()
        try:
            SECTIONS[n]()
        except Exception as e:                                  # report and continue
            item(n, 'FAILED', type(e).__name__, str(e)[:80])
        print(f'   [{n}: {time.time() - t:.0f} s]', flush=True)
    if not sys.argv[1:]:
        Path('results/paper_numbers.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
