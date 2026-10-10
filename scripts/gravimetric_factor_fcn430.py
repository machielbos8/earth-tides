# -*- coding: utf-8 -*-
#
# Diurnal gravimetric factors of anelastic PREM with the free core nutation moved to the observed
# frequency, and the tidal-gravity test of the sign of the resonant ocean-loading correction.
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
"""Gravimetric factors with the FCN at the observed frequency (paper Sect. 7, tidal gravity).

1. Anelastic hydrostatic PREM (cot dispersion), m = 1, rotating, fluid core: the gravimetric
   factor delta(sigma, theta) (earthtides.gravimetry, ICET definition) at diurnal frequencies, and
   the load Love numbers h'2, k'2 (rotating minus spherical non-rotating, as in
   scripts/otl_fcn_resonance.py).
2. Each is fitted with a resonance formula, a0 + a1 (sigma - 1) + R_F/(sigma - s_F) +
   R_I/(sigma - s_I), with our own FCN and FICN frequencies (sigma in cycles per sidereal day).
3. The FCN pole is moved to the frequency of the IERS Conventions (TN36 eq. 6.10, real part
   1.0023181, 431 sidereal days); the residues are kept. (A non-hydrostatic core flattening
   imposed on the map is not consistent with the hydrostatic pre-stress; see fcn_period.py.)
4. The resonant ocean-loading term of the gravimetric factor,
   (3 rho_w / 5 rho_bar) A21 (dh' - 3/2 dk'), with the FES2022b admittance (our sign) or
   -conj of it (the sign of the IERS empirical admittance).
5. Membach (Ducarme 2012, same series for O1 and K1), loading from results/tidal_gravity
   (CARGA, FES2022b, GOT5.5, TPXO10): rho = [dc(K1)/dc(O1)] / [theory K1/O1] - 1.
Usage: python scripts/gravimetric_factor_fcn430.py -> results/gravimetric_factor_fcn430.txt
"""
import csv
import sys
import warnings
from pathlib import Path

sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
from earthtides.model import EarthModel
from earthtides.love import model_path
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.gravimetry import surface_gravity
from earthtides import etide

SID = 1.002737909                     # sidereal days per solar day
SIG = [0.893, 0.927, 0.960, 0.980, 0.990, 0.9945, 0.9975, 0.9990, 1.0006, 1.0012, 1.0035,
       1.0060, 1.0100, 1.0400, 1.0730]
S_IERS = 1.0023181                    # TN36 eq. 6.10, real part
ADM = {'O1': -0.2856 - 0.2643j, 'K1': -0.2120 - 0.2276j}    # FES2022b, otl_fcn_resonance.txt
WAVES = {'O1': 0.927000, 'K1': 1.000000}                   # cpsd
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def body(md, geo, sigma, lat):
    w = sigma * OMEGA_EARTH
    S = TidalSolver(md.anelastic(w, dispersion='cot'), geo, m=1, K=4, N=24, omega_f=0.0,
                    coriolis=OMEGA_EARTH, omega_t=w, omega_rot=OMEGA_EARTH, wahr_frame=True,
                    dyn_fluid=True, consistent_reference=True, frame='mantle')
    S.solve()
    th, dg, dgT = surface_gravity(S)
    G = np.column_stack([dgT, dgT * np.sqrt(3) / (2 * np.sqrt(2)) * (7 * np.cos(th)**2 - 3)])
    d0, dp = np.linalg.lstsq(G, dg, rcond=None)[0]
    return np.array([d0 + dp * np.sqrt(3) / (2 * np.sqrt(2)) * (7 * np.cos(t)**2 - 3) for t in lat])


def load(md, geo, sigma):
    w = sigma * OMEGA_EARTH
    mda = md.anelastic(w, dispersion='cot')
    S = TidalSolver(mda, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                    omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                    consistent_reference=True, frame='mantle', forcing='load')
    S.solve()
    R = TidalSolver(mda, m=0, n_force=2, degrees=([2], []), omega_f=w, dyn_fluid=True,
                    forcing='load')
    R.solve()
    a, b = S.load_love(), R.load_love()
    return a['h'][2] - b['h'][2], a['k'][2] - b['k'][2]


def resonance_fit(sig, val, poles):
    """Linear least squares for a0, a1 and one residue per pole; returns (coef, rms misfit)."""
    x = np.asarray(sig)
    M = np.column_stack([np.ones_like(x), x - 1] + [1 / (x - p) for p in poles])
    c, *_ = np.linalg.lstsq(M, val, rcond=None)
    return c, np.sqrt(np.mean(np.abs(M @ c - val)**2))


def evaluate(c, poles, x):
    return c[0] + c[1] * (x - 1) + sum(ci / (x - p) for ci, p in zip(c[2:], poles))


def main():
    sys.path.insert(0, 'scripts')
    from fcn_period import fcn
    md = EarthModel(str(model_path('PREM')))
    geo = ClairautGeometry(md)
    s_F = fcn(md, geo, anelastic=True)
    s_I = 0.99840                       # weak FICN feature (Sect. 5.3); its residue is fitted
    poles = [s_F, s_I]
    out('Diurnal gravimetric factors, anelastic hydrostatic PREM, FCN moved to the IERS frequency')
    out(f'our FCN {s_F:.7f} cpsd ({1/(s_F-1):.1f} sd); IERS {S_IERS} ({1/(S_IERS-1):.1f} sd)')
    stations = {'Membach': 50.6093, 'Strasbourg': 48.6223, 'Wettzell': 49.1440,
                'Vienna': 48.2493, 'Medicina': 44.5219, 'Brussels': 50.7986, 'Metsahovi': 60.2172,
                'equator+45': 45.0}
    names = list(stations)
    colat = [np.pi / 2 - etide.geocentric(stations[n])[0] for n in names]
    D = np.array([body(md, geo, s, colat) for s in SIG])              # (nsig, nstation)
    L = np.array([load(md, geo, s) for s in SIG])                      # (nsig, 2)
    rho_bar = 3 * md.M / (4 * np.pi * md.a**3)
    fac = 0.6 * 1025.0 / rho_bar
    out()
    out('fit quality (rms misfit of the resonance formula):')
    cH, eH = resonance_fit(SIG, L[:, 0], poles)
    cK, eK = resonance_fit(SIG, L[:, 1], poles)
    out(f"  dh' {eH:.1e}   dk' {eK:.1e}   FCN residues: dh' {cH[2].real:+.3e}  dk' {cK[2].real:+.3e}")
    pred = {}
    for j, n in enumerate(names):
        c, e = resonance_fit(SIG, D[:, j], poles)
        if j == 0:
            out(f'  delta (Membach) {e:.1e}; FCN residue {c[2].real:+.3e}, FICN {c[3].real:+.3e}')
        pred[n] = {}
        for w, sw in WAVES.items():
            own = evaluate(c, poles, sw)
            iers = evaluate(c, [S_IERS, s_I], sw)
            dh = evaluate(cH, [S_IERS, s_I], sw)
            dk = evaluate(cK, [S_IERS, s_I], sw)
            ot = fac * ADM[w] * (dh - 1.5 * dk)
            pred[n][w] = dict(own=own, fcn430=iers, ot=ot, dh=dh, dk=dk)
    out()
    out('station      delta O1 (430)        delta K1 (own FCN)  delta K1 (430)       '
        'K1 ocean term (our sign)')
    for n in names:
        p = pred[n]
        out(f"{n:12s} {p['O1']['fcn430'].real:.5f}{p['O1']['fcn430'].imag:+.5f}i   "
            f"{p['K1']['own'].real:.5f}            {p['K1']['fcn430'].real:.5f}{p['K1']['fcn430'].imag:+.5f}i   "
            f"{p['K1']['ot'].real:+.5f}{p['K1']['ot'].imag:+.5f}i")
    k1 = pred['Membach']['K1']
    out(f"K1 load resonance (FCN at IERS frequency): dh' {k1['dh'].real:+.4f}  dk' {k1['dk'].real:+.4f}")

    # --- Membach and the other stations: observed ratio against the theory
    D_ = Path('results/tidal_gravity')
    st = {l.split()[3]: (float(l.split()[0]), float(l.split()[1]))
          for l in (D_ / 'Stations.dat').read_text().splitlines() if l.strip() and not l.startswith('#')}
    ldg = {}
    for m in ['FES2022b', 'GOT5.5', 'TPXO10']:
        for r in csv.reader((D_ / f'CARGA_{m}.csv').open()):
            r = [x.strip() for x in r]
            ldg[(m, r[0], r[3])] = float(r[4]), float(r[5])
    HF = {'o1': 0.26221, 'k1': 0.36870}

    def ath(s, w):
        lat = etide.geocentric(st[s][1])[0]
        return 2 * 9.81 * 0.1992 * HF[w] * np.sqrt(15) * np.sin(lat) * np.cos(lat) / 6.371e6 * 1e8

    def lvec(m, s, w):
        a_, p_ = ldg[(m, s, w)]
        return a_ / ath(s, w) * np.exp(1j * np.radians(-p_ - st[s][0]))

    K1 = {'Membach': (1.13716, 0.280), 'Strasbourg': (1.13695, 0.269), 'Wettzell': (1.13673, 0.204),
          'Vienna': (1.13392, 0.204), 'Medicina': (1.13484, 0.355), 'Brussels': (1.13712, 0.248),
          'Metsahovi': (1.13998, 0.083)}
    O1 = {'Membach': (1.14935, 0.1072, 'Ducarme 2012'), 'Strasbourg': (1.1490, 0.08, 'BB03'),
          'Wettzell': (1.1485, 0.08, 'BB03'), 'Vienna': (1.1479, 0.10, 'BB03'),
          'Medicina': (1.1493, 0.17, 'BB03'), 'Brussels': (1.1533, 0.06, 'BB03'),
          'Metsahovi': (1.1532, 0.25, 'BB03')}
    out()
    out('rho = [dc(K1)/dc(O1)]/[theory K1/O1] - 1 in per cent (in-phase parts), theory = this paper '
        'with FCN at 431 sd, no ocean resonance;')
    out('predicted rho: our sign / IERS sign of the in-phase ocean term')
    out(f"{'station':11s} {'O1 from':13s} {'FES2022b':>9s} {'GOT5.5':>8s} {'TPXO10':>8s}   "
        f"{'pred ours':>9s} {'pred IERS':>9s}")
    obs = lambda d, a: d * np.exp(1j * np.radians(a))
    for s in K1:
        p = pred[s]
        th_ratio = p['K1']['fcn430'].real / p['O1']['fcn430'].real
        rr = []
        for m in ['FES2022b', 'GOT5.5', 'TPXO10']:
            k = obs(*K1[s]) - lvec(m, s, 'k1')
            o = obs(*O1[s][:2]) - lvec(m, s, 'o1')
            rr.append(100 * ((k.real / o.real) / th_ratio - 1))
        po = 100 * p['K1']['ot'].real / p['K1']['fcn430'].real
        out(f"{s:11s} {O1[s][2]:13s} " + ' '.join(f'{v:+8.3f}' for v in rr) + f"   {po:+9.3f} {-po:+9.3f}")
    Path('results/gravimetric_factor_fcn430.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
