# -*- coding: utf-8 -*-
#
# Station-displacement budget of the choices behind the body-tide model (paper summary,
# conclusions, Sect. 8.4): each choice as the difference of two etide runs.
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
"""Displacement differences (mm) for the four stations of Table 7, 2026, hourly.

  band       PREM table minus PREM with the moduli taken at 200 s (no absorption band
             between 1 s and 200 s; table computed once with earthtides-love's default
             wave groups and cached in results/love_PREM_T0_200s.json)
  damping    IERS resonance formula (TN36 Table 7.1, eq. 6.10) minus the same with a real
             FCN eigenfrequency (residues kept complex)
  out-of-phase  IERS resonance formula minus the same with the imaginary parts of the diurnal
             h0, l0 replaced by ours (anelastic PREM, undamped): the complex residues and
             eigenfrequencies of the formula (anelasticity, core coupling) against ours
  otl        IERS Table 7.2 minus the resonance formula (the resonant ocean-loading correction
             and the small deviations of the exact computation)
  otl sign   IERS with the correction of P1 and K1 replaced by -conj (our admittance) minus IERS
  latitude   IERS model evaluated at geodetic instead of geocentric latitude minus IERS
  total      PREM table minus IERS
Radial rms/max and horizontal rms (sqrt of north^2 + east^2), means removed.
Usage: python scripts/displacement_budget.py -> results/displacement_budget.txt
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
from earthtides import etide
from earthtides.etide import Tamura, LoveModel, displacement

STATIONS = {'ONSA': (57.395, 11.926), 'KOKB': (22.126, -159.665), 'NKLG': (0.354, 9.672),
            'HOB2': (-42.805, 147.439)}
T200 = Path('results/love_PREM_T0_200s.json')
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def table_T0_200():
    """Love table of anelastic PREM with the moduli taken at 200 s (T0 = Tm = 200 s)."""
    if T200.exists():
        return str(T200)
    from earthtides import love
    from earthtides.model import EarthModel
    orig = EarthModel.anelastic

    def anel(self, omega, alpha=0.15, Tm=200.0, T0=1.0, dispersion='cot'):
        return orig(self, omega, alpha=alpha, Tm=Tm, T0=200.0, dispersion=dispersion)
    EarthModel.anelastic = anel
    try:
        love.main(['--jobs', '1', '-o', str(T200)])
    finally:
        EarthModel.anelastic = orig
    return str(T200)


def variants(cat):
    V = {'iers': LoveModel(cat, 'iers'), 'mhb': LoveModel(cat, 'mhb'), 'prem': LoveModel(cat, 'prem'),
         'prem200': LoveModel(cat, table_T0_200())}
    sig = etide._MHB_SIGMA.copy()
    etide._MHB_SIGMA[1] = sig[1].real
    V['mhb_undamped'] = LoveModel(cat, 'mhb')
    etide._MHB_SIGMA[:] = sig
    flip = LoveModel(cat, 'iers')
    f = np.abs(cat.f_cpd)
    mask = (cat.n == 2) & (cat.m == 1) & (f > 0.996) & (f < 1.0035)        # P1 ... K1, not psi1
    for key in ('h0', 'l0'):
        corr = getattr(V['iers'], key)[mask] - getattr(V['mhb'], key)[mask]
        getattr(flip, key)[mask] = getattr(V['mhb'], key)[mask] - np.conj(corr)
    V['iers_flip'] = flip
    imag = LoveModel(cat, 'mhb')                       # IERS formula with our out-of-phase part
    di = (cat.n == 2) & (cat.m == 1)
    for key in ('h0', 'l0'):
        v = getattr(imag, key)
        v[di] = v[di].real + 1j * getattr(V['prem'], key)[di].imag
    V['mhb_imag_prem'] = imag
    return V


def run(V, cat, mjd, lat, lon, key, geodetic=False):
    if geodetic:
        g = etide.geocentric
        etide.geocentric = lambda la, h=0.0: (np.radians(la), g(la, h)[1])
        try:
            return displacement(mjd, lat, lon, love=V[key], cat=cat)
        finally:
            etide.geocentric = g
    return displacement(mjd, lat, lon, love=V[key], cat=cat)


def main():
    cat = Tamura()
    V = variants(cat)
    mjd = 61041.0 + np.arange(365 * 24) / 24.0                      # 2026, hourly
    cases = [('absorption band 1-200 s', ('prem', False), ('prem200', False)),
             ('FCN damping alone', ('mhb', False), ('mhb_undamped', False)),
             ('diurnal out-of-phase part', ('mhb', False), ('mhb_imag_prem', False)),
             ('ocean-loading correction', ('iers', False), ('mhb', False)),
             ('its in-phase sign', ('iers_flip', False), ('iers', False)),
             ('geodetic latitude', ('iers', True), ('iers', False)),
             ('PREM - IERS', ('prem', False), ('iers', False))]
    out('Displacement differences, 2026 hourly, mm: radial rms / max, horizontal rms')
    out(f"{'':26s}" + ''.join(f'{s:>18s}' for s in STATIONS))
    for name, a, b in cases:
        row = []
        for st, (lat, lon) in STATIONS.items():
            ra, na, ea = run(V, cat, mjd, lat, lon, *a)
            rb, nb, eb = run(V, cat, mjd, lat, lon, *b)
            dr, dn, de = (1e3 * (x - y) for x, y in ((ra, rb), (na, nb), (ea, eb)))
            dr, dn, de = dr - dr.mean(), dn - dn.mean(), de - de.mean()
            row.append(f'{np.sqrt(np.mean(dr**2)):5.2f}/{np.abs(dr).max():4.2f} {np.sqrt(np.mean(dn**2 + de**2)):5.2f}')
        out(f'{name:26s}' + ''.join(f'{r:>18s}' for r in row))
    Path('results/displacement_budget.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
