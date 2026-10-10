# -*- coding: utf-8 -*-
#
# Uncertainty of the body-tide correction at geodetic stations in 2026 (paper Sect. 7, Fig. 7).
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
"""Hourly radial body-tide differences (mm) in 2026 at nine geodetic stations, for three
sources of uncertainty, each as the difference of etide runs:

  anelasticity  PREM table minus PREM with the moduli taken at 200 s (the absorption band
                between 1 s and 200 s, which the observations do not yet settle)
  ocean models  the resonant ocean-loading correction dh(OT) of eq. (dhot), computed with the
                degree-2, order-1 admittance of FES2014b, FES2022b, GOT5.5 and TPXO10-atlas-v2;
                each model minus the mean of the four (pooled)
  OTL sign      IERS model with the in-phase part of its ocean-loading correction reversed
                (-conj, our admittance) minus the IERS model

dh(OT) = (3/5)(rho_w/rho_mean) [h'(f) - h'_ref] A21(f): the load-Love-number increase at the
eight lines of results/otl_fcn_resonance.txt is fitted with a + b (f - 1) + R / (f - f_FCN);
A21 is interpolated linearly in frequency between Q1, O1, P1, K1 and held constant beyond.
Output: results/station_error_budget.txt and results/station_error_budget.npz (percentiles).
Usage: python scripts/station_error_budget.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
from earthtides import etide
from earthtides.etide import Tamura, LoveModel, displacement
from displacement_budget import variants

STATIONS = {'NYAL': (78.930, 11.865), 'METS': (60.217, 24.395), 'ONSA': (57.395, 11.926),
            'WTZR': (49.144, 12.879), 'KOKB': (22.126, -159.665), 'NKLG': (0.354, 9.672),
            'HRAO': (-25.890, 27.687), 'AGGO': (-34.874, -58.140), 'HOB2': (-42.805, 147.439)}
GRIDS = Path.home() / 'projects/otl_map/surfaceloads'
MODELS = {'FES2014b': 'FES2014b', 'FES2022b': 'FES2022b', 'GOT5.5': 'GOT5.5',
          'TPXO10': 'TPXO10_atlas_v2'}
WAVES = [('q1', 0.893244), ('o1', 0.929536), ('p1', 0.997262), ('k1', 1.002738)]
FAC = 0.11153                                      # (3/5) rho_w / rho_mean (otl_fcn_resonance)
F_FCN = 1.0021741 * 1.002737909                    # FCN of anelastic hydrostatic PREM [cpd]
MJD0 = 61041.0                                     # 2026-01-01
PCT = (1, 25, 50, 75, 99)
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def read_grid(path):
    """Grid of a GMT netCDF file as (z[lat, lon], lat, lon) in radians, row 0 = north."""
    import netCDF4
    f = netCDF4.Dataset(str(path))
    f.set_auto_mask(False)
    nx, ny = (int(v) for v in f['dimension'][:])
    dx, dy = (float(v) for v in f['spacing'][:])
    x0, y1 = float(f['x_range'][0]), float(f['y_range'][1])
    pix = int(getattr(f['z'], 'node_offset', 0))
    z = np.asarray(f['z'][:], float).reshape(ny, nx)
    lon = x0 + (np.arange(nx) + 0.5 * pix) * dx
    lat = y1 - (np.arange(ny) + 0.5 * pix) * dy
    keep = lon < lon[0] + 360.0 - 0.5 * dx                     # drop a repeated 360-degree column
    return z[:, keep], np.radians(lat), np.radians(lon[keep])


def admittance(model, wave, amp_tamura):
    """Degree-2, order-1 admittance of the ocean tide of one wave (complex), as in
    scripts/otl_fcn_resonance.py."""
    amp, lat, lon = read_grid(GRIDS / MODELS[model] / ('amp.' + wave))
    pha, _, _ = read_grid(GRIDS / MODELS[model] / ('pha.' + wave))
    ocean = (amp < 999.0) & (amp > 0.0)
    Z = np.where(ocean, amp * np.exp(-1j * np.radians(np.where(ocean, pha, 0.0))), 0.0)
    f, w = np.sin(2 * lat), np.cos(lat)
    num = ((Z * np.exp(-1j * lon)[None, :]).sum(1) * f * w).sum() / len(lon)
    den = (f**2 * w).sum()
    H = amp_tamura * etide.CV * etide.RE / (etide.GE / etide.RE**2)
    return num / den / H


def load_increase():
    """Fit a + b (f - 1) + R / (f - f_FCN) to dh'2 at the eight lines of otl_fcn_resonance.txt."""
    f, d = [], []
    for ln in Path('results/otl_fcn_resonance.txt').read_text().splitlines():
        w = ln.split()
        if len(w) >= 5 and w[0] in ('Q1', 'O1', 'P1', 'K1', 'psi1', 'phi1', 'J1', 'OO1'):
            try:
                f.append(float(w[1]))
                d.append(complex(w[4].replace('j', 'j')))
            except ValueError:
                pass
    f, d = np.array(f), np.array(d)
    A = np.column_stack([np.ones_like(f), f - 1, 1 / (f - F_FCN)])
    c, *_ = np.linalg.lstsq(A.astype(complex), d, rcond=None)
    out(f"load-Love-number increase: fit to {len(f)} lines, max misfit {np.abs(A @ c - d).max():.1e}")
    return lambda x: c[0] + c[1] * (x - 1) + c[2] / (x - F_FCN)


def main():
    cat = Tamura()
    V = variants(cat)
    fabs = np.abs(cat.f_cpd)
    di = (cat.n == 2) & (cat.m == 1)

    def tamura_amp(fc):
        return cat.A[np.argmin(np.abs(fabs - fc) + 10 * ~di)]

    out('Admittance A21 of the degree-2, order-1 ocean tide (ocean / equilibrium)')
    adm = {}
    for model in MODELS:
        adm[model] = np.array([admittance(model, w, tamura_amp(fc)) for w, fc in WAVES])
        out(f'  {model:9s} ' + '  '.join(f'{w.upper()} {a:+.3f}' for (w, _), a in zip(WAVES, adm[model])))
    dh = load_increase()
    fq = np.array([fc for _, fc in WAVES])
    for model in MODELS:
        L = LoveModel(cat, 'prem')
        a = adm[model]
        A21 = np.interp(fabs[di], fq, a.real) + 1j * np.interp(fabs[di], fq, a.imag)
        L.h0[di] = L.h0[di] + FAC * dh(fabs[di]) * A21
        V['otl_' + model] = L
    k1 = np.argmin(np.abs(fabs - 1.002738) + 10 * ~di)
    out('  dh(OT) at K1: ' + '  '.join(f'{m} {V["otl_" + m].h0[k1] - V["prem"].h0[k1]:+.5f}'
                                         for m in MODELS))
    mjd = MJD0 + np.arange(365 * 24) / 24.0
    kinds = ('anelasticity', 'ocean models', 'OTL sign')
    pct = np.zeros((len(STATIONS), len(kinds), len(PCT)))
    rms = np.zeros((len(STATIONS), len(kinds)))
    mx = np.zeros_like(rms)
    out()
    out('Radial differences, 2026 hourly, mm: rms / max |.|')
    out(f"{'station':8s} {'lat':>7s}" + ''.join(f'{k:>20s}' for k in kinds))
    for i, (st, (lat, lon)) in enumerate(STATIONS.items()):
        r = {k: displacement(mjd, lat, lon, love=V[k], cat=cat)[0]
             for k in ('prem', 'prem200', 'iers', 'iers_flip') + tuple('otl_' + m for m in MODELS)}
        otl = np.array([r['otl_' + m] for m in MODELS])
        series = [r['prem'] - r['prem200'], (otl - otl.mean(0)).ravel(), r['iers_flip'] - r['iers']]
        row = f'{st:8s} {lat:7.2f}'
        for j, x in enumerate(series):
            x = 1e3 * (x - x.mean())
            pct[i, j] = np.percentile(x, PCT)
            rms[i, j], mx[i, j] = np.sqrt(np.mean(x**2)), np.abs(x).max()
            row += f'{rms[i, j]:13.3f} /{mx[i, j]:5.2f}'
        out(row)
    np.savez('results/station_error_budget.npz', stations=list(STATIONS),
             lat=[v[0] for v in STATIONS.values()], kinds=list(kinds), pct=pct, percentiles=PCT,
             rms=rms, max=mx)
    Path('results/station_error_budget.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
