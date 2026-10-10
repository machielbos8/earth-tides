# -*- coding: utf-8 -*-
#
# Section 6: body-tide displacement by harmonic synthesis (HARDISP catalog) vs
# IERS DEHANTTIDEINEL.
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
# 8/10/2026 Machiel Bos
#===============================================================================
"""Section 6: body-tide displacement by harmonic synthesis (HARDISP catalog) vs IERS DEHANTTIDEINEL.

Needs skyfield + DE421 (Sun and Moon in the terrestrial frame) and the IERS routines compiled with
a stdin driver (XSTA, T, FHR, XSUN, XMON -> DXTIDE); paths in IERS_BIN. Stage 1 compares the
degree-2, frequency-independent part (DEHANTTIDEINEL with degree 3, the l1/out-of-phase
corrections and step 2 removed) with the synthesis for the same Love numbers: this checks the
normalization and phases of the chain. Stage 2 measures what the IERS model adds (degree 3, the
l1 and out-of-phase terms, step 2) and the permanent tide. Stage 3 uses the PREM admittance of
scripts/station_admittance.py: exact synthesis line by line, the same through a body-tide BLQ and
HARDISP (spline in frequency through 11 constituents), and the difference with the IERS model.
Usage: python scripts/body_tide_check.py <scratch dir with de421.bsp, dehant, dehant_d2, dehant_n3,
       hardisp>  ->  results/body_tide_check.txt
"""
import subprocess
import sys

import numpy as np
from skyfield.api import load
from skyfield.framelib import itrs

sys.path.insert(0, 'src')
from earthtides.synthesis import catalog, synthesize, body_admittance, ct, blq, BLQ_DOODSON
from earthtides.solver import ybar

W = sys.argv[1]
A_GRS, F_GRS = 6378137.0, 1 / 298.257222101
STATIONS = {'ONSA': (57.395, 11.926), 'KOKB': (22.126, -159.665), 'NKLG': (0.354, 9.672),
            'HOB2': (-42.805, 147.439)}


def ecef(lat, lon, h=0.0):
    la, lo = np.radians(lat), np.radians(lon)
    e2 = F_GRS * (2 - F_GRS)
    N = A_GRS / np.sqrt(1 - e2 * np.sin(la)**2)
    return np.array([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo),
                     (N * (1 - e2) + h) * np.sin(la)])


def epochs(days=365, step_h=1.0, start=(2026, 1, 1)):
    ts = load.timescale()
    hours = np.arange(0, days * 24, step_h)
    t = ts.utc(start[0], start[1], start[2], hours)
    return t


def sun_moon(t):
    eph = load(W + '/de421.bsp')
    earth = eph['earth']
    xs = (eph['sun'] - earth).at(t).frame_xyz(itrs).m
    xm = (eph['moon'] - earth).at(t).frame_xyz(itrs).m
    return xs.T, xm.T


def dehant(exe, x, t, xs, xm):
    T = (t.tt - 2451545.0) / 36525.0
    fhr = (np.mod(utc_jd(t) + 0.5, 1.0)) * 24.0
    inp = '\n'.join(' '.join(f'{v:.15e}' for v in (*x, T[i], fhr[i], *xs[i], *xm[i]))
                    for i in range(len(T)))
    out = subprocess.run([W + '/' + exe], input=inp, capture_output=True, text=True).stdout
    return np.array([[float(v) for v in l.split()] for l in out.splitlines()])


def utc_jd(t):
    """UTC as a Julian date (no leap second inside the interval)."""
    return t.tt - (32.184 + 37.0) / 86400.0


def to_local(dx, x):
    """ECEF displacement -> geocentric radial, south (+theta), west (-lambda)."""
    r = np.linalg.norm(x)
    th, lam = np.arccos(x[2] / r), np.arctan2(x[1], x[0])
    er = x / r
    eth = np.array([np.cos(th) * np.cos(lam), np.cos(th) * np.sin(lam), -np.sin(th)])
    elam = np.array([-np.sin(lam), np.cos(lam), 0.0])
    return np.column_stack([dx @ er, dx @ eth, -(dx @ elam)]), th, lam


lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


def stats(d):
    """rms and max of the demeaned columns (radial, south, west) in mm."""
    d = d - d.mean(0)
    return ' '.join(f'{1e3 * d[:, c].std():6.3f}/{1e3 * np.abs(d[:, c]).max():6.3f}' for c in range(3))


def hardisp(text, n):
    r = subprocess.run([W + '/hardisp', '2026', '1', '1', '0', '0', '0', str(n), '3600'],
                       input=text, capture_output=True, text=True).stdout
    v = np.array([[float(x) for x in l.split()] for l in r.splitlines() if l.strip()])
    return v[:, [0, 1, 2]]          # dU, dS, dW


def our_admittance(th, lam, A):
    """Admittances (342, 3) at a station from the stored surface fields."""
    k = np.array([np.interp(th, A['theta'], A['U'][i, c].real)
                  + 1j * np.interp(th, A['theta'], A['U'][i, c].imag)
                  for i in range(len(A['idd'])) for c in range(3)]).reshape(-1, 3)
    Z = np.zeros((len(k), 3), complex)
    for i, d in enumerate(A['idd']):
        mm = int(d[0])
        K = ct(mm, th) / ybar(2, mm, np.array([th]), 0)[0][0]
        e = np.exp(1j * mm * lam)
        Z[i] = (k[i, 0] * K * e, k[i, 1] * K * e, 1j * k[i, 2] * K * e)
    return Z


if __name__ == '__main__':
    stage = sys.argv[2] if len(sys.argv) > 2 else 'all'
    t = epochs()
    xs, xm = sun_moon(t)
    jd = utc_jd(t)
    idd, tamp = catalog()
    m = idd[:, 0]
    blq_rows = [np.where((idd == d).all(1))[0][0] for d in BLQ_DOODSON]
    out(f'Body tide 2026, hourly, {len(jd)} epochs; columns radial, south, west: rms/max in mm '
        f'(means removed)')
    A = np.load('results/station_admittance.npz') if stage in ('all', 'ours') else None
    for name, (lat, lon) in STATIONS.items():
        x = ecef(lat, lon)
        out(f'\n{name} (lat {lat:+.3f}, lon {lon:+.3f})')
        d2, th, lam = to_local(dehant('dehant_d2', x, t, xs, xm), x)
        full = to_local(dehant('dehant', x, t, xs, xm), x)[0]
        n3 = to_local(dehant('dehant_n3', x, t, xs, xm), x)[0]
        P2 = 1.5 * np.cos(th)**2 - 0.5
        h, l = 0.6078 - 0.0006 * P2, 0.0847 + 0.0002 * P2
        Zi = np.zeros((342, 3), complex)
        for mm in (0, 1, 2):
            Zi[m == mm] = np.array(body_admittance(th, lam, mm, h, l))
        u = synthesize(jd, Zi)
        out(f'  IERS degree 2, nominal (rms/max)      {stats(d2)}')
        out(f'  synthesis - IERS nominal degree 2     {stats(u - d2)}   mean '
            + ' '.join(f'{1e3 * v:+.2f}' for v in (u - d2).mean(0)) + ' (permanent tide)')
        sphi = np.cos(th)
        perm = -np.sqrt(5 / (4 * np.pi)) * 0.31460 * np.array([h * (1.5 * sphi**2 - 0.5),
                                                               -l * 3 * sphi * np.sin(th)])
        out(f'  permanent tide (IERS formula)  radial {1e3 * perm[0]:+.2f}  south {1e3 * perm[1]:+.2f} mm')
        out(f'  IERS degree 3                         {stats(full - n3)}')
        out(f'  IERS l1, out-of-phase, step 2         {stats(n3 - d2)}')
        hd = hardisp(blq(*(Zi[blq_rows].T[[0, 1, 2]])), len(jd))
        out(f'  HARDISP(nominal BLQ) - synthesis      {stats(hd - synthesize(jd, Zi, tt_utc=67.184))}')
        if A is None:
            continue
        Zo = our_admittance(th, lam, A)
        uo = synthesize(jd, Zo)
        hd = hardisp(blq(*(Zo[blq_rows].T)), len(jd))
        out(f'  PREM degree 2 (rms/max)               {stats(uo)}')
        out(f'  HARDISP(PREM BLQ) - PREM exact        {stats(hd - synthesize(jd, Zo, tt_utc=67.184))}')
        out(f'  PREM - IERS (degree 2 + IERS deg. 3) {stats(uo + (full - n3) - full)}')
    open('results/body_tide_check.txt', 'w').write('\n'.join(lines) + '\n')
