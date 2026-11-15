# -*- coding: utf-8 -*-
#
# Form of the extrapolation delta -> 0 of the viscous core (P1 review, M4).
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
# 15/11/2026 Machiel Bos
#===============================================================================
"""Referee (M4): the limit delta -> 0 is a quadratic through three deltas; the internal shear layers scale as
E^(1/4) .. E^(1/3) of the core radius, which adds terms in delta^(1/2) and delta^(2/3). Is a polynomial the
right form?

Data: results/viscous_limit_thin_Kthin_level.{txt,npz} (core K 72, level figure, delta 1000 .. 250 m, five
values; 175 m is left out, where the displacement formulation loses digits (supplement, 'The limit
delta -> 0'); Love numbers stored to 6 decimals, FCN frequency to 7). For each quantity three 3-parameter forms are
fitted by least squares to the five deltas:
   quad     a + b d + c d^2
   sqrt     a + b d + c d^(1/2)
   2/3      a + b d + c d^(2/3)
and we report the rms residual (relative to the change over the range) and the value at delta = 0. The form
that fits the six points best, and the spread of the three limits, tell how much the extrapolation depends
on the assumed form.

    .venv/bin/python scripts/delta_fit_forms.py   ->  results/delta_fit_forms.txt"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np

from provenance import Run

TAG = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == '--tag'), 'Kthin')   # Kfp: pressure unknown
KCORE = int(next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == '--K'), 72))
DMIN = float(next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == '--dmin'), 250.0))
TXT = ROOT / 'results' / f'viscous_limit_thin_{TAG}_level.txt'
NPZ = ROOT / 'results' / f'viscous_limit_thin_{TAG}_level.npz'
KEYS = ('h0', 'h2', 'l0', 'l1', 'l2', 'k0')
LINES = {'O1': 0.927, 'P1': 0.99454, 'psi1': 1.00273, 'phi1': 1.00546, 'OO1': 1.073}
FORMS = {'quad': lambda d: np.column_stack([1 + 0 * d, d, d**2]),
         'sqrt': lambda d: np.column_stack([1 + 0 * d, d, np.sqrt(d)]),
         '2/3': lambda d: np.column_stack([1 + 0 * d, d, d**(2 / 3)])}


def fcn_k72():
    t = TXT.read_text()
    blk = t[t.index(f'== core K {KCORE}'):]
    blk = blk[:blk.index('extrapolation')]
    d, s = [], []
    for m in re.finditer(r'delta\s+([\d.]+) m: FCN ([\d.]+) \+([\d.e+-]+)i', blk):
        d.append(float(m.group(1)))
        s.append(float(m.group(2)) + 1j * float(m.group(3)))
    return np.array(d), np.array(s)


def fit(d, y):
    x = d / 1000.0                                   # km
    out = {}
    for name, A in FORMS.items():
        M = A(x)
        c = np.linalg.lstsq(M, y, rcond=None)[0]
        res = y - M @ c
        out[name] = (c[0], np.sqrt(np.mean(np.abs(res)**2)))
    return out


def main(run):
    d, s = fcn_k72()
    keep = d >= DMIN
    d, s = d[keep], s[keep]
    lines = [f'delta -> 0 with three forms, {len(d)} deltas {d.max():.0f}..{d.min():.0f} m, core K {KCORE}, {TAG}, level figure',
             'rms: residual of the fit relative to span, the change of the quantity over these deltas', '']
    f = fit(d, s - 1)
    span = np.ptp((s - 1).real)
    lines.append(f'{"quantity":14s} {"span":>9s}   ' + '   '.join(f'{n:>5s}: limit / rms' for n in FORMS))
    per = lambda z: 1 / z.real
    lines.append(f'{"FCN (sd)":14s} {span:9.2e}   ' + '   '.join(
        f'{per(f[n][0]):.3f} / {f[n][1] / span:.1e}' for n in FORMS))
    lines.append(f'{"FCN Q":14s} {"":9s}   ' + '   '.join(
        f'{f[n][0].real / (2 * f[n][0].imag) if abs(f[n][0].imag) > 1e-12 else np.inf:9.0f}' for n in FORMS))
    z = np.load(NPZ)
    sweep = z['sweep']
    L = np.array([z[f'L_K{KCORE}_d{int(x)}'] for x in d])           # (ndelta, nfreq, nkey)
    for tide, fr in LINES.items():
        j = int(np.argmin(abs(sweep - fr)))
        for k, key in enumerate(KEYS):
            y = L[:, j, k].real
            sp = np.ptp(y)
            if sp < 2e-6:                                       # below the stored precision: no information
                continue
            f = fit(d, y)
            lines.append(f'{tide + " " + key:14s} {sp:9.2e}   ' + '   '.join(
                f'{f[n][0]:.6f} / {f[n][1] / sp:.1e}' for n in FORMS))
    print('\n'.join(lines))
    out = 'delta_fit_forms.txt' if TAG == 'Kthin' else f'delta_fit_forms_{TAG}_K{KCORE}.txt'
    run.write_text(str(ROOT / 'results' / out), '\n'.join(lines) + '\n')


if __name__ == '__main__':
    with Run('P1 review M4: form of the extrapolation delta -> 0 (quadratic vs delta^1/2, delta^2/3)',
             params=dict(K=KCORE, tag=TAG, dmin=DMIN, forms=list(FORMS)), inputs=[str(TXT.relative_to(ROOT)), str(NPZ.relative_to(ROOT))]) as run_:
        main(run_)
