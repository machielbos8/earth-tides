# -*- coding: utf-8 -*-
#
# Paper 1, Online Resource 1 (offsets with Wahr's theory): fingerprints (dh0, dl0, dk) of changes of the
# reference state, compared with the offset of our Love numbers from those of Dehant et al. (1999).
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
# 15/10/2026 Machiel Bos
#===============================================================================
"""Which change of the reference state explains ours - DDW in h0, l0 and k?

Elastic hydrostatic PREM, set-up of scripts/run_diurnal.py (fluid core, Coriolis, inertia, consistent
reference potential, mantle frame; k in the DDW normalisation). A single number can always be tuned to
close h0 at one line; the test is whether the same change also gives dl0 and dk. Each candidate changes
ONE thing at M2:
  no centrifugal   reference gravity without the centrifugal term (omega_rot = 0, no consistent
                   reference; the consistent reference alone changes h0 by 2e-6)
  g x (1+1e-3)     the whole reference gravity (all derivatives of Phi0) scaled
  rock, mass kept  top 15 km rho 2.6 (ocean -> rock), 15-24.4 km density lowered to keep the mass;
                   velocities rescaled so that kappa and mu do not change
  rock, mass added top 15 km rho 2.6, nothing else (the +5e-4 of Sect. 2.4)
  mu, kappa        top-layer moduli x 0.9
Millimetres: largest amplitude over latitude of the radial (dh0) and horizontal (dl0) displacement for
the Tamura amplitude of M2 and O1.
Usage: python scripts/ddw_fingerprint.py [scratch dir for model files]  ->  results/ddw_fingerprint.txt
"""
import os
import re
import sys
import tempfile
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.iers import iers_fit
from earthtides.etide import Tamura, W_functions

FR = {'M2': 1.4053e-4 / OMEGA_EARTH, 'O1': 0.92700, 'OO1': 1.07300}
CPD = {'M2': 1.932274, 'O1': 0.929536}
# Dehant et al. (1999), Tables 11, 12 (elastic hydrostatic PREM): h0, l0, k
DDW = {'M2': (0.60175, 0.08359, 0.29751), 'O1': (0.59553, 0.08298, 0.29335),
       'OO1': (0.60092, 0.08283, 0.29607)}
PREM = 'data/PREM.dat'
A_KM = 6371.0
lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


class GScaled(TidalSolver):
    """Reference gravity (all derivatives of Phi0) multiplied by gfac."""
    gfac = 1.0

    def props(self, i, s):
        p = super().props(i, s)
        p['ph0'] = {k: v * self.gfac for k, v in p['ph0'].items()}
        return p


def run(model=PREM, tide='M2', cls=TidalSolver, gfac=1.0, omega_rot=OMEGA_EARTH, cons=True):
    md = EarthModel(model)
    g = ClairautGeometry(md)
    m = 2 if tide == 'M2' else 1
    cls.gfac = gfac
    S = cls(md, g, m=m, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=FR[tide] * OMEGA_EARTH,
            omega_rot=omega_rot, wahr_frame=(m != 2), dyn_fluid=True, consistent_reference=cons,
            frame='mantle')
    r = S.solve()
    f = iers_fit(S, g)
    return np.array([f['h0'].real, f['l0'].real, (r['kk'] * (1 + g.flattening_surface / 3) ** -5).real])


def edit_model(changes, path):
    """changes: {depth label: (rho_new or None, mu factor, kappa factor)}; constant layers only."""
    txt = open(PREM).read().split('\n')
    vals, sec = {}, None
    for ln in txt:
        if re.match(r'^[A-Za-z]', ln.strip() or ' ') and '|' not in ln:
            sec = ln.strip()
        p = ln.split()
        if sec and p and p[0] in changes and len(p) == 5:
            vals[(p[0], sec)] = float(p[1])
    new = {}
    for d, (rho, fmu, fkap) in changes.items():
        r0 = vals[(d, 'rho')]
        r1 = rho if rho is not None else r0
        for h in ('h', 'v'):
            vs2, vp2 = vals[(d, 'Vs_' + h)]**2, vals[(d, 'Vp_' + h)]**2
            mu, kap = r0 * vs2, r0 * (vp2 - 4 / 3 * vs2)
            new[(d, 'Vs_' + h)] = np.sqrt(fmu * mu / r1)
            new[(d, 'Vp_' + h)] = np.sqrt((fkap * kap + 4 / 3 * fmu * mu) / r1)
        new[(d, 'rho')] = r1
    res, sec = [], None
    for ln in txt:
        if re.match(r'^[A-Za-z]', ln.strip() or ' ') and '|' not in ln:
            sec = ln.strip()
        p = ln.split()
        if sec and p and (p[0], sec) in new and len(p) == 5:
            ln = f'{p[0]:>6s}   {new[(p[0], sec)]:8.5f}' + ''.join(f'   {float(x):8.5f}' for x in p[2:])
        res.append(ln)
    open(path, 'w').write('\n'.join(res))
    return path


def shell(d_out, d_in):
    """r^3 difference of the shell between depths d_out < d_in (km)."""
    return (A_KM - d_out)**3 - (A_KM - d_in)**3


def mm_scale(tide):
    """Largest radial (per unit h) and horizontal (per unit l) amplitude over latitude, mm."""
    cat = Tamura()
    m = 2 if tide == 'M2' else 1
    s = np.flatnonzero((cat.n == 2) & (cat.m == m))
    A = abs(cat.A[s[np.argmin(abs(cat.f_cpd[s] - CPD[tide]))]])
    phi = np.radians(np.linspace(-89.5, 89.5, 359))
    W, dW = (np.array(x) for x in zip(*[W_functions(2, m, p) for p in phi]))
    hor = np.hypot(dW, m * W / np.cos(phi))
    return 1e3 * A * np.abs(W).max(), 1e3 * A * hor.max()


def main():
    tmp = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    base = {t: run(tide=t) for t in FR}
    out('Ours - Dehant et al. (1999), elastic hydrostatic PREM (1e-5):   dh0     dl0      dk')
    for t in FR:
        out(f'  {t:4s}' + ' ' * 54 + ''.join(f'{x:+8.1f}' for x in 1e5 * (base[t] - np.array(DDW[t]))))
    rho2 = 2.900 - (2.6 - 2.284) * shell(0.0, 15.0) / shell(15.0, 24.4)
    dmass = 4 * np.pi / 3 * (2.6 - 2.284) * 1e3 * shell(0.0, 15.0) * 1e9 / EarthModel(PREM).M
    cases = [('no centrifugal term in the reference gravity', dict(omega_rot=0.0, cons=False)),
             ('reference gravity x (1 + 1e-3)', dict(cls=GScaled, gfac=1.001)),
             (f'top 15 km rock (2600), 15-24.4 km rho {1e3 * rho2:.0f}: mass kept',
              dict(model=edit_model({'15.0': (2.6, 1, 1), '24.4': (rho2, 1, 1)}, os.path.join(tmp, 'kept.dat')))),
             (f'top 15 km rock (2600): mass +{100 * dmass:.2f} per cent',
              dict(model=edit_model({'15.0': (2.6, 1, 1)}, os.path.join(tmp, 'added.dat')))),
             ('shear modulus of the top 15 km x 0.9',
              dict(model=edit_model({'15.0': (None, 0.9, 1)}, os.path.join(tmp, 'mu.dat')))),
             ('bulk modulus of the top 15 km x 0.9',
              dict(model=edit_model({'15.0': (None, 1, 0.9)}, os.path.join(tmp, 'kappa.dat'))))]
    out('')
    out('Change at M2 when one thing is changed (1e-5):                     dh0     dl0      dk'
        '   dl0/dh0  dk/dh0')
    target = 1e5 * (np.array(DDW['M2']) - base['M2'])
    out(f"  {'DDW - ours (to be explained)':62s}" + ''.join(f'{x:+8.1f}' for x in target)
        + f'   {target[1] / target[0]:+7.3f} {target[2] / target[0]:+7.3f}')
    dcf = None
    for name, kw in cases:
        d = 1e5 * (run(tide='M2', **kw) - base['M2'])
        dcf = d if dcf is None else dcf
        out(f'  {name:62s}' + ''.join(f'{x:+8.1f}' for x in d) + f'   {d[1] / d[0]:+7.3f} {d[2] / d[0]:+7.3f}')
    rem = target - dcf
    out(f"  {'remainder after removing the centrifugal term':62s}" + ''.join(f'{x:+8.1f}' for x in rem))
    out('')
    out('Millimetres (largest over latitude; radial from dh0, horizontal from dl0):')
    for t in ('M2', 'O1'):
        R, H = mm_scale(t)
        d = base[t] - np.array(DDW[t])
        out(f'  {t}: offset ours - DDW  radial {abs(d[0]) * R:.3f} mm, horizontal {abs(d[1]) * H:.3f} mm'
            + (f';  remainder radial {abs(rem[0]) * 1e-5 * R:.3f} mm, horizontal {abs(rem[1]) * 1e-5 * H:.3f} mm'
               if t == 'M2' else ''))
    open('results/ddw_fingerprint.txt', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
