# -*- coding: utf-8 -*-
#
# Ekman layer below the CMB of PREM with a viscous core: solver profile vs the local Ekman solution.
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
# 18/10/2026 Machiel Bos
#===============================================================================
"""Ekman layer below the CMB of elastic hydrostatic PREM with a viscous core (delta = 1 km by default,
EarthModel.viscous_core: 40 km and 5 delta sub-layers), at psi1 (next to the FCN: large differential core
rotation), K = 12, N = 30. Circular components of the tangential displacement relative to the wall,
q+- = (u_theta - wall) +- (u_lambda - wall) (amplitudes (B, C) of Re[(A, B, -iC) e^(i(omega t + m lam))]),
against the local Ekman solution q = Q (1 - exp(-k zeta)), k delta = sqrt(i (sigma +- 2 cos theta)),
Q = the solver's value at zeta = 30 delta. Agreement within 10 delta (max |diff| / |Q|): 3-5 % at
colatitudes 45 and 80 deg for K = 12 (K 6: 5-15 %, K 9: 2-5 %; angular truncation of the
latitude-dependent layer). Paper figure fig_core_ekman (paper_pepi/make_fig_core.py).
At delta = 4 km K = 12 gives 10-13 %, K = 18 2-5 % (the 20 km inner layer changes nothing: angular
truncation again).
Usage: python scripts/ekman_profile.py [DELTA_KM [K]]  ->  results/ekman_profile[_d<DELTA_KM>_K<K>].{npz,txt}
(the suffix only when DELTA_KM is not 1 or K not 12)"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.basis import ybar
from provenance import Run

SIG, N = 1.00273, 30
DELTA = float(sys.argv[1]) * 1e3 if len(sys.argv) > 1 else 1.0e3
K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TH = np.radians([30.0, 45.0, 80.0])


def tangential(S, md, i, s, th):
    """u_theta and u_lambda amplitudes (B, C) at radii s (layer i), colatitudes th."""
    x = 2 * (s - md.r0[i]) / (md.r1[i] - md.r0[i]) - 1
    T = S._radial(i, x)[0][..., 0]
    ut = np.zeros((len(s), len(th)), complex)
    up = ut.copy()
    for j, (typ, n) in enumerate(S.fields):
        if typ not in 'VW':
            continue
        f = (S.coef[i * S._nb + j * S.N: i * S._nb + (j + 1) * S.N, None] * T).sum(0)[:, None]
        y1 = ybar(n, S.m, th, 1)[1]
        ys = S.m * ybar(n, S.m, th, 0, over_sin=True)[0]
        if typ == 'V':
            ut, up = ut + f * y1, up - f * ys
        else:
            ut, up = ut - f * ys, up + f * y1
    return ut, up


def main():
    with Run(f'P1 v2: Ekman layer below the CMB at psi1, delta {DELTA / 1e3:g} km, solver vs local solution',
             params=dict(sig=SIG, delta=DELTA, K=K, N=N, colatitudes_deg=np.degrees(TH).tolist()),
             inputs=['data/PREM.dat']) as run_:
        md = EarthModel('data/PREM.dat').viscous_core(SIG * OMEGA_EARTH, delta=DELTA)
        g = ClairautGeometry(md)
        S = TidalSolver(md, g, m=1, K=K, N=N, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=SIG * OMEGA_EARTH,
                        omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                        frame='mantle')
        S.solve()
        cmb = 3480.0e3
        top = [i for i in range(md.n_layers) if abs(md.r1[i] - cmb) < 1.0][0]
        lays = [i for i in range(md.n_layers) if md.r1[i] <= cmb + 1.0 and md.r0[i] >= cmb - 2.3e6]   # outer core
        z, ut, up, nodes = [], [], [], []
        for i in lays:
            s = np.linspace(md.r0[i], md.r1[i], 400)
            a, b = tangential(S, md, i, s, TH)
            z.append(cmb - s)
            ut.append(a)
            up.append(b)
            xk = np.cos(np.pi * np.arange(N) / (N - 1))       # Gauss-Lobatto nodes of the layer
            nodes.append(cmb - (md.r0[i] + (xk + 1) / 2 * (md.r1[i] - md.r0[i])))
        z, ut, up = np.concatenate(z), np.vstack(ut), np.vstack(up)
        o = np.argsort(z)
        z, ut, up = z[o], ut[o], up[o]
        qp, qm = (ut - ut[0]) + (up - up[0]), (ut - ut[0]) - (up - up[0])
        iq = np.argmin(abs(z - 30 * DELTA))
        lines = [f'psi1 ({SIG} cpsd), delta {DELTA / 1e3} km, K {K}, N {N}; layers (km) '
                 + ', '.join(f'{md.r0[i] / 1e3:.1f}-{md.r1[i] / 1e3:.1f}' for i in lays)]
        an = {}
        for k, t in enumerate(TH):
            for q, sg, nm in ((qp, 1, '+'), (qm, -1, '-')):
                kk = np.sqrt(1j * (SIG + sg * 2 * np.cos(t)) + 0j) / DELTA
                a = q[iq, k] * (1 - np.exp(-kk * z))
                an[(k, nm)] = a
                sel = z < 10 * DELTA
                lines.append(f'colatitude {np.degrees(t):4.0f} q{nm}: max |solver - Ekman| / |Q| (zeta < 10 delta) '
                             f'{np.max(abs(q[sel, k] - a[sel])) / abs(q[iq, k]):.3f}')
        out = 'results/ekman_profile' + ('' if DELTA == 1.0e3 and K == 12 else f'_d{DELTA / 1e3:g}_K{K}')
        run_.savez(out + '.npz', z=z, qp=qp, qm=qm, th=TH, delta=DELTA, sig=SIG, iq=iq,
                      ap=np.array([an[(k, '+')] for k in range(len(TH))]),
                      am=np.array([an[(k, '-')] for k in range(len(TH))]), nodes=np.concatenate(nodes))
        run_.write_text(out + '.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
