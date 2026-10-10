# -*- coding: utf-8 -*-
#
# Angular-momentum balance of the whole Earth in the diurnal band: a check of the rotational
# dynamics (inertia, Coriolis force, fluid core, flattening) of the solver.
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
"""Euler's equation dH/dt + Omega x H = N for the whole Earth, in the frame rotating with Omega.

Lagrangian, linearized: H = int rho0 [X x du/dt + X x (Omega x u) + u x (Omega x X)] dV0 and the
tidal torque N = -int rho0 X x grad(phiT) dV0 (internal forces exert no net torque). The fields
are Re[(A, B, -iC) exp(i(omega t + lambda))] (TidalSolver, m = 1); H and N are evaluated at
omega t = 0 and pi/2 to obtain complex amplitudes, with dH/dt = i omega H. Near the tilt-over
mode H grows as 1/(omega - Omega) while N stays finite, so the balance is a strict test.
Usage: python scripts/check_angular_momentum.py
"""
import sys
import warnings

sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
from numpy.polynomial import legendre as L
from earthtides.model import EarthModel
from earthtides.love import model_path
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.basis import ybar

NLAM = 24


def fields(S):
    """Generator over layers: weights dV (s, theta), position R, theta and the complex amplitude
    functions A, B, C at the quadrature nodes."""
    md, N, nb = S.model, S.N, S.nfld * S.N
    xg, wg = L.leggauss(N + 6)
    for i in range(md.n_layers):
        r0, r1 = max(md.r0[i], 1.0), md.r1[i]
        s = r0 + (xg + 1) * (r1 - r0) / 2
        p = S.props(i, s)
        dV = p.r**2 / p.s_r * (wg * (r1 - r0) / 2)[:, None] * S.w[None, :]   # r^2 dr dmu (per dlambda)
        A = B = C = 0.0
        for j, (uA, uB, uC, _) in enumerate(S._fields_at(i, xg, p)):
            c = S.coef[i * nb + j * N: i * nb + (j + 1) * N]
            sh = (N,) + p.r.shape
            A = A + np.tensordot(c, np.broadcast_to(uA.v, sh), axes=1)
            B = B + np.tensordot(c, np.broadcast_to(uB.v, sh), axes=1)
            C = C + np.tensordot(c, np.broadcast_to(uC.v, sh), axes=1)
        yield p, dV, A, B, C


def balance(S, omega):
    Om = np.array([0.0, 0.0, S.coriolis])
    md = S.model
    a, ga = md.a, md.g_surface
    lam = 2 * np.pi * np.arange(NLAM) / NLAM
    H = np.zeros((2, 3))
    Nt = np.zeros((2, 3))
    for p, dV, A, B, C in fields(S):
        r = p.r[..., None]
        th = S.theta[0, 0][None, :, None] + 0 * r
        st, ct = np.sin(th), np.cos(th)
        sl, cl = np.sin(lam), np.cos(lam)
        er = np.stack([st * cl, st * sl, ct + 0 * cl])
        et = np.stack([ct * cl, ct * sl, -st + 0 * cl])
        el = np.stack([-sl + 0 * st, cl + 0 * st, 0 * st * cl])
        X = r * er
        rho = p.rho[..., None]
        w = (dV[..., None] * 2 * np.pi / NLAM) * rho
        Yf = ybar(2, 1, S.theta[0, 0], 1)
        for k, wt in enumerate((0.0, np.pi / 2)):
            ph = np.exp(1j * (wt + lam))
            ur, uth, ul = (np.real(A[..., None] * ph), np.real(B[..., None] * ph),
                           np.real(-1j * C[..., None] * ph))
            vr, vth, vl = (np.real(1j * omega * A[..., None] * ph), np.real(1j * omega * B[..., None] * ph),
                           np.real(omega * C[..., None] * ph))
            u = ur * er + uth * et + ul * el
            v = vr * er + vth * et + vl * el
            Omu = np.cross(Om, u, axis=0)
            OmX = np.cross(Om, X, axis=0)
            dens = np.cross(X, v, axis=0) + np.cross(X, Omu, axis=0) + np.cross(u, OmX, axis=0)
            H[k] += np.sum(dens * w, axis=(1, 2, 3))
            # tidal potential phiT = Re[-g a (r/a)^2 Ybar21 exp(i(omega t + lambda))]
            Y, Yt = Yf[0][None, :, None], Yf[1][None, :, None]
            gr = np.real(-ga * 2 * r / a * Y * ph)
            gth = np.real(-ga * r / a * Yt * ph)
            gl = np.real(-ga * r / a * Y / st * 1j * ph)
            grad = gr * er + gth * et + gl * el
            Nt[k] += np.sum(-np.cross(X, grad, axis=0) * w, axis=(1, 2, 3))
    Hc = H[0] - 1j * H[1]
    Nc = Nt[0] - 1j * Nt[1]
    lhs = 1j * omega * Hc + np.cross(Om, Hc)
    return Hc, Nc, lhs


def main():
    md = EarthModel(str(model_path('PREM')))
    geo = ClairautGeometry(md)
    print('sigma [cpsd]   |H| Omega / |N|    |dH/dt + Omega x H - N| / |N|')
    for sig in (0.92700, 0.99454, 0.9990, 1.0002, 1.0015, 1.00546):
        w = sig * OMEGA_EARTH
        S = TidalSolver(md, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                        omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                        consistent_reference=True, frame='mantle')
        S.solve()
        Hc, Nc, lhs = balance(S, w)
        nN = np.linalg.norm(Nc)
        print(f'{sig:.5f}      {np.linalg.norm(Hc) * OMEGA_EARTH / nN:10.3e}       '
              f'{np.linalg.norm(lhs - Nc) / nN:.2e}')


if __name__ == '__main__':
    main()
