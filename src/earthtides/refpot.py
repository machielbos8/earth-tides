# -*- coding: utf-8 -*-
#
# Reference potential consistent with the model's own density.
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
# 5/10/2026 Machiel Bos
#===============================================================================
"""Reference potential consistent with the model's own density.

The momentum equation needs grad Phi0 and its second derivatives, with
Phi0 = phi0 (gravitation of the reference density) + psi (centrifugal).
The level-surface construction (W(s), Clairaut) gives them to first order in
the flattening only. For a rigid tilt of the Earth, the gravitational terms of
the momentum equation cancel against the potential perturbation phi1 only if
phi0 is exactly the potential that Poisson's equation produces for the
discretized density; otherwise an O(eps^2) residual force, multiplied by the
large nutation amplitude near the tilt-over mode (omega = Omega), leaks into
the deformation. Here phi0 is computed by solving

    lap phi0 = 4 pi G rho0(s)

on the same mapped grid (axisymmetric, degrees 0, 2, 4, ...), with phi0 and
its normal derivative continuous at the interfaces and matched to
sum c_n (a/r)^(n+1) P_n outside; psi = -Omega^2 r^2 sin^2(theta) / 2 is exact.
"""
import numpy as np
import scipy.linalg as sla

from .geometry import mapping, mapping_from_R, chain
from .model import G


class ReferencePotential:
    def __init__(self, model, geometry, omega_rot, K=4, N=24, n_theta=40):
        from .basis import cheb_basis, ybar
        from numpy.polynomial import legendre as L
        self.model, self.geom, self.Om = model, geometry, omega_rot
        self.K, self.N = K, N
        self.deg = [2 * k for k in range(K)]
        x, w = L.leggauss(n_theta)
        th = np.arccos(x)
        self.th, self.w = th, w
        Y = {n: ybar(n, 0, th, 2) for n in self.deg}
        self.Y = Y
        test = np.array([Y[n][0] * w for n in self.deg])
        md, nL = model, model.n_layers
        nb = K * N
        ncol = nL * nb + K
        H = np.zeros((ncol, ncol))
        f = np.zeros(ncol)
        col = lambda i, j: i * nb + j * N
        xin = -np.cos(np.pi * np.arange(1, N - 1) / (N - 1))
        a = md.a

        def derivs(i, xx):
            s = md.r0[i] + (np.atleast_1d(xx) + 1) * (md.r1[i] - md.r0[i]) / 2
            mp = self._map(i, s[:, None], th[None, :])
            T, T1, T2 = cheb_basis(xx, N)
            sc = 2.0 / (md.r1[i] - md.r0[i])
            A0, A1, A2 = T[:, :, None], (T1 * sc)[:, :, None], (T2 * sc**2)[:, :, None]
            out = []
            for n in self.deg:
                b0, b1, b2 = Y[n][0], Y[n][1], Y[n][2]
                out.append(chain(A0 * b0, A1 * b0, A2 * b0, A0 * b1, A1 * b1, A0 * b2, mp))
            return s, mp, out

        for i in range(nL):
            r0, r1 = md.r0[i], md.r1[i]
            row0 = i * nb
            s, mp, D = derivs(i, xin)
            r = mp['R']
            cot = 1 / np.tan(th)
            for j, d in enumerate(D):
                lap = d['rr'] + 2 * d['r'] / r + (d['tt'] + cot * d['t']) / r**2
                Q = np.einsum('nsj,kj->nsk', lap, test)
                for k in range(N - 2):
                    H[row0 + K * (1 + k): row0 + K * (2 + k), col(i, j):col(i, j) + N] = Q[:, k, :].T
            src = 4 * np.pi * G * md.rho[i](s)[:, None] * np.ones_like(th)[None, :]
            Qs = src @ test.T
            for k in range(N - 2):
                f[row0 + K * (1 + k): row0 + K * (2 + k)] = Qs[k]
            rb = row0
            if i == 0:
                T, T1, _ = cheb_basis(-1.0, N)
                for j in range(K):
                    H[rb + j, col(i, j):col(i, j) + N] = T1[:, 0]
            else:
                Tm, _, _ = cheb_basis(-1.0, N)
                Tp, _, _ = cheb_basis(1.0, N)
                for j in range(K):
                    H[rb + j, col(i - 1, j):col(i - 1, j) + N] = Tp[:, 0]
                    H[rb + j, col(i, j):col(i, j) + N] = -Tm[:, 0]
            rt = row0 + nb - K
            last = (i == nL - 1)

            def flux(ii, xx):
                _, mp_, D_ = derivs(ii, np.array([xx]))
                R_, Rt_ = mp_['R'], mp_['R_t']
                h = np.hypot(R_, Rt_)
                nr, nt = R_ / h, -Rt_ / h
                return [((d['r'] * nr + d['t'] / R_ * nt)[:, 0, :] @ test.T).T for d in D_], mp_
            Fl, mpt = flux(i, 1.0)
            for j in range(K):
                H[rt:rt + K, col(i, j):col(i, j) + N] = Fl[j]
            if not last:
                Fn, _ = flux(i + 1, -1.0)
                for j in range(K):
                    H[rt:rt + K, col(i + 1, j):col(i + 1, j) + N] -= Fn[j]
            else:
                R_, Rt_ = mpt['R'][0], mpt['R_t'][0]
                h = np.hypot(R_, Rt_)
                nr, nt = R_ / h, -Rt_ / h
                Tp, _, _ = cheb_basis(1.0, N)
                for jj, n in enumerate(self.deg):          # exterior c_n, flux rows
                    E = (a / R_)**(n + 1) * Y[n][0]
                    Er = -(n + 1) / R_ * E
                    Et = (a / R_)**(n + 1) * Y[n][1] / R_
                    H[rt:rt + K, nL * nb + jj] = -((Er * nr + Et * nt) @ test.T)
                    # continuity of phi0: extra rows
                    H[nL * nb:nL * nb + K, nL * nb + jj] = -(E @ test.T)
                for jj, n in enumerate(self.deg):
                    phin = Tp[:, 0][:, None] * Y[n][0][None, :]
                    H[nL * nb:nL * nb + K, col(i, jj):col(i, jj) + N] = (phin @ test.T).T
        rs = np.abs(H).max(axis=1)
        Hs = H / rs[:, None]
        cs = np.abs(Hs).max(axis=0)
        cs[cs == 0] = 1
        self.coef = sla.solve(Hs / cs, f / rs) / cs
        self.nb, self.derivs = nb, derivs

    def _map(self, i, s, th):
        if hasattr(self.geom, 'map_derivs'):
            return mapping_from_R(*self.geom.map_derivs(i, s, th))
        return mapping(s, th, *self.geom.eps(i, s))

    def phys_derivs(self, i, s, theta, mp):
        """Physical r/theta derivatives of Phi0 = phi0 + psi at (s, theta)."""
        from .basis import cheb_basis, ybar
        md, N = self.model, self.N
        xx = 2 * (np.atleast_1d(s) - md.r0[i]) / (md.r1[i] - md.r0[i]) - 1
        T, T1, T2 = cheb_basis(xx, N)
        sc = 2.0 / (md.r1[i] - md.r0[i])
        tot = {k: 0.0 for k in ('r', 't', 'rr', 'rt', 'tt')}
        for j, n in enumerate(self.deg):
            c = self.coef[i * self.nb + j * N: i * self.nb + (j + 1) * N]
            A0, A1, A2 = (c @ T)[:, None], (c @ T1 * sc)[:, None], (c @ T2 * sc**2)[:, None]
            b = ybar(n, 0, theta, 2)
            d = chain(A0 * b[0], A1 * b[0], A2 * b[0], A0 * b[1], A1 * b[1], A0 * b[2], mp)
            for k in tot:
                tot[k] = tot[k] + d[k]
        r = mp['R']
        st, ct, W2 = np.sin(theta), np.cos(theta), self.Om**2
        tot['r'] = tot['r'] - W2 * r * st**2
        tot['t'] = tot['t'] - W2 * r**2 * st * ct
        tot['rr'] = tot['rr'] - W2 * st**2
        tot['rt'] = tot['rt'] - 2 * W2 * r * st * ct
        tot['tt'] = tot['tt'] - W2 * r**2 * np.cos(2 * theta)
        return tot
