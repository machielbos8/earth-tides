# -*- coding: utf-8 -*-
#
# Loading Green's functions of a flattened Earth from the load Love matrix.
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
# 6/10/2026 Machiel Bos
#===============================================================================
"""Loading Green's functions of a flattened Earth from the load Love matrix.

The load Love matrix L_m(n, n') (scripts/load_love_table.py) gives the response of
the flattened Earth to a unit surface density Ybar_n'^m cos(m lam) [kg/m^2 of the
actual surface]. A point mass M at (theta_l, lam_l) is a surface density
M delta / dA; its coefficients in Ybar_n'^m cos m(lam - lam_l) are

    s_n'm = M Ybar_n'^m(theta_l) / (dA/dOmega)(theta_l) / N_n'm,
    N_n'm = int (Ybar_n'^m cos m lam)^2 dOmega = 4 pi / (2 n' + 1)   (Schmidt),

with dA/dOmega = R sqrt(R^2 + R_theta^2) on the outer surface. Each term produces the
displacement field of eq. (fields) with U_n = h'_n Phi_L / g etc., Phi_L = 4 pi G a /
(2 n' + 1). The displacement is evaluated at the station on the flattened surface and
projected on the ellipsoidal normal (up) and meridian (north); east is u_lambda.

Rigid-body terms are left out: spheroidal degree 1 (translation, frame) and toroidal
degree 1 (rotation). Load degrees 0 and 1 are not in the table either. On a sphere
(dA/dOmega = a^2, diagonal Love numbers) the sum is Farrell's
u = (G / (a g)) sum h'_n P_n(cos psi) per unit mass (with the same truncation).

All angles are geocentric colatitude theta and longitude lambda, in radians.
"""
import numpy as np

from .geometry import ClairautGeometry, mapping
from .model import EarthModel, G
from .basis import ybar


class LoadGreen:
    def __init__(self, table, model_file, nmax=None):
        d = np.load(table)
        self.H, self.L, self.T = d['H'], d['L'], d['T']
        self.sph = {k: (d[f'h_s{k}'], d[f'l_s{k}']) for k in (0, 1)}
        self.nmax = int(d['nmax']) if nmax is None else nmax
        md = EarthModel(model_file)
        self.model_file = model_file
        self.a, self.g = md.a, md.g_surface
        self.geom = ClairautGeometry(md)
        self.ilast = md.n_layers - 1

    # -- geometry of the outer surface --------------------------------------
    def surface(self, theta):
        th = np.atleast_1d(np.asarray(theta, float))
        s = np.full((1,) + th.shape, self.a)
        mp = mapping(s, th[None, :], *self.geom.eps(self.ilast, s))
        R, Rt = mp['R'][0], mp['R_t'][0]
        h = np.hypot(R, Rt)
        return R, Rt, R / h, -Rt / h, R * h            # R, R_theta, n_r, n_theta, dA/dOmega

    def normal_gravity(self, theta, omega_rot=None):
        """Normal gravity grad(Phi0).n on the outer surface of the flattened, rotating
        model (first-order level-surface reference, centrifugal force included)."""
        from .geometry import OMEGA_EARTH
        from .solver import TidalSolver
        if not hasattr(self, '_gn'):
            md = EarthModel(self.model_file)
            S = TidalSolver(md, self.geom, m=2, K=1, N=8, omega_f=0.0, n_theta=361,
                            omega_rot=OMEGA_EARTH if omega_rot is None else omega_rot)
            p = S.props(self.ilast, np.array([self.a]))
            nr, nt = S.normal(p)
            gn = (p.ph0['r'] * nr + p.ph0['t'] / p.r * nt)[0]
            th = S.theta[0, 0]
            o = np.argsort(th)
            self._gn = (th[o], gn[o])
        return np.interp(theta, *self._gn)

    # -- coefficients ---------------------------------------------------------
    def _load_coef(self, theta_l, model):
        """s_n'm * Phi_L / g for a unit mass at colatitude theta_l: dict (m, n') -> array."""
        if model == 'ell':
            dA = self.surface(theta_l)[4]
        else:
            dA = self.a**2 + 0 * np.atleast_1d(theta_l)
        out = {}
        for m in range(self.nmax + 1):
            for n in range(max(m, 2), self.nmax + 1):
                Y = ybar(n, m, np.atleast_1d(theta_l), 0)[0]
                out[m, n] = Y / dA / (4 * np.pi / (2 * n + 1)) * (4 * np.pi * G * self.a / (2 * n + 1)) / self.g
        return out

    def station_terms(self, theta_s, model='ell'):
        """For each (m, n'): the station displacement (r, theta, lambda-amplitudes) per unit
        load coefficient, as arrays [ur_c, ut_c, ul_s] multiplying cos, cos, sin(m dlam).
        model: 'ell' (flattened), 'sph0' (sphere, no rotation), 'sph1' (sphere, rotating ref.)."""
        th = np.array([theta_s], float)
        terms = {}
        isn = 1 / np.sin(th[0])
        Ycache = {}

        def Y(n, m):
            if (n, m) not in Ycache:
                y = ybar(n, m, th, 1)
                Ycache[n, m] = (y[0][0], y[1][0])
            return Ycache[n, m]
        for m in range(self.nmax + 1):
            for nf in range(max(m, 2), self.nmax + 1):
                ur = ut = ul = 0.0
                if model == 'ell':
                    for j in range(5):
                        n = nf + 2 * (j - 2)
                        if n < m or n < 0 or n == 1 or np.isnan(self.H[m, nf, j]):
                            continue
                        y0, y1 = Y(n, m)
                        ur += self.H[m, nf, j] * y0
                        lv = self.L[m, nf, j]
                        if n >= 1 and not np.isnan(lv):
                            ut += lv * y1
                            ul += -lv * m * y0 * isn
                    for j in range(6):
                        n = nf + 2 * j - 5
                        tv = self.T[m, nf, j]
                        if n < max(m, 2) or np.isnan(tv):
                            continue
                        y0, y1 = Y(n, m)
                        ut += -tv * m * y0 * isn
                        ul += tv * y1
                else:
                    hs, ls = self.sph[int(model[-1])]
                    y0, y1 = Y(nf, m)
                    ur, ut, ul = hs[nf] * y0, ls[nf] * y1, -ls[nf] * m * y0 * isn
                terms[m, nf] = np.array([ur, ut, ul])
        return terms

    def green(self, theta_s, theta_l, dlam, model='ell', terms=None, coef=None):
        """(up, north, east) displacement [m per kg] at the station for unit masses at
        (theta_l[i], station longitude - dlam[i]) -> arrays (3, len)."""
        theta_l = np.atleast_1d(theta_l)
        dlam = np.atleast_1d(dlam)
        terms = self.station_terms(theta_s, model) if terms is None else terms
        coef = self._load_coef(theta_l, 'ell' if model == 'ell' else 'sph') if coef is None else coef
        ur = ut = ul = 0.0
        for (m, n), t in terms.items():
            c = coef[m, n]
            ur = ur + c * t[0] * np.cos(m * dlam)
            ut = ut + c * t[1] * np.cos(m * dlam)
            ul = ul + c * t[2] * np.sin(m * dlam)
        if model == 'ell':
            _, _, nr, nt, _ = self.surface(theta_s)
            nr, nt = nr[0], nt[0]
        else:
            nr, nt = 1.0, 0.0
        up = ur * nr + ut * nt
        north = -(-ur * nt + ut * nr)
        return np.array([up, north, ul * np.ones_like(up)])
