# -*- coding: utf-8 -*-
#
# Level-surface geometry r = s [1 - (2/3) eps(s) P2(cos theta)].
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
"""Level-surface geometry r = s [1 - (2/3) eps(s) P2(cos theta)].

s is the mean radius of the surface. Each geometry object returns eps, eps'
and eps'' for a given layer and s; ``mapping`` turns those into the
derivatives of R(s, theta) and of its inverse S(r, theta) that the chain rule
in the solver needs.

* SphereGeometry      eps = 0
* ClairautGeometry    hydrostatic flattening from Clairaut's equation
* BumpGeometry        artificial eps that vanishes at every interface; used
                      with a *spherical* Earth to test the mesh machinery
"""
import numpy as np
from scipy.integrate import solve_ivp

from .model import G

OMEGA_EARTH = 7.292115e-5


class SphereGeometry:
    def eps(self, i, s):
        z = np.zeros_like(np.asarray(s, float))
        return z, z, z


class ClairautGeometry:
    """First-order hydrostatic flattening of the level surfaces.

    eps'' = -(6 rho / (rho_bar s)) eps' - (6/s^2)(rho/rho_bar - 1) eps,
    rho_bar = 3 M(s) / (4 pi s^3), scaled so that at the surface
    a eps' + 2 eps = (5/2) m,  m = omega^2 a^3 / (G M).
    ``scale`` multiplies eps (scale=0 gives a sphere).
    """

    def __init__(self, model, omega=OMEGA_EARTH, scale=1.0, s_start=1.0e3):
        self.model, self.omega, self.scale = model, omega, scale
        self._sol = []
        y = np.array([1.0, 0.0])
        for i in range(model.n_layers):
            s0 = max(model.r0[i], s_start)
            sol = solve_ivp(lambda s, y, i=i: self._rhs(i, s, y),
                            (s0, model.r1[i]), y, method='DOP853',
                            rtol=1e-12, atol=1e-14, dense_output=True)
            self._sol.append(sol.sol)
            y = sol.y[:, -1]
        a = model.a
        m = omega**2 * a**3 / (G * model.M)
        self.m = m
        self.norm = 2.5 * m / (a * y[1] + 2.0 * y[0])
        self.flattening_surface = self.norm * y[0]

    def _coef(self, i, s):
        rho = self.model.rho[i](s)
        rho_bar = 3.0 * self.model.mass(i, s) / (4.0 * np.pi * s**3)
        return rho, rho_bar

    def _rhs(self, i, s, y):
        rho, rho_bar = self._coef(i, s)
        e, de = y
        dde = -6.0 * rho / (rho_bar * s) * de - 6.0 / s**2 * (rho / rho_bar - 1.0) * e
        return [de, dde]

    def eps(self, i, s):
        s = np.asarray(s, float)
        # below s_start (centre) eps is constant to O(s^2)
        s_eval = np.clip(s, self._sol[i].t_min, self._sol[i].t_max)
        y = self._sol[i](s_eval.ravel())
        e, de = y[0].reshape(s.shape), y[1].reshape(s.shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho, rho_bar = self._coef(i, s_eval)
            dde = -6.0 * rho / (rho_bar * s_eval) * de \
                - 6.0 / s_eval**2 * (rho / rho_bar - 1.0) * e
        k = self.norm * self.scale
        return k * e, k * de, k * dde


class BumpGeometry:
    """eps = A_i sin^2(pi (s - r0)/(r1 - r0)) inside each layer."""

    def __init__(self, model, rel_amp=0.2):
        self.model = model
        # keep the radial excursion |r - s| below rel_amp * layer thickness
        L = model.r1 - model.r0
        smid = 0.5 * (model.r0 + model.r1)
        self.A = 1.5 * rel_amp * L / smid

    def eps(self, i, s):
        r0, r1 = self.model.r0[i], self.model.r1[i]
        k = np.pi / (r1 - r0)
        x = k * (np.asarray(s, float) - r0)
        A = self.A[i]
        return A * np.sin(x)**2, A * k * np.sin(2 * x), 2 * A * k * k * np.cos(2 * x)


class ExactSpheroidGeometry:
    """Similar, exact spheroids r = s f(theta), with flattening fl at every s.

    f(theta) = 1/sqrt(sin^2/A^2 + cos^2/C^2), A = (1-fl)^(-1/3), C = A (1-fl),
    so that s is the volume-equivalent radius. Used for the homogeneous test
    body; unlike eq. (map) it is exact at all orders in the flattening.
    """

    def __init__(self, flattening):
        self.fl = flattening
        self.A = (1 - flattening) ** (-1.0 / 3.0)
        self.C = self.A * (1 - flattening)

    def map_derivs(self, i, s, theta):
        k = 1 / self.A**2 - 1 / self.C**2
        q = np.sin(theta)**2 / self.A**2 + np.cos(theta)**2 / self.C**2
        q1 = np.sin(2 * theta) * k
        q2 = 2 * np.cos(2 * theta) * k
        f = q**-0.5
        f1 = -0.5 * q**-1.5 * q1
        f2 = 0.75 * q**-2.5 * q1**2 - 0.5 * q**-1.5 * q2
        z = 0 * s
        return s * f, f + z, 0 * (s * f), s * f1, f1 + z, s * f2

    def eps(self, i, s):                     # not used when map_derivs exists
        raise NotImplementedError


def mapping(s, theta, e, de, dde):
    """R(s,theta) and the derivatives of the inverse map S(r,theta).

    All inputs broadcast. Returns a dict with R, R_t (=dR/dtheta at fixed s)
    and s_r, s_t, s_rr, s_rt, s_tt (derivatives of S at fixed theta / r).
    """
    c, sn = np.cos(theta), np.sin(theta)
    P2 = 0.5 * (3 * c * c - 1)
    dP2 = -3.0 * c * sn
    ddP2 = -3.0 * (c * c - sn * sn)
    f = 2.0 / 3.0
    R = s * (1 - f * e * P2)
    R_s = 1 - f * (e + s * de) * P2
    R_ss = -f * (2 * de + s * dde) * P2
    R_t = -f * s * e * dP2
    R_st = -f * (e + s * de) * dP2
    R_tt = -f * s * e * ddP2
    return mapping_from_R(R, R_s, R_ss, R_t, R_st, R_tt)


def mapping_from_R(R, R_s, R_ss, R_t, R_st, R_tt):
    """Inverse-map derivatives from R(s, theta) and its derivatives."""
    s_r = 1.0 / R_s
    s_t = -R_t / R_s
    s_rr = -R_ss * s_r**3
    s_rt = -(R_st + R_ss * s_t) / R_s**2
    s_tt = -((R_tt + R_st * s_t) * R_s - R_t * (R_st + R_ss * s_t)) / R_s**2
    return dict(R=R, R_t=R_t, s_r=s_r, s_t=s_t, s_rr=s_rr, s_rt=s_rt, s_tt=s_tt)


def chain(F, Fs, Fss, Ft, Fst, Ftt, m):
    """Physical derivatives of F(S(r,theta),theta) at fixed r or theta."""
    s_r, s_t = m['s_r'], m['s_t']
    return dict(v=F,
                r=Fs * s_r,
                t=Ft + Fs * s_t,
                rr=Fss * s_r**2 + Fs * m['s_rr'],
                rt=Fss * s_r * s_t + Fst * s_r + Fs * m['s_rt'],
                tt=Ftt + 2 * Fst * s_t + Fss * s_t**2 + Fs * m['s_tt'])
