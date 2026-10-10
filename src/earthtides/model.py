# -*- coding: utf-8 -*-
#
# Radial Earth model in the astyanax file format.
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
"""Radial Earth model in the astyanax file format.

Each layer stores rho, Vp_h, Vp_v, Vs_h, Vs_v and eta as Chebyshev series
(T0..T3) on [-1, 1] mapped onto the layer [r0, r1]. Units in the file are
g/cm^3 and km/s; here everything is SI. Transversely isotropic parameters are
replaced by their Voigt isotropic equivalents, exactly as astyanax computes
kappa and mu (Rheology.cpp).

The radius in the file is interpreted as the mean radius s of a level surface
when the model is used for an ellipsoidal Earth.

``anelastic(omega)`` returns a copy with complex, frequency-dependent mu (and
lambda, keeping kappa fixed), using the same absorption-band / Q^alpha model
as astyanax (Rheology::apply_dissipation): seismic reference period T0 = 1 s,
constant Q between T0 and Tm, Q ~ (omega/omega_m)^alpha beyond Tm.
"""
import copy

import numpy as np
from numpy.polynomial import Chebyshev

G = 6.673e-11          # same value as astyanax
A_MEAN = 6371.0e3      # Earth's mean radius (m); a model's radius is read from its file


class EarthModel:
    def __init__(self, filename):
        self.name, self.r0, self.r1, self.rho, self.lam, self.mu, self.Qmu = \
            self._read(filename)
        self.is_complex = False
        self.n_layers = len(self.r0)
        self.a = self.r1[-1]
        # Mass below the bottom of each layer, then M(r) inside each layer
        self._mass_poly = []
        m_below = 0.0
        self.mass_below = np.zeros(self.n_layers)
        for i in range(self.n_layers):
            r = Chebyshev.identity(domain=[self.r0[i], self.r1[i]])
            integrand = 4.0 * np.pi * self.rho[i] * r * r
            p = integrand.integ(lbnd=self.r0[i])
            self.mass_below[i] = m_below
            self._mass_poly.append(p)
            m_below += p(self.r1[i])
        self.M = m_below
        self.g_surface = G * self.M / self.a**2

    @staticmethod
    def _read(filename):
        with open(filename) as fp:
            lines = [l.rstrip('\n') for l in fp]
        name = lines[0].split()[0]
        n = int(lines[1].split()[0])
        blocks, i = {}, 2
        while i < len(lines):
            key = lines[i].strip()
            if key in ('rho', 'Vp_h', 'Vp_v', 'Vs_h', 'Vs_v', 'eta', 'Qmu'):
                rows = [lines[i + 2 + j].split() for j in range(n)]
                blocks[key] = rows
                i += 2 + n
            else:
                i += 1
        depth = np.array([float(r[0]) for r in blocks['rho']])
        a_km = depth[0]                 # the innermost layer reaches the centre: depth = radius
        bounds = np.append(a_km - depth, a_km) * 1e3
        r0, r1 = bounds[:-1], bounds[1:]

        def cheb(key, i, scale):
            c = np.array([float(x) for x in blocks[key][i][1:5]]) * scale
            return Chebyshev(c, domain=[r0[i], r1[i]])

        Qmu = np.array([float(r[1]) for r in blocks['Qmu']]) if 'Qmu' in blocks \
            else np.full(n, np.nan)
        rho, lam, mu = [], [], []
        for i in range(n):
            d = cheb('rho', i, 1e3)
            vph, vpv = cheb('Vp_h', i, 1e3), cheb('Vp_v', i, 1e3)
            vsh, vsv = cheb('Vs_h', i, 1e3), cheb('Vs_v', i, 1e3)
            eta = cheb('eta', i, 1.0)
            A, C = d * vph * vph, d * vpv * vpv
            L, N = d * vsv * vsv, d * vsh * vsh
            F = eta * (A - 2 * L)
            kappa = (C + 4 * A - 4 * N + 4 * F) / 9.0
            m = (C + A + 6 * L + 5 * N - 2 * F) / 15.0
            rho.append(d)
            mu.append(m)
            lam.append(kappa - 2.0 * m / 3.0)
        return name, r0, r1, rho, lam, mu, Qmu

    def dmu_factor(self, omega, alpha=0.15, Tm=200.0, T0=1.0, dispersion='cot'):
        """Complex delta mu / mu per layer at angular frequency omega: constant Q between the
        seismic period T0 and Tm, Q ~ omega^alpha beyond. dispersion='cot': the exact dispersion
        of a power-law Q, cot(alpha pi / 2) (1 - x), as Wahr & Bergen (1986) and the IERS
        Conventions; 'astyanax': its small-alpha approximation 2 / (pi alpha) (1 - x), as in
        astyanax (about 2 per cent smaller for alpha = 0.15)."""
        if dispersion not in ('cot', 'astyanax'):
            raise ValueError(dispersion)
        w0, wm = 2 * np.pi / T0, 2 * np.pi / Tm
        out = np.zeros(self.n_layers, complex)
        for i, Q in enumerate(self.Qmu):
            if not np.isfinite(Q):
                continue                       # fluid / no attenuation
            if abs(alpha) < 1e-3:
                out[i] = (2 / np.pi * np.log(omega / w0) + 1j) / Q
            else:
                x = (wm / omega)**alpha
                c = 1 / np.tan(alpha * np.pi / 2) if dispersion == 'cot' else 2 / (np.pi * alpha)
                out[i] = (2 / np.pi * np.log(wm / w0) + c * (1 - x) + 1j * x) / Q
        return out

    def anelastic(self, omega, alpha=0.15, Tm=200.0, T0=1.0, dispersion='cot'):
        """Copy of the model with complex mu(omega), lambda = kappa - 2 mu / 3 (dispersion: see
        dmu_factor)."""
        new = copy.copy(self)
        d = self.dmu_factor(omega, alpha, Tm, T0, dispersion)
        new.mu = [m * (1 + di) for m, di in zip(self.mu, d)]
        new.lam = [l - 2.0 / 3.0 * m * di for l, m, di in zip(self.lam, self.mu, d)]
        new.is_complex = bool(np.any(d.imag != 0))
        new.anelastic_params = dict(omega=omega, alpha=alpha, Tm=Tm, T0=T0, dispersion=dispersion)
        return new

    def layer_of(self, r):
        return int(np.clip(np.searchsorted(self.r1, r), 0, self.n_layers - 1))

    def mass(self, i, r):
        """Mass inside radius r, r lying in layer i."""
        return self.mass_below[i] + self._mass_poly[i](r)

    def gravity(self, i, r):
        r = np.asarray(r, float)
        with np.errstate(divide='ignore', invalid='ignore'):
            g = G * self.mass(i, r) / r**2
        return np.where(r > 0, g, 0.0)
