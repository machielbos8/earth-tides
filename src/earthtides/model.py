# -*- coding: utf-8 -*-
#
# Radial Earth model in the astyanax file format.
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

``viscous_core(omega, delta=...)`` returns a copy with a Newtonian viscosity in the liquid layers,
mu = i |omega| eta, and (by default) thin extra layers at their solid boundaries that resolve the
Ekman layers; ``split_layer`` does the splitting.
"""
import copy

import numpy as np
from numpy.polynomial import Chebyshev

OMEGA_ROT = 7.292115e-5     # rad/s; = geometry.OMEGA_EARTH (geometry imports this module)

G = 6.673e-11          # same value as astyanax
A_MEAN = 6371.0e3      # Earth's mean radius (m); a model's radius is read from its file


class EarthModel:
    def __init__(self, filename):
        self.name, self.r0, self.r1, self.rho, self.lam, self.mu, self.Qmu = \
            self._read(filename)
        self.is_complex = False
        self.n_layers = len(self.r0)
        self.a = self.r1[-1]
        self._set_mass()

    def _set_mass(self):
        """Mass below the bottom of each layer, then M(r) inside each layer."""
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

    def split_layer(self, i, cuts):
        """Copy of the model with layer i split at the radii `cuts` (ascending, inside the layer). The
        sub-layers carry the same polynomials, re-expanded on their domains, so the model is unchanged;
        the solver welds them together. Layer indices above i shift by len(cuts)."""
        cuts = [float(c) for c in cuts]
        if any(not (self.r0[i] < c < self.r1[i]) for c in cuts) or cuts != sorted(cuts):
            raise ValueError('cuts must be ascending and inside layer %d' % i)
        new = copy.copy(self)
        edges = [self.r0[i]] + cuts + [self.r1[i]]
        r0, r1, rho, lam, mu, Q = [], [], [], [], [], []
        for k in range(self.n_layers):
            pieces = list(zip(edges[:-1], edges[1:])) if k == i else [(self.r0[k], self.r1[k])]
            for a, b in pieces:
                r0.append(a)
                r1.append(b)
                rho.append(self.rho[k].convert(domain=[a, b]))
                lam.append(self.lam[k].convert(domain=[a, b]))
                mu.append(self.mu[k].convert(domain=[a, b]))
                Q.append(self.Qmu[k])
        new.r0, new.r1, new.Qmu = np.array(r0), np.array(r1), np.array(Q)
        new.rho, new.lam, new.mu = rho, lam, mu
        new.n_layers = len(r0)
        new._set_mass()
        return new

    def viscous_core(self, omega, eta=None, delta=None, boundary_layers=True, d_factor=5.0,
                     d_min=4.0e4, rotation=OMEGA_ROT):
        """Copy of the model with a Newtonian viscosity in its liquid layers (mu = 0): by the
        correspondence principle the stress 2 eta d(eps)/dt of a motion exp(i omega t) is that of a
        purely imaginary shear modulus mu = i |omega| eta (bulk viscosity neglected; lambda =
        kappa - 2 mu / 3, kappa unchanged). mu != 0 makes the liquid-solid boundaries welded
        (no-slip), as a viscous fluid requires; the solver keeps inertia and Coriolis in these layers
        (|mu| < weak_fluid_mu). This regularizes the inviscid core, whose inertia-gravity spectrum is
        continuous in the tidal band (Valette 1989; manual, Sect. 'The viscous core').

        Give eta (Pa s) or the Ekman-layer thickness delta = sqrt(eta / (rho Omega)) (m), rho the mean
        density of the layer. The real core (eta ~ 1e-2 Pa s, delta ~ 0.1 m) cannot be resolved; use
        delta of a few hundred m to km. boundary_layers: next to every solid boundary of a liquid layer,
        a separate Chebyshev layer of d_min and, inside it, one of d_factor delta when that is less than
        d_min / 2 (the nodes of a Chebyshev layer cluster at its ends: a 40 km layer with N = 30 has nodes
        0.12 km apart at the wall but 2 km in its middle). Without boundary layers O1 h0 is off by 6e-4.
        d_min = 40 km: at O1 and delta = 1.5 km, d = 10 km is too thin (h(2) off by 6e-6) and 20 km
        marginal (3e-7), probably because the Ekman layer thickens near the critical latitude
        (~ R (delta / R)^(4/5), 3-7 km here). The inner 5-delta layer:
        at psi1 (next to the FCN) and delta = 0.5 km the 40 km layer alone is off by 2.5e-5 in h(2); with
        the inner layer, N = 40 and a 20 km layer agree to 2e-7 (measured on PREM). The
        model is
        frequency dependent (like anelastic): build it, and the geometry, for each tidal frequency. A complex
        omega (free-mode search) gives the analytic continuation mu = i omega eta (Re omega > 0)."""
        if (eta is None) == (delta is None):
            raise ValueError('give eta or delta')
        new = copy.copy(self)
        new.mu, new.lam = list(self.mu), list(self.lam)
        liquid = [abs(self.mu[i](0.5 * (self.r0[i] + self.r1[i]))) < 1.0 for i in range(self.n_layers)]
        cuts = {}
        for i in np.flatnonzero(liquid):
            r = Chebyshev.identity(domain=[self.r0[i], self.r1[i]])
            vol = (r * r).integ(lbnd=self.r0[i])(self.r1[i])
            rho_m = (self.rho[i] * r * r).integ(lbnd=self.r0[i])(self.r1[i]) / vol
            et = eta if eta is not None else rho_m * rotation * delta**2
            dl = np.sqrt(et / (rho_m * rotation))
            mu = 1j * (omega if np.real(omega) >= 0 else -omega) * et     # = i |omega| eta for real omega
            dom = [self.r0[i], self.r1[i]]
            new.mu[i] = Chebyshev([mu], domain=dom)
            new.lam[i] = self.lam[i] - 2.0 * mu / 3.0
            # one layer of d_min, and inside it one of d_factor delta when that is much thinner
            ds = sorted({max(d_factor * dl, d_min)} | ({d_factor * dl} if d_factor * dl < 0.5 * d_min else set()))
            c = []
            if i > 0 and not liquid[i - 1]:
                c += [self.r0[i] + d for d in ds]
            if i < self.n_layers - 1 and not liquid[i + 1]:
                c += [self.r1[i] - d for d in ds]
            c = sorted(c)
            if any(b <= a for a, b in zip(c[:-1], c[1:])):
                raise ValueError('boundary layers overlap in layer %d' % i)
            cuts[i] = c
        new.is_complex = True
        new.viscous_params = dict(omega=omega, eta=eta, delta=delta, boundary_layers=boundary_layers,
                                  d_factor=d_factor, d_min=d_min)
        if boundary_layers:
            for i in sorted(cuts, reverse=True):     # from the top: lower indices stay valid
                if cuts[i]:
                    new = new.split_layer(i, cuts[i])
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
