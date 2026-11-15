# -*- coding: utf-8 -*-
#
# Level-surface geometry r = s [1 - (2/3) eps(s) P2(cos theta)].
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
"""Level-surface coordinates: the map R(s, theta) from the label s of a surface (its mean radius) and
colatitude to the physical radius, and the derivatives the chain rule of the solver needs.

* LevelGeometry       the surfaces are level surfaces of the model's own reference potential, to all
                      orders in the flattening (iterated; the default figure since version 1.1)
* ClairautGeometry    the first-order figure r = s [1 - (2/3) eps(s) P2(cos theta)], eps from
                      Clairaut's equation (the figure of first-order theories; the default of 1.0)
* ExactLevelsGeometry exact ellipses of the Clairaut flattening (a test of the second-order terms)
* ExactSpheroidGeometry  an exact spheroid (homogeneous test bodies)
* SphereGeometry      eps = 0
* BumpGeometry        artificial eps that vanishes at every interface; used with a *spherical* Earth
                      to test the mesh machinery
``figure(model, kind)`` returns the figure of a model: 'level' or 'first-order'. Geometries with eps
return eps, eps' and eps'' per layer and s, which ``mapping`` turns into the derivatives of R and of
its inverse; LevelGeometry supplies the derivatives of R directly (``map_derivs``).
"""
import numpy as np
from numpy.polynomial import Chebyshev, chebyshev as cheb

from .model import G

OMEGA_EARTH = 7.292115e-5


class SphereGeometry:
    def eps(self, i, s):
        z = np.zeros_like(np.asarray(s, float))
        return z, z, z


class _ChebSol:
    """eps and eps' of one layer from its Chebyshev series; called like scipy's dense output."""

    def __init__(self, c, s0, s1):
        self.f = Chebyshev(c, domain=[s0, s1])
        self.df = self.f.deriv()
        self.t_min, self.t_max = s0, s1

    def __call__(self, s):
        s = np.atleast_1d(s)
        return np.vstack([self.f(s), self.df(s)])


class ClairautGeometry:
    """First-order hydrostatic flattening of the level surfaces.

    eps'' = -(6 rho / (rho_bar s)) eps' - (6/s^2)(rho/rho_bar - 1) eps,
    rho_bar = 3 M(s) / (4 pi s^3), scaled so that at the surface
    a eps' + 2 eps = (5/2) m,  m = omega^2 a^3 / (G M).
    ``scale`` multiplies eps (scale=0 gives a sphere).
    """

    _cache = {}         # the ODE depends on density and layering only, not on mu or frequency:
    #                     a sweep builds a new (anelastic / viscous) model per frequency, same key

    def __init__(self, model, omega=OMEGA_EARTH, scale=1.0, s_start=1.0e3):
        self.model, self.omega, self.scale = model, omega, scale
        key = (omega, s_start, model.a, model.M, tuple(model.r0), tuple(model.r1),
               tuple((tuple(np.asarray(f.coef).ravel()), tuple(np.asarray(f.domain)))
                     for f in model.rho))
        hit = ClairautGeometry._cache.get(key)
        if hit is None:
            # Chebyshev collocation per layer (eps, eps' continuous across interfaces), like the
            # solver; it replaced an adaptive DOP853 integration (2026-10-18; the same solution to
            # 5e-9 up to a constant factor that the surface normalisation removes, surface
            # flattening to 15 digits; ~300 times faster)
            sols = []
            y = np.array([1.0, 0.0])
            for i in range(model.n_layers):
                s0 = max(model.r0[i], s_start)
                sol = self._layer(i, s0, model.r1[i], y)
                sols.append(sol)
                y = sol(np.array([model.r1[i]]))[:, 0]
            hit = (sols, y)
            if len(ClairautGeometry._cache) > 32:
                ClairautGeometry._cache.clear()
            ClairautGeometry._cache[key] = hit
        self._sol, y = list(hit[0]), hit[1]
        a = model.a
        m = omega**2 * a**3 / (G * model.M)
        self.m = m
        self.norm = 2.5 * m / (a * y[1] + 2.0 * y[0])
        self.flattening_surface = self.norm * y[0]

    def _layer(self, i, s0, s1, y0, M=48, tol=1e-15, Mmax=512):
        """eps on [s0, s1] as a Chebyshev series: collocation of eps'' = a eps' + b eps at the
        interior Gauss-Lobatto nodes, eps(s0), eps'(s0) = y0; M doubled until the last coefficients
        fall below tol (the inner core, where the 1/s terms vary over three decades from s_start,
        takes ~384)."""
        while True:
            x = -np.cos(np.pi * np.arange(M + 1) / M)
            s = s0 + (x + 1) * (s1 - s0) / 2
            sc = 2.0 / (s1 - s0)
            rho, rho_bar = self._coef(i, s)
            a = -6.0 * rho / (rho_bar * s)
            b = -6.0 / s**2 * (rho / rho_bar - 1.0)
            I = np.eye(M + 1)
            V = cheb.chebvander(x, M)
            D1 = cheb.chebvander(x, M - 1) @ cheb.chebder(I, 1, axis=0) * sc
            D2 = cheb.chebvander(x, M - 2) @ cheb.chebder(I, 2, axis=0) * sc**2
            L = D2 - a[:, None] * D1 - b[:, None] * V
            A = np.vstack([V[:1], D1[:1], L[1:-1]])
            c = np.linalg.solve(A, np.concatenate([y0, np.zeros(M - 1)]))
            if np.max(np.abs(c[-4:])) < tol * np.max(np.abs(c)) or M >= Mmax:
                return _ChebSol(c, s0, s1)
            M *= 2

    def _coef(self, i, s):
        rho = self.model.rho[i](s)
        rho_bar = 3.0 * self.model.mass(i, s) / (4.0 * np.pi * s**3)
        return rho, rho_bar

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


def _ellipse(theta, e):
    """f = (sin^2/A^2 + cos^2/C^2)^(-1/2) with A = (1-e)^(-1/3), C = A (1-e) and its derivatives
    f_t, f_tt, f_e, f_ee, f_te."""
    u = 1 - e
    a, a1, a2 = u ** (2 / 3), -2 / 3 * u ** (-1 / 3), -2 / 9 * u ** (-4 / 3)        # 1/A^2 and d/de
    c, c1, c2 = u ** (-4 / 3), 4 / 3 * u ** (-7 / 3), 28 / 9 * u ** (-10 / 3)       # 1/C^2 and d/de
    s2, c2t, sn2 = np.sin(theta) ** 2, np.cos(theta) ** 2, np.sin(2 * theta)
    q = a * s2 + c * c2t
    qt, qtt = (a - c) * sn2, 2 * (a - c) * np.cos(2 * theta)
    qe, qee, qte = a1 * s2 + c1 * c2t, a2 * s2 + c2 * c2t, (a1 - c1) * sn2
    f, g, h = q ** -0.5, -0.5 * q ** -1.5, 0.75 * q ** -2.5
    return f, g * qt, h * qt * qt + g * qtt, g * qe, h * qe * qe + g * qee, h * qt * qe + g * qte


class ExactLevelsGeometry(ClairautGeometry):
    """Clairaut's first-order flattening eps(s), but every coordinate surface an exact ellipse,
    R = s f(theta; eps(s)), s the volume-equivalent radius (cf. ExactSpheroidGeometry)."""

    def map_derivs(self, i, s, theta):
        s = np.asarray(s, float)
        e, de, dde = self.eps(i, s)
        f, ft, ftt, fe, fee, fte = _ellipse(theta, e)
        z = 0 * (s + theta)
        return (s * f + z, f + s * fe * de + z, 2 * fe * de + s * (fee * de ** 2 + fe * dde) + z,
                s * ft + z, ft + s * fte * de + z, s * ftt + z)


class LevelGeometry(ExactLevelsGeometry):
    """Coordinate surfaces that are level surfaces of the reference potential of refpot.py: the
    hydrostatic figure of the model's density and rotation beyond first order in the flattening.

    R = s f(theta; eps(s)) + sum_n d_n(s) P_n(cos theta), n = 2, 4, 6, with d_n a Chebyshev
    interpolant per layer on M + 1 Lobatto nodes (end values shared with the neighbouring
    layer: the map stays continuous). Iteration: reference potential of the current figure ->
    height h = (Phi0 - its mean) / (dPhi0/dr) of every node's surface above the level surface
    through its mean -> d_n -= h_n, until max |h| < tol (m) apart from the part of h outside
    degrees 2 - 6 (1e-4 m for PREM), or itmax. h halves per step, because moving the surfaces
    moves the potential. ``history``: (iteration, max |h|, max of the part outside 2 - 6).
    For PREM the surface flattening becomes 1/299.72 (first-order map: 1/300.23 from
    eps = 1/299.9) and the core-mantle boundary moves by -12.6 m (P2) and +8.0 m (P4).
    eps() and flattening_surface stay those of the Clairaut figure.
    """
    _lcache = {}
    _parents = {}

    def __init__(self, model, omega=OMEGA_EARTH, M=10, tol=1e-6, itmax=30, degrees=(2, 4, 6), ref_K=4,
                 start=None, **kw):
        # degrees: of the correction; ref_K: K of the reference potential used in the iteration (degrees
        # 0 .. 2 ref_K - 2; the solver's own reference potential has K = 4)
        super().__init__(model, omega=omega, **kw)
        self._kw = dict(tol=tol, itmax=itmax, ref_K=ref_K, **kw)
        self.M, self.DEG = M, tuple(degrees)
        nL = model.n_layers
        self.d = [{n: np.zeros(M + 1) for n in self.DEG} for _ in range(nL)]
        key = (omega, M, tol, itmax, self.DEG, ref_K, self.scale, model.a, model.M, tuple(model.r0), tuple(model.r1),
               tuple((tuple(np.asarray(f.coef).ravel()), tuple(np.asarray(f.domain))) for f in model.rho))
        hit = LevelGeometry._lcache.get(key)
        if hit is not None:
            self.d, self.history = hit
            return
        from numpy.polynomial import chebyshev as Ch
        from .refpot import ReferencePotential
        self.history = []
        # warm start: a figure of the same Earth on coarser layers (the unsplit model of a viscous core, whose
        # sub-layers change with frequency and delta) restricted to these layers; one potential solve then
        # confirms it instead of ~17 iterations. start: such a figure given explicitly (restrict()).
        pkey = key[:7]
        for g0 in ([start] if start is not None else LevelGeometry._parents.get(pkey, [])):
            d0 = g0._restricted_to(model)
            if d0 is not None:
                self.d = d0
                break
        for it in range(itmax):
            rp = ReferencePotential(model, self, omega, K=ref_K)
            xk, H, hmax, rest = self._heights(rp)
            self.history.append((it, hmax, rest))
            if hmax < tol + 2 * rest:
                break
            for i in range(nL - 1):
                H[i, -1] = H[i + 1, 0] = 0.5 * (H[i, -1] + H[i + 1, 0])
            self.d = [{n: Ch.chebfit(xk, Ch.chebval(xk, self.d[i][n]) - H[i, :, j], M)
                       for j, n in enumerate(self.DEG)} for i in range(nL)]
        LevelGeometry._lcache[key] = (self.d, self.history)
        LevelGeometry._parents.setdefault(pkey, []).append(self)

    def _restricted_to(self, model):
        """The corrections d_n on the layers of `model`, when every one of them lies inside one layer of this
        figure's model with the same density there (a split of its layers); else None. Exact: a polynomial
        restricted to a sub-interval keeps its degree."""
        from numpy.polynomial import chebyshev as Ch
        md, M = self.model, self.M
        if abs(model.a - md.a) > 1e-6 or abs(model.M / md.M - 1) > 1e-12:
            return None
        xk = np.cos(np.pi * np.arange(M + 1) / M)[::-1]
        out = []
        for i in range(model.n_layers):
            r0, r1 = model.r0[i], model.r1[i]
            j = np.flatnonzero((md.r0 <= r0 + 1e-6) & (md.r1 >= r1 - 1e-6))
            if len(j) != 1:
                return None
            j = int(j[0])
            sm = r0 + np.array([0.25, 0.5, 0.75]) * (r1 - r0)
            if np.max(np.abs(model.rho[i](sm) - md.rho[j](sm))) > 1e-9 * np.max(np.abs(md.rho[j](sm))):
                return None
            s = r0 + (xk + 1) * (r1 - r0) / 2
            x = 2.0 * (s - md.r0[j]) / (md.r1[j] - md.r0[j]) - 1
            out.append({n: Ch.chebfit(xk, Ch.chebval(x, self.d[j][n]), M) for n in self.DEG})
        return out

    def restrict(self, model):
        """The level figure for `model`, a split of this figure's model into sub-layers (EarthModel.viscous_core,
        split_layer), started from this figure: for worker processes, which do not share the cache
        (build the figure of the unsplit model once, pass it to the workers, call restrict there)."""
        return LevelGeometry(model, omega=self.omega, M=self.M, degrees=self.DEG, start=self, **self._kw)

    def surface_legendre(self, s=None, i=None, nmax=8):
        """Legendre coefficients 0, 2, .., nmax of R(theta) of the coordinate surface s (default: the surface)."""
        from numpy.polynomial import legendre as L
        md = self.model
        s = md.a if s is None else s
        i = int(np.flatnonzero(np.isclose(md.r1, s))[0]) if i is None else i
        x, w = L.leggauss(40)
        R = np.asarray(self.map_derivs(i, np.array([[s]]), np.arccos(x))[0])[0]
        return {n: (2 * n + 1) / 2 * float(np.sum(w * R * L.Legendre.basis(n)(x))) for n in range(0, nmax + 1, 2)}

    def _heights(self, rp, n_theta=40):
        from numpy.polynomial import legendre as L
        from .basis import cheb_basis, ybar
        md, M = self.model, self.M
        xg, wg = L.leggauss(n_theta)
        th = np.arccos(xg)
        xk = np.cos(np.pi * np.arange(M + 1) / M)[::-1]
        P = [L.Legendre.basis(n)(xg) for n in self.DEG]
        H = np.zeros((md.n_layers, M + 1, len(self.DEG)))
        hmax = rest = 0.0
        for i in range(md.n_layers):
            for k, sk in enumerate(md.r0[i] + (xk + 1) * (md.r1[i] - md.r0[i]) / 2):
                if sk < 1.0:                                   # the centre
                    continue
                mp = mapping_from_R(*self.map_derivs(i, np.array([[sk]]), th))
                T = cheb_basis(np.array([xk[k]]), rp.N)[0]
                phi = sum((rp.coef[i * rp.nb + j * rp.N: i * rp.nb + (j + 1) * rp.N] @ T)[0]
                          * ybar(n, 0, th, 0)[0] for j, n in enumerate(rp.deg))
                phi = phi - rp.Om ** 2 * mp['R'][0] ** 2 * np.sin(th) ** 2 / 2
                h = (phi - np.sum(wg * phi) / 2) / rp.phys_derivs(i, np.array([sk]), th, mp)['r'][0]
                cn = [(2 * n + 1) / 2 * np.sum(wg * h * p) for n, p in zip(self.DEG, P)]
                H[i, k] = cn
                hmax = max(hmax, float(np.max(np.abs(h))))
                rest = max(rest, float(np.max(np.abs(h - sum(c * p for c, p in zip(cn, P))))))
        return xk, H, hmax, rest

    def surface_eps(self):
        """(eps, d eps / ds) of the P2 part of the surface, R_2(s) = -2/3 eps(s) s (for the conventions)."""
        md = self.model
        a, h = md.a, 1.0
        c = [self.surface_legendre(s=s, i=md.n_layers - 1, nmax=2)[2] for s in (a - h, a)]
        e = [-1.5 * c2 / s for c2, s in zip(c, (a - h, a))]
        return e[1], (e[1] - e[0]) / h

    def equatorial_radius(self):
        return polar_equatorial_radius(self, self.model)[1]

    def map_derivs(self, i, s, theta):
        from numpy.polynomial import chebyshev as Ch, legendre as L
        out = list(super().map_derivs(i, s, theta))
        md = self.model
        sc = 2.0 / (md.r1[i] - md.r0[i])
        x = sc * (np.asarray(s, float) - md.r0[i]) - 1
        c, sn = np.cos(theta), np.sin(theta)
        for n in self.DEG:
            a = self.d[i][n]
            if not np.any(a):
                continue
            d0, d1, d2 = Ch.chebval(x, a), Ch.chebval(x, Ch.chebder(a)) * sc, Ch.chebval(x, Ch.chebder(a, 2)) * sc ** 2
            Pn = L.Legendre.basis(n)
            P, P1, P2 = Pn(c), Pn.deriv(1)(c), Pn.deriv(2)(c)
            Pt, Ptt = -P1 * sn, P2 * sn * sn - P1 * c
            for k, v in enumerate((d0 * P, d1 * P, d2 * P, d0 * Pt, d1 * Pt, d0 * Ptt)):
                out[k] = out[k] + v
        return tuple(out)


def figure(model, kind='level', omega=OMEGA_EARTH, scale=1.0, **kw):
    """The figure of the model: 'level' (LevelGeometry: level surfaces of the reference potential, second
    order in the flattening and beyond; the default since 2026-11-15) or 'first-order' (ClairautGeometry,
    the figure of the classical theories and of every result before that date).
    scale multiplies the first-order flattening, i.e. the rotation rate by sqrt(scale); for the level
    figure the rotation rate itself is changed, omega sqrt(scale) (scale = 0: the sphere)."""
    if kind in ('first-order', 'first', 'clairaut'):
        return ClairautGeometry(model, omega=omega, scale=scale, **kw)
    if kind != 'level':
        raise ValueError(f"figure: 'first-order' or 'level', not {kind!r}")
    if scale == 0:
        return ClairautGeometry(model, omega=omega, scale=0.0)
    return LevelGeometry(model, omega=omega * np.sqrt(scale), **kw)


def polar_equatorial_radius(geom, model, i=None, s=None):
    """(polar, equatorial) radius of the coordinate surface s of layer i (default: the surface)."""
    i = model.n_layers - 1 if i is None else i
    s = model.a if s is None else s
    th = np.array([1e-6, np.pi / 2])            # not 0: maps built on a solver grid divide by sin(theta)
    if hasattr(geom, 'map_derivs'):
        R = np.asarray(geom.map_derivs(i, np.array([[s]]), th)[0])[0]
    else:
        R = mapping(np.array([[s]]), th, *geom.eps(i, np.array([[s]])))['R'][0]
    return float(R[0]), float(R[1])


def surface_eps(geom, model):
    """Ellipticity of the surface for the conventions of the Love numbers and its s-derivative, such that
    the P2 part of the surface is a (1 - 2/3 eps P2) and the equatorial radius a (1 + eps / 3) to first
    order. First-order figures: eps(a) itself (unchanged numbers). A geometry with its own
    surface_eps() (LevelGeometry: from the P2 part of its map) is asked."""
    if hasattr(geom, 'surface_eps'):
        return geom.surface_eps()
    e, de, _ = geom.eps(model.n_layers - 1, np.array([model.a]))
    return float(e[0]), float(de[0])


def equatorial_radius(geom, model):
    """Equatorial radius of the surface: a (1 + eps / 3) on a first-order figure (unchanged numbers), the
    map's own value where the geometry provides equatorial_radius()."""
    if hasattr(geom, 'equatorial_radius'):
        return geom.equatorial_radius()
    return model.a * (1 + surface_eps(geom, model)[0] / 3.0)


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
