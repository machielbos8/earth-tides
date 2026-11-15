# -*- coding: utf-8 -*-
#
# Non-hydrostatic Earth: degree-2 zonal mantle loads, their static support on the
# flattened rotating background, and the final state as data for the tidal solver.
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
# 17/10/2026 Machiel Bos
#===============================================================================
"""Non-hydrostatic Earth: a shape held up by mantle loads, with the deviatoric stress that supports it.

The excess flattening of the surface (J2) and of the core-mantle boundary (FCN period) is not
prescribed as a geometry but produced by zonal density loads in the mantle,
    drho = rho0(r) a_k P2(cos theta)   in shell k (Eulerian, fixed in the mantle),
held up by the static deformation w of the self-gravitating Earth (elastic support, or viscous flow
for the other end member). Load, deformation and the deviatoric stress that supports the shape form
one consistent state; different loads give the same J2 and CMB flattening with different stresses,
so the shells (depth ranges) and the support are the user's choices.

Three steps, one class each:
  FlatSupport  unit responses of the shells, measured on the supported state on the flattened,
               rotating background (Support: the same on a spherical background, an independent check
               that needs the Born module of a later release), and fit(): load amplitudes for targets
                   J2       excess J2 (positive = more oblate)
                   surface  excess equatorial minus polar radius of the surface [m]
                   cmb      excess equatorial minus polar radius of the CMB (mantle side) [m]
                   e_f      excess dynamical ellipticity (C_f - A_f)/A_f of the fluid core
               (exact, least squares, or regularised with target sigmas and a load prior).
  FlatState    the same loads supported on the flattened, rotating hydrostatic background with
               TidalSolver itself, so that the state is in equilibrium on the tidal mesh to second
               order in the excess; supplies drho, tau and the slip pressure to
               TidalSolver(nonhydro=...).
  FlatMap      the geometry: the background map plus the deflection of the material surfaces.
NonHydrostatic(md, settings) builds all three from a set-up table (earthtides.love). The state is
static: it does not depend on the tidal frequency or on the anelasticity of the tidal run (only the
tidal model gets complex moduli), so one state serves a whole table of waves.
"""
import copy
import numpy as np
from numpy.polynomial import Chebyshev

from .model import G

KEYS = ('J2', 'surface', 'cmb', 'e_f')


def p2(th):
    return 0.5 * (3 * np.cos(th)**2 - 1)


def mantle_layers(md, r_cmb=3480e3, r_top=6346.6e3):
    return [i for i in range(md.n_layers) if md.r0[i] >= r_cmb - 1 and md.r1[i] <= r_top + 1]


def viscous_model(base, s=2e6, eta_rel=None, r_cmb=3480e3, r_lid=6291e3, r_670=5701e3):
    """Viscous end member (instantaneous Stokes flow): copy of the
    weak-solid-core model with mu = s eta_rel(layer) [Pa] from the CMB to the base of the elastic
    lid (80 km), bulk modulus kept. Default eta_rel: lower mantle 50, upper mantle 1; s = 2e6 Pa
    (6e6 gives the convergence uncertainty, a few per cent)."""
    new = copy.copy(base)
    new.mu, new.lam = list(base.mu), list(base.lam)
    new.weak_fluid_mu = 5e4               # weak-solid core (1e4 Pa) stays fluid, the soft mantle not
    visc = [i for i in range(base.n_layers) if base.r0[i] >= r_cmb - 1 and base.r1[i] <= r_lid + 1]
    for i in visc:
        e = (eta_rel or {}).get(i, 50.0 if base.r1[i] <= r_670 + 1 else 1.0)
        mu_v = Chebyshev([s * e], domain=[base.r0[i], base.r1[i]])
        new.lam[i] = base.lam[i] + 2.0 / 3.0 * (base.mu[i] - mu_v)
        new.mu[i] = mu_v
    new.viscous_layers = visc
    return new


class Support:
    """md: the tidal model (true fluid core); md_static: the same model with a weak-solid core
    (PREMs.dat) for the static solve, or viscous_model(PREMs) for the viscous end member. shells: list of (rmin, rmax) [m] on layer boundaries, each
    carrying drho = rho0 a_k P2. fixed: Eulerian-fixed layers (default: CMB to Moho)."""

    def __init__(self, md, md_static, shells, fixed=None, N=24, n_theta=32):
        self.md, self.mds, self.shells, self.N = md, md_static, list(shells), N
        self.fixed = fixed if fixed is not None else mantle_layers(md)
        self.ioc = [i for i in range(md.n_layers) if float(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) == 0.0][-1]
        x, w = np.polynomial.legendre.leggauss(n_theta)
        self.th, self.wq = np.arccos(x), w
        self.unit = [self._unit(*sh) for sh in self.shells]
        self.layers = [u['_state'][2] for u in self.unit]

    def _p2coef(self, f):
        """P2 coefficient of f(theta) on the quadrature grid."""
        return 2.5 * np.sum(self.wq * f * p2(self.th))

    def _unit(self, rmin, rmax):
        md = self.md
        layers = [i for i in range(md.n_layers) if md.r0[i] >= rmin - 1 and md.r1[i] <= rmax + 1]

        def coef(r):
            r = np.atleast_1d(r)
            c = np.zeros((len(r), 3, 3))
            c[(r >= rmin - 1) & (r <= rmax + 1), 2, 0] = 1.0
            return c, np.zeros_like(c)

        def dlnrho(i, r, th, lam):
            if i not in layers:
                return None
            f = p2(th) + 0 * r + 0 * lam
            return f, 0 * f, -3 * np.cos(th) * np.sin(th) + 0 * r + 0 * lam, 0 * f

        try:
            from .born import DensityPotential, StaticSupport  # paper-2 machinery, spherical check only
        except ImportError as e:
            raise ImportError('nonhydro.Support (spherical background) needs earthtides.born, which is not '
                              'part of this release; use FlatSupport (NonHydrostatic does)') from e
        dP = DensityPotential(md, coef, 2, rmin, rmax, G)
        SS = StaticSupport(self.mds, dlnrho, dP, self.fixed, 2, N=self.N)
        out = self._measure(SS, dP, rmin, rmax)
        out['_state'] = (SS, dP, layers)
        return out

    def _measure(self, SS, dP, rmin, rmax):
        """The four target quantities and the pieces for the mass check."""
        md, th, lam = self.md, self.th, np.array([0.0])
        a, iL, ic = md.a, md.n_layers - 1, self.ioc
        out = {}
        f = SS.fields(iL, np.array([a]), th, lam)
        A = dP.grid(iL, np.array([a]), th, lam, value=True)
        phi20 = self._p2coef(A['v'][0, :, 0] + f['p']['v'][0, :, 0])
        out['J2'] = phi20 * a / (G * md.M)                       # code sign: Phi = -GM/r + ...
        out['surface'] = -1.5 * self._p2coef(f['r']['v'][0, :, 0])
        rc, ri = md.r1[ic], md.r0[ic]
        wc = self._p2coef(SS.fields(ic + 1, np.array([rc]), th, lam)['r']['v'][0, :, 0])
        wi = self._p2coef(SS.fields(ic - 1, np.array([ri]), th, lam)['r']['v'][0, :, 0])
        out['cmb'] = -1.5 * wc
        # core: hydrostatic final state, level surfaces zeta = -(dPhi + phi1)/g; the CMB and ICB at
        # the solid deflections. delta(C - A) = -(4 pi/5) int drho20 r^4 dr
        x, wg = np.polynomial.legendre.leggauss(64)
        s = ri + (x + 1) * (rc - ri) / 2
        Ac = dP.grid(ic, s, th, lam, value=True)['v'][:, :, 0]
        pc = SS.fields(ic, s, th, lam)['p']['v'][:, :, 0]
        zeta = np.array([self._p2coef(Ac[k] + pc[k]) for k in range(len(s))]) * -1 / md.gravity(ic, s)
        rho, drho = md.rho[ic], md.rho[ic].deriv()
        dCA = -4 * np.pi / 5 * (rho(rc) * wc * rc**4 - rho(ri) * wi * ri**4
                                - np.sum(wg * (rc - ri) / 2 * drho(s) * zeta * s**4))
        Af = 8 * np.pi / 3 * np.sum(wg * (rc - ri) / 2 * rho(s) * s**4)
        out['e_f'] = dCA / Af
        return out

    def maccullagh(self, k, n_gl=48):
        """Excess J2 of unit load k from the mass of the final state, delta(C - A)/(M a^2) with
        C - A = -int rho r^2 P2 dV (MacCullagh), independent of the potential used in _measure:
        per layer, the interior Eulerian density change (load in the loaded layers, none in the
        other fixed layers, -div(rho0 w) in the free solids, -rho0' zeta in the fluid core) plus
        its own material shifted at its bounding surfaces, rho_i (w_r(top) r_top^4 - w_r(bot) r_bot^4)."""
        md, th, lam = self.md, self.th, np.array([0.0])
        SS, dP, loaded = self.unit[k]['_state']
        x, wg = np.polynomial.legendre.leggauss(n_gl)
        P2 = p2(th)
        dCA = 0.0
        for i in range(md.n_layers):
            r0, r1 = md.r0[i], md.r1[i]
            s = max(r0, 1.0) + (x + 1) * (r1 - max(r0, 1.0)) / 2
            ws = wg * (r1 - max(r0, 1.0)) / 2
            rho = md.rho[i]
            if i == self.ioc:
                A = dP.grid(i, s, th, lam, value=True)['v'][:, :, 0]
                pc = SS.fields(i, s, th, lam)['p']['v'][:, :, 0]
                zeta = -(A + pc) / md.gravity(i, s)[:, None]
                d20 = np.array([self._p2coef(-rho.deriv()(s[j]) * zeta[j]) for j in range(len(s))])
                wb = [self._p2coef(SS.fields(j, np.array([rb]), th, lam)['r']['v'][0, :, 0])
                      for j, rb in ((i - 1, r0), (i + 1, r1))]
            else:
                f = SS.fields(i, s, th, lam)
                if i in loaded:
                    d20 = rho(s) * 1.0
                elif i in self.fixed:
                    d20 = 0 * s
                else:
                    wr, wt = f['r'], f['t']
                    S = s[:, None]
                    div = (wr['s'][:, :, 0] + 2 * wr['v'][:, :, 0] / S
                           + (wt['t'][:, :, 0] + wt['v'][:, :, 0] / np.tan(th)) / S)
                    dens = -(rho.deriv()(s)[:, None] * wr['v'][:, :, 0] + rho(s)[:, None] * div)
                    d20 = np.array([self._p2coef(dens[j]) for j in range(len(s))])
                wb = [self._p2coef(SS.fields(i, np.array([rb]), th, lam)['r']['v'][0, :, 0]) for rb in (r0, r1)]
            dCA += -4 * np.pi / 5 * (np.sum(ws * d20 * s**4)
                                      + rho(r1) * wb[1] * r1**4 - rho(r0) * wb[0] * r0**4)
        return dCA / (md.M * md.a**2)

    def response(self, amps):
        return {k: float(sum(a * u[k] for a, u in zip(amps, self.unit))) for k in KEYS}

    def fit(self, targets, sigma=None, prior=None):
        """Load amplitudes a_k for targets {key: value}.

        Without sigma/prior: exact (or least squares) fit. With sigma {key: 1-sigma uncertainty}
        and prior (1-sigma size of a_k, relative density, e.g. 5e-3): minimises
            sum_keys ((response - target)/sigma)^2 + sum_k (a_k/prior)^2,
        so that targets the support model cannot reach do not force implausible loads. For a given
        support, J2, the CMB and the surface flattening are nearly dependent (condition numbers
        1e3-1e4 for three shells), so a surface target mainly tests the support, not the loads."""
        keys = list(targets)
        M = np.array([[u[k] for u in self.unit] for k in keys])
        t = np.array([targets[k] for k in keys], float)
        if sigma is None:
            sc = np.abs(M).max(axis=1)                 # rows in very different units (J2, metres)
            return np.linalg.lstsq(M / sc[:, None], t / sc, rcond=None)[0]
        sg = np.array([sigma[k] for k in keys], float)
        A, b = M / sg[:, None], t / sg
        if prior is not None:
            n = M.shape[1]
            A = np.vstack([A, np.eye(n) / prior])
            b = np.concatenate([b, np.zeros(n)])
        return np.linalg.lstsq(A, b, rcond=None)[0]


class FlatSupport(Support):
    """Unit responses of the load shells measured on the flattened, rotating background: each shell
    is supported (FlatState: a static run of TidalSolver, forced by the weight of the load) and the
    targets are read off the final state itself -- J2 from the exterior potential of load and
    deformation, the surface and CMB flattening from the P2 part of the deflection of the material
    surfaces, e_f from C - A of the fluid core integrated over the deformed core. Linear in the load:
    one solve per shell at a small amplitude. Same fit() / response() as Support (which measures on
    the spherical background with the paper-2 machinery and remains as an independent check)."""

    AMP = 1e-4

    def __init__(self, md, md_static, shells, base, omega_rot, fixed=None, N=24, n_theta=32):
        self.md, self.mds, self.shells, self.N = md, md_static, list(shells), N
        self.base, self.omega_rot = base, omega_rot
        self.fixed = fixed if fixed is not None else mantle_layers(md)
        self.ioc = [i for i in range(md.n_layers) if float(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) == 0.0][-1]
        x, w = np.polynomial.legendre.leggauss(n_theta)
        self.th, self.wq = np.arccos(x), w
        self.layers = [[i for i in range(md.n_layers) if md.r0[i] >= a - 1 and md.r1[i] <= b + 1]
                       for a, b in self.shells]
        self.unit = []
        for k in range(len(self.shells)):
            amps = np.zeros(len(self.shells))
            amps[k] = self.AMP
            st = FlatState(self, amps, base, omega_rot, N=N)
            self.unit.append({key: v / self.AMP for key, v in self._measure_flat(st).items()})

    def _CA(self, geo, drho, ri, rc, n_gl=48):
        """C - A and A of the fluid core on the map geo with extra density drho (or None)."""
        md, ic = self.md, self.ioc
        x, wg = np.polynomial.legendre.leggauss(n_gl)
        s = ri + (x + 1) * (rc - ri) / 2
        ws = wg * (rc - ri) / 2
        th, wq = self.th, self.wq                   # wq: weights in cos(theta)
        R, R_s = (np.asarray(v) for v in (geo.map_derivs(ic, s[:, None], th[None, :])[:2]
                                          if hasattr(geo, 'map_derivs') else
                                          _bg_R(geo, ic, s, th)))
        rho = md.rho[ic](s)[:, None] + (0.0 if drho is None else drho)
        c2, sn2 = np.cos(th)**2, np.sin(th)**2
        dV = R**2 * R_s * ws[:, None] * wq[None, :] * 2 * np.pi
        CA = np.sum(rho * R**2 * (0.5 * sn2 - c2) * dV)
        A = np.sum(rho * R**2 * (0.5 * sn2 + c2) * dV)
        return CA, A

    def _measure_flat(self, st):
        md, th = self.md, self.th
        a, iL, ic = md.a, md.n_layers - 1, self.ioc
        rc, ri = md.r1[ic], md.r0[ic]
        nL = md.n_layers
        rp = st.rp_load
        c_load = rp.coef[nL * rp.nb + rp.deg.index(2)]
        Ts = st.Ts
        nb = Ts.nfld * Ts.N
        c_def = Ts.coef[nL * nb + Ts.deg.index(2)]
        out = {'J2': (c_load + c_def) * a / (G * md.M)}
        out['surface'] = -1.5 * self._p2coef(st.eta2(iL, np.array([a]), th)[0])
        out['cmb'] = -1.5 * self._p2coef(st.eta2(ic + 1, np.array([rc]), th)[0])
        geo = FlatMap(self.base, st)
        x, wg = np.polynomial.legendre.leggauss(48)
        s = ri + (x + 1) * (rc - ri) / 2
        mp = {'R': 0 * s[:, None] + 0 * th[None, :]}
        d = st.drho(ic, s[:, None], th, mp)[0]
        CA1, _ = self._CA(geo, d, ri, rc)
        CA0, A0 = self._CA(self.base, None, ri, rc)
        out['e_f'] = (CA1 - CA0) / A0
        return out


def _bg_R(base, i, s, th):
    """R and R_s of the background map on the grid (s, theta)."""
    if hasattr(base, 'map_derivs'):                          # any figure with its own map (geometry.LevelGeometry)
        R, R_s = base.map_derivs(i, s[:, None], th[None, :])[:2]
        return np.asarray(R), np.asarray(R_s)
    e, de, _ = base.eps(i, s[:, None])
    P2 = 0.5 * (3 * np.cos(th)**2 - 1)[None, :]
    S = s[:, None]
    return S * (1 - 2.0 / 3.0 * e * P2), 1 - 2.0 / 3.0 * (e + S * de) * P2


def _p2(th):
    c, s = np.cos(th), np.sin(th)
    return 0.5 * (3 * c * c - 1), -3 * c * s, -3 * (c * c - s * s)




# ---------------------------------------------------------------------------------------------
# The final state supported on the FLATTENED, rotating background (equilibrium on the tidal mesh)
# ---------------------------------------------------------------------------------------------
class FlatState:
    """Final state of the mantle loads a_k (amps, shells of sup) supported on the flattened, rotating
    hydrostatic background itself, so that it is in equilibrium on the mesh of the tidal solver to
    second order in the excess (a support on the spherical background is consistent only to
    eps x excess, which the tilt-over mode amplifies: angular-momentum balance 1e-1 instead of 1e-3).

    Static solve: TidalSolver(md_static, base, m = 0, degrees 0..6, forcing none, consistent
    reference, rotating, Eulerian-fixed mantle) with the body force drho grad Phi0 + rho0 grad dPhi of
    the load drho = a rho0(s) P2 (mapped coordinates); dPhi by Poisson's equation for the load alone
    on the same mapped grid. Final state on the tidal map R = R_bg(s, theta) + eta(s, theta):
      solids  eta = w_r (all harmonics); tau = Gamma:eps(w) + w_theta (d_theta p0)/r I (the tangential
              part of w moves the point off the coordinate surface of the flattened p0); extra
              density: load + rho0' s_r w_r (fixed), -rho0 div w - w_theta rho0' s_t / r (free solids)
      core    hydrostatic: level surfaces zeta = -dPhi_E / d_r Phi0, eta = zeta + q (q linear between
              the boundary mismatches), extra density rho0' s_r q, pressure dp = p0' s_r q
    with p0' = -rho0 (g - 2/3 Omega^2 s) of the background."""

    def __init__(self, sup, amps, base, omega_rot, N=24, degs=(0, 2, 4, 6)):
        from numpy.polynomial import Chebyshev as Ch
        from .refpot import ReferencePotential
        from .solver import TidalSolver
        md, mds = sup.md, sup.mds
        self.md, self.mds, self.sup, self.base, self.Om, self.N = md, mds, sup, base, omega_rot, N
        self.ioc = sup.ioc
        self.fixed = list(sup.fixed)
        self.load = {i: float(sum(a for a, lay in zip(amps, sup.layers) if i in lay))
                     for i in range(md.n_layers)}

        def loadf(i, s2, th, mp):
            a = self.load.get(i, 0.0)
            if a == 0.0:
                return None
            return (a * md.rho[i](s2) * _p2(th)[0] + 0 * mp['R'],)
        self.loadf = loadf
        mz = copy.copy(md)
        mz.rho = [Ch([0.0], domain=[md.r0[i], md.r1[i]]) for i in range(md.n_layers)]
        self.rp_load = ReferencePotential(mz, base, 0.0, drho=loadf)
        Ts = TidalSolver(mds, base, m=0, N=N, n_force=2, degrees=(list(degs), []), forcing='none',
                         omega_f=0.0, omega_rot=omega_rot, consistent_reference=True,
                         fixed_density_layers=self.fixed)
        Ts.solve()
        f = np.zeros_like(Ts.f)
        nb, nfld = Ts.nfld * N, Ts.nfld
        xin = -np.cos(np.pi * np.arange(1, N - 1) / (N - 1))
        th = Ts.theta[0]
        for i in range(md.n_layers):
            r0, r1 = md.r0[i], md.r1[i]
            s = r0 + (xin + 1) * (r1 - r0) / 2
            p = Ts.props(i, s)
            dP = self.rp_load.phys_derivs(i, s, th, p['map'])
            d = loadf(i, s[:, None], th, p['map'])
            d = 0.0 if d is None else d[0]
            Fr = d * p.ph0['r'] + p.rho * dP['r']
            Ft = (d * p.ph0['t'] + p.rho * dP['t']) / p.r
            Q = Ts._project(Fr, Ft, 0 * Fr, 0 * Fr)
            for k in range(N - 2):
                f[i * nb + nfld * (1 + k): i * nb + nfld * (2 + k)] += Q[k]
        Ts.coef = Ts._solve_linear(Ts.Hsp, f)
        self.Ts = Ts
        self._ev = {}
        self._cache = {}
        # viscous support (viscous_model): in the viscous layers w ~ 1/s is no deflection; the map is
        # interpolated linearly in s between anchors (density jumps and the edges of the viscous
        # region), where the deflections are finite (dynamic topography)
        self.viscous = list(getattr(mds, 'viscous_layers', []))
        self.anchors = []
        if self.viscous:
            v = sorted(self.viscous)
            rad = [md.r0[v[0]]] + [md.r1[j] for j in v[:-1]
                                   if abs(float(md.rho[j](md.r1[j])) - float(md.rho[j + 1](md.r1[j]))) > 1.0] \
                + [md.r1[v[-1]]]
            self.anchors = rad

    # -- evaluation of the static solution at arbitrary theta ---------------------------------
    def _solver_at(self, th):
        from .basis import ybar
        key = (len(th), float(th[0]), float(th[-1]))
        if key not in self._ev:
            T2 = copy.copy(self.Ts)
            T2.theta = th[None, None, :]
            T2.ang = {n: dict(Y=ybar(n, 0, th, 3), mYs=[0 * th] * 3) for n in self.Ts.ang}
            self._ev[key] = T2
        return self._ev[key]

    def fields(self, i, s1d, th):
        """props p (background map) and the summed jets (v, r, t, rr, rt, tt) of w_r, w_theta, phi1."""
        from .jets import _D
        th = np.asarray(th, float).ravel()
        s1d = np.atleast_1d(np.asarray(s1d, float))
        key = ('f', i, tuple(np.round(s1d, 4)), len(th), float(th[0]))
        if key in self._cache:
            return self._cache[key]
        T2 = self._solver_at(th)
        md, N = self.md, self.N
        x = 2 * (s1d - md.r0[i]) / (md.r1[i] - md.r0[i]) - 1
        p = T2.props(i, s1d)
        nb = T2.nfld * N
        tot = [{k: 0.0 for k in ('v', 'r', 't', 'rr', 'rt', 'tt')} for _ in range(4)]
        for j, fl in enumerate(T2._fields_at(i, x, p)):
            c = T2.coef[i * nb + j * N: i * nb + (j + 1) * N]
            for comp, d in enumerate(fl):
                for k in tot[comp]:
                    v = d[k] if k in d else 0.0
                    if np.ndim(v):
                        tot[comp][k] = tot[comp][k] + np.tensordot(c, np.broadcast_to(v, (N,) + p.r.shape), axes=1)
        out = (p,) + tuple(_D({k: v + 0 * p.r for k, v in t.items()}) for t in tot)
        self._cache[key] = out
        return out

    def _p0(self, i, s):
        """Background pressure gradient p0'(s) and p0''(s). First-order figure: -rho0 (g - 2/3 Omega^2 s).
        A base with its own map (level figure): p0 is constant on the coordinate surfaces, p0' = -rho0 dPhi0/ds
        with dPhi0/ds = (d_r Phi0) R_s of the reference potential (mean over theta; on a level figure it does
        not depend on theta)."""
        md, G4 = self.md, 4 * np.pi * G
        rho = md.rho[i]
        if hasattr(self.base, 'map_derivs'):
            sa = np.asarray(s, float)
            s1 = sa[:, 0] if sa.ndim == 2 else np.atleast_1d(sa)
            p = self.Ts.props(i, s1)
            m, w = p['map'], np.asarray(self.Ts.w).ravel()
            mean = lambda f: (f * w[None, :]).sum(axis=1) / w.sum()
            W1 = mean(p.ph0['r'] / m['s_r'])
            W1s = mean(p.ph0['rr'] / m['s_r']**2 - p.ph0['r'] * m['s_rr'] / m['s_r']**3)
            a, b = -rho(s1) * W1, -rho.deriv()(s1) * W1 - rho(s1) * W1s
            return (a[:, None] + 0 * sa, b[:, None] + 0 * sa) if sa.ndim == 2 else (a, b)
        g = md.gravity(i, s)
        W1 = g - 2.0 / 3.0 * self.Om**2 * s
        W1s = G4 * rho(s) - 2 * g / s - 2.0 / 3.0 * self.Om**2
        return -rho(s) * W1, -rho.deriv()(s) * W1 - rho(s) * W1s

    def _load_value(self, i, s1d, th):
        """dPhi of the load (value) at (s, theta) on the background map."""
        from .basis import cheb_basis, ybar
        rp, md = self.rp_load, self.md
        x = 2 * (np.atleast_1d(s1d) - md.r0[i]) / (md.r1[i] - md.r0[i]) - 1
        T = cheb_basis(x, rp.N)[0]
        v = 0.0
        for j, n in enumerate(rp.deg):
            c = rp.coef[i * rp.nb + j * rp.N: i * rp.nb + (j + 1) * rp.N]
            v = v + (c @ T)[:, None] * ybar(n, 0, th, 0)[0][None, :]
        return v

    def _zeta(self, i, s1d, th):
        """Radial deflection of the core's level surfaces, -dPhi_E / d_r Phi0 (background map)."""
        p, _, _, _, P = self.fields(i, s1d, th)
        return -(self._load_value(i, s1d, th) + P.v) / p.ph0['r']

    def _q(self, s1d, th):
        md, i = self.md, self.ioc
        r0, r1 = md.r0[i], md.r1[i]
        th = np.asarray(th, float).ravel()
        DI = self.eta2(i - 1, np.array([r0]), th)[0] - self._zeta(i, np.array([r0]), th)[0]
        DC = self.eta2(i + 1, np.array([r1]), th)[0] - self._zeta(i, np.array([r1]), th)[0]
        x = ((np.atleast_1d(s1d) - r0) / (r1 - r0))[:, None]
        return DI[None, :] * (1 - x) + DC[None, :] * x

    # -- the map ------------------------------------------------------------------------------
    def _eta_material(self, i, s1d, th):
        """Deflection of the material surfaces, w_r - (w_theta / r) dR_bg/dtheta|_s."""
        p, A, B, _, _ = self.fields(i, s1d, th)
        return A.v - B.v / p.r * p['map']['R_t']

    def _interval(self, s):
        """Anchor interval (ra, rb) containing s (viscous layers)."""
        a = self.anchors
        k = int(np.clip(np.searchsorted(a, s, side='right') - 1, 0, len(a) - 2))
        return a[k], a[k + 1]

    def _layer_at(self, r, above):
        md = self.md
        j = int(np.searchsorted(md.r1, r - 1.0)) if not above else int(np.searchsorted(md.r1, r + 1.0))
        return min(j, md.n_layers - 1)

    def _eta_viscous(self, s1d, th):
        """Linear interpolation between the material deflections at the anchors."""
        s1d = np.atleast_1d(np.asarray(s1d, float))
        th = np.asarray(th, float).ravel()
        out = np.zeros((len(s1d), len(th)))
        for k, sv in enumerate(s1d):
            ra, rb = self._interval(sv)
            ea = self._eta_material(self._layer_at(ra, True), np.array([ra]), th)[0]
            eb = self._eta_material(self._layer_at(rb, False), np.array([rb]), th)[0]
            x = (sv - ra) / (rb - ra)
            out[k] = (1 - x) * ea + x * eb
        return out

    def eta2(self, i, s1d, th):
        """eta(s, theta) on the background map (array (ns, ntheta))."""
        if i == self.ioc:
            return self._zeta(i, s1d, th) + self._q(s1d, th)
        if i in self.viscous:
            return self._eta_viscous(s1d, th)
        return self._eta_material(i, s1d, th)

    def _eta_jet(self, i, s, t, p):
        """eta as a jet (value, d/dr, d/dtheta at fixed r) on the background map."""
        from .jets import J
        m = p['map']
        md = self.md
        h, kk = 1e-4 * (md.r1[i] - md.r0[i]), 1e-5
        e0, es, _ = _sderivs(lambda ss: self.eta2(i, ss, t), s, md.r0[i], md.r1[i])
        et = (self.eta2(i, s, t + kk) - self.eta2(i, s, t - kk)) / (2 * kk)
        return J(e0, es * m['s_r'], et + es * m['s_t'])

    # -- data for TidalSolver(nonhydro=...) ----------------------------------------------------
    def _sth(self, s2, th):
        return np.atleast_1d(np.asarray(s2, float)).ravel(), np.asarray(th, float).ravel()

    def drho(self, i, s2, th, mp):
        from .jets import J
        s, t = self._sth(s2, th)
        p, A, B, _, _ = self.fields(i, s, t)
        m = p['map']
        md = self.md
        rho = md.rho[i]
        S = s[:, None] + 0 * p.r
        r0 = J(rho(S), rho.deriv()(S) * m['s_r'], rho.deriv()(S) * m['s_t'])
        r1 = J(rho.deriv()(S), rho.deriv(2)(S) * m['s_r'], rho.deriv(2)(S) * m['s_t'])
        sr = J(m['s_r'], m['s_rr'], m['s_rt'])
        st = J(m['s_t'], m['s_rt'], m['s_tt'])
        R = J(p.r, 1.0, 0.0)
        Aj, Bj = J(A.v, A.r, A.t), J(B.v, B.r, B.t)
        if i == self.ioc:
            from .geometry import chain
            q = self._q(s, t)
            h = 1e-3 * (md.r1[i] - md.r0[i])
            qs = (self._q(s + h, t) - self._q(s - h, t)) / (2 * h)
            kk = 1e-5
            qt = (self._q(s, t + kk) - self._q(s, t - kk)) / (2 * kk)
            qq = J(q, qs * m['s_r'], qt + qs * m['s_t'])
            d = r1 * sr * qq
        elif i in self.fixed:
            a = self.load.get(i, 0.0)
            P, Pt, _ = _p2(t)
            ld = J(a * rho(S) * P, a * rho.deriv()(S) * P * m['s_r'],
                   a * (rho(S) * Pt + rho.deriv()(S) * P * m['s_t']))
            if i in self.viscous:
                eta = self._eta_jet(i, s, t, p)
            else:
                Rt = J(m['R_t'], *self._Rt_derivs(i, s, t, p))
                eta = Aj - Bj / R * Rt
            d = ld + r1 * sr * eta
        else:
            e = self.Ts._stress(A, B, _ZERO, p)
            d = -1.0 * r0 * e.D
        return d.v, d.r, d.t

    def tau(self, i, s2, th, mp):
        from .jets import J
        if i == self.ioc:
            return None
        s, t = self._sth(s2, th)
        p, A, B, _, _ = self.fields(i, s, t)
        m = p['map']
        e = self.Ts._stress(A, B, _ZERO, p)
        out = dict(rr=e.trr, tt=e.ttt, pp=e.tpp, rt=e.trt)
        if i in self.viscous:
            # the map is not material here: the stress at the mapped point is the Eulerian stress of
            # the final state there, Gamma:eps(w) + ((w - eta r).grad p0) I (zero for the material map)
            S = s[:, None] + 0 * p.r
            p1, p2 = self._p0(i, S)
            P0 = J(p1, p2 * m['s_r'], p2 * m['s_t'])
            sr = J(m['s_r'], m['s_rr'], m['s_rt'])
            st = J(m['s_t'], m['s_rt'], m['s_tt'])
            R = J(p.r, 1.0, 0.0)
            eta = self._eta_jet(i, s, t, p)
            c = (J(A.v, A.r, A.t) - eta) * P0 * sr + J(B.v, B.r, B.t) * P0 * st / R
            out = {k: (v + c if k in ('rr', 'tt', 'pp') else v) for k, v in out.items()}
        return out

    def core_dp(self, s1d, th):
        """Static pressure change of the fluid core at the mapped points (s, theta) (array (ns, ntheta)),
        relative to the background p0(s): the isotropic stress of the weak-solid core of the static solve,
        moved to the mapped point as in tau() for the viscous layers, dp = -(sigma_iso + (w - eta r).grad p0).
        It balances the solids' tau_nn at the CMB and the ICB (degree 0 included), which the gauge form
        p0' s_r q does not (2026-10-31, task C: 1.5-2.4e5 Pa in degree 0; the welded viscous FCN overshoots)."""
        s = np.atleast_1d(np.asarray(s1d, float))
        t = np.asarray(th, float).ravel()
        i = self.ioc
        p, A, B, _, _ = self.fields(i, s, t)
        e = self.Ts._stress(A, B, _ZERO, p)
        iso = (e.trr.v + e.ttt.v + e.tpp.v) / 3.0
        p1, _ = self._p0(i, s)
        m = p['map']
        eta = self.eta2(i, s, t)
        c = p1[:, None] * ((A.v - eta) * m['s_r'] + B.v * m['s_t'] / p.r)
        return -(iso + c)

    def _Rt_derivs(self, i, s, t, p):
        """Physical d/dr and d/dtheta (fixed r) of R_t = dR/dtheta|_s of the background map."""
        m = p['map']
        if hasattr(self.base, 'map_derivs'):
            d = self.base.map_derivs(i, s[:, None], t[None, :])
            Rt_s, Rt_t = np.asarray(d[4]), np.asarray(d[5])
            return Rt_s * m['s_r'], Rt_t + Rt_s * m['s_t']
        e, de, _ = self.base.eps(i, s[:, None])
        c, sn = np.cos(t)[None, :], np.sin(t)[None, :]
        dP2, ddP2 = -3 * c * sn, -3 * (c * c - sn * sn)
        S = s[:, None]
        Rt_s = -2.0 / 3.0 * (e + S * de) * dP2
        Rt_t = -2.0 / 3.0 * S * e * ddP2
        m = p['map']
        return Rt_s * m['s_r'], Rt_t + Rt_s * m['s_t']

    def slip(self, ifl, rb, th):
        s = np.array([rb], float)
        t = np.asarray(th, float).ravel()
        p1, _ = self._p0(ifl, s)
        def dpf(tt):
            return p1[0] * self.fields(ifl, s, tt)[0]['map']['s_r'][0] * self._q(s, tt)[0]
        kk = 1e-5
        return dpf(t), (dpf(t + kk) - dpf(t - kk)) / (2 * kk)


from .jets import _D as _Dj
_ZERO = _Dj(v=0.0, r=0.0, t=0.0, rr=0.0, rt=0.0, tt=0.0)


def _sderivs(f, s, r0, r1, rel=1e-4):
    """f(s), f'(s), f''(s) by finite differences whose stencil stays inside [r0, r1] (one-sided,
    second order, at the ends): outside its layer the background map is clipped."""
    s = np.asarray(s, float)
    h = rel * (r1 - r0)
    f0 = f(s)
    out1, out2 = np.zeros_like(f0), np.zeros_like(f0)
    lo, hi = s - h < r0, s + h > r1
    mid = ~(lo | hi)
    if mid.any():
        fp, fm = f(s[mid] + h), f(s[mid] - h)
        out1[mid], out2[mid] = (fp - fm) / (2 * h), (fp - 2 * f0[mid] + fm) / h**2
    for sel, sg in ((lo, 1.0), (hi, -1.0)):
        if sel.any():
            f1, f2, f3 = f(s[sel] + sg * h), f(s[sel] + 2 * sg * h), f(s[sel] + 3 * sg * h)
            out1[sel] = sg * (-3 * f0[sel] + 4 * f1 - f2) / (2 * h)
            out2[sel] = (2 * f0[sel] - 5 * f1 + 4 * f2 - f3) / h**2
    return f0, out1, out2


class FlatMap:
    """Geometry for TidalSolver: the background map R_bg(s, theta) (Clairaut, eps of base) plus the
    deflection eta(s, theta) of a FlatState (all harmonics), with s and theta derivatives (eta
    differentiated numerically: it is smooth in each layer)."""

    def __init__(self, base, state):
        self.base, self.state = base, state
        if state.base is not base:
            raise ValueError('FlatMap: the state was built on another figure than this base')
        if hasattr(base, 'map_derivs'):                      # conventions from the map (geometry.surface_eps)
            self.surface_eps = self._surface_eps
            self.equatorial_radius = self._equatorial_radius

    def _surface_eps(self):
        from .geometry import surface_eps
        md = self.state.md
        e, de = surface_eps(self.base, md)
        d = self._deps(md.n_layers - 1, np.array([md.a]))
        return e + float(d[0][0]), de + float(d[1][0])

    def _equatorial_radius(self):
        from .geometry import polar_equatorial_radius
        return polar_equatorial_radius(self, self.state.md)[1]

    def map_derivs(self, i, s, th):
        s = np.asarray(s, float)
        th = np.asarray(th, float)
        S1 = np.unique(s.ravel())
        T1 = th.ravel() if th.ndim else np.atleast_1d(th)
        if hasattr(self.base, 'map_derivs'):
            R, R_s, R_ss, R_t, R_st, R_tt = (np.asarray(v) + np.zeros((len(S1), len(T1)))
                                             for v in self.base.map_derivs(i, S1[:, None], T1[None, :]))
        else:
            e, de, dde = self.base.eps(i, S1[:, None])
            c, sn = np.cos(T1)[None, :], np.sin(T1)[None, :]
            P2, dP2, ddP2 = 0.5 * (3 * c * c - 1), -3 * c * sn, -3 * (c * c - sn * sn)
            f = 2.0 / 3.0
            Sc = S1[:, None]
            R = Sc * (1 - f * e * P2)
            R_s = 1 - f * (e + Sc * de) * P2
            R_ss = -f * (2 * de + Sc * dde) * P2
            R_t = -f * Sc * e * dP2
            R_st = -f * (e + Sc * de) * dP2
            R_tt = -f * Sc * e * ddP2
        st = self.state
        k = 1e-4
        E = lambda ss, tt: st.eta2(i, ss, tt)
        e0, es, ess = _sderivs(lambda ss: E(ss, T1), S1, st.md.r0[i], st.md.r1[i])
        et, etm = E(S1, T1 + k), E(S1, T1 - k)
        _, est_p, _ = _sderivs(lambda ss: E(ss, T1 + k), S1, st.md.r0[i], st.md.r1[i])
        _, est_m, _ = _sderivs(lambda ss: E(ss, T1 - k), S1, st.md.r0[i], st.md.r1[i])
        R = R + e0
        R_s = R_s + es
        R_ss = R_ss + ess
        R_t = R_t + (et - etm) / (2 * k)
        R_tt = R_tt + (et - 2 * e0 + etm) / k**2
        R_st = R_st + (est_p - est_m) / (2 * k)
        shape = np.broadcast(s, th).shape
        idx = np.searchsorted(S1, np.atleast_1d(s).ravel())
        out = [np.broadcast_to(X[idx].reshape(shape), shape) for X in (R, R_s, R_ss, R_t, R_st, R_tt)]
        return tuple(out)

    def eps(self, i, s):
        """Effective ellipticity (P2 part of the map) and its first two s-derivatives: the
        background's plus -3/2 eta20 / s (used for the conventions at the surface)."""
        s = np.atleast_1d(np.asarray(s, float))
        e, de, dde = self.base.eps(i, s)
        x, w = np.polynomial.legendre.leggauss(24)
        th = np.arccos(x)
        P2 = 0.5 * (3 * x * x - 1)
        n20 = lambda ss: 2.5 * (self.state.eta2(i, ss, th) * (w * P2)[None, :]).sum(axis=-1)
        n, dn, ddn = _sderivs(n20, s, self.state.md.r0[i], self.state.md.r1[i])
        return (e - 1.5 * n / s, de - 1.5 * (dn / s - n / s**2),
                dde - 1.5 * (ddn / s - 2 * dn / s**2 + 2 * n / s**3))

    def _deps(self, i, s):
        """The state's part of eps() and of its first s-derivative: -3/2 eta20 / s."""
        s = np.atleast_1d(np.asarray(s, float))
        x, w = np.polynomial.legendre.leggauss(24)
        th = np.arccos(x)
        P2 = 0.5 * (3 * x * x - 1)
        n20 = lambda ss: 2.5 * (self.state.eta2(i, ss, th) * (w * P2)[None, :]).sum(axis=-1)
        n, dn, _ = _sderivs(n20, s, self.state.md.r0[i], self.state.md.r1[i])
        return -1.5 * n / s, -1.5 * (dn / s - n / s**2)


SUPPORTS = ('elastic', 'viscous')


class NonHydrostatic:
    """Non-hydrostatic state from a set-up table (dict), e.g.
        support = "elastic"
        shells_km = [[3480, 5701], [5701, 6346.6]]
        targets = {J2 = 1.14e-5, e_f = 9.73e-5}
        sigma = {...}, prior = 5e-3          (optional: regularised fit)
    md: tidal model (true fluid core); md_static: the same layers with a weak-solid core (default:
    PREMs for PREM). Attributes: support, amps, response (the fitted targets), state, geometry."""

    def __init__(self, md, settings, base, omega_rot, md_static=None):
        st = dict(settings)
        bad = [k for k in st if k not in ('support', 'shells_km', 'targets', 'sigma', 'prior', 'model_static',
                                          'viscosity_scale', 'eta_lower', 'eta_upper')]
        if bad:
            raise ValueError(f'unknown nonhydrostatic keys {bad}')
        kind = st.get('support', 'elastic')
        if kind not in SUPPORTS:
            raise ValueError(f'support must be one of {SUPPORTS}')
        if md_static is None:
            raise ValueError('a static model with a weak-solid core is needed (e.g. PREMs)')
        shells = [(a * 1e3, b * 1e3) for a, b in st.get('shells_km', [[3480, 5701], [5701, 6346.6]])]
        targets = st.get('targets')
        if not targets:
            raise ValueError('nonhydrostatic: targets missing')
        bad = [k for k in targets if k not in KEYS]
        if bad:
            raise ValueError(f'unknown targets {bad}; allowed {KEYS}')
        if kind == 'viscous':
            # instantaneous Stokes flow as the limit of the static support: mantle mu = s eta_rel
            # (viscosity_scale s [Pa], relative viscosities of the lower and upper mantle; 80 km
            # elastic lid); s = 2e6 converged to a few per cent
            lower, upper = st.get('eta_lower', 50.0), st.get('eta_upper', 1.0)
            eta_rel = {i: (lower if md_static.r1[i] <= 5701e3 + 1 else upper) for i in range(md_static.n_layers)}
            vm = viscous_model(md_static, st.get('viscosity_scale', 2e6), eta_rel)
            md_static = vm
        self.support = FlatSupport(md, md_static, shells, base, omega_rot)
        if 'sigma' in st or 'prior' in st:
            self.amps = self.support.fit(targets, st.get('sigma'), st.get('prior'))
        else:
            if len(targets) != len(shells):
                raise ValueError('exact fit: as many targets as shells (or give sigma and prior)')
            self.amps = self.support.fit(targets)
        self.response = self.support.response(self.amps)
        self.state = FlatState(self.support, self.amps, base, omega_rot)
        self.geometry = FlatMap(base, self.state)

    def summary(self):
        """Fitted amplitudes and the resulting targets (for the table metadata)."""
        return dict(amplitudes=[float(a) for a in self.amps],
                    response={k: float(v) for k, v in self.response.items()})
