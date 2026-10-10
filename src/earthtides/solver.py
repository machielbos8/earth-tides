# -*- coding: utf-8 -*-
#
# Tidal response of an axisymmetric (ellipsoidal) elastic, self-gravitating
# Earth for a degree-nf, order-m forcing.
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
"""Tidal response of an axisymmetric (ellipsoidal) elastic, self-gravitating
Earth for a degree-nf, order-m forcing.

Unknowns, as functions of the level-surface label s and colatitude theta
(physical spherical components at the point r = R(s, theta)), with the
lambda dependence cos(m lambda) for u_r, u_theta, phi1 and sin(m lambda) for
u_lambda (Ybar = Schmidt semi-normalised P_n^m(cos theta)):

    u_r   =  sum U_n Ybar_n
    u_th  =  sum V_n dYbar_n/dth  -  sum W_n m Ybar_n / sin th
    u_lam = -sum V_n m Ybar_n / sin th  +  sum W_n dYbar_n/dth
    phi1  =  sum P_n Ybar_n

U, V, P run over the spheroidal degrees n >= m with n - nf even; W over the
toroidal degrees with n - nf odd (absent for m = 0). Each radial function is
a Chebyshev series per layer (as in astyanax). The equations are evaluated
pointwise in physical coordinates, using first-order automatic
differentiation ("jets") for the stress divergence, and projected onto the
scalar / spheroidal / toroidal test functions by Gauss-Legendre quadrature.
No coupling coefficient is derived by hand.

Field equations (hydrostatic pre-stress, Lagrangian form, Phi0 = potential
energy with grad Phi0 = +g r_hat for a sphere):

    div tau - rho grad(u . grad Phi0) + rho (div u) grad Phi0
            - rho grad(phi1 + phiT) + rho omega^2 u  [+ alpha rho y_hat x r] = 0
    lap phi1 + 4 pi G div(rho u) = 0

Interfaces between two solids (or two fluids) are welded: u, phi1, n.tau and the gravity flux
n.grad phi1 + 4 pi G rho u.n are continuous. At a fluid-solid interface (true fluid, mu = 0) the
fluid slides: u.n, n.tau.n, phi1 and the flux are continuous, the tangential traction vanishes on
the solid side and the fluid's tangential momentum equation takes the freed rows (_fs_interface;
Tromp & Mitrovica 1999, eq. 3.7). A fluid needs inertia: its static limit is undetermined
(Longman 1963), so true-fluid models are solved at a tidal frequency; PREMs (weak-solid core) is for
static runs. Free surface n.tau = 0 (load: -g sigma n), phi1 matched to sum c_n (a/r)^(n+1) Ybar_n.
The manual (docs/manual) explains the assembly step by step.

m = 1: a degree-2 order-1 tide torques the bulges of mantle and core;
without Coriolis there is no static balance (the real Earth nutates, and the
core's differential tilt is the FCN mode, at zero frequency without
rotation). As in Mathews et al., the deformation is computed with the
rigid rotation of each region (inner core, outer core, mantle) removed: per
region an unknown inertial force alpha_k rho (y x r) and the Tisserand
condition int_region rho r x u dV = 0.

phiT = -g a (r/a)^nf Ybar_nf cos(m lambda); on a sphere h = U_nf(a)/a,
l = V_nf(a)/a, k = -c_nf/(g a).
"""

import numpy as np
from numpy.polynomial import legendre as L
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from .basis import cheb_basis, ybar, legendre_theta, _trig_derivs, _schmidt_reduced  # noqa: F401
from .geometry import mapping, mapping_from_R, chain, SphereGeometry
from .jets import J, _D, ZERO
from .model import G


class TidalSolver:
    """Tidal (or load) response of a layered, possibly flattened and rotating Earth for one order m
    and one forcing degree n_force (manual, chapters 'How the solver works' and 'API reference').

    Resolution: K spheroidal (KT toroidal) degrees per field, N Chebyshev polynomials per layer,
    n_theta Gauss-Legendre points; or explicit degrees=([spheroidal], [toroidal]).
    Forcing: forcing='tide' (potential -g a (r/a)^nf Ybar cos m lam), 'load' (unit surface mass
    density Ybar cos m lam) or 'none'.
    Dynamics: omega_f (rad/s) inertia of a non-rotating run; coriolis (rotation rate) and omega_t
    (signed tidal frequency) for a rotating run; dyn_fluid adds them in the liquid layers;
    omega_rot is the rotation rate of the reference state; consistent_reference computes Phi0
    from the model's own density; gravity=False switches self-gravitation off.
    Frame: frame='mantle' | 'surface' for the rigid rotation removed from the output; wahr_frame
    lets the regions rotate freely (rotating runs with m = 0, 1).
    Linear algebra: linear_solver='sparse' (SuperLU) | 'dense' (LAPACK).
    Centre (spheroidal degree 1 only): centre='fixed' pins the centre, U_1 = V_1 = 0 (the frame of
    static runs, where a translation is free); centre='free' lets it move, U_1 = V_1, dU_1/ds = 0,
    P_1 = 0, so that inertia determines the translation (needs omega_f or omega_t; manual,
    'Degree 1').
    Experimental (paper 2): lateral, all_degrees, fixed_density_layers, eulerian_interfaces,
    background.
    solve() returns results(): U, V, W, k per degree and h, l, kk for the forcing degree.
    """

    def __init__(self, model, geometry=None, m=0, K=3, KT=None, N=24,
                 n_theta=None, n_force=2, omega_f=1.4053e-4, omega_rot=0.0,
                 background='level', tisserand=None, regions=None,
                 coriolis=0.0, omega_t=0.0, wahr_frame=False, dyn_fluid=False,
                 gravity=True, consistent_reference=False, frame='surface',
                 forcing='tide', deg_window=None, lateral=None, all_degrees=False,
                 degrees=None, fixed_density_layers=(), eulerian_interfaces=False,
                 linear_solver='sparse', centre='fixed', load_inertia=True):
        self.model = model
        self.geom = geometry if geometry is not None else SphereGeometry()
        self.m, self.N, self.nf = m, N, n_force
        self.omega_f, self.omega_rot = omega_f, omega_rot
        self.background = background
        # rotating-frame dynamics in the solid layers only: Coriolis (rate
        # `coriolis`, rad/s) and inertia at the signed tidal frequency omega_t
        self.coriolis, self.omega_t = coriolis, omega_t
        # gravity=False: no pre-stress, no self-gravitation (pure elasticity)
        self.gravity = gravity
        self.G = G if gravity else 0.0
        # consistent_reference: Phi0 from a Poisson solve for the model density
        # plus the exact centrifugal potential (see refpot.py)
        # frame: 'surface' = best-fit rotation of the surface displacement (Wahr
        # 1981, eq. 4.3); 'mantle' = mean (Tisserand) rotation of the mantle volume,
        # the terrestrial frame of Mathews et al. (1995) and de Vries & Wahr (1991)
        self.frame = frame
        # forcing='tide': degree-n_force tidal potential (body force);
        # forcing='load': unit surface mass density Ybar_nf^m cos(m lam) [kg/m^2]
        # on the outer surface (normal traction -g sigma, flux jump 4 pi G sigma)
        # forcing='none': no right-hand side (the matrix for a user right-hand side, born.py)
        if forcing not in ('tide', 'load', 'none'):
            raise ValueError(forcing)
        self.forcing = forcing
        # fixed_density_layers: layers where the EULERIAN density is held fixed (static support of
        # a prescribed final density, born.StaticSupport): the terms div(rho u) grad Phi0 and the
        # volume source of Poisson's equation are dropped there (interface masses remain)
        self.fixed_density_layers = tuple(fixed_density_layers)
        # eulerian_interfaces: between two fixed-density layers the welded conditions are on the
        # EULERIAN traction (tau - rho0 g u.n n) and on n.grad phi1 alone: a deflected density jump
        # carries no load or surface mass there (phase boundaries crossed by flow; whole-mantle
        # flow in the viscous limit of born.StaticSupport). Default: Lagrangian traction and
        # flux with 4 pi G rho u.n (material interfaces).
        self.eulerian_interfaces = eulerian_interfaces
        # linear_solver: 'sparse' (SuperLU with COLAMD ordering; the matrix is block-banded by
        # layer, bordered by a few dense rows and columns, ~8 per cent non-zero) or 'dense' (LAPACK).
        # Both give residuals at round-off; they differ only in the poorly determined directions of
        # the matrix (high degrees in the fluid core, condition number up to ~1e14 at M2), which
        # change the Love numbers by < 1e-9.
        if linear_solver not in ('sparse', 'dense'):
            raise ValueError(linear_solver)
        self.linear_solver = linear_solver
        # centre: 'fixed' (U_1 = V_1 = 0 at s = 0) or 'free' (U_1 = V_1, dU_1/ds = 0: the centre
        # moves, the translation follows from inertia; a static run would be singular)
        if centre not in ('fixed', 'free'):
            raise ValueError(centre)
        if centre == 'free' and omega_f == 0.0 and omega_t == 0.0:
            raise ValueError("centre='free' needs inertia (omega_f or omega_t): a static "
                             "translation is undetermined; use centre='fixed'")
        self.centre = centre
        # load_inertia: in a rotating run the load material must move to redistribute its mass; the
        # inertial and Coriolis forces of that motion add the tangential traction
        # -(d Lambda / dt) Omega x r (Sasao & Wahr 1981, eq. 4.14), which supplies the torque
        # d(c_M . Omega)/dt and decides the sign of the free-core-nutation resonance of the load
        # Love numbers. False: pressure and potential only (the non-rotating load problem).
        self.load_inertia = load_inertia
        self.refpot = None
        if consistent_reference:
            from .refpot import ReferencePotential
            self.refpot = ReferencePotential(model, self.geom, omega_rot)
        # lateral(i, s, theta) -> (f, f_s, f_theta): relative shear-modulus variation
        # mu = mu0(s) (1 + f) in layer i (zero layers: return 0) and its derivatives;
        # axisymmetric (m stays separate). all_degrees: keep both parities (needed when
        # the variation is not symmetric about the equator, e.g. f ~ cos theta).
        self.lateral = lateral
        if degrees is not None:         # explicit (spheroidal, toroidal) degree lists
            self.deg, degW = list(degrees[0]), list(degrees[1])
            KT = len(degW)
        elif all_degrees:
            self.deg = list(range(m, m + K))
            KT = K if KT is None else KT
            degW = list(range(max(m, 1), max(m, 1) + KT))
        elif deg_window is None:        # K degrees upward from m
            self.deg = [n for n in range(m, m + 2 * K + 2) if (n - n_force) % 2 == 0][:K]
            KT = K if KT is None else KT
            degW = [n for n in range(max(m, 1), max(m, 1) + 2 * KT + 2) if (n - n_force) % 2][:KT]
        else:                           # n_force +- 2 deg_window only (high-degree loads)
            w = deg_window
            self.deg = [n for n in range(n_force - 2 * w, n_force + 2 * w + 1, 2) if n >= m]
            degW = [n for n in range(n_force - 2 * w - 1, n_force + 2 * w + 2, 2) if n >= max(m, 1)]
        self.degV = [n for n in self.deg if n >= 1]
        self.degW = [] if (m == 0 and not coriolis and degrees is None) else degW
        self.K = len(self.deg)
        self.fields = [('U', n) for n in self.deg] + [('V', n) for n in self.degV] \
            + [('W', n) for n in self.degW] + [('P', n) for n in self.deg]
        self.nfld = len(self.fields)
        self.tilt = (m in (0, 1) and 1 in self.degW) if tisserand is None else tisserand
        # Tisserand regions: contiguous groups of solid / (weak-)fluid layers;
        # each gets its own rigid-rotation inertial force and zero-rotation constraint
        if regions is None:
            fluid = self.fluid_layers()
            regions, cur = [], [0]
            for i in range(1, model.n_layers):
                if fluid[i] == fluid[i - 1]:
                    cur.append(i)
                else:
                    regions.append(cur)
                    cur = [i]
            regions.append(cur)
        # wahr_frame: solid regions rotate freely (their rigid rotation is set by
        # inertia + Coriolis, i.e. the forced nutation / rotation-rate change, whose
        # centrifugal and Coriolis effects stay in the body tide); only the static
        # weak-solid core keeps a Tisserand constraint. The rigid rotation is then
        # removed from the *surface* displacement only (Wahr 1981, eqs. 4.3, 4.6).
        self.wahr_frame, self.dyn_fluid = wahr_frame, dyn_fluid
        # layers with mu = 0 exactly are true fluids: slip at fluid-solid interfaces
        self.true_fluid = [abs(model.mu[i](0.5 * (model.r0[i] + model.r1[i]))) < 1.0
                           for i in range(model.n_layers)]
        if wahr_frame:
            fl = self.fluid_layers()
            regions = [] if dyn_fluid else [r for r in regions if all(fl[i] for i in r)]
        self.regions = regions if self.tilt else []
        self.H_dtype = complex if getattr(model, 'is_complex', False) else float
        n_theta = n_theta or 2 * max(self.deg + self.degW + [n_force]) + 24
        x, w = L.leggauss(n_theta)
        th = np.arccos(x)
        self.theta = th[None, None, :]
        self.w = w
        self.ang = {}
        for n in sorted(set(self.deg + self.degW)):
            y = ybar(n, m, th, 3)
            ys = ybar(n, m, th, 2, over_sin=True) if m > 0 else [0 * th] * 3
            self.ang[n] = dict(Y=y, mYs=[m * v for v in ys])
        # test functions (times quadrature weights); vector tests are (theta, lambda)
        self.testP = np.array([self.ang[n]['Y'][0] * w for n in self.deg])
        self.testV = [(self.ang[n]['Y'][1] * w, -self.ang[n]['mYs'][0] * w) for n in self.degV]
        self.testW = [(-self.ang[n]['mYs'][0] * w, self.ang[n]['Y'][1] * w) for n in self.degW]

    def fluid_layers(self):
        """(Weak-)fluid layers: mu below the model's weak_fluid_mu (default 1e8 Pa; lowered for
        the viscous-limit static models of prestress.FinalState, whose mantle mu is small)."""
        md = self.model
        cut = getattr(md, 'weak_fluid_mu', 1e8)
        return [abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) < cut for i in range(md.n_layers)]

    # ------------------------------------------------------------------
    def props(self, i, s):
        """Material, reference potential and geometry at s (Ns,) x theta."""
        md = self.model
        s2 = s[:, None]
        th = self.theta[0]
        if hasattr(self.geom, 'map_derivs'):
            m = mapping_from_R(*self.geom.map_derivs(i, s2, th))
        else:
            m = mapping(s2, th, *self.geom.eps(i, s2))
        r = m['R']
        p = _D(r=r, R_t=m['R_t'], s_r=m['s_r'], cot=1 / np.tan(th), isn=1 / np.sin(th),
               cos=np.cos(th) + 0 * r, sin=np.sin(th))
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.background == 'level':
                for name, f in (('rho', md.rho[i]), ('lam', md.lam[i]), ('mu', md.mu[i])):
                    d = f.deriv()(s2)
                    p[name] = f(s2) + 0 * r
                    p[name + '_r'] = d * m['s_r']
                    p[name + '_t'] = d * m['s_t']
                gs = md.gravity(i, s2)
                W1 = gs - 2.0 / 3.0 * self.omega_rot**2 * s2
                W2 = 4 * np.pi * self.G * md.rho[i](s2) - 2 * gs / s2 \
                    - 2.0 / 3.0 * self.omega_rot**2
                ph0 = chain(0.0, W1, W2, 0.0, 0.0, 0.0, m)
            else:   # spherical Earth on a (possibly distorted) mesh
                for name, f in (('rho', md.rho[i]), ('lam', md.lam[i]), ('mu', md.mu[i])):
                    p[name] = f(r)
                    p[name + '_r'] = f.deriv()(r)
                    p[name + '_t'] = 0 * r
                g = md.gravity(i, r)
                ph0 = dict(r=g, t=0 * r, rr=4 * np.pi * self.G * md.rho[i](r) - 2 * g / r,
                           rt=0 * r, tt=0 * r)
        if self.lateral is not None:
            f, f_s, f_t = (np.broadcast_to(v, r.shape) for v in self.lateral(i, s2, th))
            mu0, mu0_r, mu0_t = p['mu'], p['mu_r'], p['mu_t']
            p['mu'] = mu0 * (1 + f)
            p['mu_r'] = mu0_r * (1 + f) + mu0 * f_s * m['s_r']
            p['mu_t'] = mu0_t * (1 + f) + mu0 * (f_t + f_s * m['s_t'])
        if self.refpot is not None and self.background == 'level':
            ph0 = self.refpot.phys_derivs(i, s, th, m)
        if not self.gravity:
            ph0 = {k: 0 * r for k in ('r', 't', 'rr', 'rt', 'tt')}
        p['ph0'] = ph0
        p['map'] = m
        p['fixed'] = 1.0 if i in self.fixed_density_layers else 0.0
        p['solid'] = 0.0 if (self.fluid_layers()[i] and not self.dyn_fluid) else 1.0
        return p

    def _radial(self, i, x):
        """Chebyshev polynomials of layer i and their first two derivatives with respect to s."""
        T, T1, T2 = cheb_basis(x, self.N)
        sc = 2.0 / (self.model.r1[i] - self.model.r0[i])
        return T[:, :, None], (T1 * sc)[:, :, None], (T2 * sc * sc)[:, :, None]

    @staticmethod
    def _comp(A, ang, mp):
        """Chain-rule derivatives of radial (A, A', A'') x angular (b, b', b'')."""
        A0, A1, A2 = A
        b0, b1, b2 = ang[0], ang[1], ang[2]
        return _D(chain(A0 * b0, A1 * b0, A2 * b0, A0 * b1, A1 * b1, A0 * b2, mp))

    def _fields_at(self, i, x, p):
        """Per field: (u_r, u_theta, u_lambda, phi1) amplitude-derivative dicts."""
        A = self._radial(i, x)
        mp = p['map']
        out = []
        for typ, n in self.fields:
            a = self.ang[n]
            Y, mYs = a['Y'], a['mYs']
            neg = [-v for v in mYs]
            if typ == 'U':
                out.append((self._comp(A, Y, mp), ZERO, ZERO, ZERO))
            elif typ == 'P':
                out.append((ZERO, ZERO, ZERO, self._comp(A, Y, mp)))
            elif typ == 'V':
                out.append((ZERO, self._comp(A, Y[1:], mp), self._comp(A, neg, mp), ZERO))
            else:   # W
                out.append((ZERO, self._comp(A, neg, mp), self._comp(A, Y[1:], mp), ZERO))
        return out

    # ------------------------------------------------------------------
    def _stress(self, uA, uB, uC, p, jets=True):
        """Strain and stress amplitudes as jets (needs 2nd derivatives of u)."""
        m = self.m
        if jets:
            mk = lambda d: (J(d.v, d.r, d.t), J(d.r, d.rr, d.rt), J(d.t, d.rt, d.tt))
        else:
            mk = lambda d: (J(d.v), J(d.r), J(d.t))
        A, A_r, A_t = mk(uA)
        B, B_r, B_t = mk(uB)
        Cc, C_r, C_t = mk(uC)
        sn = p.sin
        R = J(p.r, 1.0, 0.0)
        cot = J(p.cot, 0.0, -1.0 / sn**2)
        isn = J(p.isn, 0.0, -p.cos / sn**2)
        lam = J(p.lam, p.lam_r, p.lam_t)
        mu = J(p.mu, p.mu_r, p.mu_t)
        err = A_r
        ett = (B_t + A) / R
        epp = (m * Cc * isn + A + B * cot) / R
        ert = 0.5 * (A_t / R + B_r - B / R)
        erp = 0.5 * ((-m) * A * isn / R + C_r - Cc / R)
        etp = 0.5 * ((C_t - Cc * cot) / R - m * B * isn / R)
        D = err + ett + epp
        lD = lam * D
        return _D(D=D, A=A, B=B, C=Cc,
                  trr=lD + 2 * mu * err, ttt=lD + 2 * mu * ett, tpp=lD + 2 * mu * epp,
                  trt=2 * mu * ert, trp=2 * mu * erp, ttp=2 * mu * etp)

    def operator(self, uA, uB, uC, ph, p):
        """Residuals E_r, E_theta, E_lambda of the momentum equation and E_phi of Poisson's
        equation for one trial field (displacement amplitudes uA, uB, uC and potential ph, with
        their derivatives) at the points of props p: elasticity, the gravitational terms of the
        hydrostatic reference state, self-gravitation, inertia and the Coriolis force."""
        m, r, cot, isn = self.m, p.r, p.cot, p.isn
        e = self._stress(uA, uB, uC, p)
        trr, ttt, tpp, trt, trp, ttp = e.trr, e.ttt, e.tpp, e.trt, e.trp, e.ttp
        div_r = trr.r + trt.t / r + m * trp.v * isn / r \
            + (2 * trr.v - ttt.v - tpp.v + trt.v * cot) / r
        div_t = trt.r + ttt.t / r + m * ttp.v * isn / r \
            + ((ttt.v - tpp.v) * cot + 3 * trt.v) / r
        div_p = trp.r + ttp.t / r - m * tpp.v * isn / r \
            + (3 * trp.v + 2 * ttp.v * cot) / r
        ph0 = p.ph0
        R = J(r, 1.0, 0.0)
        gr = J(ph0['r'], ph0['rr'], ph0['rt'])
        gt = J(ph0['t'], ph0['rt'], ph0['tt']) / R
        q = e.A * gr + e.B * gt
        rho, w2, D = p.rho, self.omega_f**2, e.D.v
        Er = div_r - rho * q.r + rho * D * gr.v - rho * ph.r + rho * w2 * e.A.v
        Et = div_t - rho * q.t / r + rho * D * gt.v - rho * ph.t / r + rho * w2 * e.B.v
        Ep = div_p + rho * m * q.v * isn / r + rho * m * ph.v * isn / r + rho * w2 * e.C.v
        if self.coriolis or self.omega_t:
            # fields are Re[(A, B, -iC) exp(i(omega t + m lambda))]; the terms
            # rho omega^2 u - 2 i omega rho Omega x u become real in (A, B, C)
            rs_, wt = rho * p.solid, self.omega_t
            cw = 2.0 * wt * self.coriolis
            sn, cs = p.sin, p.cos
            Er = Er + rs_ * (wt**2 * e.A.v + cw * sn * e.C.v)
            Et = Et + rs_ * (wt**2 * e.B.v + cw * cs * e.C.v)
            Ep = Ep + rs_ * (wt**2 * e.C.v + cw * (sn * e.A.v + cs * e.B.v))
        Eg = ph.rr + 2 * ph.r / r + (ph.tt + cot * ph.t) / r**2 - (m * isn / r)**2 * ph.v \
            + 4 * np.pi * self.G * (rho * D + p.rho_r * e.A.v + p.rho_t / r * e.B.v)
        if p.get('fixed', 0.0):              # Eulerian density fixed: drop div(rho u) terms
            dru = rho * D + p.rho_r * e.A.v + p.rho_t / r * e.B.v
            Er = Er - dru * gr.v
            Et = Et - dru * gt.v
            Eg = Eg - 4 * np.pi * self.G * dru
        return Er, Et, Ep, Eg

    def normal(self, p):
        """Unit normal (r, theta components) of the level surface at the points of props p."""
        nr, nt = p.r, -p.R_t
        h = np.hypot(nr, nt)
        return nr / h, nt / h

    def interface(self, uA, uB, uC, ph, p, eulerian=False):
        """n.tau (r, theta, lambda) and gravity flux n.grad phi1 + 4 pi G rho u.n; eulerian=True:
        n.(tau - rho0 g u.n I) and n.grad phi1 (see eulerian_interfaces)."""
        e = self._stress(uA, uB, uC, p, jets=False)
        nr, nt = self.normal(p)
        tr = e.trr.v * nr + e.trt.v * nt
        tt = e.trt.v * nr + e.ttt.v * nt
        tp = e.trp.v * nr + e.ttp.v * nt
        un = e.A.v * nr + e.B.v * nt
        if eulerian:
            gn = p.ph0['r'] * nr + p.ph0['t'] / p.r * nt
            return tr - p.rho * gn * un * nr, tt - p.rho * gn * un * nt, tp, ph.r * nr + ph.t / p.r * nt
        fl = ph.r * nr + ph.t / p.r * nt + 4 * np.pi * self.G * p.rho * un
        return tr, tt, tp, fl

    def _project(self, Er, Et, Ep, Eg):
        """Residual arrays (..., Nth) -> stacked equations (..., nfld)."""
        Er, Et, Ep, Eg = np.broadcast_arrays(Er, Et, Ep, Eg)
        cols = [np.einsum('...j,kj->...k', Er, self.testP)]
        for tests in (self.testV, self.testW):
            if tests:
                cols.append(np.stack([Et @ a + Ep @ b for a, b in tests], axis=-1))
        cols.append(np.einsum('...j,kj->...k', Eg, self.testP))
        return np.concatenate(cols, axis=-1)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _bnd(self, i, x, p):
        """Boundary quantities of every trial field of layer i at end x = +-1."""
        nr, nt = self.normal(p)
        out = []
        for uA, uB, uC, ph in self._fields_at(i, np.array([x]), p):
            e = self._stress(uA, uB, uC, p, jets=False)
            tr = e.trr.v * nr + e.trt.v * nt
            tt = e.trt.v * nr + e.ttt.v * nt
            tp = e.trp.v * nr + e.ttp.v * nt
            tnn = tr * nr + tt * nt
            un = e.A.v * nr + e.B.v * nt
            fl = ph.r * nr + ph.t / p.r * nt + 4 * np.pi * self.G * p.rho * un
            if self.true_fluid[i]:            # fluid: tangential momentum equation
                _, Et, Ep, _ = self.operator(uA, uB, uC, ph, p)
            else:                              # solid: tangential traction
                Et, Ep = tt - tnn * nt, tp
            out.append(dict(un=un, tnn=tnn, Et=Et, Ep=Ep, fl=fl, phi=ph.v))
        return out

    def _tidal_force_rows(self, p):
        """Projected tidal body force rho grad(phiT) at the points of props p."""
        md, nfo, m = self.model, self.nf, self.m
        a, ga = md.a, md.g_surface
        Yf = ybar(nfo, m, self.theta[0, 0], 1)
        rr = p.r
        phT = -ga * a * (rr / a)**nfo * Yf[0]
        return self._project(p.rho * nfo * phT / rr,
                             p.rho * (-ga * (rr / a)**(nfo - 1) * Yf[1]),
                             p.rho * (-m) * phT * p.isn / rr, 0 * rr)

    def _fs_interface(self, H, f, iL, rows_b, rows_t, col):
        """Rows for a fluid-solid interface between layer iL (below) and iL+1.

        Bottom rows of the upper layer: normal displacement and phi1 continuous,
        tangential condition of the upper layer. Top rows of the lower layer:
        normal traction and gravity flux continuous, tangential condition of the
        lower layer. Tangential condition: zero tangential traction (solid) or the
        tangential momentum equation at the boundary node (fluid). Valid on level
        surfaces (Tromp & Mitrovica 1999, eq. 3.7).
        """
        md, N = self.model, self.N
        iU, rI = iL + 1, md.r1[iL]
        pL, pU = self.props(iL, np.array([rI])), self.props(iU, np.array([rI]))
        BL, BU = self._bnd(iL, 1.0, pL), self._bnd(iU, -1.0, pU)
        zero = 0.0
        for j, b in enumerate(BL):
            H[rows_b, col(iL, j):col(iL, j) + N] = self._project(b['un'], zero, zero, b['phi'])[:, 0, :].T
            H[rows_t, col(iL, j):col(iL, j) + N] = self._project(b['tnn'], b['Et'], b['Ep'], b['fl'])[:, 0, :].T
        for j, b in enumerate(BU):
            H[rows_b, col(iU, j):col(iU, j) + N] = self._project(-b['un'], b['Et'], b['Ep'], -b['phi'])[:, 0, :].T
            H[rows_t, col(iU, j):col(iU, j) + N] = self._project(-b['tnn'], zero, zero, -b['fl'])[:, 0, :].T
        # tidal force in the fluid tangential rows
        nS0, nS1 = len(self.deg), len(self.deg) + len(self.degV) + len(self.degW)
        for lay, pp, rows in ((iL, pL, rows_t), (iU, pU, rows_b)):
            if self.true_fluid[lay] and self.forcing == 'tide':
                fr = self._tidal_force_rows(pp)[0]
                f[rows.start + nS0: rows.start + nS1] = fr[nS0:nS1]

    # ------------------------------------------------------------------
    # Assembly of the linear system (manual, 'Assembling the linear system'). Unknowns: per
    # layer i and field j the N Chebyshev coefficients (columns col(i, j) .. + N), then the K
    # exterior coefficients c_n, then one rigid-rotation force per Tisserand region. Rows per
    # layer: nfld bottom rows, nfld * (N - 2) interior rows, nfld top rows.
    def solve(self):
        """Assemble the system, solve it and return results()."""
        H, f = self._allocate()
        for i in range(self.model.n_layers):
            self._interior_rows(H, f, i)
            self._bottom_rows(H, i)
            self._top_rows(H, f, i)
        for ir, lays in enumerate(self.regions):
            H[self._c_alpha + ir, :self.model.n_layers * self._nb] = self._tisserand_row(lays)
        x = self._solve_linear(H, f)
        self.coef, self.H, self.f = x, H, f
        return self.results()

    def _allocate(self):
        """Empty matrix and right-hand side; index helpers of the layout."""
        md, N, nfld, K = self.model, self.N, self.nfld, self.K
        self._nb = nfld * N                                     # unknowns per layer
        self._nS = len(self.deg) + len(self.degV) + len(self.degW)   # traction slots
        self._c_alpha = md.n_layers * self._nb + K               # first rigid-rotation column
        ncol = self._c_alpha + len(self.regions)
        dt = complex if getattr(md, 'is_complex', False) else float
        return np.zeros((ncol, ncol), dt), np.zeros(ncol, dt)

    def _col(self, i, j):
        """First column of field j of layer i."""
        return i * self._nb + j * self.N

    def _interior_rows(self, H, f, i):
        """Collocation at the N - 2 interior Gauss-Lobatto nodes of layer i: the projected
        residual of every trial field, the tidal force and the rigid-rotation force."""
        md, N, nfld, m = self.model, self.N, self.nfld, self.m
        col, row0 = self._col, i * self._nb
        r0, r1 = md.r0[i], md.r1[i]
        xin = -np.cos(np.pi * np.arange(1, N - 1) / (N - 1))
        s = r0 + (xin + 1) * (r1 - r0) / 2
        p = self.props(i, s)
        for j, fl in enumerate(self._fields_at(i, xin, p)):
            Q = self._project(*self.operator(*fl, p))        # (N, Ns, nfld)
            for k in range(N - 2):
                H[row0 + nfld * (1 + k): row0 + nfld * (2 + k), col(i, j):col(i, j) + N] = Q[:, k, :].T
        Qf = self._tidal_force_rows(p)
        reg = [ir for ir, lays in enumerate(self.regions) if i in lays]
        if reg:
            rr = p.r
            if m == 1:      # rigid rotation about the y axis
                Qa = self._project(0 * rr, p.rho * rr, -p.rho * rr * p.cos, 0 * rr)
            else:           # m = 0: rigid rotation about the spin axis
                Qa = self._project(0 * rr, 0 * rr, p.rho * rr * p.sin, 0 * rr)
        for k in range(N - 2):
            rows = slice(row0 + nfld * (1 + k), row0 + nfld * (2 + k))
            if self.forcing == 'tide':
                f[rows] = Qf[k]
            if reg:
                H[rows, self._c_alpha + reg[0]] = Qa[k]

    def _bottom_rows(self, H, i):
        """Bottom rows of layer i: regularity at the centre (U = V = W = 0; P' = 0, but P = 0 for
        degree 1, where the regular potential grows as s) or continuity of every field with layer
        i - 1 (welded interface); a liquid-solid interface is filled by _fs_interface from the top
        rows of layer i - 1. For degree 1, centre='fixed' (U = V = 0) also fixes the frame: the
        centre does not move (exact for static runs, where a translation is free); centre='free'
        imposes U = V and dU/ds = 0, the regular behaviour U, V = const + O(s^2), and leaves the
        translation to inertia (manual, 'Degree 1')."""
        md, N, nfld = self.model, self.N, self.nfld
        col, rb = self._col, i * self._nb
        if i == 0:
            T, T1, _ = cheb_basis(-1.0, N)
            sc = 2.0 / (md.r1[i] - md.r0[i])
            for j, (typ, n) in enumerate(self.fields):
                H[rb + j, col(i, j):col(i, j) + N] = (T1[:, 0] * sc) if (typ == 'P' and n != 1) else T[:, 0]
            if self.centre == 'free' and 1 in self.deg and 1 in self.degV:
                jU, jV = self.fields.index(('U', 1)), self.fields.index(('V', 1))
                H[rb + jU, col(i, jV):col(i, jV) + N] = -T[:, 0]          # U_1 - V_1 = 0
                H[rb + jV, col(i, jV):col(i, jV) + N] = 0.0
                H[rb + jV, col(i, jU):col(i, jU) + N] = T1[:, 0] * sc     # dU_1/ds = 0
        elif self.true_fluid[i - 1] != self.true_fluid[i]:
            pass                       # filled by _fs_interface (layer i-1)
        else:
            Tm, _, _ = cheb_basis(-1.0, N)
            Tp, _, _ = cheb_basis(1.0, N)
            for j in range(nfld):
                H[rb + j, col(i - 1, j):col(i - 1, j) + N] = Tp[:, 0]
                H[rb + j, col(i, j):col(i, j) + N] = -Tm[:, 0]

    def _top_rows(self, H, f, i):
        """Top rows of layer i: continuity of traction and gravity flux with layer i + 1 (welded),
        the liquid-solid conditions, or the free surface (_surface_rows)."""
        md, N, nfld = self.model, self.N, self.nfld
        col, nb = self._col, self._nb
        rt = i * nb + nb - nfld
        last = (i == md.n_layers - 1)
        if not last and self.true_fluid[i] != self.true_fluid[i + 1]:
            self._fs_interface(H, f, i, slice((i + 1) * nb, (i + 1) * nb + nfld),
                               slice(rt, rt + nfld), col)
            return
        nrow = self._nS if last else nfld
        eul = (self.eulerian_interfaces and not last and i in self.fixed_density_layers
               and i + 1 in self.fixed_density_layers)
        r1 = md.r1[i]
        pt = self.props(i, np.array([r1]))
        Ft = [self.interface(*fl, pt, eul) for fl in self._fields_at(i, np.array([1.0]), pt)]
        for j, F in enumerate(Ft):
            H[rt:rt + nrow, col(i, j):col(i, j) + N] = self._project(*F)[:, 0, :].T[:nrow]
        if not last:
            pn = self.props(i + 1, np.array([r1]))
            Fn = [self.interface(*fl, pn, eul) for fl in self._fields_at(i + 1, np.array([-1.0]), pn)]
            for j, F in enumerate(Fn):
                H[rt:rt + nfld, col(i + 1, j):col(i + 1, j) + N] = -self._project(*F)[:, 0, :].T
        else:
            self._surface_rows(H, f, i, rt, pt, Ft)

    def _surface_rows(self, H, f, i, rt, pt, Ft):
        """Free surface: zero traction (or the weight of a load), phi1 continuous with the
        exterior potential sum c_n (a/r)^(n+1) Ybar_n, and continuity of the gravity flux (rows
        after the layers), with the surface mass of a load."""
        md, N, K = self.model, self.N, self.K
        col, nS, nL, nb = self._col, self._nS, md.n_layers, self._nb
        a, nfo, m = md.a, self.nf, self.m
        r = pt.r
        nr, nt = self.normal(pt)
        Tp, _, _ = cheb_basis(1.0, N)
        jP0 = nS
        for jj, n in enumerate(self.deg):          # phi1 continuity
            phin = Tp[:, 0][:, None] * self.ang[n]['Y'][0][None, :]
            H[rt + jP0:rt + jP0 + K, col(i, jP0 + jj):col(i, jP0 + jj) + N] = (phin @ self.testP.T).T
        for jj, n in enumerate(self.deg):          # exterior c_n
            E = (a / r)**(n + 1) * self.ang[n]['Y'][0]
            Er_ = -(n + 1) / r * E
            Et_ = (a / r)**(n + 1) * self.ang[n]['Y'][1] / r
            cc = nL * nb + jj
            H[rt + jP0:rt + jP0 + K, cc] = -(E[0] @ self.testP.T)
            H[nL * nb:nL * nb + K, cc] = -((Er_ * nr + Et_ * nt)[0] @ self.testP.T)
        for j, F in enumerate(Ft):                 # flux continuity
            fl = np.broadcast_to(F[3], (N,) + np.shape(F[3])[-2:])
            H[nL * nb:nL * nb + K, col(i, j):col(i, j) + N] = \
                np.einsum('nj,kj->kn', fl[:, 0, :], self.testP)
        if self.forcing == 'load':                 # surface mass sigma = Ybar_nf^m
            sig = ybar(nfo, m, self.theta[0, 0], 0)[0][None, :]
            gn = pt.ph0['r'] * nr + pt.ph0['t'] / r * nt
            tl = 0 * r
            if self.load_inertia and self.coriolis and self.omega_t:
                # -(dLambda/dt) Omega x r, eastward; amplitude in the (A, B, -iC) convention
                tl = self.omega_t * self.coriolis * r * pt.sin * sig
            Ql = self._project(-gn * sig * nr, -gn * sig * nt, tl, 0 * r)[0]
            f[rt:rt + nS] = Ql[:nS]
            f[nL * nb:nL * nb + K] = -4 * np.pi * self.G * (sig[0] @ self.testP.T)

    def _solve_linear(self, H, f):
        """Scale rows, then columns, to unit maximum (the Tisserand rows are ~1e30 times larger
        and must not dominate the column scaling) and solve: sparse LU (SuperLU, COLAMD) or
        dense LAPACK (manual, 'Solving the system')."""
        if self.linear_solver == 'sparse':
            Hc = sps.csr_matrix(H)                       # one pass over the dense matrix
            rs = abs(Hc).max(axis=1).toarray().ravel()
            if np.any(rs == 0):
                raise RuntimeError('empty row %s' % np.where(rs == 0)[0][:10])
            Hc = sps.diags(1.0 / rs) @ Hc
            cs = abs(Hc).max(axis=0).toarray().ravel()
            cs[cs == 0] = 1.0
            A = (Hc @ sps.diags(1.0 / cs)).tocsc()
            return spla.splu(A, permc_spec='COLAMD').solve(f / rs) / cs
        rs = np.abs(H).max(axis=1)
        if np.any(rs == 0):
            raise RuntimeError('empty row %s' % np.where(rs == 0)[0][:10])
        Hs = H / rs[:, None]
        cs = np.abs(Hs).max(axis=0)
        cs[cs == 0] = 1.0
        return sla.solve(Hs / cs, f / rs) / cs

    def _tisserand_row(self, layers):
        """int rho r (u_th - u_lam cos th) dV = y-component of int rho r x u dV / pi."""
        md, N, nb = self.model, self.N, self.nfld * self.N
        row = np.zeros(md.n_layers * nb, self.H_dtype)
        xg, wg = L.leggauss(N + 4)
        for i in layers:
            r0, r1 = md.r0[i], md.r1[i]
            s = r0 + (xg + 1) * (r1 - r0) / 2
            p = self.props(i, s)
            # dV = R^2 R_s ds sin th dth dlam,  R_s = 1/s_r,  ds = (r1-r0)/2 dx
            wv = p.rho * p.r**3 / p.s_r * (wg * (r1 - r0) / 2)[:, None] * self.w[None, :]
            for j, (uA, uB, uC, _) in enumerate(self._fields_at(i, xg, p)):
                w_ = uB.v - uC.v * p.cos if self.m == 1 else uC.v * p.sin
                integrand = np.broadcast_to(w_, (N,) + p.r.shape)
                row[i * nb + j * N: i * nb + (j + 1) * N] = np.sum(integrand * wv, axis=(1, 2))
        return row

    # ------------------------------------------------------------------
    def surface_coefficients(self):
        """Values at the surface (s = a) of every field (dict keyed by (type, degree)) and the
        exterior coefficients c_n of phi1."""
        md, N, nb = self.model, self.N, self.nfld * self.N
        i = md.n_layers - 1
        Tp, _, _ = cheb_basis(1.0, N)
        vals = {}
        for j, fld in enumerate(self.fields):
            c = self.coef[i * nb + j * N: i * nb + (j + 1) * N]
            v = c @ Tp[:, 0]
            vals[fld] = complex(v) if np.iscomplexobj(v) else float(v)
        cext = {n: self.coef[md.n_layers * nb + jj] for jj, n in enumerate(self.deg)}
        return vals, cext

    def _tilt_potential(self, n=2):
        """Degree-n exterior coefficient of -(xi x x).grad(phi0): the potential change
        of the flattened Earth tilted rigidly by the removed rotation xi (Wahr 1981,
        eq. 4.5). For m = 1 and a zonal phi0 = sum c_n (a/r)^(n+1) P_n it is
        sqrt(n (n+1) / 2) xi c_n (dP_n/dtheta = -sqrt(n(n+1)/2) Ybar_n^1): sqrt(3) xi c2 for
        degree 2 and sqrt(10) xi c4 for degree 4, which, with xi ~ 1/(sigma - 1), dominates the
        degree-4 coefficient near K1. Degrees above 2 need the consistent reference potential."""
        from .model import G as Gc
        th = np.linspace(0.02, np.pi - 0.02, 401)
        self.surface_field(th, remove_rigid=True)          # sets self._xi
        md = self.model
        if self.refpot is not None:
            rp = self.refpot
            if n not in rp.deg:
                return 0.0
            cn = rp.coef[md.n_layers * rp.nb + rp.deg.index(n)]
        elif n == 2:                                        # Clairaut: J2 = (2f - m)/3
            fl = getattr(self.geom, 'flattening_surface', 0.0)
            mq = getattr(self.geom, 'm', 0.0)
            cn = Gc * md.M * (2 * fl - mq) / 3 / md.a
        else:
            return 0.0
        return np.sqrt(n * (n + 1) / 2.0) * self._xi * cn

    def results(self):
        """Love numbers: U, V, W (surface values / a) and k per degree, and h, l, kk for the
        forcing degree; alpha = rigid-rotation forces of static runs. For wahr_frame with m = 1
        the potential of the rigidly tilted flattened Earth is removed from k (Wahr 1981, 4.5)."""
        vals, cext = self.surface_coefficients()
        a, ga = self.model.a, self.model.g_surface
        out = dict(U={n: vals[('U', n)] / a for n in self.deg},
                   V={n: vals[('V', n)] / a for n in self.degV},
                   W={n: vals[('W', n)] / a for n in self.degW},
                   k={n: -cext[n] / (ga * a) for n in self.deg})
        if self.wahr_frame and self.m == 1:
            for n in out['k']:
                if n >= 2:
                    out['k'][n] = out['k'][n] - self._tilt_potential(n) / (-ga * a)
        nf = self.nf
        out['h'], out['l'], out['kk'] = out['U'][nf], out['V'].get(nf, 0.0), out['k'][nf]
        if self.regions:
            out['alpha'] = [complex(v) for v in self.coef[-len(self.regions):]]
        return out

    def load_love(self):
        """Load Love numbers (Farrell 1972) of a solve with forcing='load'.

        Normalisation by Phi_L = 4 pi G a / (2 nf + 1) (geodetic sign, a the mean
        radius), the load potential on a sphere. For every degree n of the response:
        h'_n = g U_n(a) / Phi_L, l'_n = g V_n(a) / Phi_L, k'_n = potential of the
        deformation / Phi_L, i.e. the exterior coefficient minus that of the load
        itself, which is computed on the actual (flattened) surface: on an
        ellipsoid the load's own degree-n potential differs from Phi_L by O(n f),
        large compared with k'. On a sphere only n = nf is nonzero.
        """
        from .model import G as Gc
        vals, cext = self.surface_coefficients()
        md = self.model
        a, g = md.a, md.g_surface
        PhiL = 4 * np.pi * Gc * a / (2 * self.nf + 1)
        cL = self.load_potential()
        out = dict(h={n: g * vals[('U', n)] / PhiL for n in self.deg},
                   l={n: g * vals[('V', n)] / PhiL for n in self.degV},
                   k={n: -(cext[n] - cL[n]) / PhiL for n in self.deg})
        if self.wahr_frame and self.m == 1:
            # as for the body tide (results): remove the potential of the rigidly tilted,
            # flattened Earth, so that k' is the potential of the deformation alone
            for n in out['k']:
                if n >= 2:
                    out['k'][n] = out['k'][n] + self._tilt_potential(n) / PhiL
        out['W'] = {n: g * vals[('W', n)] / PhiL for n in self.degW}
        return out

    def load_potential(self):
        """Exterior coefficients c_n (phi = sum c_n (a/r)^(n+1) Ybar_n^m cos m lam, code
        sign) of the unit surface density Ybar_nf^m cos m lam on the outer surface."""
        from .model import G as Gc
        md = self.model
        th = self.theta[0, 0]
        p = self.props(md.n_layers - 1, np.array([md.a]))
        R, Rt = p.r[0], p.R_t[0]
        dA = R * np.hypot(R, Rt)                      # dA / (dlambda dmu)
        sig = ybar(self.nf, self.m, th, 0)[0]
        pim = 2 * np.pi if self.m == 0 else np.pi
        a = md.a
        return {n: -Gc * a * pim * np.sum(self.w * (R / a)**n * dA / a**2 * self.ang[n]['Y'][0] * sig)
                for n in self.deg}

    def surface_field(self, theta, remove_rigid=None):
        """Amplitudes of u_r, u_theta (x cos m lam) and u_lambda (x sin m lam)
        at the deformed-surface points (R(a,theta), theta). With remove_rigid
        (default: wahr_frame) the degree-1 toroidal term, the rigid rotation of
        the surface, is dropped (Wahr's separation of nutation / rotation rate)."""
        vals, _ = self.surface_coefficients()
        if remove_rigid is None:
            remove_rigid = self.wahr_frame
        th = np.asarray(theta, float)
        m = self.m
        ur = sum(vals[('U', n)] * ybar(n, m, th, 0)[0] for n in self.deg)
        ut, up = 0 * th, 0 * th
        for n in self.degV:
            y1 = ybar(n, m, th, 1)[1]
            ys = m * ybar(n, m, th, 0, over_sin=True)[0] if m else 0 * th
            ut = ut + vals[('V', n)] * y1
            up = up - vals[('V', n)] * ys
        for n in self.degW:
            y1 = ybar(n, m, th, 1)[1]
            ys = m * ybar(n, m, th, 0, over_sin=True)[0] if m else 0 * th
            ut = ut - vals[('W', n)] * ys
            up = up + vals[('W', n)] * y1
        if remove_rigid and m in (0, 1):
            if self.frame == 'mantle':
                ut, up = self._remove_mantle_rotation(th, ut, up)
            else:
                ut, up = self._remove_rigid_rotation(th, ut, up)
        return ur, ut, up

    def _rigid_pattern(self, i, s, th):
        """Rigid rotation (theta, lambda amplitudes) at (s, theta) of layer i."""
        if hasattr(self.geom, 'map_derivs'):
            R = self.geom.map_derivs(i, s, th)[0]
        else:
            R = mapping(s, th, *self.geom.eps(i, np.asarray(s, float)))['R']
        if self.m == 1:
            return R, -R * np.cos(th)
        return 0 * R, R * np.sin(th)

    def _remove_mantle_rotation(self, th, ut, up):
        """Subtract the mean rotation of the mantle volume (Tisserand frame of the
        mantle): xi = int rho r x u dV / int rho r x (rigid) dV over the mantle."""
        from numpy.polynomial import legendre as L
        md, N, nb = self.model, self.N, self.nfld * self.N
        fl = self.fluid_layers()
        mantle = [i for i in range(md.n_layers) if i > 0 and not fl[i] and fl[i - 1] or
                  (i > 0 and not fl[i] and any(fl[:i]))]
        num = den = 0.0
        xg, wg = L.leggauss(N + 4)
        for i in mantle:
            r0, r1 = md.r0[i], md.r1[i]
            s = r0 + (xg + 1) * (r1 - r0) / 2
            p = self.props(i, s)
            wv = p.rho * p.r**3 / p.s_r * (wg * (r1 - r0) / 2)[:, None] * self.w[None, :]
            if self.m == 1:
                proj = lambda b_t, b_p: b_t - b_p * p.cos
            else:
                proj = lambda b_t, b_p: b_p * p.sin
            # displacement of the solution in this layer
            ut_l, up_l = 0.0, 0.0
            for j, (uA, uB, uC, _) in enumerate(self._fields_at(i, xg, p)):
                c = self.coef[i * nb + j * N: i * nb + (j + 1) * N]
                ut_l = ut_l + np.tensordot(c, np.broadcast_to(uB.v, (N,) + p.r.shape), axes=1)
                up_l = up_l + np.tensordot(c, np.broadcast_to(uC.v, (N,) + p.r.shape), axes=1)
            bt, bp = self._rigid_pattern(i, s[:, None], self.theta[0])
            num = num + np.sum(proj(ut_l, up_l) * wv)
            den = den + np.sum(proj(bt, bp) * wv)
        xi = num / den
        self._xi = xi
        bt, bp = self._rigid_pattern(md.n_layers - 1, md.a, th)
        return ut - xi * bt, up - xi * bp

    def _remove_rigid_rotation(self, th, ut, up):
        """Subtract the rigid rotation that best fits the surface displacement
        (Wahr 1981, eqs. 4.3, 4.6). On an ellipsoid the rotation is proportional
        to the physical radius R(a, theta), so it is not a pure W_1 field."""
        md = self.model
        i = md.n_layers - 1
        if hasattr(self.geom, 'map_derivs'):
            R = self.geom.map_derivs(i, md.a, th)[0]
        else:
            R = mapping(md.a, th, *self.geom.eps(i, np.array(md.a)))['R']
        if self.m == 1:                     # rotation about an equatorial axis
            bt, bp = R, -R * np.cos(th)
        else:                               # m = 0: spin about the polar axis
            bt, bp = 0 * th, R * np.sin(th)
        w = np.sin(th) * np.gradient(th)    # surface weight
        xi = np.sum(w * (ut * bt + up * bp)) / np.sum(w * (bt * bt + bp * bp))
        self._xi = xi
        return ut - xi * bt, up - xi * bp


def ZonalSolver(model, geometry=None, **kw):
    """m = 0 solver (kept for the earlier scripts and tests)."""
    return TidalSolver(model, geometry, m=0, **kw)
