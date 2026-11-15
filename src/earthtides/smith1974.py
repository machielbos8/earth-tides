# -*- coding: utf-8 -*-
#
# Smith (1974) scalar equations for a non-gravitating, rotating, slightly
# elliptical solid body.
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
# 16/10/2026 Machiel Bos
#===============================================================================
"""Smith (1974) scalar equations for a rotating, slightly elliptical solid body (class Smith: without
self-gravitation; class SmithGrav: self-gravitating, hydrostatic initial state; class SmithLayered: the
same for welded solid layers with density jumps). One correction to the printed equations: the sign of
the radial rho_2 term of (5.26).

An independent check of the solver, and of the theory behind the Love numbers of Wahr (1981) and
Dehant et al. (1999), which integrate these equations. Transcribed from Smith (1974, eqs 5.19-5.36,
5.44-5.46) for a body without self-gravitation and without initial stress (no initial potential:
eta_3 = 0, g = g~ = 0, phi_2 = 0), one solid layer whose properties vary with radius, and equidensity
surfaces of constant ellipticity eps: lambda_2 = (2/3) eps r d lambda_0/dr (5.21), and likewise mu_2,
beta_2 and rho_2. Truncated as in the programs that use them: spheroidal degrees sph (U, V, P, Q),
toroidal degrees tor (W, R).

Dimensionless: radius a = 1, the moduli divided by mu_ref, the density by rho_ref. Time dependence
exp(i w t); tidal potential phi = r^2 D^2_{m0}. Signs as Smith and Dehant et al. (1999, table 4):
sigma_h = -(V/2L) grad_1 D, tau = -(W/2L) i (r x grad_1 D), with D^l_{m0} = (4 pi/(2l+1))^{1/2} Y_l^m.
"""
import numpy as np
from functools import lru_cache
from scipy.integrate import solve_ivp
from scipy.special import lpmv, factorial
from sympy.physics.wigner import wigner_3j

SGN = {'+': 1, '0': 0, '-': -1}


@lru_cache(None)
def J(l, l1, l2, n, n1, n2, m, m1, m2):
    """Smith's J-square, eq. (3.11)."""
    n, n1, n2 = (SGN[x] if isinstance(x, str) else x for x in (n, n1, n2))
    if l < 0 or l2 < 0:
        return 0.0
    return float((2*l + 1)*(-1)**(m + n)*wigner_3j(l, l1, l2, -n, n1, n2)*wigner_3j(l, l1, l2, -m, m1, m2))


def d1(f, r, h=1e-3):
    """First derivative, 5-point stencil (smooth, error ~ h^4)."""
    return (f(r - 2*h) - 8*f(r - h) + 8*f(r + h) - f(r + 2*h))/(12*h)


def d2(f, r, h=1e-3):
    """Second derivative, 5-point stencil."""
    return (-f(r - 2*h) + 16*f(r - h) - 30*f(r) + 16*f(r + h) - f(r + 2*h))/(12*h*h)


def LN(N, k):
    v = (k + N)*(k + 1 - N)/2.0
    return np.sqrt(v) if v > 0 else 0.0


class Smith:
    """lam, mu, rho: callables of r (dimensionless); w2 = rho_ref w^2 a^2/mu_ref,
    kap = 2 rho_ref w Omega a^2/mu_ref; eps: ellipticity of the equidensity surfaces and the surface."""

    def __init__(self, m, lam, mu, rho, w2, kap, eps, sph=(2, 4), tor=(1, 3), dr=1e-6):
        self.m, self.w2, self.kap, self.eps = m, w2, kap, eps
        self.f = {'lam': lam, 'mu': mu, 'rho': rho}
        self.dr = dr
        self.sph = [l for l in sph if l >= m]
        self.tor = [l for l in tor if l >= max(m, 1)]
        self.idx = {}
        for l in self.sph:
            for c in 'UVPQ':
                self.idx[(c, l)] = len(self.idx)
        for l in self.tor:
            for c in 'WR':
                self.idx[(c, l)] = len(self.idx)
        self.n = len(self.idx)
        self._forc = True

    def mat(self, r):
        """lam0, mu0, rho0 and their ellipticity parts (5.21)."""
        out = {}
        for k, fn in self.f.items():
            out[k] = fn(r)
            out[k + '2'] = 2/3*self.eps*r*d1(fn, r)
        out['beta'], out['beta2'] = out['lam'] + 2*out['mu'], out['lam2'] + 2*out['mu2']
        return out

    def get(self, y, c, l):
        i = self.idx.get((c, l))
        return 0.0 if i is None else y[i]

    def _eta(self, y, r, l, forcing):
        """rho-free parts of eta_1 + eta_2 for degree l: (U, V) components if spheroidal, W if toroidal."""
        m, g = self.m, (lambda c, k: self.get(y, c, k))
        if l in self.sph:
            L0 = LN(0, l)
            phi, dphi = (r**2, 2*r) if (forcing and l == 2) else (0.0, 0.0)
            e1U = -self.w2*g('U', l) + dphi                                       # (5.31)
            e1V = -self.w2*g('V', l) - 2*L0/r*phi                                 # (5.32)
            e2U = e2V = 0.0
            for lp in range(abs(l - 1), l + 2):                                   # (5.34), (5.35)
                even = (abs(l - lp) % 2 == 0)
                e2U += J(l, 1, lp, 0, '+', '-', m, 0, m)*(g('V', lp) if even else -g('W', lp))
                e2V -= (J(l, 1, lp, '-', '-', 0, m, 0, m)*(2*g('U', lp) if even else 0.0)
                        + J(l, 1, lp, '-', 0, '-', m, 0, m)*(g('V', lp) if even else -g('W', lp)))
            return e1U + self.kap*e2U, e1V + self.kap*e2V
        if l in self.tor:
            e2W = 0.0
            for lp in range(abs(l - 1), l + 2):                                   # (5.36)
                even = (abs(l - lp) % 2 == 0)
                e2W += (J(l, 1, lp, '-', '-', 0, m, 0, m)*(0.0 if even else 2*g('U', lp))
                        + J(l, 1, lp, '-', 0, '-', m, 0, m)*(-g('W', lp) if even else g('V', lp)))
            return -self.w2*g('W', l) + self.kap*e2W                              # (5.33)
        return None

    def rhs(self, r, y, forcing=True):
        m, p = self.m, self.mat(r)
        lam, mu, beta, rho = p['lam'], p['mu'], p['beta'], p['rho']
        lam2, mu2, beta2, rho2 = p['lam2'], p['mu2'], p['beta2'], p['rho2']
        g = lambda c, k: self.get(y, c, k)
        d = np.zeros(self.n, complex)
        dU = {}
        for l in self.sph:                                                        # (5.22)
            L0 = LN(0, l)
            v = g('P', l)/beta - lam/(beta*r)*L0*g('V', l) - 2*lam/(beta*r)*g('U', l)
            for lp in self.sph:
                v -= J(l, 2, lp, 0, 0, 0, m, 0, m)*(beta2/beta**2*g('P', lp)
                     - 2*(lam*mu2 - lam2*mu)/(r*beta**2)*(LN(0, lp)*g('V', lp) + 2*g('U', lp)))
            dU[l] = v
        Delta = {l: dU[l] + 2/r*g('U', l) + LN(0, l)/r*g('V', l) for l in self.sph}   # (5.19)
        S = {l: (LN(0, l)*g('V', l) + 2*g('U', l)) for l in self.sph}
        eta = {l: self._eta(y, r, l, forcing) for l in self.sph + self.tor}
        present = self.sph + self.tor
        for l in self.sph:
            L0, Lm1, L2 = LN(0, l), LN(-1, l), LN(2, l)
            dV = g('Q', l)/mu + g('V', l)/r + 2*L0/r*g('U', l)                     # (5.23)
            dP = (-2/r*g('P', l) - L0/r*g('Q', l) + 2/r*(lam*Delta[l] + mu/r*S[l])
                  + rho*eta[l][0])                                                 # (5.28)
            dQ = (-3/r*g('Q', l) + 2*mu/r**2*Lm1*L2*g('V', l) + 2*L0/r*(mu/r*S[l] + lam*Delta[l])
                  + rho*eta[l][1])                                                 # (5.29)
            for lp in present:
                if abs(lp - l) > 2:
                    continue
                even = (abs(l - lp) % 2 == 0)
                Jp = J(l, 2, lp, '+', 0, '+', m, 0, m)
                dV -= mu2/mu**2*Jp*(g('Q', lp) if even else g('R', lp))
                if lp in self.sph:
                    J0 = J(l, 2, lp, 0, 0, 0, m, 0, m)
                    dP += 2/r*J0*(lam2*Delta[lp] + mu2/r*S[lp]) + rho2*J0*eta[lp][0]
                    dQ += 2*L0/r*J0*(mu2/r*S[lp] + lam2*Delta[lp])
                J2 = J(l, 2, lp, 2, 0, 2, m, 0, m)
                dQ += 2*Lm1/r**2*J2*mu2*LN(2, lp)*(g('V', lp) if even else g('W', lp))
                dQ += rho2*Jp*(eta[lp][1] if lp in self.sph else eta[lp])
            for c, v in (('U', dU[l]), ('V', dV), ('P', dP), ('Q', dQ)):
                d[self.idx[(c, l)]] = v
        for l in self.tor:
            Lm1, L2 = LN(-1, l), LN(2, l)
            dW = g('R', l)/mu + g('W', l)/r                                        # (5.24)
            dR = -3/r*g('R', l) + 2/r**2*Lm1*L2*mu*g('W', l) + rho*eta[l]           # (5.30)
            for lp in present:
                if abs(lp - l) > 2:
                    continue
                even = (abs(l - lp) % 2 == 0)
                Jp = J(l, 2, lp, '+', 0, '+', m, 0, m)
                dW -= mu2/mu**2*Jp*(g('R', lp) if even else g('Q', lp))
                J2 = J(l, 2, lp, 2, 0, 2, m, 0, m)
                dR += 2*Lm1/r**2*J2*mu2*LN(2, lp)*(g('W', lp) if even else g('V', lp))
                dR += rho2*Jp*(eta[lp] if lp in self.tor else eta[lp][1])
            d[self.idx[('W', l)]] = dW
            d[self.idx[('R', l)]] = dR
        return d

    def bcs(self, r, y):
        """Free-surface conditions (5.44)-(5.46)."""
        m, p = self.m, self.mat(r)
        lam, mu = p['lam'], p['mu']
        h2 = -2/3*r*self.eps
        dy = self.rhs(r, y, forcing=self._forc)
        g = lambda c, k: self.get(y, c, k)
        dg = lambda c, k: self.get(dy, c, k)
        def Sv(l):
            if l not in self.sph:
                return 0.0
            Dl = dg('U', l) + 2/r*g('U', l) + LN(0, l)/r*g('V', l)
            return lam*Dl + mu/r*(LN(0, l)*g('V', l) + 2*g('U', l))
        present = self.sph + self.tor
        out = []
        for l in self.sph:
            pp = g('P', l)                                                          # (5.44)
            for lp in self.sph:
                pp += J(l, 2, lp, 0, 0, 0, m, 0, m)*h2*dg('P', lp)
            out.append(pp)
            q = g('Q', l)                                                           # (5.45)
            for lp in present:
                if abs(lp - l) > 2:
                    continue
                even = (abs(l - lp) % 2 == 0)
                q += J(l, 2, lp, '+', 0, '+', m, 0, m)*h2*(dg('Q', lp) if even else dg('R', lp))
                q += np.sqrt(3)*h2/r*J(l, 2, lp, '+', '+', 0, m, 0, m)*(1 + (-1)**(lp - l))*Sv(lp)
                q += np.sqrt(3)*h2/r*J(l, 2, lp, '+', '-', 2, m, 0, m)*2*mu/r*LN(2, lp)*(g('V', lp) if even else g('W', lp))
            out.append(q)
        for l in self.tor:
            rr = g('R', l)                                                          # (5.46)
            for lp in present:
                if abs(lp - l) > 2:
                    continue
                even = (abs(l - lp) % 2 == 0)
                rr += J(l, 2, lp, '+', 0, '+', m, 0, m)*h2*(dg('R', lp) if even else dg('Q', lp))
                rr += np.sqrt(3)*h2/r*J(l, 2, lp, '+', '+', 0, m, 0, m)*(1 - (-1)**(lp - l))*Sv(lp)
                rr += np.sqrt(3)*h2/r*J(l, 2, lp, '+', '-', 2, m, 0, m)*2*mu/r*LN(2, lp)*(g('W', lp) if even else g('V', lp))
            out.append(rr)
        return np.array(out)

    def solve(self, r_start=1e-3):
        """Shoot the regular solutions from the centre and impose the free-surface conditions."""
        n = self.n
        Pidx = [self.idx[k] for k in self.idx if k[0] in 'PQR']
        T = lambda r: np.diag([1/r if i in Pidx else 1.0 for i in range(n)])
        A = lambda r: np.column_stack([self.rhs(r, np.eye(n)[k], forcing=False) for k in range(n)])
        r0 = 1e-6
        dT = np.diag([-1/r0**2 if i in Pidx else 0.0 for i in range(n)])
        ev, V = np.linalg.eig(r0*(np.linalg.inv(T(r0)) @ (A(r0) @ T(r0) - dT)))
        keep = [k for k in np.argsort(-ev.real) if ev[k].real > 0.5][:n//2]
        f = lambda r, y, forc: self.rhs(r, y, forcing=forc)
        run = lambda y0, forc: solve_ivp(f, (r_start, 1.0), y0, args=(forc,), method='DOP853',
                                         rtol=1e-12, atol=1e-16).y[:, -1]
        sols = [run((T(r_start) @ (r_start**ev[k].real*V[:, k])).astype(complex), False) for k in keep]
        yp = run(np.zeros(n, complex), True)
        self._forc = True
        b0 = self.bcs(1.0, yp)
        self._forc = False
        B = np.column_stack([self.bcs(1.0, y) for y in sols])
        self._forc = True
        c = np.linalg.solve(B, -b0)
        self.y = yp + sum(ci*yi for ci, yi in zip(c, sols))
        return self.y

    def surface(self, th):
        """u_r, u_theta, u_lambda/i on the surface r = 1 - (2/3) eps P2 (first-order Taylor step)."""
        m, y = self.m, self.y
        dy = self.rhs(1.0, y, forcing=True)
        c = np.cos(th)
        delta = -2/3*self.eps*0.5*(3*c**2 - 1)
        val = lambda k, l: self.get(y, k, l) + delta*self.get(dy, k, l)
        def D(l, dd=1e-6):
            N = np.sqrt((2*l + 1)/(4*np.pi)*factorial(l - m)/factorial(l + m))*np.sqrt(4*np.pi/(2*l + 1))
            return N*lpmv(m, l, c), N*(lpmv(m, l, np.cos(th + dd)) - lpmv(m, l, np.cos(th - dd)))/(2*dd)
        ur, ut, ul = 0*th + 0j, 0*th + 0j, 0*th + 0j
        for l in self.sph:
            Dv, dD = D(l); L = LN(0, l)
            ur = ur + val('U', l)*Dv
            ut = ut - val('V', l)/(2*L)*dD
            ul = ul - val('V', l)/(2*L)*m/np.sin(th)*Dv
        for l in self.tor:
            Dv, dD = D(l); L = LN(0, l)
            ut = ut - val('W', l)/(2*L)*m/np.sin(th)*Dv
            ul = ul - val('W', l)/(2*L)*dD
        return ur, ut, ul, 1 + delta


class SmithGrav(Smith):
    """Smith (1974) with self-gravitation and the hydrostatic initial state (eqs 5.25-5.26, 5.31-5.32 in
    full, 5.37-5.39, 5.42-5.43). Spheroidal unknowns U, V, P, Q, phi1, g1; the free-space potential
    outside decays as r^-(l+1). Gh = G rho_ref^2 a^2/mu_ref, Om2 = rho_ref Omega^2 a^2/mu_ref;
    g0(r): initial gravity (dimensionless, a rho_ref/mu_ref); eps(r): hydrostatic ellipticity.
    The tidal potential acts as a body force on rho_0 + rho' (it is not part of phi1).
    RHO2_SIGN: sign of the radial rho_2 term of (5.26); -1 corrected (default), +1 as printed, 0 dropped
    (the test is in the Supplementary Material of the paper, Sect. S4)."""
    RHO2_SIGN = -1.0

    def __init__(self, m, lam, mu, rho, w2, kap, eps, Gh, Om2, g0, sph=(2, 4), tor=(1, 3), dr=1e-6):
        super().__init__(m, lam, mu, rho, w2, kap, 0.0, sph=sph, tor=tor, dr=dr)
        self.epsf, self.Gh, self.Om2, self.g0 = eps, Gh, Om2, g0
        self.idx = {}
        for l in self.sph:
            for c in ('U', 'V', 'P', 'Q', 'F', 'G'):            # F = phi1, G = g1
                self.idx[(c, l)] = len(self.idx)
        for l in self.tor:
            for c in 'WR':
                self.idx[(c, l)] = len(self.idx)
        self.n = len(self.idx)

    def mat(self, r):
        self.eps = self.epsf(r)
        out = super().mat(r)
        d = self.dr
        phi2 = lambda x: 2/3*self.epsf(x)*x*self.g0(x)                     # (5.21)
        out['g'], out['gt'] = self.g0(r), self.g0(r) - 2/3*self.Om2*r
        out['phi2'] = phi2(r)
        out['dphi2'] = d1(phi2, r)
        out['ddphi2'] = d2(phi2, r)
        return out

    def _eta(self, y, r, l, forcing):
        res = super()._eta(y, r, l, forcing)
        if l in self.sph:                                                   # g, g~, phi1 parts of (5.31), (5.32)
            p = self._p
            L0 = LN(0, l)
            g = lambda c, k: self.get(y, c, k)
            eU = res[0] + g('G', l) - 2/r*(p['g'] + p['gt'])*g('U', l) - L0/r*p['gt']*g('V', l)
            eV = res[1] - 2*L0/r*(g('F', l) + p['gt']*g('U', l))
            return eU, eV
        return res

    def rhs(self, r, y, forcing=True):
        m = self.m
        self._p = p = self.mat(r)
        d = super().rhs(r, y, forcing)                    # elastic, inertia, Coriolis, rho2 parts (uses _eta above)
        g = lambda c, k: self.get(y, c, k)
        rho, rho2, G4 = p['rho'], p['rho2'], 4*np.pi*self.Gh
        drho2 = self._drho2(r)
        phi2, dphi2, ddphi2 = p['phi2'], p['dphi2'], p['ddphi2']
        dU = {l: d[self.idx[('U', l)]] for l in self.sph}
        dV = {l: d[self.idx[('V', l)]] for l in self.sph}
        dW = {l: d[self.idx[('W', l)]] for l in self.tor}
        Delta = {l: dU[l] + 2/r*g('U', l) + LN(0, l)/r*g('V', l) for l in self.sph}
        present = self.sph + self.tor
        hz = lambda lp, even: (g('V', lp) if even else -g('W', lp))
        dhz = lambda lp, even: (dV.get(lp, 0.0) if even else -dW.get(lp, 0.0))
        for l in self.sph:
            L0 = LN(0, l)
            dF = g('G', l) - G4*rho*g('U', l)                                  # (5.25)
            dG = l*(l + 1)/r**2*g('F', l) - 2/r*g('G', l) - G4/r*rho*L0*g('V', l)   # (5.26)
            e3U = e3V = 0.0
            for lp in present:
                if abs(lp - l) > 2:
                    continue
                even = (abs(l - lp) % 2 == 0)
                J0, Jpm = J(l, 2, lp, 0, 0, 0, m, 0, m), J(l, 2, lp, 0, '+', '-', m, 0, m)
                Jp, Jpp = J(l, 2, lp, '+', 0, '+', m, 0, m), J(l, 2, lp, '+', '+', 0, m, 0, m)
                if lp in self.sph:
                    # (5.26): Smith prints +4 pi G (d_r + 2/r)(rho_2 U); Poisson's equation with
                    # g_1 = d_r phi_1 + 4 pi G rho_0 U (5.18) gives the minus sign, as in Huang et al.
                    # (2004, eq. 12); with + the l+ of a body with a density gradient is 3% off (first order)
                    dG += self.RHO2_SIGN*G4*J0*(drho2*g('U', lp) + rho2*dU[lp] + 2/r*rho2*g('U', lp))
                    e3U += J0*(dU[lp]*dphi2 + g('U', lp)*ddphi2 - Delta[lp]*dphi2)      # (5.37)
                    e3V += 2*L0/r*(-J0*g('U', lp)*dphi2)
                    e3V += Jpp*np.sqrt(3)/r*phi2*(1 + (-1)**(lp - l))*Delta[lp]        # (5.38)
                e3U += np.sqrt(3)*Jpm*((dphi2/r - phi2/r**2)*hz(lp, even) + phi2/r*dhz(lp, even))
                e3V += 2*L0/r*(-np.sqrt(3)/r*phi2*Jpm*hz(lp, even))
                dG -= G4/r*L0*rho2*Jp*(g('V', lp) if even else g('W', lp))
            d[self.idx[('F', l)]] = dF
            d[self.idx[('G', l)]] = dG
            d[self.idx[('P', l)]] += rho*e3U
            d[self.idx[('Q', l)]] += rho*e3V
        for l in self.tor:
            e3W = 0.0
            for lp in self.sph:                                                 # (5.39)
                if abs(lp - l) > 2:
                    continue
                e3W += np.sqrt(3)/r*phi2*J(l, 2, lp, '+', '+', 0, m, 0, m)*(1 + (-1)**(lp + 1 + l))*Delta[lp]
            d[self.idx[('R', l)]] += rho*e3W
        return d

    def _drho2(self, r):
        r2 = lambda x: 2/3*self.epsf(x)*x*d1(self.f['rho'], x)
        return d1(r2, r)

    def bcs(self, r, y, c_ext=None):
        """Tractions (5.44)-(5.46) and the gravity conditions (5.42)-(5.43) against the exterior
        potential sum_l c_l r^-(l+1); returns residuals, with c_ext as extra unknowns."""
        m, p = self.m, self.mat(r)
        self._p = p
        out = list(Smith.bcs(self, r, y))
        h2 = -2/3*r*self.eps
        dy = self.rhs(r, y, forcing=self._forc)
        g = lambda c, k: self.get(y, c, k)
        dg = lambda c, k: self.get(dy, c, k)
        G4, rho = 4*np.pi*self.Gh, p['rho']
        drho = d1(self.f['rho'], r)
        cl = dict(zip(self.sph, c_ext))
        for l in self.sph:
            gi = g('G', l)
            fi, fe, ge = g('F', l), cl[l]*r**-(l + 1), -(l + 1)*cl[l]*r**-(l + 2)
            for lp in self.sph + self.tor:
                if abs(lp - l) > 2:
                    continue
                even = (abs(l - lp) % 2 == 0)
                J0, Jpm = J(l, 2, lp, 0, 0, 0, m, 0, m), J(l, 2, lp, 0, '+', '-', m, 0, m)
                if lp in self.sph:
                    gi += J0*G4*h2*(-LN(0, lp)*rho/r*g('V', lp) - drho*g('U', lp))      # (5.42)
                    fi += J0*h2*dg('F', lp)                                          # (5.43)
                    fe += J0*h2*(-(lp + 1)*cl[lp]*r**-(lp + 2))
                gi += np.sqrt(3)*h2/r*Jpm*(-G4*rho*g('V', lp) if even else G4*rho*g('W', lp))
            out += [gi - ge, fi - fe]
        return np.array(out)

    def solve(self, r_start=1e-3):
        n, ns = self.n, len(self.sph)
        Pidx = [self.idx[k] for k in self.idx if k[0] in 'PQRG']
        T = lambda r: np.diag([1/r if i in Pidx else 1.0 for i in range(n)])
        A = lambda r: np.column_stack([self.rhs(r, np.eye(n)[k], forcing=False) for k in range(n)])
        r0 = 1e-6
        dT = np.diag([-1/r0**2 if i in Pidx else 0.0 for i in range(n)])
        ev, V = np.linalg.eig(r0*(np.linalg.inv(T(r0)) @ (A(r0) @ T(r0) - dT)))
        keep = [k for k in np.argsort(-ev.real) if ev[k].real > 0.5][:n//2]
        f = lambda r, y, forc: self.rhs(r, y, forcing=forc)
        run = lambda y0, forc: solve_ivp(f, (r_start, 1.0), y0, args=(forc,), method='DOP853',
                                         rtol=1e-12, atol=1e-16).y[:, -1]
        sols = [run((T(r_start) @ (r_start**ev[k].real*V[:, k])).astype(complex), False) for k in keep]
        yp = run(np.zeros(n, complex), True)
        z = np.zeros(ns)
        self._forc = True
        b0 = self.bcs(1.0, yp, z)
        self._forc = False
        cols = [self.bcs(1.0, y, z) for y in sols]
        cols += [self.bcs(1.0, np.zeros(n, complex), np.eye(ns)[k]) for k in range(ns)]
        self._forc = True
        c = np.linalg.solve(np.column_stack(cols), -b0)
        self.y = yp + sum(ci*yi for ci, yi in zip(c[:len(sols)], sols))
        self.cext = c[len(sols):]
        return self.y


class SmithLayered(SmithGrav):
    """SmithGrav for a body of welded solid layers: across each interface the combinations of
    Smith (1974, eqs 5.41a-c, 5.42, 5.43, 5.44-5.46) are continuous. layers: list of
    (r_top, lam, mu, rho), callables of r (dimensionless) valid in that layer and smooth beyond it;
    eps3(i, r) -> (eps, eps', eps'') of the hydrostatic equidensity surfaces in layer i;
    g0(i, r): initial gravity."""

    def __init__(self, m, layers, w2, kap, eps3, Gh, Om2, g0, sph=(2, 4), tor=(1, 3), dr=1e-6):
        self.layers, self.eps3, self.g0i, self.i = layers, eps3, g0, 0
        _, lam, mu, rho = layers[0]
        super().__init__(m, lam, mu, rho, w2, kap, lambda r: self.eps3(self.i, r)[0], Gh, Om2,
                         lambda r: self.g0i(self.i, r), sph=sph, tor=tor, dr=dr)

    def mat(self, r):
        _, lam, mu, rho = self.layers[self.i]
        self.f = {'lam': lam, 'mu': mu, 'rho': rho}
        e, de, dde = self.eps3(self.i, r)
        self.eps = e
        out = Smith.mat(self, r)
        G4 = 4*np.pi*self.Gh
        g = self.g0i(self.i, r)
        dg = G4*rho(r) - 2*g/r
        ddg = G4*d1(rho, r) - 2*dg/r + 2*g/r**2
        A, dA, ddA = e*r, de*r + e, dde*r + 2*de                           # phi_2 = (2/3) eps r g (5.21)
        out['g'], out['gt'] = g, g - 2/3*self.Om2*r
        out['phi2'], out['dphi2'] = 2/3*A*g, 2/3*(dA*g + A*dg)
        out['ddphi2'] = 2/3*(ddA*g + 2*dA*dg + A*ddg)
        out['e3'] = (e, de, dde)
        return out

    def _drho2(self, r):
        e, de, dde = self._p['e3']
        rho = self.f['rho']
        return 2/3*(de*r*d1(rho, r) + e*d1(rho, r) + e*r*d2(rho, r))

    def cont(self, r, y):
        """The quantities continuous across a welded interface, on the side of layer self.i."""
        m, p = self.m, self.mat(r)
        self._p = p
        out = list(Smith.bcs(self, r, y))                                   # (5.44)-(5.46)
        h2 = -2/3*r*self.eps
        dy = self.rhs(r, y, forcing=self._forc)
        g = lambda c, k: self.get(y, c, k)
        dg = lambda c, k: self.get(dy, c, k)
        G4, rho, drho = 4*np.pi*self.Gh, p['rho'], d1(self.f['rho'], r)
        present = self.sph + self.tor
        for l in self.sph:
            u, v, gi, fi = g('U', l), g('V', l), g('G', l), g('F', l)
            for lp in present:
                if abs(lp - l) > 2:
                    continue
                even = (abs(l - lp) % 2 == 0)
                J0, Jpm = J(l, 2, lp, 0, 0, 0, m, 0, m), J(l, 2, lp, 0, '+', '-', m, 0, m)
                Jp = J(l, 2, lp, '+', 0, '+', m, 0, m)
                if lp in self.sph:
                    u += J0*h2*dg('U', lp)                                          # (5.41a)
                    gi += J0*G4*h2*(-LN(0, lp)*rho/r*g('V', lp) - drho*g('U', lp))  # (5.42)
                    fi += J0*h2*dg('F', lp)                                         # (5.43)
                v += Jp*h2*(dg('V', lp) if even else dg('W', lp))                   # (5.41b)
                gi += np.sqrt(3)*h2/r*Jpm*(-G4*rho*g('V', lp) if even else G4*rho*g('W', lp))
            out += [u, v, gi, fi]
        for l in self.tor:
            w = g('W', l)                                                           # (5.41c)
            for lp in present:
                if abs(lp - l) > 2:
                    continue
                even = (abs(l - lp) % 2 == 0)
                w += J(l, 2, lp, '+', 0, '+', m, 0, m)*h2*(dg('W', lp) if even else dg('V', lp))
            out.append(w)
        return np.array(out)

    def _affine(self, r, i):
        self.i = i
        self._forc = False
        M = np.column_stack([self.cont(r, np.eye(self.n)[k]) for k in range(self.n)])
        self._forc = True
        return M, self.cont(r, np.zeros(self.n, complex))

    def solve(self, r_start=1e-3):
        n, ns = self.n, len(self.sph)
        self.i = 0
        Pidx = [self.idx[k] for k in self.idx if k[0] in 'PQRG']
        T = lambda r: np.diag([1/r if i in Pidx else 1.0 for i in range(n)])
        A = lambda r: np.column_stack([self.rhs(r, np.eye(n)[k], forcing=False) for k in range(n)])
        r0 = 1e-6
        dT = np.diag([-1/r0**2 if i in Pidx else 0.0 for i in range(n)])
        ev, V = np.linalg.eig(r0*(np.linalg.inv(T(r0)) @ (A(r0) @ T(r0) - dT)))
        keep = [k for k in np.argsort(-ev.real) if ev[k].real > 0.5][:n//2]
        f = lambda r, y, forc: self.rhs(r, y, forcing=forc)
        run = lambda y0, forc, r1, r2: solve_ivp(f, (r1, r2), y0, args=(forc,), method='DOP853',
                                                 rtol=1e-12, atol=1e-16).y[:, -1]
        sols = [(T(r_start) @ (r_start**ev[k].real*V[:, k])).astype(complex) for k in keep]
        yp = np.zeros(n, complex)
        rb = r_start
        for i, (rt, *_) in enumerate(self.layers):
            self.i = i
            sols = [run(y, False, rb, rt) for y in sols]
            yp = run(yp, True, rb, rt)
            rb = rt
            if i + 1 < len(self.layers):                                    # welded interface
                Mm, bm = self._affine(rt, i)
                Mp, bp = self._affine(rt, i + 1)
                sols = [np.linalg.solve(Mp, Mm @ y) for y in sols]
                yp = np.linalg.solve(Mp, Mm @ yp + bm - bp)
        self.i = len(self.layers) - 1
        z = np.zeros(ns)
        self._forc = True
        b0 = self.bcs(1.0, yp, z)
        self._forc = False
        cols = [self.bcs(1.0, y, z) for y in sols]
        cols += [self.bcs(1.0, np.zeros(n, complex), np.eye(ns)[k]) for k in range(ns)]
        self._forc = True
        c = np.linalg.solve(np.column_stack(cols), -b0)
        self.y = yp + sum(ci*yi for ci, yi in zip(c[:len(sols)], sols))
        self.cext = c[len(sols):]
        return self.y
