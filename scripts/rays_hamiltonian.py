# -*- coding: utf-8 -*-
#
# Rays of gravito-inertial waves in PREM's liquid core by Hamiltonian ray tracing (position and wave
# vector together), against the straight inertial-wave rays of Fig. 3 (paper_pepi,
# make_fig_characteristics.py). Supersedes scripts/rays_buoyancy.py, whose turning-point rule did not
# converge with the step.
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
# 02/11/2026 Machiel Bos
#===============================================================================
"""Sub-acoustic waves in the meridional plane (s, z) of a core with buoyancy frequency N(r) obey, for a
wave vector k = (p, q), the local dispersion relation

    W(x, k) = (q^2 + n^2(r) c^2) / |k|^2 = fh^2,   c = p z/r - q s/r = (k x r^),
    n^2 = N^2/(2 Omega)^2,  fh = omega/(2 Omega) = f/2 (f in cpsd, as in Fig. 3),

with N^2 = -g rho'/rho - g^2/c_s^2 (c_s^2 = lambda/rho). The density gradient alone enters at lower
order and does not change W; buoyancy does. A ray follows Hamilton's equations dx/dt = dW/dk,
dk/dt = -dW/dx (here in the arc length of phase space, |k| renormalised to 1 every step and its growth
accumulated);
the group velocity dW/dk is perpendicular to k, so the ray is a characteristic, straight for n = 0. A
turning surface (the boundary of a region where the equation is elliptic, W = fh^2 has no real k) is
passed smoothly: k turns and the ray bends back, with no special rule. At a wall the tangential part of
k is kept and the normal part is the other root of W = fh^2 (phase matching), the reflected group
velocity pointing into the fluid. RK4 with a fixed phase-space step H, each step projected back onto
W = fh^2, landing on the wall by bisection.

Focusing is measured by the growth of |k| (rate per wall reflection, over the second half of the ray):
positive on an attractor, where k grows exponentially (for n = 0 it equals minus the Lyapunov exponent of
two neighbouring rays). Sphere, inner-core radius ratio 1221.5/3480, no flattening, as Fig. 3.
Usage: python scripts/rays_hamiltonian.py [H [WORKERS]]  ->  results/rays_hamiltonian_h<H>.txt, .pdf"""
import sys
import math
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from multiprocessing import Pool

import numpy as np

from earthtides.model import EarthModel

OMEGA = 7.292115e-5
ETA = 1221.5 / 3480.0
H = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-3
FIGF = (0.907, 1.92700)
SCAN = np.round(np.arange(0.880, 0.9501, 0.0025), 4)
NREF, NFIG = 2000, 6000
P0 = (0.62, 0.15)
D0 = (0.5, -0.8)


def buoyancy(md, nx=20001, smooth_km=10.0):
    """r / r_cmb, n^2 = N^2/(2 Omega)^2 and dn^2/d(r / r_cmb) of the outer core (PREM's: one layer). A core in
    several layers (stable layers, convergence_stratification.layered) has steps in n^2 at their edges; there
    n^2 is smoothed over smooth_km (running mean), so that the rays bend through them with the fixed RK4 step."""
    liq = [i for i in range(md.n_layers) if abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) < 1.0]
    oc = [liq[0]]                                   # the contiguous liquid layers from the ICB up
    while oc[-1] + 1 in liq:
        oc.append(oc[-1] + 1)
    r = np.linspace(md.r0[oc[0]], md.r1[oc[-1]], nx)
    n2 = np.empty_like(r)
    for i in oc:
        k = (r >= md.r0[i]) & (r <= md.r1[i])
        rho, drho = md.rho[i](r[k]), md.rho[i].deriv()(r[k])
        g = md.gravity(i, r[k])
        n2[k] = (-g * drho / rho - g ** 2 * rho / md.lam[i](r[k])) / (2 * OMEGA) ** 2
    if len(oc) > 1:
        w = max(1, int(round(smooth_km * 1e3 / (r[1] - r[0]))))
        n2 = np.convolve(np.pad(n2, w, mode='edge'), np.ones(2 * w + 1) / (2 * w + 1), mode='same')[w:-w]
    x = r / md.r1[oc[-1]]
    return x, n2, np.gradient(n2, x, edge_order=2), md.r1[oc[-1]]


class Rays:
    def __init__(self, x, n2, dn2, fh, eta):
        self.x0, self.dx, self.nx = x[0], x[1] - x[0], len(x)
        self.n2, self.dn2 = list(n2), list(dn2)
        self.f2, self.eta = fh * fh, eta

    def _n2(self, r):
        u = min(max((r - self.x0) / self.dx, 0.0), self.nx - 1.000001)
        i = int(u)
        w = u - i
        return (self.n2[i] + w * (self.n2[i + 1] - self.n2[i]),
                self.dn2[i] + w * (self.dn2[i + 1] - self.dn2[i]))

    def W(self, s, z, p, q):
        r = math.hypot(s, z)
        n2, _ = self._n2(r)
        c = (p * z - q * s) / r
        return (q * q + n2 * c * c) / (p * p + q * q)

    def disc(self, s, z):
        """> 0 where the equation is hyperbolic (real characteristics), < 0 elliptic."""
        r = math.hypot(s, z)
        a, b = s / r, z / r
        n2, _ = self._n2(r)
        return self.f2 * (1.0 - self.f2) + n2 * (a * a * self.f2 - b * b * (1.0 - self.f2))

    def project(self, y, it=3):
        """Back onto W = fh^2 along the phase-space gradient of W (RK4 drifts off it where a ray grazes a
        turning surface; the gradient does not vanish there, unlike dW/dk)."""
        for _ in range(it):
            s, z, p, q = y
            r = math.hypot(s, z)
            a, b = s / r, z / r
            n2, dn2 = self._n2(r)
            K2 = p * p + q * q
            c = p * b - q * a
            Wv = (q * q + n2 * c * c) / K2
            vp = (2.0 * n2 * c * b - 2.0 * p * Wv) / K2
            vq = (2.0 * q - 2.0 * n2 * c * a - 2.0 * q * Wv) / K2
            pa = (p * a + q * b) / r
            Ws = (dn2 * a * c * c - 2.0 * n2 * c * b * pa) / K2
            Wz = (dn2 * b * c * c + 2.0 * n2 * c * a * pa) / K2
            d = (Wv - self.f2) / (Ws * Ws + Wz * Wz + vp * vp + vq * vq)
            y = (s - d * Ws, z - d * Wz, p - d * vp, q - d * vq)
        return y

    def rhs(self, s, z, p, q):
        """d(s, z, p, q)/d(sigma), sigma the arc length in phase space (x, k) at |k| = 1: Hamilton's
        equations divided by |(dW/dk, dW/dx)|. (In plain arc length they are singular at a turning point,
        where the group velocity vanishes while k keeps turning.)"""
        r = math.hypot(s, z)
        a, b = s / r, z / r
        n2, dn2 = self._n2(r)
        K2 = p * p + q * q
        c = p * b - q * a
        Wv = (q * q + n2 * c * c) / K2
        vp = (2.0 * n2 * c * b - 2.0 * p * Wv) / K2
        vq = (2.0 * q - 2.0 * n2 * c * a - 2.0 * q * Wv) / K2
        pa = (p * a + q * b) / r
        Ws = (dn2 * a * c * c - 2.0 * n2 * c * b * pa) / K2
        Wz = (dn2 * b * c * c + 2.0 * n2 * c * a * pa) / K2
        v = math.sqrt(vp * vp + vq * vq + Ws * Ws + Wz * Wz)
        return vp / v, vq / v, -Ws / v, -Wz / v

    def rk4(self, y, h):
        k1 = self.rhs(*y)
        k2 = self.rhs(*(y[i] + 0.5 * h * k1[i] for i in range(4)))
        k3 = self.rhs(*(y[i] + 0.5 * h * k2[i] for i in range(4)))
        k4 = self.rhs(*(y[i] + h * k3[i] for i in range(4)))
        return tuple(y[i] + h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4))

    def start(self, p0, d0):
        """A wave vector at p0 on W = fh^2 whose ray goes along d0 (of the two families, the closer)."""
        s, z = p0
        best = None
        ang = np.linspace(0, np.pi, 36001)                     # roots of W - fh^2 in the angle of k
        F = np.array([self.W(s, z, math.cos(t), math.sin(t)) for t in ang]) - self.f2
        for i in np.nonzero(np.sign(F[:-1]) != np.sign(F[1:]))[0]:
            lo, hi = ang[i], ang[i + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if np.sign(self.W(s, z, math.cos(mid), math.sin(mid)) - self.f2) == np.sign(F[i]):
                    lo = mid
                else:
                    hi = mid
            p, q = math.cos(lo), math.sin(lo)
            vs, vz, _, _ = self.rhs(s, z, p, q)
            c = vs * d0[0] + vz * d0[1]
            if best is None or abs(c) > abs(best[0]):
                best = (c, (p, q) if c > 0 else (-p, -q))
        return best[1]

    def reflect(self, s, z, p, q, R):
        """Reflected wave vector at the wall of radius R: tangential part kept, other normal root."""
        r = math.hypot(s, z)
        a, b = s / r, z / r
        mu, ka = p * a + q * b, p * b - q * a                 # k = mu r^ + ka (b, -a)
        A = b * b - self.f2                                    # 0 at the critical latitude
        mu2 = 2.0 * ka * a * b / A - mu                        # sum of the two roots of W = fh^2 in mu
        return mu2 * a + ka * b, mu2 * b - ka * a

    def trace(self, nref, path_from=None):
        """Wall-reflection points of nref segments; growth of log|k| at each reflection; the path (every
        point after reflection path_from, if given); largest |W - fh^2|; smallest discriminant met (near 0:
        the ray grazed a turning surface). A ray whose reflected group velocity would leave the fluid (at a
        wall where the discriminant is ~0, a turning surface meeting the wall) stops there: self.stopped =
        (reflections done, s, z, discriminant)."""
        s, z = P0
        p, q = self.start(P0, D0)
        y = (s, z, p, q)
        pts, logk, path, lk, werr, dmin = [(s, z)], [0.0], [], 0.0, 0.0, np.inf
        self.stopped = None
        while len(pts) <= nref:
            yn = self.rk4(y, H)
            rn = math.hypot(yn[0], yn[1])
            if self.eta < rn < 1.0:
                y = self.project(yn)
                if not self.eta < math.hypot(y[0], y[1]) < 1.0:
                    y = yn
            else:                                              # land on the wall: bisect the step
                R = 1.0 if rn >= 1.0 else self.eta
                lo, hi = 0.0, H
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    rm = math.hypot(*self.rk4(y, mid)[:2])
                    if (rm < 1.0) if R == 1.0 else (rm > self.eta):
                        lo = mid
                    else:
                        hi = mid
                s, z, p, q = self.rk4(y, lo)
                r = math.hypot(s, z)
                s, z = s * R / r, z * R / r
                p, q = self.reflect(s, z, p, q, R)
                for _ in range(3):                             # W = fh^2 at the wall point: correct k only
                    w = self.W(s, z, p, q) - self.f2
                    e = 1e-7
                    gp = (self.W(s, z, p + e, q) - self.W(s, z, p - e, q)) / (2 * e)
                    gq = (self.W(s, z, p, q + e) - self.W(s, z, p, q - e)) / (2 * e)
                    g2 = gp * gp + gq * gq
                    if g2 > 1e-12:
                        p, q = p - w * gp / g2, q - w * gq / g2
                vs, vz, _, _ = self.rhs(s, z, p, q)
                nin = (-s, -z) if R == 1.0 else (s, z)
                if vs * nin[0] + vz * nin[1] < 0:              # no reflection into the fluid: stop here
                    self.stopped = (len(pts) - 1, s, z, self.disc(s, z))
                    break
                y = (s + 1e-9 * vs, z + 1e-9 * vz, p, q)
                pts.append((y[0], y[1]))
            s, z, p, q = y
            kn = math.hypot(p, q)
            lk += math.log(kn)
            y = (s, z, p / kn, q / kn)
            dmin = min(dmin, self.disc(s, z))
            if len(logk) < len(pts):
                logk.append(lk)
            werr = max(werr, abs(self.W(*y) - self.f2))
            if path_from is not None and len(pts) > path_from:
                path.append((s, z))
        return np.array(pts), np.array(logk), np.array(path), werr, dmin


def growth(logk):
    """Rate of log|k| per wall reflection over the second half."""
    n = len(logk)
    if n < 200:
        return np.nan
    k = np.arange(n // 2, n)
    return np.polyfit(k, logk[n // 2:], 1)[0]


def cmb_points(P, n=200):
    q = P[-n:]
    q = q[np.abs(np.hypot(*q.T) - 1.0) < 1e-6]
    return np.unique(np.round(np.degrees(np.arctan2(q[:, 0], q[:, 1])), 1))


def job(args):
    f, which, nref, path_from = args
    x, n2, dn2, rc = buoyancy(EarthModel('data/PREM.dat'))
    if which == 'straight':
        n2, dn2 = 0 * n2, 0 * dn2
    R = Rays(x, n2, dn2, f / 2, ETA)
    P, lk, path, werr, dmin = R.trace(nref, path_from)
    return dict(f=f, which=which, growth=growth(lk), pts=len(cmb_points(P)), cmb=cmb_points(P),
                werr=werr, dmin=dmin, path=path, P=P, stopped=R.stopped, nref=nref)


def main():
    from provenance import Run
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    md = EarthModel('data/PREM.dat')
    x, n2, dn2, rc = buoyancy(md)
    out = f'results/rays_hamiltonian_h{H:g}'
    with Run('rays of Fig. 3 with PREM buoyancy N^2(r), Hamiltonian ray tracing: bending and focusing',
             params=dict(H=H, eta=ETA, fig_freqs=FIGF, scan=list(SCAN), nref=NREF, nfig=NFIG, p0=P0, d0=D0,
                         omega=OMEGA), inputs=['data/PREM.dat']) as run_:
        lines = ['Hamiltonian ray tracing of gravito-inertial waves in PREM\'s outer core (sphere, eta = '
                 f'1221.5/3480, RK4 step {H:g} r_cmb, start {P0} towards {D0}) against straight inertial-wave '
                 'rays (N^2 = 0, Fig. 3)', '', 'n^2 = N^2/(2 Omega)^2 of PREM\'s outer core:',
                 '  r [km]   ' + ''.join(f'{v:8.0f}' for v in x[::2500] * rc / 1e3),
                 '  n^2      ' + ''.join(f'{v:+8.3f}' for v in n2[::2500]), '']
        # check: n = 0 reproduces the straight rays of the figure script
        sys.path.insert(0, 'paper_pepi')
        from make_fig_characteristics import trace as straight_trace
        Pn, _, _, _, _ = Rays(x, 0 * n2, 0 * dn2, FIGF[0] / 2, ETA).trace(50)
        Ps = straight_trace(np.arcsin(FIGF[0] / 2), ETA, 50, P0)
        lines.append(f'check, N^2 = 0 vs the straight rays of Fig. 3: largest difference of the first 50 '
                     f'reflection points {np.abs(Pn - Ps).max():.1e} r_cmb')
        jobs = [(f, w, NREF, None) for f in SCAN for w in ('straight', 'PREM')]
        jobs += [(f, w, NFIG, NFIG - 60) for f in FIGF for w in ('straight', 'PREM')]
        with Pool(nproc) as pool:
            res = pool.map(job, jobs)
        R = {(r['f'], r['which'], r['nref']): r for r in res}
        lines += ['', f'growth of log|k| per wall reflection (> 0: focused onto an attractor), distinct CMB '
                  f'colatitudes (0.1 deg) of the last 200 reflections (few: an attractor), smallest discriminant on '
                  f'the PREM ray (0: grazes a turning surface), largest |W - fh^2|; {NREF} reflections',
                  f'{"f [cpsd]":>9s} {"straight":>9s} {"pts":>4s} {"PREM N^2":>9s} {"pts":>4s} {"disc":>7s} '
                  f'{"W err":>8s}']
        for f in SCAN:
            a, b = R[f, 'straight', NREF], R[f, 'PREM', NREF]
            lines.append(f'{f:9.4f} {a["growth"]:+9.4f} {a["pts"]:4d} {b["growth"]:+9.4f} {b["pts"]:4d} '
                         f'{b["dmin"]:+7.3f} {max(a["werr"], b["werr"]):8.1e}'
                         + ''.join(f'  {r["which"]} stopped after {r["stopped"][0]} reflections at ({r["stopped"][1]:.3f}, '
                                   f'{r["stopped"][2]:.3f}), discriminant {r["stopped"][3]:+.1e}'
                                   for r in (a, b) if r['stopped']))
        lines += ['', f'frequencies of Fig. 3, {NFIG} reflections: growth, CMB colatitudes [deg] of the last 200 '
                  'reflection points; figure: the last 60 segments, shaded where the equation is elliptic']
        fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=(1, 1, 0.8)))
        th = np.linspace(0, 2 * np.pi, 400)
        for ax, f in zip(axs[:2], FIGF):
            Rp = Rays(x, n2, dn2, f / 2, ETA)
            S_, Z_ = np.meshgrid(np.linspace(0, 1, 401), np.linspace(-1, 1, 801))
            Dm = np.vectorize(lambda a, b: Rp.disc(a, b) if ETA < math.hypot(a, b) < 1 else 1.0)(S_ + 1e-9, Z_)
            for sg in (1, -1):
                ax.contourf(sg * S_, Z_, Dm, levels=[-1e9, 0], colors=['#f3d9c9'])
            for w, col, lw, lab in (('straight', '#8a8984', 1.6, 'straight ($N=0$)'),
                                    ('PREM', '#eb6834', 0.8, 'PREM $N^2$')):
                r = R[f, w, NFIG]
                ax.plot(*r['path'].T, color=col, lw=lw, label=lab)
                lines.append(f'  {f:.5f} {w:8s}: growth {r["growth"]:+.4f}, smallest discriminant {r["dmin"]:+.3f}, '
                             f'W err {r["werr"]:.1e}, CMB colatitudes ({r["pts"]}) '
                             + ' '.join(f'{c:.1f}' for c in r['cmb'][:40]) + (' ...' if r['pts'] > 40 else '')
                             + (f'; stopped after {r["stopped"][0]} reflections' if r['stopped'] else ''))
            ax.plot(np.cos(th), np.sin(th), color='#0b0b0b', lw=0.8)
            ax.fill(ETA * np.cos(th), ETA * np.sin(th), color='#d9d8d3', lw=0)
            ax.set_aspect('equal')
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.axis('off')
            ax.set_title(f'{f} cpsd', fontsize=8)
        axs[0].legend(fontsize=6, loc='lower left')
        axs[2].plot(n2, x * rc / 1e3, color='#2a78d6')
        axs[2].axvline(0, color='#8a8984', lw=0.6)
        axs[2].set_xlabel(r'$N^2/(2\Omega)^2$')
        axs[2].set_ylabel('radius (km)')
        fig.tight_layout()
        fig.savefig(run_.output(out + '.pdf'))
        txt = '\n'.join(lines)
        run_.write_text(out + '.txt', txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
