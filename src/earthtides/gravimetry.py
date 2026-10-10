# -*- coding: utf-8 -*-
#
# Tidal gravimetric factor of a TidalSolver solution.
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
# 9/10/2026 Machiel Bos
#===============================================================================
"""Gravimetric factor delta (ICET definition, Dehant et al. 1999, Sect. 2).

A gravimeter fixed to the ground measures the magnitude of the specific force F = -grad(Phi)
- a, with a its acceleration relative to the frame of the solution. To first order the change
of |F| along the plumb line n = grad(Phi0)/|grad(Phi0)| (upward) is

    dg = n.grad(phi1 + phiT) + n.H_ext.u + n.a_rel,

with H_ext the Hessian of the reference potential Phi0 just outside the surface (the inside value
minus 4 pi G rho s s, s the normal of the surface: the normal-normal second derivative jumps
with the surface density) and
a_rel = -omega^2 u - 2 i omega Omega x u in the code convention of TidalSolver.operator. The
gravimetric factor is dg divided by the tidal attraction n.grad(phiT) at the same point. Nothing
is removed: gravity does not depend on the reference frame, and a rigid rotation of the whole
Earth changes none of it.
The latitude dependence is fitted with delta(theta) = delta0 + delta_plus f(theta) (Agnew 2015,
eq. 20, after Dehant et al. 1999): f = sqrt(3)/(2 sqrt 2) (7 cos^2 theta - 3) for m = 1 and
sqrt(3)/2 (7 cos^2 theta - 1) for m = 2.
"""
import numpy as np

from .basis import ybar


def surface_gravity(S, inertia=True):
    """Colatitudes (the Gauss-Legendre nodes of S) and the gravity change dg and the tidal
    attraction dgT at the deformed surface, amplitudes of cos(m lambda), in m/s^2 per unit
    forcing of S (phiT = -g a (r/a)^n Ybar_n^m cos m lambda)."""
    md = S.model
    i = md.n_layers - 1
    a, ga, nf, m = md.a, md.g_surface, S.nf, S.m
    p = S.props(i, np.array([a]))
    th = S.theta[0, 0]
    r = p.r[0]
    vals, cext = S.surface_coefficients()
    # displacement (A, B, C) at the surface: same sums as surface_field, no rigid rotation removed
    A = sum(vals[('U', n)] * ybar(n, m, th, 0)[0] for n in S.deg)
    B, C = 0 * th, 0 * th
    for n in S.degV:
        ys = m * ybar(n, m, th, 0, over_sin=True)[0] if m else 0 * th
        B = B + vals[('V', n)] * ybar(n, m, th, 1)[1]
        C = C - vals[('V', n)] * ys
    for n in S.degW:
        ys = m * ybar(n, m, th, 0, over_sin=True)[0] if m else 0 * th
        B = B - vals[('W', n)] * ys
        C = C + vals[('W', n)] * ybar(n, m, th, 1)[1]
    # potential of the deformation, exterior series, and the tidal potential
    p1r = sum(-(n + 1) / r * cext[n] * (a / r)**(n + 1) * ybar(n, m, th, 0)[0] for n in S.deg)
    p1t = sum(cext[n] * (a / r)**(n + 1) * ybar(n, m, th, 1)[1] / r for n in S.deg)
    Yf = ybar(nf, m, th, 1)
    pTr = -ga * nf * (r / a)**(nf - 1) * Yf[0]
    pTt = -ga * (r / a)**(nf - 1) * Yf[1]
    # Hessian of Phi0 (r, theta block; Phi0 axisymmetric), outside the surface
    ph0 = {k: np.broadcast_to(v, p.r.shape)[0] for k, v in p.ph0.items()}
    # the plumb line: the direction of grad Phi0 itself, not the normal of the mapped surface.
    # They differ by O(f^2), and a rigid tilt changes gravity tangentially by g xi, which near
    # K1 (xi ~ 1/(omega - Omega)) leaks into dg through that angle.
    gr, gt = ph0['r'], ph0['t'] / r
    nr, nt = gr / np.hypot(gr, gt), gt / np.hypot(gr, gt)
    dgT = nr * pTr + nt * pTt
    Hrr = ph0['rr']
    Hrt = ph0['rt'] / r - ph0['t'] / r**2
    Htt = ph0['tt'] / r**2 + ph0['r'] / r
    # the jump of the Hessian is across the surface itself: geometric normal (the plumb line differs
    # by O(f^2), which matters for the large tangential displacement of a rigid tilt)
    jump = 4 * np.pi * S.G * md.rho[i](a)
    sr, st = (v[0] for v in S.normal(p))
    Hrr, Hrt, Htt = Hrr - jump * sr * sr, Hrt - jump * sr * st, Htt - jump * st * st
    free_air = nr * (Hrr * A + Hrt * B) + nt * (Hrt * A + Htt * B)
    dg = nr * p1r + nt * p1t + dgT + free_air
    if inertia:
        w2 = S.omega_f**2 + S.omega_t**2
        cw = 2.0 * S.omega_t * S.coriolis
        ar = -w2 * A - cw * np.sin(th) * C
        at = -w2 * B - cw * np.cos(th) * C
        dg = dg + nr * ar + nt * at
    return th, dg, dgT


def gravimetric_factor(S, inertia=True):
    """(delta0, delta_plus, rms misfit) of the gravimetric factor, fitted over latitude."""
    th, dg, dgT = surface_gravity(S, inertia)
    c = np.cos(th)
    if S.m == 1:
        f = np.sqrt(3) / (2 * np.sqrt(2)) * (7 * c**2 - 3)
    elif S.m == 2:
        f = np.sqrt(3) / 2 * (7 * c**2 - 1)
    else:
        f = 0 * c
    G = np.column_stack([dgT, dgT * f]) if S.m else dgT[:, None]
    x, *_ = np.linalg.lstsq(G, dg, rcond=None)
    mis = np.sqrt(np.mean(np.abs(dg - G @ x)**2)) / np.sqrt(np.mean(np.abs(dgT)**2))
    return x[0], (x[1] if S.m else 0.0), mis
