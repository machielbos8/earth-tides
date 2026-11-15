# -*- coding: utf-8 -*-
#
# Dynamic deformation of a homogeneous, non-gravitating, rotating elastic
# spheroid under a degree-2 tidal potential, as a polynomial series.
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
"""Dynamic deformation of a homogeneous, non-gravitating, rotating elastic spheroid
under a degree-2 tidal potential, as a polynomial series.

Independent of the solver: Cartesian coordinates, no mapping, no vector spherical harmonics.
Dimensionless (volume-equivalent radius 1, mu = 1). Complex amplitudes of exp(i sigma t); body
force grad V, with V the complex solid harmonic of degree 2 and order m (m = 1: z(x + iy),
m = 2: (x + iy)^2). The equation of motion in the rotating frame,

    (lam + 1) grad div u + lap u + grad V + rho_s u - i kappa z x u = 0,
    rho_s = rho sigma^2 a^2 / mu,   kappa = 2 rho Omega sigma a^2 / mu,

with zero traction on x^2/A^2 + y^2/A^2 + z^2/C^2 = 1. The solution is an entire function; u is
expanded in all monomials of odd total degree <= D and the equations are imposed by least squares
at points in the volume and on the surface. The residual measures the truncation error (1e-13 for
D = 9 with Earth-like parameters). Inertia and Coriolis are kept to all orders; the static m = 1
problem has no solution on a spheroid (the potential exerts a torque on the bulge), the dynamic one
does: the torque is balanced by the inertia of the tilt.
"""
import numpy as np
from itertools import product

def monos(D):
    return [(i, j, k) for i, j, k in product(range(D+1), repeat=3) if i+j+k <= D and (i+j+k) % 2 == 1]

def mono_eval(e, P, d=(0, 0, 0)):
    """d-th derivative of x^i y^j z^k at points P (n,3)."""
    out = np.ones(len(P), complex)
    for ax in range(3):
        p, q = e[ax], d[ax]
        if q > p: return np.zeros(len(P), complex)
        c = 1.0
        for t in range(q): c *= (p - t)
        out = out * c * P[:, ax]**(p - q)
    return out

def solve(m, lam, rho_s, kappa, A=1.0, C=1.0, D=9, seed=0):
    rng = np.random.default_rng(seed)
    E = monos(D); nb = 3*len(E)
    # volume points inside the spheroid, surface points on it
    nv, ns = 3*nb//3 + 200, nb//2 + 200
    v = rng.normal(size=(nv, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    v *= rng.uniform(0, 1, (nv, 1))**(1/3); v[:, :2] *= A; v[:, 2] *= C
    th = np.arccos(rng.uniform(-1, 1, ns)); la = rng.uniform(0, 2*np.pi, ns)
    S = np.column_stack([A*np.sin(th)*np.cos(la), A*np.sin(th)*np.sin(la), C*np.cos(th)])
    n = S / np.array([A**2, A**2, C**2]); n /= np.linalg.norm(n, axis=1)[:, None]
    # grad V for V = r^2 Y_2m e^{im lam}
    xs, ys, zs = (v[:, k] for k in range(3)); w = xs + 1j*ys
    if m == 1:  gV = np.column_stack([zs, 1j*zs, w])            # V = z (x + i y)
    else:       gV = np.column_stack([2*w, 2j*w, 0*w])          # V = (x + i y)^2
    unit = np.eye(3, dtype=int)
    Mv = np.zeros((3*nv, nb), complex); Ms = np.zeros((3*ns, nb), complex)
    for b, (comp, e) in enumerate(product(range(3), E)):
        d1 = [mono_eval(e, v, tuple(unit[i])) for i in range(3)]
        lap = sum(mono_eval(e, v, tuple(2*unit[i])) for i in range(3))
        val = mono_eval(e, v)
        for j in range(3):
            gd = mono_eval(e, v, tuple(unit[j] + unit[comp]))       # d_j d_comp
            row = (lam + 1)*gd + (lap + rho_s*val if j == comp else 0)
            # -i kappa (z x u)_j : (z x u) = (-u_y, u_x, 0)
            if j == 0 and comp == 1: row = row + 1j*kappa*val
            if j == 1 and comp == 0: row = row - 1j*kappa*val
            Mv[j*nv:(j+1)*nv, b] = row
        s1 = [mono_eval(e, S, tuple(unit[i])) for i in range(3)]
        for j in range(3):
            # sigma_ij n_i = lam div u n_j + (d_i u_j + d_j u_i) n_i
            div = s1[comp]                                   # d_comp of the component
            t = lam*div*n[:, j]
            if j == comp: t = t + sum(s1[i]*n[:, i] for i in range(3))
            t = t + s1[j]*n[:, comp]
            Ms[j*ns:(j+1)*ns, b] = t
    Mat = np.vstack([Mv, Ms]); rhs = np.concatenate([-gV.T.ravel(), np.zeros(3*ns)])
    sc = np.abs(Mat).max(axis=0); sc[sc == 0] = 1
    c = np.linalg.lstsq(Mat/sc, rhs, rcond=None)[0]/sc
    res = np.abs(Mat @ c - rhs).max()/np.abs(rhs).max()
    def u(P):
        out = np.zeros((len(P), 3), complex)
        for b, (comp, e) in enumerate(product(range(3), E)):
            out[:, comp] += c[b]*mono_eval(e, P)
        return out
    return u, res

def surface(u, A, C, th):
    R = 1/np.sqrt(np.sin(th)**2/A**2 + np.cos(th)**2/C**2)
    P = np.column_stack([R*np.sin(th), 0*th, R*np.cos(th)])
    U = u(P)
    er = P/R[:, None]; et = np.column_stack([np.cos(th), 0*th, -np.sin(th)])
    return (U*er).sum(1), (U*et).sum(1), U[:, 1], R
