# -*- coding: utf-8 -*-
#
# Exact static deformation of a homogeneous, non-gravitating elastic spheroid
# under a quadratic (degree-2) tidal potential.
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
"""Exact static deformation of a homogeneous, non-gravitating elastic spheroid
under a quadratic (degree-2) tidal potential.

For an ellipsoidal body of uniform material a body force that is linear in the
coordinates produces a displacement that is a cubic polynomial (Chree 1895;
the same property gives Eshelby's uniform-strain result). With the parity of a
degree-2, order 0 or 2 forcing,

    u_x = x (a0 + a1 x^2 + a2 y^2 + a3 z^2)
    u_y = y (b0 + b1 x^2 + b2 y^2 + b3 z^2)
    u_z = z (c0 + c1 x^2 + c2 y^2 + c3 z^2),

which also excludes rigid translations and rotations. The 12 coefficients follow
from the Navier equation, imposed identically, and zero traction on the surface
(x^2 + y^2)/A^2 + z^2/C^2 = 1, imposed identically after eliminating z^2.

The tidal potential (potential energy per unit mass) is the one of the solver,
phiT = -g a (r/a)^2 Ybar_2^m cos(m lambda):
    m = 0:  phiT = -(g/a) (z^2 - (x^2 + y^2)/2)
    m = 2:  phiT = -(g/a) (sqrt(3)/2) (x^2 - y^2)
and the body force is f = -rho grad(phiT).
"""
import numpy as np
import sympy as sp


def solve(A, C, lam, mu, rho, g, a, m=0):
    """Return a function u(x, y, z) -> (ux, uy, uz) of the exact solution.

    Solved in units of the mean radius a (keeps the polynomial coefficients
    of order one); with x = a x' the physical displacement is a^2 u'(x').
    """
    u1 = _solve_unit(A / a, C / a, lam, mu, rho, g, m)
    return lambda X_, Y_, Z_: [a**2 * v for v in u1(X_ / a, Y_ / a, Z_ / a)]


def _solve_unit(A, C, lam, mu, rho, g, m):
    a = 1.0
    x, y, z = sp.symbols('x y z', real=True)
    co = sp.symbols('a0:4 b0:4 c0:4', real=True)
    a_, b_, c_ = co[0:4], co[4:8], co[8:12]
    q = lambda k: k[0] + k[1] * x**2 + k[2] * y**2 + k[3] * z**2
    u = sp.Matrix([x * q(a_), y * q(b_), z * q(c_)])
    X = (x, y, z)
    grad = sp.Matrix(3, 3, lambda i, j: sp.diff(u[j], X[i]))     # grad[i, j] = d_i u_j
    div = sum(sp.diff(u[i], X[i]) for i in range(3))
    eps = (grad + grad.T) / 2
    sig = lam * div * sp.eye(3) + 2 * mu * eps
    if m == 0:
        phiT = -(g / a) * (z**2 - (x**2 + y**2) / 2)
    else:
        phiT = -(g / a) * (sp.sqrt(3) / 2) * (x**2 - y**2)
    f = [-rho * sp.diff(phiT, X[i]) for i in range(3)]
    eqs = []
    # Navier: div sigma + f = 0, identically in x, y, z
    for j in range(3):
        nav = sp.expand(sum(sp.diff(sig[i, j], X[i]) for i in range(3)) + f[j])
        eqs += sp.Poly(nav, x, y, z).coeffs()
    # traction-free on the spheroid, imposed at many surface points (the traction
    # is a cubic polynomial, so enough points make the condition exact)
    n = [x / A**2, y / A**2, z / C**2]
    trac = [sp.expand(sum(sig[i, j] * n[i] for i in range(3))) for j in range(3)]
    rng = np.random.default_rng(0)
    for th, la in zip(np.arccos(rng.uniform(-1, 1, 40)), rng.uniform(0, 2 * np.pi, 40)):
        pt = {x: A * np.sin(th) * np.cos(la), y: A * np.sin(th) * np.sin(la), z: C * np.cos(th)}
        eqs += [t.subs(pt) for t in trac]
    M, rhs = sp.linear_eq_to_matrix([e for e in eqs if e != 0], co)
    scale = np.abs(np.array(M.evalf(), dtype=float)).max(axis=1, keepdims=True)
    Mn = np.array(M.evalf(), dtype=float) / scale
    rn = np.array(rhs.evalf(), dtype=float).ravel() / scale.ravel()
    sol, res, rank, _ = np.linalg.lstsq(Mn, rn, rcond=None)
    resid = np.linalg.norm(Mn @ sol - rn) / max(np.linalg.norm(rn), 1e-300)
    if rank < 12 or resid > 1e-10:
        raise RuntimeError(f'no exact polynomial solution: rank {rank}, residual {resid:.1e}')
    fu = sp.lambdify((x, y, z), list(u.subs(dict(zip(co, sol)))), 'numpy')
    return lambda X_, Y_, Z_: [np.broadcast_to(v, np.shape(X_)) for v in fu(X_, Y_, Z_)]


def surface_field(usol, A, C, theta):
    """u_r, u_theta of the exact solution at the surface points, lambda = 0."""
    st, ct = np.sin(theta), np.cos(theta)
    R = 1.0 / np.sqrt(st**2 / A**2 + ct**2 / C**2)
    X, Z = R * st, R * ct
    ux, uy, uz = usol(X, 0 * X, Z)
    ur = ux * st + uz * ct
    ut = ux * ct - uz * st
    return ur, ut, R
