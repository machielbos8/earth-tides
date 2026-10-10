# -*- coding: utf-8 -*-
#
# Example 3, analytic part: Love numbers of a homogeneous, compressible, self-
# gravitating sphere from the exact solution in spherical Bessel functions
# (Love 1911; Pekeris & Jarosch 1958; Dahlen & Tromp 1998, Sect. 8.7; Bos &
# Scherneck 2013, Appendix 2).
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
# 8/10/2026 Machiel Bos
#===============================================================================
"""Example 3, analytic part: Love numbers of a homogeneous, compressible, self-gravitating sphere
from the exact solution in spherical Bessel functions (Love 1911; Pekeris & Jarosch 1958;
Dahlen & Tromp 1998, Sect. 8.7; Bos & Scherneck 2013, Appendix 2).

For degree n and frequency omega there are three solutions regular at the centre, two with
spherical Bessel functions j_n(gamma_+- r) and one with r^n (B&S eqs 1.131-1.143), with

    gamma^2 = omega^2/(2 v_b^2) + (omega^2 + 16 pi G rho / 3)/(2 v_a^2)
              +- 1/2 { [omega^2/v_b^2 - (omega^2 + 16 pi G rho / 3)/v_a^2]^2
                       + (8 pi G k rho / (3 v_a v_b))^2 }^(1/2),     k = sqrt(n (n + 1)).

NOTE: B&S eq. 1.132 as printed has 16 pi G rho / 3 instead of (omega^2 + 16 pi G rho / 3) inside the
square root; only the form above reproduces their Table 1.3 (at degree 2 the misprint changes h by
6e-4 at the M2 frequency). The three amplitudes follow from the surface conditions for a unit
potential: radial stress R = 0 (body tide) or R = (2n+1) g / (4 pi G a) (load), tangential stress
S = 0, and the gravity condition Q = (2n+1)/a. Then h = -g U(a), l = -g V(a), k = P(a) - 1 (the minus sign
because the unit potential of these formulas has the opposite sign to the tidal potential).
Run:  python analytic.py   (model b of Bos & Scherneck 2013, M2 frequency)
"""
import numpy as np
from scipy.special import spherical_jn

G, a = 6.673e-11, 6.371e6
RHO, MU, LAM = 5517.0, 146e9, 347e9
W_M2 = 1.40526e-4
TABLE_1_3 = {2: (0.52221, 0.28413, 0.60384, -0.58502, -0.02167, -0.44057),
             10: (0.10622, 0.01229, 0.14818, -0.88125, 0.14981, -0.91403)}


def solutions(n, w, rho=RHO, mu=MU, lam=LAM, r=a):
    """(U, V, P, R, S, Q) at radius r of the three regular solutions of degree n."""
    kap = lam + 2 / 3 * mu
    va2, vb2 = (lam + 2 * mu) / rho, mu / rho
    gpr = np.pi * G * rho
    k2 = n * (n + 1)
    A, B = w**2 / vb2, (w**2 + 16 / 3 * gpr) / va2
    out = []
    for sgn in (1, -1):
        g2 = 0.5 * (A + B) + sgn * 0.5 * np.sqrt((A - B)**2 + (8 * gpr * np.sqrt(k2) / (3 * np.sqrt(va2 * vb2)))**2)
        gm = np.sqrt(complex(g2))
        z = 0.75 / gpr * vb2 * (g2 - w**2 / vb2)          # zeta
        xi = z - (n + 1)
        jn, jn1 = spherical_jn(n, gm * r), spherical_jn(n + 1, gm * r)
        out.append((n * xi / r * jn - z * gm * jn1,
                    xi / r * jn + gm * jn1,
                    -4 * np.pi * G * rho * z * jn,
                    -((kap + 4 / 3 * mu) * z * g2 - 2 * n * (n - 1) * mu * xi / r**2) * jn
                    + 2 * mu * (2 * z + k2) * gm / r * jn1,
                    mu * (g2 + 2 * (n - 1) * xi / r**2) * jn - 2 * mu * (z + 1) * gm / r * jn1,
                    -4 * np.pi * G * rho / r * (k2 + (n + 1) * z) * jn))
    out.append((n * r**(n - 1), r**(n - 1), (w**2 - 4 / 3 * gpr * n) * r**n,
                2 * n * (n - 1) * mu * r**(n - 2), 2 * (n - 1) * mu * r**(n - 2),
                ((2 * n + 1) * w**2 - 8 / 3 * gpr * n * (n - 1)) * r**(n - 1)))
    return out


def love(n, w=W_M2, load=False):
    """h, n l, n k (body tide) or h', n l', n k' (load) of the homogeneous sphere."""
    g = 4 / 3 * np.pi * G * RHO * a
    sol = solutions(n, w)
    M = np.array([[s[3], s[4], s[5]] for s in sol]).T            # rows R, S, Q; columns solutions
    rhs = [(2 * n + 1) / a * g / (4 * np.pi * G) if load else 0.0, 0.0, (2 * n + 1) / a]
    c = np.linalg.solve(M, rhs)
    U, V, P = (sum(c[i] * sol[i][j] for i in range(3)) for j in range(3))
    h, l, k = -(g * U).real, -(g * V).real, (P - 1).real     # the same signs for body tide and load
    return h, n * l, n * k


if __name__ == '__main__':
    print('Homogeneous sphere (model b), M2: exact solution vs Bos & Scherneck (2013) Table 1.3')
    print(f"{'n':>4s} {'h':>9s} {'n l':>9s} {'n k':>9s} {'h_':>9s} {'n l_':>9s} {'n k_':>9s}".replace('_', "'"))
    for n, ref in TABLE_1_3.items():
        v = love(n) + love(n, load=True)
        print(f'{n:4d} ' + ' '.join(f'{x:+9.5f}' for x in v))
        print('     ' + ' '.join(f'{x:+9.5f}' for x in ref) + '   (Table 1.3)')
