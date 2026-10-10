# -*- coding: utf-8 -*-
#
# The IERS empirical ocean admittance A21 = fFCN fOD (TN36 Sect. 6.2.1, after Mathews et al. 2002)
# against the admittance of the IERS's own FES2004 ocean tide coefficients (TN36 Sect. 6.3).
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
"""IERS empirical A21 against the IERS's own FES2004 coefficients, all in TN36 conventions.

TN36 eq. 6.8b (checked with its K1 example) puts the tide generating potential of a diurnal wave
in the fully normalized P21 as  A1 Hf sin(theta_f + lambda) = |A1| Hf cos(theta_f + chi_f + lambda)
(A1 < 0, Hf > 0, chi_f = pi/2, Table 6.6), and multiplies complex Love numbers onto the complex
amplitude of exp(+i(theta_f + lambda)): a lag is a negative imaginary part. The equilibrium height
of that term is Hbar = R |A1| Hf.
TN36 eqs. 6.19-6.20 write the prograde ocean tide as Chat sin(theta_f + chi_f + lambda + eps), so in
the same convention zeta21 = Chat exp(i(eps - pi/2)) and A21 = zeta21 / Hbar (eq. 6.11).
Input: fes2004.dat from iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/
Usage: python scripts/check_iers_admittance.py [path/to/fes2004.dat]
"""
import sys
from pathlib import Path

import numpy as np

A1, R = -3.1274e-8, 6.371e6                        # TN36 eq. 6.8, mean radius
# frequency [cpd], |Hf| [m] (Cartwright-Tayler; the signs of Hf and chi_f cancel in A21)
WAVES = {'Q1': (0.893244, 0.05020), 'O1': (0.929536, 0.26221), 'P1': (0.997262, 0.12203),
         'K1': (1.002738, 0.36870)}
# TN36 Table 6.4 resonance parameters (alpha = 0..3) and eq. 6.10 resonance frequencies [cpsd]
SIG = np.array([-0.0026010 - 0.0001361j, 1.0023181 + 0.000025j, 0.999026 + 0.000780j])
K21 = [0.29954, -0.77896e-3, 0.90963e-4, -0.11416e-5]
H21L = [-0.99500, 1.6583e-3, 2.8018e-4, 5.5852e-7]
K21L = [-0.30808, 8.1874e-4, 1.4116e-4, 3.4618e-7]
H21 = [0.60671, -0.15777e-2, 0.18053e-3, -0.18616e-5]  # TN36 Table 7.1, h(0)


def res(c, s):
    return (c[0] + sum(c[a + 1] / (s - SIG[a]) for a in range(3))).real


def f_iers(f_cpd, rho_ratio=0.1115):
    """TN36 Sect. 6.2.1: A21 = fFCN fOD."""
    s = f_cpd / 1.002737909
    f_od = (1.3101 - 0.8098j) - (1.1212 - 0.6030j) * s
    f_eqm = (1 + res(K21, s) - res(H21, s)) / (1 - rho_ratio * (1 + res(K21L, s) - res(H21L, s)))
    return (0.1732 + 0.9687 * f_eqm) * f_od


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / 'scratch/earthtides_iers/fes2004.dat')
    fes = {}
    for line in path.read_text().splitlines()[3:]:
        w = line.split()
        if len(w) > 9 and w[1] in WAVES and w[2:4] == ['2', '1']:
            fes[w[1]] = float(w[8]) / 100.0, np.radians(float(w[9]))      # Chat [m], eps+
    print('wave  A21 FES2004 (TN36 conventions)      A21 IERS fFCN fOD          -conj(FES2004)')
    ph = lambda z: np.degrees(np.angle(z))
    for name, (f, hf) in WAVES.items():
        chat, eps = fes[name]
        a_iers = f_iers(f)
        a_fes = chat * np.exp(1j * (eps - np.pi / 2)) / (R * abs(A1) * hf)
        print(f'{name:4s}  {a_fes:+.4f} ({abs(a_fes):.3f}, {ph(a_fes):+6.1f})  '
              f'{a_iers:+.4f} ({abs(a_iers):.3f}, {ph(a_iers):+6.1f})  '
              f'{-np.conj(a_fes):+.4f} ({ph(-np.conj(a_fes)):+6.1f})')


if __name__ == '__main__':
    main()
