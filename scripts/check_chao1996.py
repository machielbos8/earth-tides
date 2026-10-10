# -*- coding: utf-8 -*-
#
# The ocean tidal angular momentum of Chao et al. (1996) against the ocean admittances: the origin of
# the sign of the in-phase part of the empirical admittance of the IERS Conventions (eq. 6.11).
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
"""Chao et al. (1996, JGR 101, 20151; ~/papers/earthtides/Chao_etal_1996.pdf), Table 3, model C: x, y
components of the mass (height) term of the oceanic tidal angular momentum, Omega c with
c = c13 + i c23 = -a^4 rho_w int zeta sin(theta) cos(theta) exp(i lambda) dOmega, amplitude (1e24 kg m2/s)
and Greenwich phase lag relative to the Doodson argument with the 90-degree shifts of their Table 1.
For zeta = Re[A21 Hbar exp(i(xi + lambda))] sqrt(15) sin cos:  c = -K (A21 Hbar)^* exp(-i xi),
K = (4 pi / sqrt 15) a^4 rho_w. The retrograde part c_r (coefficient of exp(-i xi)) of their x, y gives
A21 = -conj(c_r) / (K Hbar Omega); the IERS formula has the phase of c_r itself.
Usage: python scripts/check_chao1996.py
"""
import numpy as np

CHAO = {'Q1': ((1.1, 342), (2.5, 218)), 'O1': ((4.4, 334), (11.3, 222)),
        'P1': ((1.4, 316), (4.3, 222)), 'K1': ((4.5, 312), (13.5, 222))}
HF = {'Q1': 5.02, 'O1': 26.22, 'P1': 12.20, 'K1': 36.87}                  # cm, Cartwright-Tayler
FES2022B = {'Q1': -0.3283 - 0.3145j, 'O1': -0.2856 - 0.2643j, 'P1': -0.2167 - 0.2216j, 'K1': -0.2120 - 0.2276j}
IERS = {'Q1': 0.3119 - 0.2732j, 'O1': 0.2714 - 0.2514j, 'P1': 0.1982 - 0.2135j, 'K1': 0.1992 - 0.2181j}
A, RHO, OM = 6.371e6, 1035.0, 7.292115e-5


def main():
    K = 4 * np.pi / np.sqrt(15) * A**4 * RHO
    ph = lambda z: np.degrees(np.angle(z))
    print('wave  A21 from Chao (converted)   FES2022b             IERS fFCN fOD        phase of Chao c_r')
    for w, ((ax, gx), (ay, gy)) in CHAO.items():
        cr = (ax * np.exp(1j * np.radians(gx)) + 1j * ay * np.exp(1j * np.radians(gy))) / 2 * 1e24
        a21 = -np.conj(cr) / (K * 0.1992 * HF[w] / 100 * OM)
        print(f'{w:4s}  {a21:+.3f} ({ph(a21):+6.1f})   {FES2022B[w]:+.3f} ({ph(FES2022B[w]):+6.1f})   '
              f'{IERS[w]:+.3f} ({ph(IERS[w]):+6.1f})   {ph(cr):+6.1f}')


if __name__ == '__main__':
    main()
