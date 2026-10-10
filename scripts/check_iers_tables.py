# -*- coding: utf-8 -*-
#
# Which sign of the ocean admittance the IERS tables 6.5a and 7.2 and DEHANTTIDEINEL contain.
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
# 10/10/2026 Machiel Bos
#===============================================================================
"""The resonant ocean-tide terms in the tables of the IERS Conventions (TN36), separated from the
resonance formulae and compared with eq. (6.11) for both signs of the in-phase admittance:

  Table 6.5a (Chapter 6, geopotential): delta k_f = (k21(sigma) - k21 nominal) + delta k(OT)
  Table 7.2 (Chapter 7, displacement): h(0), l(0) = resonance formula (Table 7.1) + delta h,l(OT)
  Table 7.3a / DEHANTTIDEINEL STEP2DIU: radial K1 correction from Table 7.2

eq. (6.11): delta L(OT) = [L'(sigma) - L'(nom)] (4 pi G rho_w R / 5 g) A21(sigma), with the load
Love numbers of TN36 Table 6.4 and A21 = f_FCN f_OD (empirical) or -conj (our sign, Sect. 8.2).
Usage: python scripts/check_iers_tables.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from check_iers_admittance import f_iers
from earthtides import etide

SIG = np.array([-0.0026010 - 0.0001361j, 1.0023181 + 0.000025j, 0.999026 + 0.000780j])   # eq. 6.10
K0 = [0.29954 - 0.1412e-2j, -0.77896e-3 - 0.3711e-4j, 0.90963e-4 - 0.2963e-5j,
      -0.11416e-5 + 0.5325e-7j]                                                          # Table 6.4
LOAD = {'k': ([-0.30808, 8.1874e-4, 1.4116e-4, 3.4618e-7], -0.3075),
        'h0': ([-0.99500, 1.6583e-3, 2.8018e-4, 5.5852e-7], -1.001),
        'l0': ([0.02315, 2.3232e-4, -8.4659e-6, 1.0724e-8], 0.0295)}
FAC = 4 * math.pi * 6.674e-11 * 1025.0 * 6.371e6 / (5 * 9.820)
# Table 6.5a: deg/hr, delta k_f (units 1e-5 in the table)
T65A = {'O1': (13.94303, -0.00083 + 0.00007j), 'P1': (14.95893, -0.01138 + 0.00077j),
        'K1': (15.04107, -0.04084 + 0.00262j)}


def res(c, s):
    return c[0] + sum(c[a + 1] / (s - SIG[a]) for a in range(3))


def ot(key, f_cpd, our_sign):
    c, nom = LOAD[key]
    A = f_iers(f_cpd)
    return (res(c, f_cpd / 1.002737909).real - nom) * FAC * (-np.conj(A) if our_sign else A)


def main():
    print('Table 6.5a (Chapter 6): delta k_f = body + ocean term')
    for name, (dh, tab) in T65A.items():
        f = dh / 15
        body = res(K0, f / 1.002737909) - (0.29830 - 0.00144j)
        print(f'  {name}: table {tab:+.5f}  empirical {body + ot("k", f, False):+.5f}  '
              f'our sign {body + ot("k", f, True):+.5f}')
    k_our = res(K0, 15.04107 / 15 / 1.002737909) - (0.29830 - 0.00144j) + ot('k', 15.04107 / 15, True)
    print(f'  K1 in-phase amplitude A1 dk_f H_f [1e-12]: table 470.9, our sign '
          f'{470.9 * k_our.real / T65A["K1"][1].real:.1f}')
    print('Table 7.2 (Chapter 7): table minus resonance formula = ocean term')
    for name, f in (('O1', 0.929536), ('P1', 0.997262), ('K1', 1.002738)):
        for key in ('h0', 'l0'):
            d = complex(etide.table72(key, f)) - complex(etide.mhb_resonance(key, f))
            print(f'  {name} {key}: table - formula {d:+.5f}  empirical {ot(key, f, False):+.5f}  '
                  f'our sign {ot(key, f, True):+.5f}')
    mm = 3 * math.sqrt(5 / (24 * math.pi)) * 0.36870 * 0.5 * 1e3     # mm per unit h, sin 2 phi = 1
    h_iers = complex(etide.table72('h0', 1.002738))
    h_our = complex(etide.mhb_resonance('h0', 1.002738)) + ot('h0', 1.002738, True)
    print(f'Table 7.3a / STEP2DIU, K1 radial in-phase [mm]: table 12.00, from Table 7.2 '
          f'{-(h_iers.real - 0.6078) * mm:.2f}, our sign {-(h_our.real - 0.6078) * mm:.2f}')


if __name__ == '__main__':
    main()
