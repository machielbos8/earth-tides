# -*- coding: utf-8 -*-
#
# Offset of h(2) between the inviscid and the viscous core in the fine sweeps of Fig. 4 (P1 v2), against
# the tail of the FCN resonance predicted from the resonance shares.
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
# 01/11/2026 Machiel Bos
#===============================================================================
"""h(2) inviscid (core K 8, 16, 24) minus viscous (delta 4 km, core K 24) at frequencies away from the inviscid
spikes, from results/convergence_viscous_coreK*_N30_d{0,4}_s0.0005.txt, and the prediction
(share_slip - share_visc) * R_h0 / (sigma_FCN - sigma) with the shares of P1 Sect. 5.3 (+0.0040, +0.0015),
the FCN residue of h0 (1.8306e-4 cpsd, lab notes 2026-10-2x) and sigma_FCN = 1 + 1/457.3 sd (sidereal).
Usage: python scripts/core_sweep_offset.py  ->  results/core_sweep_offset.txt"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
from provenance import Run

SHARE_SLIP, SHARE_VISC, R_H0, P_FCN = 0.0040, 0.0015, 1.8306e-4, 457.3
F = 'results/convergence_viscous_coreK%d_N30_d%d_s0.0005.txt'
FN = 'results/convergence_viscous_coreK%d_N30_d0_s0.0005_noslipterm.txt'   # inviscid, without the slip term
                                                                          # (files of commit da66985)
SIG = ((2, (1.90, 1.93, 1.96)), (1, (0.90, 0.92, 0.93, 0.95)))

if __name__ == '__main__':
    ins = [FN % K for K in (8, 16, 24)] + [F % (24, 4)]
    with Run('P1 v2: inviscid-viscous h(2) offset in the Fig. 4 sweeps vs the FCN tail',
             params=dict(share_slip=SHARE_SLIP, share_visc=SHARE_VISC, R_h0=R_H0, P_fcn_sd=P_FCN),
             inputs=ins) as run:
        v = np.genfromtxt(F % (24, 4), skip_header=1, usecols=range(9))
        sf = 1 + 1 / P_FCN
        lines = ['m sigma predicted(1e-6) inv-visc K8 K16 K24 (1e-6)']
        for m, ss in SIG:
            for s in ss:
                pred = (SHARE_SLIP - SHARE_VISC) * R_H0 / (sf - s) * 1e6 if m == 1 else 0.0
                row = []
                for K in (8, 16, 24):
                    i = np.genfromtxt(FN % K, skip_header=1, usecols=range(9))
                    pick = lambda d: d[(d[:, 0] == m) & (abs(d[:, 1] - s) < 1e-6), 5][0]
                    row.append((pick(i) - pick(v)) * 1e6)
                lines.append(f'{m} {s:.3f} {pred:7.1f} ' + ' '.join(f'{x:+6.1f}' for x in row))
        print('\n'.join(lines))
        run.write_text('results/core_sweep_offset.txt', '\n'.join(lines) + '\n')
