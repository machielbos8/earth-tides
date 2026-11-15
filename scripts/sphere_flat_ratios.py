# -*- coding: utf-8 -*-
#
# Short-wavelength energy of the core flow, sphere against flattened PREM (P1 v2 Sect. S9, Fig. S7)
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
# 10/11/2026 Machiel Bos
#===============================================================================
"""Ratios sphere / flattened of the energy of the degrees >= 20 of the solver's core displacement (maps of
core_flow_attractor.py --full-maps, delta 0.25 km, K 168, 16 sub-layers): total (sum over the map points, the
same grid in both), median and peak, at 0.9175 cycles per sidereal day and at M2 (1.927). Was computed by hand
(lab notes 2026-11-06); M2 recomputed at the right frequency.
Usage: python scripts/sphere_flat_ratios.py  ->  results/sphere_flat_ratios.txt"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np

FLAT = {0.9175: 'results/core_flow_attractor_K168_S16N15.npz', 1.927: 'results/core_flow_attractor_M2.npz'}
SPH = 'results/core_flow_attractor_sphere.npz'


def main():
    from provenance import Run
    with Run('sphere / flattened short-wavelength energy of the core flow', inputs=list(FLAT.values()) + [SPH]) as run:
        s = np.load(SPH)
        lines = ['sphere / flattened, energy of degrees >= 20 (delta 0.25 km, K 168): total, median, peak']
        for f, src in FLAT.items():
            k = f'E_PREM_{f}_d0.25_K168_S16_N15'
            a, b = np.load(src)[k], s[k]
            assert a.shape == b.shape
            lines.append(f'{f:7.4f}  total {b.sum() / a.sum():.2f}  median {np.median(b) / np.median(a):.2f}  '
                         f'peak {b.max() / a.max():.2f}')
        run.write_text('results/sphere_flat_ratios.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
