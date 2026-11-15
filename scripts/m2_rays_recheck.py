# -*- coding: utf-8 -*-
#
# M2 rays at the right frequency: attractor growth for PREM, layered and neutral cores (P1 v2 Sect. 5.2)
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
"""The M2 rays of Sect. 5.2 were traced at sigma = 1.93227, M2's frequency in cycles per solar day; in cycles
per sidereal day it is 1.92700. This script traces the ray of core_flow_attractor.loop (2000 reflections, start
at the critical latitude of the inner core) at both frequencies for the PREM, layered and neutral cores and
gives the growth of log|k| per reflection (> 0: focusing), the number of reflections in which a beam halves in
width (ln 2 / growth) and the attractor period p (None: the ray has not closed).
Usage: python scripts/m2_rays_recheck.py  ->  results/m2_rays_recheck.txt"""
import sys
import warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np

import core_flow_attractor as cfa

FREQS = (1.93227, 1.9322736 / 1.002737909)
CORES = ('PREM', 'layered', 'neutral')


def main():
    from provenance import Run
    with Run('M2 ray attractors at 1.93227 (solar-day value used before) and 1.92700 cpsd',
             params=dict(freqs=FREQS, cores=CORES, nref=cfa.NREF), inputs=['data/PREM.dat']) as run:
        lines = [f'ray of core_flow_attractor.loop, {cfa.NREF} reflections; growth of log|k| per reflection',
                 f'{"core":8s} {"sigma":>8s} {"growth":>9s} {"halving":>8s} {"p":>4s}']
        for core in CORES:
            for f in FREQS:
                _, _, gr, p = cfa.loop(f, core)
                half = np.log(2) / gr if gr > 1e-4 else np.inf
                lines.append(f'{core:8s} {f:8.5f} {gr:+9.5f} {half:8.1f} {str(p):>4s}')
                print(lines[-1], flush=True)
        run.write_text('results/m2_rays_recheck.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
