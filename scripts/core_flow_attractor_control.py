# -*- coding: utf-8 -*-
#
# Rotated-loop control for saved core_flow_attractor runs (the control column came after the first scan).
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
# 04/11/2026 Machiel Bos
#===============================================================================
"""From the saved maps (every second point) of results/core_flow_attractor_<TAG>.npz: E/A within DIST/2 of
the loop and its mirror without the walls, and the same on the loop turned by +-20 deg about the centre
(core_flow_attractor.rotated). Close to the full-grid E2int/A of the run's own table.
Usage: python scripts/core_flow_attractor_control.py TAG  ->  results/core_flow_attractor_<TAG>_control.txt"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np

import core_flow_attractor as cfa


def main():
    from provenance import Run
    tag = sys.argv[1]
    src = f'results/core_flow_attractor_{tag}.npz'
    with Run('rotated-loop control from saved core_flow_attractor maps', params=dict(tag=tag),
             inputs=[src]) as run_:
        d = np.load(src)
        lines = [f'{src}: E/A within {cfa.DIST / 2} r_cmb of the loop and its mirror, without the walls, '
                 f'own loop and turned by +-{np.degrees(cfa.ROT):g} deg (saved maps, every second point)',
                 f'{"case":>34s} {"own":>6s} {"turned":>6s}']
        for k in sorted(x[2:] for x in d.files if x.startswith('E_')):
            r, th, E, L = d[f'r_{k}'], d[f'th_{k}'], d[f'E_{k}'].astype(float), d[f'loop_{k}']
            sh = cfa.shares(None, L, r, th, E)
            lines.append(f'{k:>34s} {sh[6] / sh[7]:6.2f} {cfa.rotated(L, r, th, E):6.2f}')
        txt = '\n'.join(lines)
        run_.write_text(f'results/core_flow_attractor_{tag}_control.txt', txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
