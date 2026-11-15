# -*- coding: utf-8 -*-
#
# Where no wave of a given frequency can travel in PREM's liquid core (P1 v2, figure of rays and solver flow).
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
# 05/11/2026 Machiel Bos
#===============================================================================
"""The discriminant of the local dispersion relation (rays_hamiltonian.Rays.disc; negative: no wave of that
frequency, as rays_figure_paths.py) on a quarter of the meridional plane, and the volume share of the
no-wave region, at the frequencies of the figure.
Usage: python scripts/rays_disc_grid.py [F ...] [--core layered]  ->  results/rays_disc_grid[_layered].npz, .txt
(--core layered: core_flow_attractor.base('layered'), the stable layers below the CMB and above the ICB)"""
import sys
import math
_argv, sys.argv = sys.argv, sys.argv[:1]     # rays_hamiltonian reads its step H from argv at import
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
import rays_hamiltonian as rh
sys.argv = _argv
import numpy as np

from earthtides.model import EarthModel


def main():
    from provenance import Run
    argv = sys.argv[1:]
    core = argv.pop(argv.index('--core') + 1) if '--core' in argv else 'PREM'
    argv = [a for a in argv if a != '--core']
    freqs = [float(a) for a in argv] or [0.9175, 1.92700]         # M2 in cycles per sidereal day
    sfx = '' if core == 'PREM' else '_' + core
    with Run(f'no-wave regions (discriminant < 0) of the core ({core}) on a quarter of the meridional plane',
             params=dict(freqs=freqs, core=core), inputs=['data/PREM.dat']) as run_:
        if core == 'PREM':
            md = EarthModel('data/PREM.dat')
        else:
            from core_flow_attractor import base
            md = base(core)
        x, n2, dn2, rc = rh.buoyancy(md)
        s = np.linspace(0, 1, 601)
        S, Z = np.meshgrid(s, s)
        r = np.hypot(S, Z)
        inside = (r > rh.ETA) & (r < 1)
        from rays_figure_paths import intervals
        out = dict(s=s, eta=rh.ETA, x=x, n2=n2, rc=rc, freqs=np.array(freqs))
        lines = [f'regions without waves (discriminant < 0), outer core {core}; radii in km; volume share on a quarter '
                 'grid (weight s)']
        for f in freqs:
            fh2 = (f / 2) ** 2
            R = rh.Rays(x, n2, dn2, f / 2, rh.ETA)
            D = np.ones_like(S)
            for idx in zip(*np.nonzero(inside)):
                D[idx] = R.disc(S[idx], Z[idx])
            out[f'disc_{f}'] = D
            share = (S * (D < 0) * inside).sum() / (S * inside).sum()
            eq = intervals(x * rc / 1e3, n2 < fh2 - 1)
            ax = intervals(x * rc / 1e3, n2 > fh2)
            cm = np.linspace(0.05, 90, 1799)
            dc = np.array([R.disc(0.9999 * math.sin(math.radians(c)), 0.9999 * math.cos(math.radians(c))) for c in cm])
            col = intervals(cm, dc < 0)
            lines += ['', f'{f} cpsd: fh^2 = {fh2:.4f}; no waves in {100 * share:.1f} % of the core volume',
                      '  equatorial plane: r ' + (', '.join(f'{a:.0f}-{b:.0f}' for a, b in eq) or 'none'),
                      '  rotation axis: r ' + (', '.join(f'{a:.0f}-{b:.0f}' for a, b in ax) or 'none'),
                      '  just below the CMB: colatitudes ' + (', '.join(f'{a:.1f}-{b:.1f}' for a, b in col) or 'none')
                      + ' deg',
                      f'  critical latitude arcsin(fh) = {math.degrees(math.asin(f / 2)):.1f} deg']
        run_.savez(f'results/rays_disc_grid{sfx}.npz', **out)
        txt = '\n'.join(lines)
        run_.write_text(f'results/rays_disc_grid{sfx}.txt', txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
