# -*- coding: utf-8 -*-
#
# Check data/PREM.dat against Table I of Dziewonski & Anderson (1981).
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
# 16/11/2026 Machiel Bos
#===============================================================================
"""Compare every layer of data/PREM.dat (and PREMs.dat) with the polynomials of PREM, Table I of Dziewonski &
Anderson (1981, PEPI 25, 297; transcribed from the scan ~/papers/Green/PREM_dziewonski_anderson_1981.pdf, p. 308),
x = r / 6371 km, reference period 1 s: density, Vpv, Vph, Vsv, Vsh, eta at 41 radii per layer, and Q_mu.
Our top layer (6356-6371 km) is PREM's upper crust (6356-6368) and ocean (6368-6371) merged; its mass is checked
against theirs instead.

    .venv/bin/python scripts/check_prem_table1.py  ->  results/check_prem_table1.txt"""
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import Chebyshev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import earthtides

DATA = Path(earthtides.__file__).resolve().parent / 'data'   # the package in use (run_all --installed has no src/)

# Table I: (r0, r1 [km], rho, Vpv, Vph, Vsv, Vsh, eta, Q_mu); polynomials in x as coefficient lists (ascending)
ISO = None
T1 = [
    (0.0, 1221.5, [13.0885, 0, -8.8381], [11.2622, 0, -6.3640], ISO, [3.6678, 0, -4.4475], ISO, [1.0], 84.6),
    (1221.5, 3480.0, [12.5815, -1.2638, -3.6426, -5.5281], [11.0487, -4.0362, 4.8023, -13.5732], ISO, [0.0], ISO,
     [1.0], np.inf),
    (3480.0, 3630.0, [7.9565, -6.4761, 5.5283, -3.0807], [15.3891, -5.3181, 5.5242, -2.5514], ISO,
     [6.9254, 1.4672, -2.0834, 0.9783], ISO, [1.0], 312.0),
    (3630.0, 5600.0, [7.9565, -6.4761, 5.5283, -3.0807], [24.9520, -40.4673, 51.4832, -26.6419], ISO,
     [11.1671, -13.7818, 17.4575, -9.2777], ISO, [1.0], 312.0),
    (5600.0, 5701.0, [7.9565, -6.4761, 5.5283, -3.0807], [29.2766, -23.6027, 5.5242, -2.5514], ISO,
     [22.3459, -17.2473, -2.0834, 0.9783], ISO, [1.0], 312.0),
    (5701.0, 5771.0, [5.3197, -1.4836], [19.0957, -9.8672], ISO, [9.9839, -4.9324], ISO, [1.0], 143.0),
    (5771.0, 5971.0, [11.2494, -8.0298], [39.7027, -32.6166], ISO, [22.3512, -18.5856], ISO, [1.0], 143.0),
    (5971.0, 6151.0, [7.1089, -3.8045], [20.3926, -12.2569], ISO, [8.9496, -4.4597], ISO, [1.0], 143.0),
    (6151.0, 6291.0, [2.6910, 0.6924], [0.8317, 7.2180], [3.5908, 4.6172], [5.8582, -1.4678], [-1.0839, 5.7176],
     [3.3687, -2.4778], 80.0),
    (6291.0, 6346.6, [2.6910, 0.6924], [0.8317, 7.2180], [3.5908, 4.6172], [5.8582, -1.4678], [-1.0839, 5.7176],
     [3.3687, -2.4778], 600.0),
    (6346.6, 6356.0, [2.900], [6.800], ISO, [3.900], ISO, [1.0], 600.0),
]
UPPER_CRUST = (6356.0, 6368.0, 2.600, 5.800, 3.200, 600.0)
OCEAN = (6368.0, 6371.0, 1.020, 1.450, 0.0)
KEYS = ('rho', 'Vp_v', 'Vp_h', 'Vs_v', 'Vs_h', 'eta')


def read_blocks(path):
    lines = [l.rstrip('\n') for l in open(path)]
    n = int(lines[1].split()[0])
    out, i = {}, 2
    while i < len(lines):
        key = lines[i].strip()
        if key in KEYS + ('Qmu',):
            out[key] = [lines[i + 2 + j].split() for j in range(n)]
            i += 2 + n
        else:
            i += 1
    depth = np.array([float(r[0]) for r in out['rho']])
    a = depth[0]
    bounds = np.append(a - depth, a)
    return n, bounds, out


def ours(blocks, bounds, key, i, r):
    c = np.array([float(x) for x in blocks[key][i][1:5]])
    return Chebyshev(c, domain=[bounds[i], bounds[i + 1]])(r)


def main(run):
    lines = ['data/PREM.dat and data/PREMs.dat against Table I of Dziewonski & Anderson (1981)', '']
    for fname in ('PREM.dat', 'PREMs.dat'):
        n, bounds, B = read_blocks(DATA / fname)
        lines.append(f'== {fname}: {n} layers, boundaries {list(np.round(bounds, 1))}')
        for i, row in enumerate(T1):
            r0, r1 = row[0], row[1]
            if abs(bounds[i] - r0) > 1e-6 or abs(bounds[i + 1] - r1) > 1e-6:
                lines.append(f'  layer {i}: boundaries {bounds[i]}-{bounds[i+1]} != PREM {r0}-{r1}')
                continue
            r = np.linspace(r0, r1, 41)
            x = r / 6371.0
            polys = dict(zip(KEYS, row[2:8]))
            polys['Vp_h'] = polys['Vp_h'] or polys['Vp_v']
            polys['Vs_h'] = polys['Vs_h'] or polys['Vs_v']
            diffs = []
            for key in KEYS:
                ref = np.polynomial.polynomial.polyval(x, polys[key])
                d = np.max(np.abs(ours(B, bounds, key, i, r) - ref))
                if d > 5e-5:
                    diffs.append(f'{key} {d:.2e}')
            q = float(B['Qmu'][i][1])
            qref = row[8]
            qok = (np.isnan(q) and not np.isfinite(qref)) or q == qref
            lines.append(f'  {r0:7.1f}-{r1:7.1f}: max |diff| > 5e-5: {", ".join(diffs) if diffs else "none"}; '
                         f'Q_mu {q} vs PREM {qref}{"" if qok else "   <-- DIFFERS"}')
        # merged top layer
        i = n - 1
        r = np.linspace(bounds[i], bounds[i + 1], 41)
        rho = ours(B, bounds, 'rho', i, r)
        vp, vs = ours(B, bounds, 'Vp_v', i, r), ours(B, bounds, 'Vs_v', i, r)
        m_ours = rho.mean() * (bounds[i + 1]**3 - bounds[i]**3)
        m_prem = (UPPER_CRUST[2] * (UPPER_CRUST[1]**3 - UPPER_CRUST[0]**3) + OCEAN[2] * (OCEAN[1]**3 - OCEAN[0]**3))
        lines.append(f'  top layer {bounds[i]:.1f}-{bounds[i+1]:.1f}: rho {rho.mean():.4f}, Vp {vp.mean():.4f}, '
                     f'Vs {vs.mean():.4f}, Q_mu {B["Qmu"][i][1]}; mass / PREM upper crust + ocean = '
                     f'{m_ours / m_prem:.6f}; PREM upper crust rho 2.600 Vp 5.800 Vs 3.200 Q_mu 600')
        lines.append('')
    run.write_text(str(ROOT / 'results' / 'check_prem_table1.txt'), '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    from provenance import Run
    with Run('data/PREM.dat and PREMs.dat against Table I of Dziewonski & Anderson (1981)',
             inputs=[str(DATA / f) if not (DATA / f).is_relative_to(ROOT) else str((DATA / f).relative_to(ROOT))
                     for f in ('PREM.dat', 'PREMs.dat')]) as run_:
        main(run_)
