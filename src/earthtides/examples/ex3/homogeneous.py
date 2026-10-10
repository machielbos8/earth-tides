# -*- coding: utf-8 -*-
#
# Example 3: Love numbers of homogeneous spherical Earth models (Bos &
# Scherneck 2013, Sect. 1.7).
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
# 7/10/2026 Machiel Bos
#===============================================================================
"""Example 3: Love numbers of homogeneous spherical Earth models (Bos & Scherneck 2013, Sect. 1.7).

Two models of Alterman, Jarosch & Pekeris (1959), as tabulated by Bos & Scherneck (2013), with
G = 6.673e-11, a = 6371 km and forcing at the M2 frequency w = 1.40526e-4 rad/s:
  model b: homogeneous, rho 5517 kg/m^3, mu 146 GPa, lambda 347 GPa. Gravity is exactly
           proportional to r, so the spherical-Bessel solution is exact: their Table 1.3 gives the
           body-tide (h, n l, n k) and load (h', n l', n k') Love numbers for degrees 1-1000.
  model a: homogeneous mantle (rho 4460, mu 174, lambda 231) over a fluid core of radius 0.55 a
           (rho 11020, lambda 950); their Table 1.4 gives h2, 2 l2, 2 k2 and the load numbers for
           forcing periods 6, 12 and 24 h. Their per-layer analytical solutions assume g ~ r, which
           does not hold for this layered model, so their numbers are approximate (h 4 %, l 15 %);
           the reference is therefore astyanax computelove (radial ODEs, exact gravity), which we
           match to 1e-8. Their static row (boundary-layer construction for the fluid) is not repeated.
Our solver: spherical geometry, one degree per solve, Chebyshev nodes per layer N.
Run:  python homogeneous.py   ->  homogeneous.out (and the model files modelb.dat, modela.dat)
"""
import math
from pathlib import Path

import numpy as np

from earthtides.model import EarthModel
from earthtides.solver import TidalSolver

HERE = Path(__file__).resolve().parent
W_M2 = 1.40526e-4
BS_B = {2: (0.52221, 0.28413, 0.60384, -0.58502, -0.02167, -0.44057),
        10: (0.10622, 0.01229, 0.14818, -0.88125, 0.14981, -0.91403),
        100: (0.01167, 0.00014, 0.01736, -1.00537, 0.22378, -1.14968)}
BS_A = {'6h': (0.69216, 0.27579, 0.72544, -0.87766, -0.08390, -0.65887),
        '12h': (0.68037, 0.27254, 0.71343, -0.86444, -0.08128, -0.64730),
        '24h': (0.67741, 0.27174, 0.71048, -0.86035, -0.08067, -0.64433)}
# astyanax computelove (radial ODEs, exact self-gravitation), model a, custom omega, elastic
AST_A = {'6h': (0.71859879140, 0.23479124104, 0.73442418162, -0.94395237053, -0.042624867171, -0.70277340110),
         '12h': (0.70546259101, 0.23232720448, 0.72154117763, -0.92830591975, -0.040785430174, -0.68938400421),
         '24h': (0.70210575931, 0.23173593891, 0.71835815355, -0.92339606038, -0.040428985442, -0.68585336486)}


def write_model(name, layers):
    """layers: list of (depth of the layer bottom [km], rho, mu, lambda) from the centre outwards."""
    def blk(title, vals):
        rows = ''.join(f'{d:8.2f}  {v:16.10f}    0.00000    0.00000    0.00000\n' for d, v in vals)
        return f'\n{title}\n{"-" * len(title)}\n' + rows
    rho = [(d, r / 1e3) for d, r, mu, la in layers]
    vp = [(d, math.sqrt((la + 2 * mu) / r) / 1e3) for d, r, mu, la in layers]
    vs = [(d, math.sqrt(mu / r) / 1e3) for d, r, mu, la in layers]
    q = ''.join(f'{d:8.2f}   {"nan" if mu == 0 else "600.0"}\n' for d, r, mu, la in layers)
    s = (f'{name}\t| homogeneous test model (Bos & Scherneck 2013)\n{len(layers)}       | Number of layers\n'
         + blk('rho', rho) + blk('Vp_h', vp) + blk('Vp_v', vp) + blk('Vs_h', vs) + blk('Vs_v', vs)
         + blk('eta', [(d, 1.0) for d, *_ in layers]) + '\nQmu\n---\n' + q)
    path = HERE / f'{name}.dat'
    path.write_text(s)
    return EarthModel(str(path))


def love(md, n, omega, N=24):
    """Body-tide (h, l, k) and load (h', l', k') Love numbers of degree n."""
    S = TidalSolver(md, m=0, N=N, n_force=n, degrees=([n], []), omega_f=omega)
    r = S.solve()
    L = TidalSolver(md, m=0, N=N, n_force=n, degrees=([n], []), omega_f=omega, forcing='load')
    L.solve()
    ll = L.load_love()
    return r['h'], r['l'], r['kk'], ll['h'][n], ll['l'][n], ll['k'][n]


lines = []


def out(s=''):
    print(s, flush=True)
    lines.append(s)


if __name__ == '__main__':
    mb = write_model('modelb', [(6371.0, 5517.0, 146e9, 347e9)])
    ma = write_model('modela', [(6371.0, 11020.0, 0.0, 950e9), (6371.0 * 0.45, 4460.0, 174e9, 231e9)])
    hdr = f"{'':10s}{'h':>10s}{'n l':>10s}{'n k':>10s}{'h_':>10s}{'n l_':>10s}{'n k_':>10s}"
    out('Model b (homogeneous; exact analytical solution), M2 frequency')
    out(hdr.replace('_', "'"))
    for n, ref in BS_B.items():
        N = {2: 24, 10: 32, 100: 96}[n]
        v = love(mb, n, W_M2, N)
        v = (v[0], n * v[1], n * v[2], v[3], n * v[4], n * v[5])
        out(f"{'n=' + str(n) + ' ours':10s}" + ''.join(f'{x:+10.5f}' for x in v))
        out(f"{'     B&S':10s}" + ''.join(f'{x:+10.5f}' for x in ref))
    out()
    out('Model a (homogeneous mantle, fluid core 0.55 a), degree 2; B&S assume g ~ r in each layer')
    out(hdr.replace('_', "'"))
    for T, ref in BS_A.items():
        omega = 2 * math.pi / (float(T[:-1]) * 3600)
        v = love(ma, 2, omega)
        v = (v[0], 2 * v[1], 2 * v[2], v[3], 2 * v[4], 2 * v[5])
        out(f"{T + ' ours':10s}" + ''.join(f'{x:+10.5f}' for x in v))
        out(f"{'     B&S':10s}" + ''.join(f'{x:+10.5f}' for x in ref))
        out(f"{'  astyanax':10s}" + ''.join(f'{x:+10.5f}' for x in AST_A[T]))
        out(f"{'  diff':10s}" + ''.join(f'{x - y:+10.1e}' for x, y in zip(v, AST_A[T])))
    (HERE / 'homogeneous.out').write_text('\n'.join(lines) + '\n')
