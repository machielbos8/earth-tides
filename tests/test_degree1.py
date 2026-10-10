# -*- coding: utf-8 -*-
#
# Degree 1: regularity of phi1 at the centre (P = 0, not P' = 0) and the frame, against the exact
# solution of a homogeneous sphere (examples/ex3/analytic.py).
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
# 8/10/2026 Machiel Bos
#===============================================================================
import sys
from pathlib import Path

import numpy as np

from earthtides.model import EarthModel
from earthtides.solver import TidalSolver

EX3 = Path(__file__).resolve().parents[1] / 'examples' / 'ex3'


def test_degree1_load_static_homogeneous():
    """Static degree-1 load: h' - k' and l' - k' (independent of the translation) exact."""
    sys.path.insert(0, str(EX3))
    import analytic as A
    L = TidalSolver(EarthModel(str(EX3 / 'modelb.dat')), m=0, N=24, n_force=1, degrees=([1], []),
                    omega_f=0.0, forcing='load')
    L.solve()
    ll = L.load_love()
    h, l, k = ll['h'][1], ll['l'][1], ll['k'][1]
    ex = A.love(1, w=1e-8, load=True)
    assert abs((h - k) - (ex[0] - ex[2])) < 1e-7
    assert abs((l - k) - (ex[1] - ex[2])) < 1e-7


def _modelb():
    sys.path.insert(0, str(EX3))
    import analytic as A
    return A, EarthModel(str(EX3 / 'modelb.dat'))


def test_degree1_free_centre_load_dynamic():
    """centre='free' at M2: the moving centre of the exact solution (k'1 = 0 follows)."""
    A, mb = _modelb()
    L = TidalSolver(mb, m=0, N=24, n_force=1, degrees=([1], []), omega_f=A.W_M2, forcing='load',
                    centre='free')
    L.solve()
    ll = L.load_love()
    ex = A.love(1, load=True)
    for got, ref in zip((ll['h'][1], ll['l'][1], ll['k'][1]), ex):
        assert abs(got - ref) < 1e-7


def test_degree1_free_centre_tide_translation():
    """A degree-1 tidal potential is a uniform force: the free centre gives the exact translation."""
    A, mb = _modelb()
    r = TidalSolver(mb, m=0, N=24, n_force=1, degrees=([1], []), omega_f=A.W_M2, centre='free').solve()
    ex = A.love(1)
    for got, ref in zip((r['h'], r['l'], r['kk']), ex):
        assert abs(got - ref) < 1e-6 * abs(ref)


def test_degree1_free_centre_needs_inertia():
    import pytest
    _, mb = _modelb()
    with pytest.raises(ValueError):
        TidalSolver(mb, m=0, n_force=1, degrees=([1], []), omega_f=0.0, centre='free')


def test_load_fcn_resonance_sign():
    """Rotating PREM with fluid core: the degree-2 load Love number h'2 resonates with the free core
    nutation with the sign of Sasao & Wahr (1981, eq. 4.19) and IERS TN36 Table 6.4 (positive residue
    in cpsd, retrograde positive): more negative below the FCN, less negative above. This needs the
    tangential traction -(dLambda/dt) Omega x r of the moving load (load_inertia=True)."""
    from earthtides.love import model_path
    from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
    md = EarthModel(str(model_path('PREM')))
    g = ClairautGeometry(md)
    p = 1 / 457.2
    hp = []
    for x in (0.75, 1.3):
        fr = 1 + x * p
        S = TidalSolver(md, g, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=fr * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                        consistent_reference=True, frame='mantle', forcing='load')
        S.solve()
        hp.append(S.load_love()['h'][2].real)
    assert hp[0] < -1.05 and hp[1] > -0.95


def test_degree4_tilt_potential_removed():
    """Near K1 the rigid tilt xi ~ 1/(sigma - 1) of the flattened figure adds sqrt(10) xi c4 to the
    degree-4 exterior coefficient; with it removed, k4 keeps its size through the band and at O1
    equals the IERS k21(+) (Mathews; TN36 Table 6.4) times sqrt(9/5) (Schmidt vs full norm)."""
    import numpy as np
    from earthtides.model import EarthModel
    from earthtides.love import model_path
    from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
    from earthtides.solver import TidalSolver
    md = EarthModel(str(model_path('PREM')))
    geo = ClairautGeometry(md)
    k = {}
    for s in (0.92700, 1.0015):
        S = TidalSolver(md, geo, m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=s * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True,
                        dyn_fluid=True, consistent_reference=True, frame='mantle')
        S.solve()
        r = S.results()
        k[s] = (r['k'][2].real, r['k'][4].real)
    sig = np.array([-0.0026010, 1.0023181, 0.999026])
    kp = lambda s: -0.804e-3 + sum(c / (s - p) for c, p in zip([0.209e-5, -0.182e-6, -0.713e-9], sig))
    k0 = lambda s: 0.29954 + sum(c / (s - p) for c, p in zip([-0.77896e-3, 0.90963e-4, -0.11416e-5], sig))
    ratio_ours = k[0.927][1] / k[0.927][0] / np.sqrt(9 / 5)
    assert abs(ratio_ours / (kp(0.927) / k0(0.927)) - 1) < 0.01
    assert abs(k[1.0015][1]) < 2e-3          # raw value, with the tilt, is +4.9e-3


def test_angular_momentum_balance():
    """Euler's equation dH/dt + Omega x H = N for the whole Earth (scripts/check_angular_momentum.py):
    at O1 the remainder is 0.25 f of the tidal torque (second order in the flattening)."""
    import sys
    import numpy as np
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from check_angular_momentum import balance
    from earthtides.model import EarthModel
    from earthtides.love import model_path
    from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
    from earthtides.solver import TidalSolver
    md = EarthModel(str(model_path('PREM')))
    w = 0.927 * OMEGA_EARTH
    S = TidalSolver(md, ClairautGeometry(md), m=1, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=w, omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve()
    H, N, lhs = balance(S, w)
    assert np.linalg.norm(lhs - N) / np.linalg.norm(N) < 1.0e-3
