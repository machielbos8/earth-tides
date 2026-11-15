# -*- coding: utf-8 -*-
#
# Sparse assembly of the system matrix (_BlockMatrix) and per-layer N in the fluid core (fluid_N).
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
# 18/10/2026 Machiel Bos
#===============================================================================
"""_BlockMatrix builds the same matrix as numpy assignment (each entry written once); fluid_N = N changes nothing and a coarser
mantle N leaves the Love numbers of a viscous core unchanged (CLAUDE.md, 'Memory')."""
import numpy as np
import pytest

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, _BlockMatrix, resolution_per_layer
from earthtides.iers import iers_fit

PREM = 'data/PREM.dat'


def test_block_matrix_matches_numpy():
    """int / slice indices, broadcasting, explicit zeros, chunked conversion; every entry written
    once (the assembly never overwrites); a second write raises, within and across chunks."""
    rng = np.random.default_rng(1)
    shape = (40, 37)
    writes = [((slice(3, 9), slice(5, 12)), rng.normal(size=(6, 7)) + 1j),
              ((4, slice(13, 20)), rng.normal(size=7)),
              ((slice(10, 15), 7), rng.normal(size=5)),
              ((slice(16, 20), slice(0, 4)), 0.0),               # explicit zeros: dropped
              ((5, slice(20, 30)), 2.5),
              ((-1, -2), 3.0),
              ((slice(30, 39), slice(30, 37)), rng.normal(size=(9, 7)))]
    for chunk in (10**9, 20):                                    # one chunk / many chunks
        D = np.zeros(shape, complex)
        B = _BlockMatrix(shape, complex)
        B.CHUNK = chunk
        for key, val in writes:
            D[key] = val
            B[key] = val
        A = B.tocsr()
        assert np.array_equal(A.toarray(), D)
        assert A.nnz == np.count_nonzero(D)
    for chunk in (10**9, 20):                                    # double write: within / across chunks
        B = _BlockMatrix(shape, complex)
        B.CHUNK = chunk
        B[slice(0, 5), slice(0, 5)] = 1.0
        B[slice(10, 15), slice(10, 15)] = 1.0
        B[2, 3] = 2.0
        with pytest.raises(RuntimeError):
            B.tocsr()


def _love(N, fluid_N=None, m=2, sig=1.93, K=4):
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=1.0e3)
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=m, K=K, N=N, fluid_N=fluid_N, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2),
                    dyn_fluid=True, consistent_reference=True, frame='mantle')
    S.solve()
    f = iers_fit(S, g)
    return S, np.array([f[k] for k in ('h0', 'h2', 'l2', 'l1')])


def test_fluid_N_equal_to_N_is_identical():
    _, a = _love(24)
    _, b = _love(24, fluid_N=24)
    assert np.array_equal(a, b)


@pytest.mark.parametrize('m, sig', [(2, 1.93), (1, 0.93)])
def test_fluid_N_coarse_mantle(m, sig):
    """Mantle and inner core at N = 24, core at 30 = uniform N = 30 (measured 2026-10-18 at K 6:
    1e-12 in h(2), 6e-13 in h0; the core N itself moves h(2) by 1.6e-9)."""
    S, split = _love(24, fluid_N=30, m=m, sig=sig)
    _, full = _love(30, m=m, sig=sig)
    assert S.Nl.count(30) == sum(S.fluid_layers()) and S.Nl.count(24) == len(S.Nl) - S.Nl.count(30)
    assert np.max(np.abs(split - full)) < 1e-10


def _love_k(K, fluid_K=None, m=2, sig=1.93, N=24):
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=1.0e3)
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=m, K=K, N=N, fluid_K=fluid_K, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2),
                    dyn_fluid=True, consistent_reference=True, frame='mantle')
    S.solve()
    f = iers_fit(S, g)
    return S, np.array([f[k] for k in ('h0', 'h2', 'l2', 'l1')])


def test_fluid_K_equal_to_K_is_identical():
    _, a = _love_k(4)
    _, b = _love_k(4, fluid_K=4)
    assert np.array_equal(a, b)


def test_fluid_K_coarse_solids():
    """Solids at K = 4, core at 6 = uniform K = 6 at M2 (viscous core). Measured 2026-10-18 (N 30):
    K4/K8 vs K8 1.3e-9 in h(2) at M2; in the diurnal band the solids need K = 6 (K6/K8 vs K8 7e-11,
    K4/K8 7.5e-8)."""
    S, split = _love_k(4, fluid_K=6)
    _, full = _love_k(6)
    fl = S.fluid_layers()
    assert all(len(B.U) == (6 if fl[i] else 4) for i, B in enumerate(S.B))
    assert len(S.degP) == 6
    assert np.max(np.abs(split - full)) < 1e-8


def test_resolution_rules():
    """Rules select by midpoint depth and fluid/solid (core sub-layers included); later rules win;
    a uniform rule set is the plain call, bit for bit."""
    md = EarthModel(PREM).viscous_core(1.93 * OMEGA_EARTH, delta=1.0e3)
    spec = {'K': 6, 'N': 24, 'rules': [{'depth_km': [0, 220], 'N': 12},
                                       {'fluid': True, 'K': 12, 'N': 30},
                                       {'depth_km': [5150, 6371], 'K': 4}]}
    Kl, Nl = resolution_per_layer(md, spec)
    depth = [(md.a - 0.5 * (r0 + r1)) / 1e3 for r0, r1 in zip(md.r0, md.r1)]
    fl = [abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) < 1e8 for i in range(md.n_layers)]
    for d, f, k, n in zip(depth, fl, Kl, Nl):
        if f:
            assert (k, n) == (12, 30)
        elif d > 5150:
            assert (k, n) == (4, 24)
        elif d < 220:
            assert (k, n) == (6, 12)
        else:
            assert (k, n) == (6, 24)
    with pytest.raises(ValueError):
        resolution_per_layer(md, {'K': 4, 'rules': [{'layer': 3, 'K': 8}]})


def test_uniform_layer_lists_are_plain():
    md = EarthModel(PREM).viscous_core(1.93 * OMEGA_EARTH, delta=1.0e3)
    Kl, Nl = resolution_per_layer(md, {'K': 4, 'N': 24})
    _, a = _love_k(4)
    g = ClairautGeometry(md)
    S = TidalSolver(md, g, m=2, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=1.93 * OMEGA_EARTH, omega_rot=OMEGA_EARTH, dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve()
    f = iers_fit(S, g)
    assert np.array_equal(a, np.array([f[k] for k in ('h0', 'h2', 'l2', 'l1')]))


@pytest.mark.parametrize('m, sig', [(2, 1.93), (1, 0.93)])
def test_block_solver_matches_superlu(m, sig):
    """linear_solver='block' (dense layer blocks + iterative refinement) is accepted (residual at
    the SuperLU level) and gives SuperLU's Love numbers; measured 2026-10-18: <= 5e-13 in h(2)
    (viscous K6/N30), 4e-14 at K8/N44."""
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=2.0e3)
    g = ClairautGeometry(md)
    out = {}
    for ls in ('block', 'sparse'):
        S = TidalSolver(md, g, m=m, K=4, N=24, linear_solver=ls, omega_f=0.0, coriolis=OMEGA_EARTH,
                        omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2),
                        dyn_fluid=True, consistent_reference=True, frame='mantle')
        S.solve()
        f = iers_fit(S, g)
        out[ls] = (np.array([f[k] for k in ('h0', 'h2', 'l2', 'l1')]), S.__dict__.get('_block_info'))
    info = out['block'][1]
    assert info is not None and info['residual'] < 1e-11          # block path used, not the fallback
    assert out['sparse'][1] is None
    assert np.max(np.abs(out['block'][0] - out['sparse'][0])) < 1e-10


def test_static_weak_core_m0_on_block_path():
    """Static PREMs (weak-solid core), m = 0: eliminated from the centre the inner core's degree-0
    potential is a free mode of its block (singular, SuperLU took over; NaN before the pivot
    check). Eliminated from the surface (_BlockLU.ORDER = 'down') every layer inherits the
    potential's value through its top interface: the block path solves it, equal to SuperLU."""
    md = EarthModel('data/PREMs.dat')
    out = {}
    for ls in ('block', 'sparse'):
        S = TidalSolver(md, m=0, K=3, N=24, omega_f=0.0, linear_solver=ls)
        r = S.solve()
        out[ls] = (r, S.__dict__.get('_block_info'))
    info = out['block'][1]
    assert info is not None and not info['shifted'] and info['residual'] < 1e-11
    for k in ('h', 'l', 'kk'):
        assert abs(out['block'][0][k] - out['sparse'][0][k]) < 1e-12


def test_static_rotation_regions_use_superlu():
    """Static runs with rotation regions go to SuperLU by rule (not as a fallback): the block path
    solves them, but leaves the rigid-rotation component W ~1e-12..2e-10 off (SuperLU 1e-15;
    2026-10-18, notes_solver_history.md). Same numbers as linear_solver='sparse', bit for bit."""
    md = EarthModel('data/PREMs.dat')
    out = {}
    for ls in ('block', 'sparse'):
        S = TidalSolver(md, m=1, K=3, N=24, omega_f=0.0, linear_solver=ls)
        r = S.solve()
        out[ls] = (r, S.__dict__.get('_block_info'), len(S.regions), S._solver_used)
    assert out['block'][2] > 0 and out['block'][1] is None
    assert out['block'][3] == 'sparse (rotation regions)'
    for k in ('h', 'l', 'kk'):
        assert out['block'][0][k] == out['sparse'][0][k]


def test_solver_used_is_recorded():
    """S._solver_used says which path solved the system: 'block' for a dynamic run."""
    md = EarthModel(PREM).viscous_core(1.93 * OMEGA_EARTH, delta=2.0e3)
    S = TidalSolver(md, ClairautGeometry(md), m=2, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=1.93 * OMEGA_EARTH, omega_rot=OMEGA_EARTH, dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    S.solve()
    assert S._solver_used == 'block'


@pytest.mark.parametrize('flat', [False, True])
def test_static_true_fluid_core(flat):
    """Static PREM with a true (inviscid) fluid core is ill-posed (static core paradox; use PREMs
    for static runs). Whichever path solves it (the block path on the sphere, SuperLU for the
    flattened Earth, 2026-10-18), the result is finite and the same."""
    md = EarthModel(PREM)
    g = ClairautGeometry(md) if flat else None
    out = {}
    for ls in ('block', 'sparse'):
        S = TidalSolver(md, g, m=2, K=4, N=24, omega_f=0.0, linear_solver=ls)
        out[ls] = S.solve()
    for k in ('h', 'l', 'kk'):
        assert np.isfinite(out['block'][k]) and abs(out['block'][k] - out['sparse'][k]) < 1e-8


@pytest.mark.parametrize('m, sig', [(2, 1.93), (1, 0.93)])
def test_block_orders_agree(m, sig):
    """Elimination from the centre ('up') and from the surface ('down', the default) solve the
    same system: equal to round-off (measured: <= 2e-13 in h(2))."""
    import earthtides.solver as SV
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=2.0e3)
    g = ClairautGeometry(md)
    res = {}
    for order in ('up', 'down'):
        SV._BlockLU.ORDER = order
        try:
            S = TidalSolver(md, g, m=m, K=4, N=24, omega_f=0.0, coriolis=OMEGA_EARTH,
                            omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2),
                            dyn_fluid=True, consistent_reference=True, frame='mantle')
            S.solve()
            f = iers_fit(S, g)
        finally:
            SV._BlockLU.ORDER = 'down'
        assert S._block_info is not None
        res[order] = np.array([f[k] for k in ('h0', 'h2', 'l2', 'l1')])
    assert np.max(np.abs(res['up'] - res['down'])) < 1e-11
