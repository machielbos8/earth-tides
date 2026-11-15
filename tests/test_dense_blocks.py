# -*- coding: utf-8 -*-
#
# Hybrid assembly (dense layer blocks + sparse rest, _HybridMatrix) against the sparse assembly
# (_BlockMatrix): plan docs/plan_solver_memory_2026-11-02.md, step 2.
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
# 02/11/2026 Machiel Bos
#===============================================================================
"""(a) _HybridMatrix stores the same entries as _BlockMatrix for any write pattern, and a double write
raises; (b) the assembled system of real configurations is identical; (c) the scaled blocks are
bit-identical to the scaled CSR; (d) Love numbers agree with the sparse assembly to round-off; (e) the
SuperLU and LAPACK routes and the rotation-region rule are unchanged."""
import numpy as np
import pytest

from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, SphereGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, _BlockMatrix, _HybridMatrix, resolution_per_layer
from earthtides.iers import iers_fit

PREM, PREMS = 'data/PREM.dat', 'data/PREMs.dat'


def _writes(rng):
    """Writes in the patterns of the assembly: inside one block, straddling two layers, border rows
    and columns, int and slice indices, broadcast scalars, explicit zeros, a non-contiguous row."""
    return [((slice(1, 4), slice(0, 5)), rng.normal(size=(3, 5)) + 1j),        # block 0
            ((slice(5, 9), slice(6, 12)), rng.normal(size=(4, 6))),             # block 1
            ((4, slice(3, 9)), rng.normal(size=6)),                              # straddles blocks 0, 1
            ((slice(12, 15), slice(9, 15)), 2.0),                               # straddles 1, 2 (scalar)
            ((slice(16, 18), slice(15, 18)), 0.0),                               # explicit zeros
            ((18, slice(0, 18)), rng.normal(size=18)),                           # border row
            ((slice(0, 18), 19), rng.normal(size=18)),                           # border column
            ((-1, -1), 3.0),
            ((0, 0), 1.5)]


@pytest.mark.parametrize('chunk', [10**9, 7])
def test_hybrid_matches_block_matrix(chunk):
    rng = np.random.default_rng(3)
    shape, off = (20, 20), [0, 4, 9, 15, 18]
    D = np.zeros(shape, complex)
    A = _BlockMatrix(shape, complex)
    H = _HybridMatrix(shape, complex, off)
    A.CHUNK = H.rest.CHUNK = chunk
    for key, val in _writes(rng):
        D[key] = val
        A[key] = val
        H[key] = val
    Ah, Ab = H.finish().tocsr(), A.tocsr()
    assert np.array_equal(Ah.toarray(), D)
    assert (Ah != Ab).nnz == 0 and Ah.nnz == Ab.nnz
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):          # blocks hold exactly the diagonal part
        assert np.array_equal(H.blocks[i], D[a:b, a:b])
    assert H.rest[:18, :18].multiply(sps_block_mask(off, 18)).nnz == 0


def sps_block_mask(off, n):
    import scipy.sparse as sps
    return sps.block_diag([sps.csr_matrix(np.ones((b - a, b - a))) for a, b in zip(off[:-1], off[1:])], format='csr')[:n, :n]


def test_hybrid_double_write_raises():
    H = _HybridMatrix((10, 10), float, [0, 5, 9])
    H[slice(0, 3), slice(0, 3)] = 1.0
    with pytest.raises(RuntimeError):
        H[2, slice(2, 4)] = 2.0                                     # (2, 2) again, inside block 0
    H = _HybridMatrix((10, 10), float, [0, 5, 9])
    H[np.int64(1), 7] = 1.0                                         # outside the blocks: rest
    H[1, 7] = 1.0
    with pytest.raises(RuntimeError):
        H.finish()


def _configs():
    sig1, sig2 = 0.93, 1.93
    rot = dict(omega_f=0.0, coriolis=OMEGA_EARTH, omega_rot=OMEGA_EARTH, dyn_fluid=True,
               consistent_reference=True, frame='mantle')
    return [
        ('viscous m1', lambda: EarthModel(PREM).viscous_core(sig1 * OMEGA_EARTH, delta=2e3),
         dict(m=1, K=4, N=20, omega_t=sig1 * OMEGA_EARTH, wahr_frame=True, **rot), True),
        ('viscous m2', lambda: EarthModel(PREM).viscous_core(sig2 * OMEGA_EARTH, delta=2e3),
         dict(m=2, K=4, N=20, omega_t=sig2 * OMEGA_EARTH, **rot), True),
        ('inviscid m2', lambda: EarthModel(PREM),
         dict(m=2, K=3, N=16, omega_t=sig2 * OMEGA_EARTH, **rot), True),
        ('static PREMs m0', lambda: EarthModel(PREMS),
         dict(m=0, K=3, N=16, omega_f=0.0, omega_rot=OMEGA_EARTH, consistent_reference=True), True),
        ('static PREMs m1 regions', lambda: EarthModel(PREMS),
         dict(m=1, K=3, N=16, omega_f=0.0, omega_rot=OMEGA_EARTH, consistent_reference=True), True),
        ('static PREMs m2 sphere', lambda: EarthModel(PREMS),
         dict(m=2, K=3, N=16, omega_f=0.0), False),
        ('load m2', lambda: EarthModel(PREMS),
         dict(m=2, K=3, N=16, omega_f=0.0, forcing='load', omega_rot=OMEGA_EARTH,
              consistent_reference=True), True),
    ]


def _solve(make, kw, flat, **extra):
    md = make()
    S = TidalSolver(md, ClairautGeometry(md) if flat else SphereGeometry(), **kw, **extra)
    S.solve()
    return S


def _love(S):
    r = S.results()
    return np.array([r['h'], r['l'], r['kk']], complex)


@pytest.mark.parametrize('name,make,kw,flat', _configs(), ids=[c[0] for c in _configs()])
def test_same_system_and_numbers(name, make, kw, flat):
    a = _solve(make, kw, flat, assembly='sparse')
    b = _solve(make, kw, flat, assembly='hybrid')
    if b._solver_used == 'block':
        assert b._Hsp is None                   # the block path never built the CSR
    assert (a.Hsp != b.Hsp).nnz == 0            # (b) identical system, entry for entry
    assert np.array_equal(a.f, b.f)
    assert a._solver_used == b._solver_used
    la, lb = _love(a), _love(b)
    # (d) the system is identical (above); the solves differ only by round-off, which depends on the
    # platform's BLAS threads (Windows: 6e-13 relative), hence 1e-10
    assert np.max(np.abs(la - lb)) <= 1e-10 * np.max(np.abs(la)), (la, lb)


def test_scaled_blocks_bit_identical():
    """(c) the hybrid path's scaled blocks and rest equal the scaled CSR of the sparse path exactly."""
    sig = 0.93
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=2e3)
    S = TidalSolver(md, ClairautGeometry(md), m=1, K=3, N=16, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True,
                    consistent_reference=True, frame='mantle')
    seen = {}
    orig = TidalSolver._block_path

    def spy(self, A, b, rs, cs, factor=None):
        seen['rs'], seen['cs'], seen['F'] = rs, cs, factor
        return orig(self, A, b, rs, cs, factor)
    TidalSolver._block_path = spy
    try:
        S.solve()
    finally:
        TidalSolver._block_path = orig
    Hc, rs, cs = TidalSolver._scale_csr(S.Hsp)
    assert np.array_equal(rs, seen['rs']) and np.array_equal(cs, seen['cs'])
    D = Hc.toarray()
    Hh, off = S._Hh, S._off
    irs, ics = 1.0 / rs, 1.0 / cs
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        B = np.multiply(Hh.blocks[i], irs[a:b, None]) * ics[None, a:b]
        assert np.array_equal(B, D[a:b, a:b])


@pytest.mark.parametrize('solver', ['sparse', 'dense'])
def test_other_linear_solvers_unchanged(solver, monkeypatch):
    """(e) SuperLU and LAPACK through the CSR of the hybrid: the same numbers as with the sparse
    assembly, bit for bit."""
    monkeypatch.setenv('EARTHTIDES_MAX_GB', '8')
    sig = 1.93
    make = lambda: EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=2e3)
    kw = dict(m=2, K=3, N=16, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH,
              omega_rot=OMEGA_EARTH, dyn_fluid=True, consistent_reference=True, frame='mantle',
              linear_solver=solver)
    a = _solve(make, kw, True, assembly='sparse')
    b = _solve(make, kw, True, assembly='hybrid')
    assert getattr(a, '_solver_used', None) == getattr(b, '_solver_used', None)   # (dense: unset)
    assert np.array_equal(a.coef, b.coef)


def test_layer_coef_and_reuse_cache():
    """Factor reuse across neighbouring frequencies works on the hybrid path (BlockCache)."""
    from earthtides.solver import BlockCache
    cache = BlockCache()
    out = []
    for sig in (0.930, 0.9305):
        md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=2e3)
        Kl, Nl = resolution_per_layer(md, {'K': 3, 'N': 16})
        S = TidalSolver(md, ClairautGeometry(md), m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0,
                        coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH,
                        wahr_frame=True, dyn_fluid=True, consistent_reference=True, frame='mantle',
                        factor_cache=cache)
        S.solve()
        ref = TidalSolver(md, ClairautGeometry(md), m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0,
                          coriolis=OMEGA_EARTH, omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH,
                          wahr_frame=True, dyn_fluid=True, consistent_reference=True, frame='mantle',
                          assembly='sparse')
        ref.solve()
        out.append((iers_fit(S, ClairautGeometry(md))['h2'], iers_fit(ref, ClairautGeometry(md))['h2']))
    assert cache.factored + cache.reused == 2
    for h, r in out:
        assert abs(h - r) <= 1e-10                  # round-off only (platform BLAS)


def test_thickness_selector():
    """resolution_per_layer: 'thickness_km' picks the Ekman sub-layers of viscous_core (40 km and
    5 delta next to each liquid-solid wall), not the core interior, at any delta."""
    for delta in (0.5e3, 1e3, 4e3):
        md = EarthModel(PREM).viscous_core(0.93 * OMEGA_EARTH, delta=delta)
        Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 24, 'rules': [
            {'fluid': True, 'K': 20, 'N': 30}, {'fluid': True, 'thickness_km': [0, 50], 'N': 16}]})
        th = [(md.r1[i] - md.r0[i]) / 1e3 for i in range(md.n_layers)]
        fl = [abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) < 1e8 for i in range(md.n_layers)]
        for i in range(md.n_layers):
            if fl[i]:
                assert Kl[i] == 20 and Nl[i] == (16 if th[i] <= 50 else 30)
            else:
                assert (Kl[i], Nl[i]) == (6, 24)
        assert sum(f and t <= 50 for f, t in zip(fl, th)) == (4 if delta < 4e3 else 2)
    with pytest.raises(ValueError):
        resolution_per_layer(md, {'rules': [{'thick_km': [0, 1]}]})
