# -*- coding: utf-8 -*-
#
# Banded angular coupling (_BandBlock, _BandLU, TidalSolver(band=B)): plan
# docs/plan_banded_coupling_2026-11-02.md, step 3.
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
"""(a) degree order and bandwidths against brute force; (b) a _BandBlock holds exactly the masked dense
block for random writes, a double write raises, scaling / mat-vec / CSR agree with the dense block;
(c) _BandLU with a Woodbury update solves the updated system; (d) band >= every degree distance gives
the full-coupling numbers; (e) band storage gives the numbers of the same band on dense storage;
(f) the SuperLU fallback is refused above BAND_FALLBACK_MAX; band needs the hybrid assembly."""
import numpy as np
import pytest

from earthtides import solver as sv
from earthtides.model import EarthModel
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, _BandBlock, _BandLU, _degree_order, resolution_per_layer
from earthtides.iers import iers_fit

PREM = 'data/PREM.dat'


def _layer_degrees(rng, N=3, nfield=7):
    """Degrees of the rows and columns of a small layer: fields of random degrees, N per field."""
    n = rng.integers(0, 9, size=7) if nfield == 7 else np.arange(nfield) // 2
    nf = len(n)
    return n[np.arange(nf * N) % nf], np.repeat(n, N)


def _masked(rd, cd, B, rng, dtype=complex):
    A = rng.normal(size=(len(rd), len(cd))) + (1j * rng.normal(size=(len(rd), len(cd))) if dtype is complex else 0)
    A[np.abs(rd[:, None] - cd[None, :]) > B] = 0.0
    return A


@pytest.mark.parametrize('B', [0, 1, 2, 5])
def test_degree_order_and_bandwidth(B):
    rng = np.random.default_rng(B)
    rd, cd = _layer_degrees(rng)
    rp, cp, kl, ku = _degree_order(rd, cd, B)
    assert np.all(np.diff(rd[rp]) >= 0) and np.all(np.diff(cd[cp]) >= 0)
    assert sorted(rp) == list(range(len(rd))) and sorted(cp) == list(range(len(cd)))
    inb = np.abs(rd[rp][:, None] - cd[cp][None, :]) <= B
    i, j = np.nonzero(inb)
    assert kl == max(0, (i - j).max()) and ku == max(0, (j - i).max())


def test_band_block_holds_masked_block():
    rng = np.random.default_rng(1)
    rd, cd = _layer_degrees(rng)
    B, n = 2, len(rd)
    D = _masked(rd, cd, B, rng)
    full = D + (np.abs(rd[:, None] - cd[None, :]) > B) * 7.0            # out-of-band values are discarded
    Bb = _BandBlock(rd, cd, B, complex)
    for rows, cols in ((np.arange(0, 5), np.arange(n)), (np.arange(5, n), np.arange(0, 9)),
                       (np.arange(5, n), np.arange(9, n))):
        Bb.add(rows, cols, full[np.ix_(rows, cols)])
    with pytest.raises(RuntimeError):
        Bb.add(np.array([0]), np.array([0]), np.ones((1, 1)))
    Bb.finish()
    assert np.array_equal(Bb.tocsr().toarray(), D)
    assert Bb.dropped == 7.0
    x = rng.normal(size=n) + 1j
    assert np.allclose(Bb @ x, D @ x, rtol=0, atol=1e-13)
    assert np.array_equal(Bb.row_absmax(), np.abs(D).max(axis=1))
    irs = 1 / np.abs(D).max(axis=1)
    assert np.allclose(Bb.col_absmax(irs), np.abs(D * irs[:, None]).max(axis=0), rtol=1e-15)


@pytest.mark.parametrize('nfield, B', [(7, 3), (40, 2)])
def test_band_lu_with_woodbury(nfield, B):
    rng = np.random.default_rng(2)
    rd, cd = _layer_degrees(rng, nfield=nfield)
    n = len(rd)
    D = _masked(rd, cd, B, rng) + 5 * np.diag(np.abs(rd - cd) <= B)   # (the band keeps only these)
    Bb = _BandBlock(rd, cd, B, complex)
    Bb.add(np.arange(n), np.arange(n), D)
    Bb.finish()
    one = np.ones(n)
    F = _BandLU(Bb, one, one, complex)
    assert F.ok and (nfield == 7 or F.nc >= 4)                  # several chunks in the big case
    Y = rng.normal(size=(n, 3)) + 0j
    assert np.allclose(D @ F.solve(Y), Y, atol=1e-12)
    U, V = rng.normal(size=(n, 2)), rng.normal(size=(2, n))             # update: full, not banded
    F.update(U.astype(complex), V.astype(complex))
    assert np.allclose((D - U @ V) @ F.solve(Y), Y, atol=1e-11)


def _solver(m, sig, Kc, **kw):
    md = EarthModel(PREM).viscous_core(sig * OMEGA_EARTH, delta=2e3)
    g = ClairautGeometry(md)
    Kl, Nl = resolution_per_layer(md, {'K': 4, 'N': 16, 'rules': [{'fluid': True, 'K': Kc, 'N': 20}]})
    S = TidalSolver(md, g, m=m, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH,
                    omega_t=sig * OMEGA_EARTH, omega_rot=OMEGA_EARTH, wahr_frame=(m != 2), dyn_fluid=True,
                    consistent_reference=True, frame='mantle', **kw)
    S.solve()
    f = iers_fit(S, g)
    return S, np.array([complex(f[k]) for k in ('h0', 'h2', 'l2', 'l1')])


@pytest.mark.parametrize('m, sig', [(1, 0.93), (2, 1.93)])
def test_wide_band_is_full_coupling(m, sig, monkeypatch):
    _, L0 = _solver(m, sig, 8)
    monkeypatch.setattr(sv._HybridMatrix, 'BAND_STORAGE', 4.0)        # band storage everywhere it fits
    S, L1 = _solver(m, sig, 8, band=100)
    assert S._solver_used == 'block'
    assert 'band' in [l[3] for l in S._band_info['layers']]
    assert max(l[4] for l in S._band_info['layers']) == 0.0
    assert np.abs(L1 - L0).max() < 1e-11


@pytest.mark.parametrize('m, sig', [(1, 0.93), (2, 1.93)])
def test_band_storage_matches_dense_storage(m, sig, monkeypatch):
    _, Ld = _solver(m, sig, 8, band=3)                                 # band applied on dense blocks
    monkeypatch.setattr(sv._HybridMatrix, 'BAND_STORAGE', 4.0)
    S, Lb = _solver(m, sig, 8, band=3)
    assert [l[3] for l in S._band_info['layers']].count('band') >= 3
    assert S._band_info['layers'][2][4] > 0                            # something was dropped
    assert np.abs(Lb - Ld).max() < 1e-11


def test_fallback_guard_and_arguments(monkeypatch):
    with pytest.raises(ValueError):
        _solver(1, 0.93, 6, band=4, assembly='sparse')
    monkeypatch.setattr(TidalSolver, 'BAND_FALLBACK_MAX', 10)
    monkeypatch.setattr(TidalSolver, '_solve_hybrid', lambda self, H, f: (_ for _ in ()).throw(sv._NotBlock()))
    with pytest.raises(RuntimeError, match='BAND_FALLBACK_MAX'):
        _solver(1, 0.93, 6, band=4)


def test_banded_projection_equals_masked_full_assembly():
    """The assembly projects each trial field only on the test degrees within the band: the same matrix
    as the full assembly with |n - n'| > B removed (layer rows and columns), to round-off."""
    S0, _ = _solver(1, 0.93, 8)
    S1, _ = _solver(1, 0.93, 8, band=3)
    A0, A1 = S0.Hsp.tocoo(), S1.Hsp.tocsr()
    nt = S0._ntot
    rd, cd = np.full(A0.shape[0], -100), np.full(A0.shape[0], -100)
    for i, (Bi, N) in enumerate(zip(S0.B, S0.Nl)):
        n = np.array([d for _, d in Bi.fields])
        a, b = S0._off[i], S0._off[i + 1]
        rd[a:b], cd[a:b] = n[np.arange(b - a) % len(n)], np.repeat(n, N)
    r, c, v = A0.row, A0.col, A0.data.copy()
    v[(r < nt) & (c < nt) & (np.abs(rd[r] - cd[c]) > 3)] = 0.0
    M = sv.sps.csr_matrix((v, (r, c)), shape=A0.shape)
    M.eliminate_zeros()
    D = abs(M - A1)
    assert D.max() <= 1e-13 * abs(M).max()
    assert M.nnz == A1.nnz
