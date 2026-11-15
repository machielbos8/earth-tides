# -*- coding: utf-8 -*-
#
# Tidal response of an axisymmetric (ellipsoidal) elastic, self-gravitating
# Earth for a degree-nf, order-m forcing.
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
# 5/10/2026 Machiel Bos
#===============================================================================
"""Tidal response of an axisymmetric (ellipsoidal) elastic, self-gravitating
Earth for a degree-nf, order-m forcing.

Unknowns, as functions of the level-surface label s and colatitude theta
(physical spherical components at the point r = R(s, theta)), with the
lambda dependence cos(m lambda) for u_r, u_theta, phi1 and sin(m lambda) for
u_lambda (Ybar = Schmidt semi-normalised P_n^m(cos theta)):

    u_r   =  sum U_n Ybar_n
    u_th  =  sum V_n dYbar_n/dth  -  sum W_n m Ybar_n / sin th
    u_lam = -sum V_n m Ybar_n / sin th  +  sum W_n dYbar_n/dth
    phi1  =  sum P_n Ybar_n

U, V, P run over the spheroidal degrees n >= m with n - nf even; W over the
toroidal degrees with n - nf odd (absent for m = 0). Each radial function is
a Chebyshev series per layer (as in astyanax). The equations are evaluated
pointwise in physical coordinates, using first-order automatic
differentiation ("jets") for the stress divergence, and projected onto the
scalar / spheroidal / toroidal test functions by Gauss-Legendre quadrature.
No coupling coefficient is derived by hand.

Field equations (hydrostatic pre-stress, Lagrangian form, Phi0 = potential
energy with grad Phi0 = +g r_hat for a sphere):

    div tau - rho grad(u . grad Phi0) + rho (div u) grad Phi0
            - rho grad(phi1 + phiT) + rho omega^2 u  [+ alpha rho y_hat x r] = 0
    lap phi1 + 4 pi G div(rho u) = 0

Interfaces between two solids (or two fluids) are welded: u, phi1, n.tau and the gravity flux
n.grad phi1 + 4 pi G rho u.n are continuous. At a fluid-solid interface (true fluid, mu = 0) the
fluid slides: u.n, n.tau.n, phi1 and the flux are continuous, the tangential traction vanishes on
the solid side and the fluid's tangential momentum equation takes the freed rows (_fs_interface;
Tromp & Mitrovica 1999, eq. 3.7). A fluid needs inertia: its static limit is undetermined
(Longman 1963), so true-fluid models are solved at a tidal frequency; PREMs (weak-solid core) is for
static runs. Free surface n.tau = 0 (load: -g sigma n), phi1 matched to sum c_n (a/r)^(n+1) Ybar_n.
The manual explains the assembly step by step.

m = 1: a degree-2 order-1 tide torques the bulges of mantle and core;
without Coriolis there is no static balance (the real Earth nutates, and the
core's differential tilt is the FCN mode, at zero frequency without
rotation). As in Mathews et al., the deformation is computed with the
rigid rotation of each region (inner core, outer core, mantle) removed: per
region an unknown inertial force alpha_k rho (y x r) and the Tisserand
condition int_region rho r x u dV = 0.

phiT = -g a (r/a)^nf Ybar_nf cos(m lambda); on a sphere h = U_nf(a)/a,
l = V_nf(a)/a, k = -c_nf/(g a).
"""

import os
import warnings

import numpy as np
from numpy.polynomial import legendre as L
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from .basis import cheb_basis, ybar, legendre_theta, _trig_derivs, _schmidt_reduced  # noqa: F401
from .geometry import mapping, mapping_from_R, chain, SphereGeometry
from .jets import J, _D, ZERO
from .model import G


def _check_dense(ncol, dtype):
    """A dense ncol x ncol matrix peaks at ~1.7 ncol^2 itemsize with its copies (viscous PREM K6/N30:
    2.1 GB -> 3.0-3.4 GB RSS); a request beyond the physical memory can freeze a laptop.
    EARTHTIDES_MAX_GB raises the per-process limit deliberately."""
    peak = 1.7 * ncol**2 * np.dtype(dtype).itemsize / 1e9
    limit = float(os.environ.get('EARTHTIDES_MAX_GB', '8'))
    if peak > limit:
        raise MemoryError('dense matrix needs ~%.1f GB (ncol %d, %s) > EARTHTIDES_MAX_GB = %g; '
                          'mind the number of parallel workers' % (peak, ncol, np.dtype(dtype).name, limit))


def _unit(n, k):
    e = np.zeros(n)
    e[k] = 1.0
    return e


class _QField(tuple):
    """(u_r, u_theta, u_lambda, phi1) of a 'Q' trial field (all zero) with its Pi jets in .q."""


class _NotBlock(Exception):
    """The block path cannot solve the system (TidalSolver._block_path): not block-tridiagonal by
    layer, or GMRES did not converge."""


class _BlockMatrix:
    """Write-only sparse matrix for the assembly: H[rows, cols] = block, numpy indexing (int or slice
    per axis, broadcasting). Every entry is written ONCE (the assembly never overwrites; a second
    write to an entry raises in tocsr). Triplets go to CSR in scipy's compiled COO -> CSR conversion,
    in chunks; the system is ~6 per cent non-zero, so the dense ncol^2 array is never formed."""

    CHUNK = 4_000_000

    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, np.dtype(dtype)
        self._r, self._c, self._v = [], [], []
        self._pending = 0
        self._parts = []                         # CSR of the chunks converted so far

    @staticmethod
    def _index(k, n):
        if isinstance(k, slice):
            return np.arange(*k.indices(n), dtype=np.int32), True
        if isinstance(k, np.ndarray):                 # integer index array (banded projection)
            return k.astype(np.int32), True
        k = int(k)
        return np.array([k + n if k < 0 else k], np.int32), False

    def __setitem__(self, key, val):
        (rows, rs), (cols, cs) = self._index(key[0], self.shape[0]), self._index(key[1], self.shape[1])
        target = tuple(len(a) for a, s in ((rows, rs), (cols, cs)) if s)
        v = np.broadcast_to(np.asarray(val, dtype=self.dtype), target).reshape(len(rows), len(cols))
        self.add(rows, cols, v)

    def add(self, rows, cols, v):
        """Write the (len(rows), len(cols)) array v at the index arrays rows x cols."""
        self._r.append(np.repeat(rows, len(cols)))
        self._c.append(np.tile(cols, len(rows)))
        self._v.append(np.asarray(v, dtype=self.dtype).ravel())
        self._pending += v.size
        if self._pending > self.CHUNK:
            self._flush()

    def _flush(self):
        if not self._pending:
            return
        r, c, v = np.concatenate(self._r), np.concatenate(self._c), np.concatenate(self._v)
        self._r, self._c, self._v, self._pending = [], [], [], 0
        A = sps.csr_matrix((v, (r, c)), shape=self.shape, dtype=self.dtype)   # sums duplicates
        if A.nnz != len(v):
            raise RuntimeError('assembly wrote %d entries twice' % (len(v) - A.nnz))
        self._parts.append(A)

    def tocsr(self):
        self._flush()
        parts, self._parts = self._parts, []
        A = parts[0]
        if len(parts) > 1:
            # overlap between chunks, on the structure (sparse + drops exact zeros, so count ones)
            ind = [sps.csr_matrix((np.ones(B.nnz, np.int8), B.indices, B.indptr), shape=self.shape)
                   for B in parts]
            I = ind[0]
            for B in ind[1:]:
                I = I + B
            if I.nnz and I.data.max() > 1:
                raise RuntimeError('assembly wrote entries twice (across chunks)')
            del ind, I
            for B in parts[1:]:
                A = A + B
        A.eliminate_zeros()
        A.sort_indices()
        return A


class _HybridMatrix:
    """Write-only system matrix for the assembly, by layer (plan_solver_memory_2026-11-02): the
    diagonal block of every layer (rows and columns off[i]..off[i+1]) is a dense array, everything
    else (interface rows reaching into a neighbouring layer, the border rows and columns: exterior
    c_n, Tisserand rows, rigid-rotation forces) goes to a _BlockMatrix (`rest`). The core blocks are
    full (every degree couples to every degree; collocation is dense in radius), so a dense array
    stores them in 16 B per entry instead of the CSR's 12 B plus its copies. Same write semantics as
    _BlockMatrix: int or slice per axis, broadcasting, every entry written once (a second write
    raises). tocsr() builds the full CSR (SuperLU, S.Hsp, diagnostics); it equals the CSR of
    _BlockMatrix entry for entry.
    band = (B, rdeg, cdeg) (plan_banded_coupling_2026-11-02): the coupling between degrees |n - n'| > B is
    dropped, in the layer blocks and in the rows between layers; rdeg / cdeg: degree of every layer row /
    column (length off[-1]). A layer whose band is narrow (2 kl + ku + 1 < n / 2) is stored as a
    _BandBlock, the others stay dense with the band applied in finish(). band_info(): what was dropped."""

    BAND_STORAGE = 0.5       # band storage when its rows 2 kl + ku + 1 < BAND_STORAGE x block size (tests: 2)

    def __init__(self, shape, dtype, off, band=None):
        self.shape, self.dtype = shape, np.dtype(dtype)
        self.off = [int(o) for o in off]
        self.nL, self.nt = len(off) - 1, int(off[-1])
        self.band = band
        self.blocks, self._done = [], []
        for a, b in zip(off[:-1], off[1:]):
            if band is not None:
                Bb = _BandBlock(band[1][a:b], band[2][a:b], band[0], self.dtype)
                if 2 * Bb.kl + Bb.ku + 1 < self.BAND_STORAGE * (b - a):
                    self.blocks.append(Bb)
                    self._done.append(None)
                    continue
            self.blocks.append(np.zeros((b - a, b - a), self.dtype, order='F'))
            self._done.append(np.zeros((b - a, b - a), bool))
        self.rest = _BlockMatrix(shape, dtype)
        self._dropped_rest = 0.0

    def _layer(self, idx):                           # layer of every index (nL = border)
        return np.searchsorted(self.off, idx, side='right') - 1 if len(idx) else idx

    def __setitem__(self, key, val):
        n0, n1 = self.shape
        (rows, rsl), (cols, csl) = _BlockMatrix._index(key[0], n0), _BlockMatrix._index(key[1], n1)
        target = tuple(len(a) for a, s in ((rows, rsl), (cols, csl)) if s)
        v = np.broadcast_to(np.asarray(val, dtype=self.dtype), target).reshape(len(rows), len(cols))
        lr, lc = self._layer(rows), self._layer(cols)
        for L in np.unique(lr):
            ri = np.flatnonzero(lr == L)
            inb = (lc == L) if L < self.nL else np.zeros(len(cols), bool)
            if inb.any():
                ci = np.flatnonzero(inb)
                rr, cc = rows[ri] - self.off[L], cols[ci] - self.off[L]
                if isinstance(self.blocks[L], _BandBlock):
                    self.blocks[L].add(rr, cc, v[np.ix_(ri, ci)])
                elif _contiguous(rr) and _contiguous(cc):          # the common case: one rectangle
                    s0, s1 = slice(rr[0], rr[-1] + 1), slice(cc[0], cc[-1] + 1)
                    if self._done[L][s0, s1].any():
                        raise RuntimeError('assembly wrote an entry of layer block %d twice' % L)
                    self._done[L][s0, s1] = True
                    self.blocks[L][s0, s1] = v[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
                else:
                    ix = np.ix_(rr, cc)
                    if self._done[L][ix].any():
                        raise RuntimeError('assembly wrote an entry of layer block %d twice' % L)
                    self._done[L][ix] = True
                    self.blocks[L][ix] = v[np.ix_(ri, ci)]
            if (~inb).any():
                co = np.flatnonzero(~inb)
                vr = v[np.ix_(ri, co)]
                if self.band is not None and L < self.nL:      # rows between layers: banded too
                    c = cols[co]
                    lay = c < self.nt
                    if lay.any():
                        out = np.zeros(vr.shape, bool)
                        out[:, lay] = np.abs(self.band[1][rows[ri]][:, None]
                                             - self.band[2][c[lay]][None, :]) > self.band[0]
                        if out.any():
                            self._dropped_rest = max(self._dropped_rest, float(np.abs(vr[out]).max()))
                            vr = np.where(out, 0, vr)
                self.rest.add(rows[ri], cols[co], vr)

    def finish(self):
        """End of the assembly: drop the write masks; the band on the dense blocks; rest to CSR."""
        self._done = None
        self._dropped = [0.0] * self.nL
        for i, B in enumerate(self.blocks):
            if isinstance(B, _BandBlock):
                B.finish()
                self._dropped[i] = B.dropped / max(B.kept, 1e-300)
            elif self.band is not None:
                a, b = self.off[i], self.off[i + 1]
                out = np.abs(self.band[1][a:b, None] - self.band[2][None, a:b]) > self.band[0]
                if out.any():
                    A = np.abs(B)
                    self._dropped[i] = float(A[out].max()) / max(float(A.max()), 1e-300)
                    B[out] = 0.0
        self.rest = self.rest.tocsr()
        return self

    def band_info(self):
        """B; per layer (size, kl, ku, 'band' | 'dense', largest dropped |entry| / largest kept);
        the largest dropped |entry| of the rows between layers (unscaled)."""
        if self.band is None:
            return None
        lay = []
        for i, B in enumerate(self.blocks):
            n = self.off[i + 1] - self.off[i]
            if isinstance(B, _BandBlock):
                lay.append((n, B.kl, B.ku, 'band', self._dropped[i]))
            else:
                lay.append((n, None, None, 'dense', self._dropped[i]))
        return dict(B=self.band[0], layers=lay, dropped_between_layers=self._dropped_rest)

    def tocsr(self):
        """The full matrix as CSR (as _BlockMatrix.tocsr: explicit zeros dropped, sorted indices)."""
        rest = self.rest if sps.issparse(self.rest) else self.rest.tocsr()
        nB = self.shape[0] - self.nt
        D = sps.block_diag([B.tocsr() if isinstance(B, _BandBlock) else sps.csr_matrix(B) for B in self.blocks]
                           + ([sps.csr_matrix((nB, nB), dtype=self.dtype)] if nB else []), format='csr')
        A = (D + rest).astype(self.dtype)
        A.eliminate_zeros()
        A.sort_indices()
        return A


def _contiguous(a):
    return len(a) > 0 and a[-1] - a[0] == len(a) - 1 and (len(a) < 3 or np.all(np.diff(a) == 1))


def _degree_order(rdeg, cdeg, B):
    """Degree order of a layer block for the band (plan_banded_coupling_2026-11-02): rows and columns
    sorted by degree (stable: field order, then row kind / Chebyshev index within a degree), and the
    lower / upper bandwidths of the entries with |n_row - n_col| <= B in that order. Every degree has
    as many rows as columns (N per field), so the diagonal runs through the degree groups."""
    rperm = np.argsort(rdeg, kind='stable')
    cperm = np.argsort(cdeg, kind='stable')
    rs, cs = rdeg[rperm], cdeg[cperm]
    kl = ku = 0
    for d in np.unique(rs):
        r1 = np.searchsorted(rs, d, 'right') - 1                    # last row of degree d
        c0 = np.searchsorted(cs, d - B, 'left')                      # first column within the band
        r0 = np.searchsorted(rs, d, 'left')
        c1 = np.searchsorted(cs, d + B, 'right') - 1
        kl, ku = max(kl, r1 - c0), max(ku, c1 - r0)
    return rperm, cperm, int(kl), int(ku)


class _BandBlock:
    """Diagonal block of a layer restricted to the angular band |n_row - n_col| <= B, stored as a LAPACK
    band in degree order (plan_banded_coupling_2026-11-02): ab[ku + i - j, j] = A[i, j] for the permuted
    indices i, j. Writes in the layer's own (field-major) order; entries outside the band are discarded
    (the largest discarded and kept |entry| are recorded); every kept entry is written once."""

    def __init__(self, rdeg, cdeg, B, dtype):
        self.n = len(rdeg)
        self.B, self.dtype = B, np.dtype(dtype)
        self.rdeg, self.cdeg = np.asarray(rdeg), np.asarray(cdeg)
        self.rperm, self.cperm, self.kl, self.ku = _degree_order(self.rdeg, self.cdeg, B)
        self.rpos, self.cpos = np.argsort(self.rperm), np.argsort(self.cperm)
        self.ab = np.zeros((self.kl + self.ku + 1, self.n), self.dtype)
        self._done = np.zeros(self.ab.shape, bool)
        self.kept = self.dropped = 0.0
        self._dia = None

    @property
    def shape(self):
        return (self.n, self.n)

    def add(self, rr, cc, v):
        """Write the (len(rr), len(cc)) values v at local rows rr and columns cc."""
        keep = np.abs(self.rdeg[rr][:, None] - self.cdeg[cc][None, :]) <= self.B
        a = np.abs(v)
        if (~keep).any():
            self.dropped = max(self.dropped, float(a[~keep].max()))
        if not keep.any():
            return
        self.kept = max(self.kept, float(a[keep].max()))
        pr, pc = self.rpos[rr], self.cpos[cc]
        t = (self.ku + pr[:, None] - pc[None, :])[keep]
        j = np.broadcast_to(pc[None, :], keep.shape)[keep]
        if self._done[t, j].any():
            raise RuntimeError('assembly wrote an entry of a band block twice')
        self._done[t, j] = True
        self.ab[t, j] = v[keep]

    def finish(self):
        self._done = None

    def _diagonals(self):                            # (t, permuted columns j, permuted rows i)
        n, ku = self.n, self.ku
        for t in range(self.ab.shape[0]):
            j0, j1 = max(0, ku - t), min(n, n + ku - t)
            if j1 > j0:
                yield t, slice(j0, j1), slice(j0 + t - ku, j1 + t - ku)

    def row_absmax(self):
        """max_j |A[i, j]| per local row (layer order)."""
        m = np.zeros(self.n)
        for t, js, is_ in self._diagonals():
            np.maximum(m[is_], np.abs(self.ab[t, js]), out=m[is_])
        return m[self.rpos]

    def col_absmax(self, irs):
        """max_i |A[i, j] irs[i]| per local column (layer order); irs in layer order."""
        w = irs[self.rperm]
        m = np.zeros(self.n)
        for t, js, is_ in self._diagonals():
            np.maximum(m[js], np.abs(self.ab[t, js] * w[is_]), out=m[js])
        return m[self.cpos]

    def dense(self, i0, i1, j0, j1, wr=None, wc=None, dt=None):
        """Dense A[i0:i1, j0:j1] in the PERMUTED (degree) order, times wr[i] wc[j] (permuted) if given."""
        I = np.arange(i0, i1)[:, None]
        J = np.arange(j0, j1)[None, :]
        T = self.ku + I - J
        ok = (T >= 0) & (T < self.ab.shape[0])
        out = np.zeros((i1 - i0, j1 - j0), self.dtype if dt is None else dt, order='F')
        JJ = np.broadcast_to(J, T.shape)
        out[ok] = self.ab[T[ok], JJ[ok]]
        if wr is not None:
            out *= wr[i0:i1, None]
            out *= wc[None, j0:j1]
        return out

    def __matmul__(self, x):
        """A x for a local vector x (layer order)."""
        if self._dia is None:
            self._dia = sps.dia_matrix((self.ab, self.ku - np.arange(self.ab.shape[0])), shape=self.shape)
        y = self._dia @ x[self.cperm]
        return y[self.rpos]

    def tocsr(self):
        """The block as CSR in layer order (explicit zeros dropped)."""
        D = sps.dia_matrix((self.ab, self.ku - np.arange(self.ab.shape[0])), shape=self.shape).tocoo()
        A = sps.csr_matrix((D.data, (self.rperm[D.row], self.cperm[D.col])), shape=self.shape)
        A.eliminate_zeros()
        return A


class _BandLU:
    """LU of a scaled band block, solves in layer order, with an optional rank-|q| update S = A - U V
    applied by the Sherman-Morrison-Woodbury formula: the Schur complement of the coupling to the layer
    eliminated before (not banded, so not stored in the band). Blocked band LU with partial pivoting
    (as LAPACK's blocked gbtrf): the band is cut into chunks of at least max(kl, ku) unknowns (degree
    order); step k factors the tall panel of chunk k's columns over the rows still active (the rest of
    chunk k after the previous step, and chunk k + 1) with getrf, so rows may move across a chunk
    boundary, and updates the next two chunks' columns (pivoting fills one chunk to the right). All
    operations dense level 3 (getrf, trsm, gemm), also in the solves with many right-hand sides; gbtrs
    (first version, 2026-11-02) solved those column by column: 17 of 38 s at core K 48. Block elimination
    without pivoting across chunks failed (exactly singular Schur complement of a chunk)."""

    def __init__(self, blk, irs, ics, dt, shift=None):
        n, w = blk.n, max(blk.kl, blk.ku, 1)
        nc = max(1, n // w)
        e = np.linspace(0, n, nc + 1).round().astype(int)           # chunk edges, each >= w unknowns
        wr, wc = irs[blk.rperm], ics[blk.cperm]
        self.edges, self.rperm, self.cpos, self.nc = e, blk.rperm, blk.cpos, nc
        self.steps = []
        self.ok = True

        def cols(k):                                 # columns of chunks k .. k + 2
            return e[k], e[min(k + 3, nc)]

        def rows_of(k, c0, c1):                      # original rows of chunk k on columns c0 .. c1
            M = blk.dense(e[k], e[k + 1], c0, c1, wr, wc, dt)
            if shift is not None:
                for i in range(e[k], e[k + 1]):
                    if c0 <= i < c1:
                        M[i - e[k], i - c0] += shift
            return M

        c0, c1 = cols(0)
        carry = rows_of(0, c0, c1)                   # active rows, columns from chunk k on
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', sla.LinAlgWarning)
            for k in range(nc):
                c0, c1 = cols(k)
                ck = e[k + 1] - e[k]
                if k < nc - 1:
                    nxt = rows_of(k + 1, c0, c1)
                    if carry.shape[1] < c1 - c0:     # the carry has no entries in the newest chunk
                        carry = np.hstack([carry, np.zeros((carry.shape[0], c1 - c0 - carry.shape[1]), carry.dtype)])
                    M = np.vstack([carry, nxt])
                else:
                    M = carry
                lu, piv = sla.lu_factor(M[:, :ck], overwrite_a=False, check_finite=False)
                perm = np.arange(M.shape[0])
                for a_, b_ in enumerate(piv):        # LAPACK swaps -> one permutation
                    perm[a_], perm[b_] = perm[b_], perm[a_]
                M = M[perm]
                d = np.abs(np.diag(lu))                # lu: L11 \ U11 packed on top, L21 below
                self.ok &= bool(np.all(np.isfinite(d)) and d.min() > 1e-13 * max(d.max(), 1e-300))
                U12 = sla.solve_triangular(lu[:ck], M[:ck, ck:], lower=True, unit_diagonal=True,
                                           check_finite=False) if M.shape[1] > ck else None
                if k < nc - 1:
                    carry = M[ck:, ck:] - lu[ck:] @ U12
                del M
                self.steps.append((perm, lu, U12))
        self.wood = None

    def solve_A(self, F):
        e, nc = self.edges, self.nc
        G = F[self.rperm]
        one = G.ndim == 1
        if one:
            G = G[:, None]
        dt = np.result_type(G.dtype, self.steps[0][1].dtype)
        Y = []
        carry = G[e[0]:e[1]].astype(dt)
        for k in range(nc):                          # forward: P, unit L11, L21
            perm, lu, _ = self.steps[k]
            ck = e[k + 1] - e[k]
            v = np.vstack([carry, G[e[k + 1]:e[k + 2]]]) if k < nc - 1 else carry
            v = v[perm]
            y = sla.solve_triangular(lu[:ck], v[:ck], lower=True, unit_diagonal=True, check_finite=False)
            Y.append(y)
            if k < nc - 1:
                carry = v[ck:] - lu[ck:] @ y
        X = [None] * nc
        for k in range(nc - 1, -1, -1):              # backward: U11 x_k = y_k - U12 [x_k+1; x_k+2]
            _, lu, U12 = self.steps[k]
            r = Y[k]
            if U12 is not None:
                r = r - U12 @ np.vstack(X[k + 1:min(k + 3, nc)])
            X[k] = sla.solve_triangular(lu[:e[k + 1] - e[k]], r, lower=False, check_finite=False)
        x = np.vstack(X)[self.cpos]
        return x[:, 0] if one else x

    def update(self, U, V):
        """S = A - U V from now on: W = A^-1 U, capacitance I - V W."""
        W = self.solve_A(U)
        C = np.eye(U.shape[1], dtype=W.dtype) - V @ W
        self.wood = (W, V, sla.lu_factor(C, check_finite=False))

    def solve(self, F):
        X = self.solve_A(F)
        if self.wood is not None:
            W, V, luC = self.wood
            X = X + W @ sla.lu_solve(luC, V @ X, check_finite=False)
        return X


class _ScaledHybrid:
    """The row- and column-scaled hybrid system, diag(1/rs) H diag(1/cs), for GMRES: the mat-vec
    uses the unscaled blocks (kept for diagnostics), (H (x / cs)) / rs; equal to the scaled matrix
    times x up to round-off."""

    def __init__(self, Hh, rs, cs):
        self.Hh, self.rs, self.cs = Hh, rs, cs
        self.dtype = Hh.dtype
        self.shape = Hh.shape

    def __matmul__(self, x):
        z = x / self.cs
        y = self.Hh.rest @ z
        y = y.astype(np.result_type(y.dtype, self.dtype, z.dtype), copy=False)
        for B, a, b in zip(self.Hh.blocks, self.Hh.off[:-1], self.Hh.off[1:]):
            y[a:b] += B @ z[a:b]
        return y / self.rs


class BlockCache:
    """Factors of the last block-LU factorization, carried from one frequency to the next of a
    sweep: TidalSolver(..., factor_cache=cache) with the same cache for neighbouring frequencies
    (same model layering and resolution). Each solve first tries GMRES preconditioned with the
    cached factors (at most max_reuse_iter iterations), and factors anew only when that fails."""

    def __init__(self, max_reuse_iter=12):
        self.F = self.rs = self.cs = self.key = self.x = None
        self.max_reuse_iter = max_reuse_iter
        self.factored = self.reused = 0


class _BlockLU:
    """Block LU of the scaled system by layers: the rows and columns of layer i form the dense
    block A_ii; a layer couples to its neighbours only through its bottom and top rows; the border
    (exterior c_n, rigid-rotation forces) is eliminated with a Schur complement. Pivoting inside
    each block only (LAPACK getrf): an approximate inverse (residual ~1e-7..1e-6), used as the
    preconditioner of _gmres. A block that is singular on its own (a static fluid or weak-solid
    layer; a free rigid rotation) is factored with a diagonal shift of SHIFT of its largest entry.
    get_block (hybrid assembly): A holds only the entries outside the diagonal blocks, and
    get_block(i) returns a fresh scaled copy of block i (Fortran order), factored in place; a shift
    retry asks for the block again instead of keeping a copy. A band layer (TidalSolver(band=B)):
    get_block(i) returns (_BandBlock, row / column scales, dtype); the layer is factored as a band
    (_BandLU, block tridiagonal in chunks of the band) without
    the Schur update of the layer eliminated before it, and that update (rank |q|, not banded) is applied
    with the Woodbury formula (plan_banded_coupling_2026-11-02)."""

    SHIFT = 1e-8

    ORDER = 'down'           # better pivots everywhere; fixes static m = 0

    def __init__(self, A, off, nL, nt, dt, order=None, get_block=None):
        n = A.shape[0]
        nB = n - nt
        R = [slice(off[i], off[i + 1]) for i in range(nL)]
        self.R, self.nL, self.nt, self.nB, self.n, self.dt = R, nL, nt, nB, n, dt
        self.shifted = False
        # order of elimination: 'up' from the centre outwards, 'down' from the surface inwards (the
        # potential's value is fixed at the surface: going inwards every layer inherits it through
        # its top interface, so the inner core's degree-0 potential is no longer a free mode)
        order = self.ORDER if order is None else order
        if order not in ('up', 'down'):
            raise ValueError(order)
        self.order = order
        # one pass over the CSR rows of each layer: diagonal block, couplings to the layers below /
        # above (few non-zero rows), border columns; anything else is not block-tridiagonal
        lcol = np.full(n, nL, np.int64)              # layer of every column (nL = border)
        for i in range(nL):
            lcol[R[i]] = i
        indptr, ind, val = A.indptr, A.indices, A.data
        diag, low, up = [], [None] * nL, [None] * nL
        BcL = np.zeros((nt, nB), dt) if nB else None

        def rows_dense(rr, cc, vv, ncols):           # (non-zero local rows, dense rows x ncols)
            u = np.unique(rr)
            M = np.zeros((len(u), ncols), dt)
            M[np.searchsorted(u, rr), cc] = vv
            return u, M

        for i in range(nL):
            a0, a1 = indptr[off[i]], indptr[off[i + 1]]
            rr = np.repeat(np.arange(off[i + 1] - off[i]), np.diff(indptr[off[i]:off[i + 1] + 1]))
            cc, vv = ind[a0:a1], val[a0:a1]
            lc = lcol[cc]
            d = lc - i
            border = lc == nL
            if np.any(~border & (np.abs(d) > 1)):
                raise _NotBlock
            k = d == 0
            if get_block is None:
                S = np.zeros((off[i + 1] - off[i],) * 2, dt, order='F')     # column-major: getrf in place
                S[rr[k], cc[k] - off[i]] = vv[k]
                diag.append(S)
            elif np.any(k):
                raise RuntimeError('hybrid matrix: entries of layer block %d outside the block' % i)
            k = d == -1
            low[i] = rows_dense(rr[k], cc[k] - off[i - 1], vv[k], off[i] - off[i - 1]) if i else None
            k = (d == 1) & ~border
            if i < nL - 1:
                up[i] = rows_dense(rr[k], cc[k] - off[i + 1], vv[k], off[i + 2] - off[i + 1])
            if nB and np.any(border):
                BcL[off[i] + rr[border], cc[border] - nt] = vv[border]
        del lcol
        self.low, self.up = low, up
        # elimination sequence; for layer i: p = the layer eliminated before it, its rows coupled
        # to p (to_prev) and p's rows coupled to i (from_prev); 'nxt' = the layer after it
        if order == 'up':
            seq = list(range(nL))
            self.prev = {i: i - 1 for i in seq[1:]}
            self.to_prev = {i: low[i] for i in seq[1:]}
            self.from_prev = {i: up[i - 1] for i in seq[1:]}
            self.to_next = {i: up[i] for i in seq[:-1]}
        else:
            seq = list(range(nL - 1, -1, -1))
            self.prev = {i: i + 1 for i in seq[1:]}
            self.to_prev = {i: up[i] for i in seq[1:]}
            self.from_prev = {i: low[i + 1] for i in seq[1:]}
            self.to_next = {i: low[i] for i in seq[:-1]}
        self.seq = seq
        # factorization: layer blocks (Schur complements of the coupling to the previous layer)
        lus = [None] * nL
        self.lus = lus

        def coupling(i):                             # (rows r of i, L_r S_p^-1 on columns q, U_q) or None
            if i in self.prev:
                p = self.prev[i]
                q, Uq = self.from_prev[i]            # rows of p coupled to layer i
                r, Lr = self.to_prev[i]              # rows of layer i coupled to p
                if len(r) and len(q):
                    E = np.zeros((off[p + 1] - off[p], len(q)), dt)
                    E[q, np.arange(len(q))] = 1.0
                    Z = self._lsolve(p, E)           # columns q of S_p^-1
                    return r, Lr @ Z, Uq
            return None

        def schur(i, S):
            c = coupling(i)
            if c is not None:
                r, LZ, Uq = c
                S[r] -= LZ @ Uq
            return S

        for i in seq:
            if get_block is None:
                S = diag[i]
                diag[i] = None
                lus[i] = self._factor(schur(i, S))
            else:
                blk = get_block(i)
                if isinstance(blk, tuple):           # band layer: band LU + Woodbury (not banded)
                    lus[i] = self._factor_band(blk)
                    c = coupling(i)
                    if c is not None:
                        r, LZ, Uq = c
                        U = np.zeros((off[i + 1] - off[i], LZ.shape[1]), dt)
                        U[r] = LZ
                        lus[i].update(U, Uq)
                else:
                    lus[i] = self._factor(schur(i, blk), rebuild=lambda i=i: schur(i, get_block(i)))
        # border: Schur complement of the exterior coefficients / rigid-rotation forces
        if nB:
            self.C = A[nt:, :nt]
            self.XB = self._layers(BcL)
            SB = A[nt:, nt:].toarray() - self.C @ self.XB
            # equilibrate rows, then columns: with rotation regions and a weak-solid core the
            # entries span ~30 decades (a rigid-rotation force on the weak core moves it ~1e18 times
            # more than the potential coefficients), singular-looking to a pivot test but
            # condition ~300 once scaled (static PREMs m = 1, 2026-10-18)
            self.rB = np.abs(SB).max(axis=1)
            self.rB[self.rB == 0] = 1.0
            SB = SB / self.rB[:, None]
            self.cB = np.abs(SB).max(axis=0)
            self.cB[self.cB == 0] = 1.0
            self.luB = self._factor(np.asfortranarray((SB / self.cB).astype(dt)))

    def _lsolve(self, i, F):                         # S_i^-1 F
        lu = self.lus[i]
        if isinstance(lu, _BandLU):
            return lu.solve(F)
        return sla.lu_solve(lu, F, check_finite=False)

    def _factor_band(self, blk, rebuild=None):
        """Band layer (plan_banded_coupling_2026-11-02): blk = (_BandBlock, irs, ics, dtype), factored
        from the unscaled band; a singular one again with the shift (of the scaled block's largest entry,
        which is 1 after the row and column scaling)."""
        Bb, irs, ics, dt = blk
        F = _BandLU(Bb, irs, ics, dt)
        if not F.ok:
            F = _BandLU(Bb, irs, ics, dt, shift=self.SHIFT)
            self.shifted = True
        return F

    def _factor(self, S, rebuild=None):
        """getrf; a singular block (pivot below 1e-13 of the largest) again with a diagonal shift.
        rebuild(): the block again (instead of a copy kept for the retry)."""
        S0 = S.copy(order='F') if rebuild is None else None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', sla.LinAlgWarning)
            lu = sla.lu_factor(S, overwrite_a=True, check_finite=False)
        d = np.abs(np.diag(lu[0]))
        if d.size and (not np.all(np.isfinite(d)) or d.min() <= 1e-13 * d.max()):
            if S0 is None:
                S0 = rebuild()
            S0[np.diag_indices(S0.shape[0])] += self.SHIFT * max(np.abs(S0).max(), 1e-300)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sla.LinAlgWarning)
                lu = sla.lu_factor(S0, overwrite_a=True, check_finite=False)
            self.shifted = True
        return lu

    def _layers(self, F):                            # T^-1 F for the layer part (F: nt x k)
        R, lus, nL, seq = self.R, self.lus, self.nL, self.seq
        Y = [None] * nL
        for i in seq:                                # forward, in elimination order
            Yi = F[R[i]].copy()
            if i in self.prev and len(self.to_prev[i][0]):
                r, Lr = self.to_prev[i]
                Yi[r] -= Lr @ self._lsolve(self.prev[i], Y[self.prev[i]])
            Y[i] = Yi
        X = [None] * nL
        for k in range(nL - 1, -1, -1):              # backward
            i = seq[k]
            Yi = Y[i]
            if k < nL - 1:
                q, Uq = self.to_next[i]
                if len(q):
                    Yi[q] -= Uq @ X[seq[k + 1]]
            X[i] = self._lsolve(i, Yi)
        return np.vstack(X)

    def apply(self, rhs):
        """Approximate A^-1 rhs."""
        nt, nB = self.nt, self.nB
        dt = np.result_type(self.dt, rhs.dtype)
        xf = self._layers(rhs[:nt, None].astype(dt))[:, 0]
        y = np.empty(self.n, dt)
        if nB:
            yB = sla.lu_solve(self.luB, (rhs[nt:] - self.C @ xf) / self.rB, check_finite=False) / self.cB
            y[:nt] = xf - self.XB @ yB
            y[nt:] = yB
        else:
            y[:] = xf
        return y


class _LayerLU:
    """Direct LU of the scaled system by layers, with partial pivoting ACROSS the interfaces (direct,
    no preconditioner). The matrix is block tridiagonal by layer plus a
    border (exterior c_n, rigid-rotation forces). The layers are eliminated in sequence (surface inwards by
    default, as _BlockLU). Step for layer i: the panel holds every row that still has entries in the
    columns of layer i: the rows of layer i not used before, the rows left over from the previous step, and
    the rows of the NEXT layer that reach into layer i (its interface rows); getrf takes the pivots from all
    of them, so an interface row of the neighbour can be a pivot (_BlockLU pivots inside a layer only, which
    makes it an approximate inverse that needs GMRES). The rows not used as pivots, as many as were taken
    from the next layer, go on to the next step with their fill in that layer's columns. The border rows
    reach into every layer; they are updated at every step and take part in the pivoting of the last
    step, which is one square block (last layer + border). This is LAPACK-style band LU with the layers as
    the band: no iteration, no fallback. Dense panels: memory sum_i n_i (n_i + n_next) entries.
    A: scaled CSR; a row that reaches beyond its neighbouring layer raises _NotBlock."""

    def __init__(self, A, off, nL, nt, dt, order='down', diag_rows=None):
        """A: the scaled CSR; or, with diag_rows, only its part outside the layer blocks (the `rest` of the
        hybrid matrix) and diag_rows(i, rows) = the scaled rows `rows` (local) of the diagonal block of layer
        i, dense. Then the CSR of the whole system is never formed: a block is read once, straight into the
        panel."""
        n = A.shape[0]
        nB = n - nt
        if not sps.isspmatrix_csr(A):
            A = sps.csr_matrix(A)
        indptr, ind = A.indptr, A.indices

        def rows_of(ids_):
            """(panel row, column, value) of the entries of the rows ids_ in A."""
            sub = A[ids_]
            return np.repeat(np.arange(len(ids_)), np.diff(sub.indptr)), sub.indices, sub.data
        seq = list(range(nL - 1, -1, -1)) if order == 'down' else list(range(nL))
        lay = np.searchsorted(off, np.arange(n), side='right') - 1     # layer of every index (nL = border)
        for i in range(nL):                          # block tridiagonal by layer, plus the border
            cl = lay[ind[indptr[off[i]]:indptr[off[i + 1]]]]
            if np.any((np.abs(cl - i) > 1) & (cl != nL)):
                raise _NotBlock
        del cl
        cb0 = nt
        state = np.zeros(n, np.int8)                 # 0 untouched, 1 carried, 2 pivot row
        state[nt:] = 1
        self.n, self.nt, self.nB, self.dt = n, nt, nB, dt
        self.cb = np.arange(nt, n)
        self.steps = []
        carry_ids = np.zeros(0, np.int64)            # left-over layer rows, values on [C_i | border]
        carry = np.zeros((0, 0), dt)
        Bd = None                                    # border rows: current values on [C_i | border]
        growth = 0.0

        def absmax(X):                               # without a temporary of the size of X
            return max((float(np.abs(X[:, j:j + 256]).max()) for j in range(0, X.shape[1], 256)), default=0.0) \
                if X.size else 0.0

        for k, i in enumerate(seq):
            c0, c1 = off[i], off[i + 1]
            ni = c1 - c0
            last = k == nL - 1
            nx = None if last else seq[k + 1]
            n0, n1 = (off[nx], off[nx + 1]) if not last else (0, 0)
            nn = n1 - n0
            own = np.arange(c0, c1)
            own = own[state[own] == 0]
            if last:
                pulled = np.zeros(0, np.int64)
            else:                                    # rows of the next layer that reach into this one
                a0, a1 = indptr[n0], indptr[n1]
                seg = ind[a0:a1]
                hit = (seg >= c0) & (seg < c1)
                rr = np.repeat(np.arange(n0, n1), np.diff(indptr[n0:n1 + 1]))[hit]
                pulled = np.unique(rr)
                pulled = pulled[state[pulled] == 0]
                del seg, hit, rr
                if len(pulled):                      # they may not reach beyond their own layer as well
                    sub = A[pulled]
                    cl = lay[sub.indices]
                    if np.any((cl != i) & (cl != nx) & (cl != nL)):
                        raise _NotBlock
                    del sub, cl
            fresh = np.concatenate([own, pulled])
            m_c = len(carry_ids)
            ids = np.concatenate([carry_ids, fresh])
            m = len(ids)
            # the panel: P = pivot columns (dense, factored in place); border columns Qb (dense, few);
            # next-layer columns only for the rows that have entries there (interface rows): rows Rn, values Qn
            if last:                                 # the last panel is square with the border: one array
                M = np.zeros((m + nB, ni + nB), dt, order='F')
                P = M[:m, :ni]
            else:
                P = np.zeros((m, ni), dt, order='F')
            Qb = np.zeros((m, nB), dt)
            rr, cc, vv = rows_of(ids)
            q = (cc >= n0) & (cc < n1) if nn else np.zeros(len(cc), bool)
            ppan = m_c + len(own) + np.arange(len(pulled))       # panel rows of the pulled rows
            Rn = np.unique(np.concatenate([rr[q], ppan])) if diag_rows is not None else np.unique(rr[q])
            Qn = np.zeros((len(Rn), nn), dt)
            Qn[np.searchsorted(Rn, rr[q]), cc[q] - n0] = vv[q]
            if diag_rows is not None:                # the layer blocks: own rows into P, pulled rows into Qn
                for a_ in range(0, len(own), 1024):
                    P[m_c + a_:m_c + min(a_ + 1024, len(own))] = diag_rows(i, own[a_:a_ + 1024] - c0)
                if len(pulled):
                    Qn[np.searchsorted(Rn, ppan)] = diag_rows(nx, pulled - n0)
            q = (cc >= c0) & (cc < c1) & (rr >= m_c)             # carried rows: current values instead
            P[rr[q], cc[q] - c0] = vv[q]
            q = (cc >= cb0) & (rr >= m_c)
            Qb[rr[q], cc[q] - cb0] = vv[q]
            del rr, cc, vv, q
            if m_c:
                P[:m_c] = carry[:, :ni]
                Qb[:m_c] = carry[:, ni:]
            del carry
            # border rows on the same columns
            BP = np.zeros((nB, ni), dt)
            BQn = np.zeros((nB, nn), dt)
            BQb = np.zeros((nB, nB), dt)
            if nB:
                rr, cc, vv = rows_of(self.cb)
                if nn:
                    q = (cc >= n0) & (cc < n1)
                    BQn[rr[q], cc[q] - n0] = vv[q]
                if Bd is None:
                    q = (cc >= c0) & (cc < c1)
                    BP[rr[q], cc[q] - c0] = vv[q]
                    q = cc >= cb0
                    BQb[rr[q], cc[q] - cb0] = vv[q]
                else:
                    BP[:] = Bd[:, :ni]
                    BQb[:] = Bd[:, ni:]
                del rr, cc, vv
            state[fresh] = 1
            if last:
                if m != ni:
                    raise _NotBlock
                M[:m, ni:] = Qb
                M[m:, :ni] = BP
                M[m:, ni:] = BQb
                del P, Qb
                ids = np.concatenate([ids, self.cb])
                amax = absmax(M)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', sla.LinAlgWarning)
                    lu = sla.lu_factor(M, overwrite_a=True, check_finite=False)
                d = np.abs(np.diag(lu[0]))
                if not np.all(np.isfinite(d)) or d.min() == 0.0:
                    raise _NotBlock
                growth = max(growth, absmax(lu[0]) / max(amax, 1e-300))
                self.final = (ids, lu, np.concatenate([np.arange(c0, c1), self.cb]))
                break
            if m < ni:
                raise _NotBlock
            amax = max(absmax(P), absmax(Qb), absmax(Qn))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sla.LinAlgWarning)
                lu, piv = sla.lu_factor(P, overwrite_a=True, check_finite=False)
            del P
            d = np.abs(np.diag(lu))
            if not np.all(np.isfinite(d)) or d.min() == 0.0:
                raise _NotBlock
            at = np.arange(m)                        # at[p] = panel row now at position p (LAPACK's swaps)
            for a_, b_ in enumerate(piv):
                if a_ != b_:
                    at[a_], at[b_] = at[b_], at[a_]
            pos = np.empty(m, np.int64)
            pos[at] = np.arange(m)                   # position of every panel row
            ids = ids[at]
            Qb = Qb[at]
            pn = pos[Rn]                             # positions of the rows with next-layer entries
            top = pn < ni
            pt, Qt = pn[top], Qn[top]                # pivot rows among them: U12 = L11^-1 E_pt Qt
            # W = L11^-1 E_pt (ni x |pt|): the next-layer part of U12 is W Qt, never formed
            E = np.zeros((ni, len(pt)), dt, order='F')
            E[pt, np.arange(len(pt))] = 1.0
            W = sla.solve_triangular(lu[:ni], E, lower=True, unit_diagonal=True, overwrite_b=True,
                                     check_finite=False)
            U12b = sla.solve_triangular(lu[:ni], Qb[:ni], lower=True, unit_diagonal=True, check_finite=False)
            L21 = lu[ni:]
            carry = np.zeros((m - ni, nn + nB), dt)
            carry[pn[~top] - ni, :nn] = Qn[~top]
            carry[:, :nn] -= (L21 @ W) @ Qt
            carry[:, nn:] = Qb[ni:] - L21 @ U12b
            del Qb, Qn
            growth = max(growth, max(absmax(U12b), absmax(carry)) / max(amax, 1e-300))
            state[ids[:ni]] = 2
            carry_ids = ids[ni:]
            if nB:                                   # border rows: L_b = B[:, C_i] U11^-1, then their update
                Lb = sla.solve_triangular(lu[:ni], BP.T, trans='T', lower=False, check_finite=False).T
                Bd = np.hstack([BQn - (Lb @ W) @ Qt, BQb - Lb @ U12b])
            else:
                Lb = np.zeros((0, ni), dt)
            del W, E
            self.steps.append((ids, ni, lu, pt, Qt, U12b, Lb, (c0, c1), (n0, n1)))
        self.growth = growth

    def solve(self, rhs):
        dt = np.result_type(self.dt, rhs.dtype)
        g = rhs.astype(dt).copy()
        cb = self.cb
        Y = []
        for ids, ni, lu, pt, Qt, U12b, Lb, ci, cn in self.steps:
            v = g[ids]
            y = sla.solve_triangular(lu[:ni], v[:ni], lower=True, unit_diagonal=True, check_finite=False)
            g[ids[ni:]] = v[ni:] - lu[ni:] @ y
            if self.nB:
                g[cb] -= Lb @ y
            Y.append(y)
        x = np.zeros(self.n, dt)
        ids, lu, cols = self.final
        x[cols] = sla.lu_solve(lu, g[ids], check_finite=False)
        for (ids, ni, lu, pt, Qt, U12b, Lb, ci, cn), y in zip(self.steps[::-1], Y[::-1]):
            u = np.zeros(ni, dt)                     # U12 [x_next; x_border] = L11^-1 E_pt (Qt x_next) + U12b x_b
            u[pt] = Qt @ x[cn[0]:cn[1]]
            u = sla.solve_triangular(lu[:ni], u, lower=True, unit_diagonal=True, check_finite=False)
            x[ci[0]:ci[1]] = sla.solve_triangular(lu[:ni], y - u - U12b @ x[cb], lower=False, check_finite=False)
        return x

    def stored_gb(self):
        """Memory kept by the factors (GB) and, per step, the share of pivot rows that reach into the next layer."""
        b = self.final[1][0].nbytes
        kept = []
        for ids, ni, lu, pt, Qt, U12b, Lb, ci, cn in self.steps:
            b += lu.nbytes + Qt.nbytes + U12b.nbytes + Lb.nbytes
            kept.append(len(pt) / max(ni, 1))
        return b / 1e9, kept


class _DegBand:
    """LU with partial pivoting of a tall matrix (mb rows, n columns, mb >= n) that is banded in the angular
    degree: the columns come in groups of equal degree (ascending), the rows are sorted by degree, and a row
    of degree d has entries only in the columns of degree d - B .. d + B (B None: no band, one dense LU done
    group by group). Step g factors the columns of group g over the window of rows that can have entries
    there: every row of degree <= deg_g + B not used as a pivot before. The pivots are taken from the whole
    window (getrf), the pivot rows are swapped to its top, the rest stays in the window; rows in the window
    reach at most degree deg_g + 2 B (induction: a new row reaches its degree + B, fill comes from pivot
    rows), so only the column groups up to there are updated. This is LAPACK's band LU (gbtrf) with the
    degree groups as blocks, for more rows than columns: the mb - n rows never taken stay in the window to
    the end, and are the last rows afterwards (rowat[p] = row now at position p).
    Storage, per column group c: W[c] = rows lo[c] .. hi[c] of that column block: U above (pivot rows of
    the groups deg_c - 2 B .. c), the LU of its panel, L below (the window of step c):
    about (3 B + 1) w^2 entries per degree instead of (K w)^2 for the layer.
    load(c, rows): the entries of the given rows (sorted-row indices) in column group c, dense."""

    def __init__(self, rowdeg, gdeg, gw, B, load, dt):
        mb, G = len(rowdeg), len(gdeg)
        coff = np.r_[0, np.cumsum(gw)].astype(np.int64)
        n = int(coff[-1])
        if mb < n:
            raise _NotBlock
        if B is None:
            hi = np.full(G, mb, np.int64)
            gmin = np.zeros(G, np.int64)
        else:
            hi = np.searchsorted(rowdeg, gdeg + B, 'right').astype(np.int64)
            hi[-1] = mb
            gmin = np.searchsorted(gdeg, gdeg - 2 * B, 'left').astype(np.int64)
        lo = coff[gmin]
        self.mb, self.n, self.G, self.coff, self.hi, self.lo, self.gw = mb, n, G, coff, hi, lo, gw
        self.rowat = rowat = np.arange(mb)
        self.W = W = [None] * G
        self.perm = []
        self.umax = 0.0
        nxt = 0
        for g in range(G):
            p0, pe, w = int(coff[g]), int(hi[g]), int(gw[g])
            while nxt < G and gmin[nxt] <= g:        # the column groups this step can reach: original entries
                c = nxt
                blk = np.zeros((hi[c] - lo[c], gw[c]), dt, order='F')
                rows = rowat[lo[c]:hi[c]]
                sel = np.arange(len(rows)) if B is None else np.flatnonzero(np.abs(rowdeg[rows] - gdeg[c]) <= B)
                if len(sel):
                    blk[sel] = load(c, rows[sel])
                W[c] = blk
                nxt += 1
            m = pe - p0
            if m < w:
                raise _NotBlock
            d = p0 - int(lo[g])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sla.LinAlgWarning)
                lu, piv = sla.lu_factor(np.array(W[g][d:], order='F'), overwrite_a=True, check_finite=False)
            dg = np.abs(np.diag(lu))
            if not np.all(np.isfinite(dg)) or dg.min() == 0.0:
                raise _NotBlock
            W[g][d:] = lu
            at = np.arange(m)
            for a_, b_ in enumerate(piv):
                if a_ != b_:
                    at[a_], at[b_] = at[b_], at[a_]
            chg = np.flatnonzero(at != np.arange(m))
            src = at[chg]
            self.perm.append((chg, src))
            if len(chg):
                seg = rowat[p0:pe]
                seg[chg] = seg[src]
            L11, L21 = lu[:w], lu[w:]
            for c in range(g + 1, G):
                if gmin[c] > g:
                    break
                T = W[c][p0 - lo[c]:pe - lo[c]]
                if len(chg):
                    T[chg] = T[src]
                U = sla.solve_triangular(L11, T[:w], lower=True, unit_diagonal=True, check_finite=False)
                T[:w] = U
                if m > w:
                    T[w:] -= L21 @ U
                if U.size:
                    self.umax = max(self.umax, float(np.abs(U).max()))
            del lu, L11, L21

    def fwd(self, V, g0=0):
        """In place: rows (in the sorted order) -> positions, top n = L11^-1 (P V), below = what is left of the
        rows never used as pivots. g0: first step (V zero in the rows of the windows before it)."""
        for g in range(g0, self.G):
            p0, pe, w = int(self.coff[g]), int(self.hi[g]), int(self.gw[g])
            chg, src = self.perm[g]
            seg = V[p0:pe]
            if len(chg):
                seg[chg] = seg[src]
            d = p0 - int(self.lo[g])
            blk = self.W[g]
            y = sla.solve_triangular(blk[d:d + w], seg[:w], lower=True, unit_diagonal=True, check_finite=False)
            seg[:w] = y
            if pe - p0 > w:
                seg[w:] -= blk[d + w:] @ y
        return V

    def first_step(self, p):
        """First step whose window holds the row at (initial) position p."""
        return int(np.searchsorted(self.hi, p, 'right'))

    def bwd(self, Y):
        """In place: U11^-1 Y (n rows)."""
        for g in range(self.G - 1, -1, -1):
            p0, w, l0 = int(self.coff[g]), int(self.gw[g]), int(self.lo[g])
            d = p0 - l0
            blk = self.W[g]
            x = sla.solve_triangular(blk[d:d + w], Y[p0:p0 + w], lower=False, check_finite=False)
            Y[p0:p0 + w] = x
            if d:
                Y[l0:p0] -= blk[:d] @ x
        return Y

    def rsolve_u(self, Bm):
        """Bm U11^-1 for a few dense rows Bm (k, n)."""
        Z = np.array(Bm.T, order='F')
        if Z.shape[1] == 0:
            return Z.T
        for g in range(self.G):
            p0, w, l0 = int(self.coff[g]), int(self.gw[g]), int(self.lo[g])
            d = p0 - l0
            blk = self.W[g]
            r = Z[p0:p0 + w] - blk[:d].T @ Z[l0:p0] if d else Z[p0:p0 + w]
            Z[p0:p0 + w] = sla.solve_triangular(blk[d:d + w], r, lower=False, trans='T', check_finite=False)
        return Z.T

    def nbytes(self):
        return sum(b.nbytes for b in self.W)


class _BandLayerLU:
    """_LayerLU with banded layer panels (2026-11-13, branch banded-layers): the same direct LU by layers
    with partial pivoting across the interfaces, memory ~K instead of K^2.
    What stood in the way: in _LayerLU the rows left over from the panel of a layer carry fill in ALL
    columns of the next layer (U12 = L11^-1 x the next-layer part of the interface rows), so the next panel
    has dense rows that must be pivot candidates: no band. Here every row that couples two layers gets an
    auxiliary unknown y for its next-layer side:
        row:  a . x_this + q . x_next + b . x_border = g    ->    a . x_this + y + b . x_border = g   (in this panel)
                                                                  -q . x_next + y = 0             (in the next)
    Then the panel of a layer has (i) its own block rows and the rows of the next layer that reach into it,
    all banded in degree on the layer's columns (the second halves, -q, of the rows split before are block
    rows too), and (ii) a border of dense columns: y of the interface before (Yin), y of the interface
    after (Yout), the global border. Step for a layer:
      1. _DegBand: LU of the band rows on the layer's columns, pivots from all of them;
      2. the rows it leaves (as many as there are y's, give or take), and those left by the layer before
         (which have no entries on this layer's columns, only on Yin), are reduced on [Yin | Yout | border]:
         Schur complement by band solves, then a dense LU with partial pivoting on the Yin columns;
      3. the rows left then (entries on Yout and the border only) go to the next layer.
    The border rows are eliminated along (never pivots), and close the last step: one dense square LU on
    [Yin | border]. No iteration; the caller refines with the same factors.
    R: the scaled part of the system outside the layer blocks (CSR); diag_cols(i, rows, cols): scaled
    entries of the block of layer i (local indices), dense; rdeg, cdeg: degree of every layer row / column
    (None: no band, every layer one group); B: the band."""

    def __init__(self, R, off, nL, nt, dt, diag_cols, rdeg=None, cdeg=None, B=None, order='down'):
        n = R.shape[0]
        nB = n - nt
        if not sps.isspmatrix_csr(R):
            R = sps.csr_matrix(R)
        if rdeg is None:
            rdeg, cdeg, B = np.zeros(nt, np.int64), np.zeros(nt, np.int64), None
        rdeg, cdeg = np.asarray(rdeg), np.asarray(cdeg)
        indptr, ind = R.indptr, R.indices
        seq = list(range(nL - 1, -1, -1)) if order == 'down' else list(range(nL))
        lay = np.searchsorted(off, np.arange(n), side='right') - 1
        for i in range(nL):
            cl = lay[ind[indptr[off[i]]:indptr[off[i + 1]]]]
            if np.any((np.abs(cl - i) > 1) & (cl != nL)):
                raise _NotBlock
        self.n, self.nt, self.nB, self.dt = n, nt, nB, dt
        state = np.zeros(n, np.int8)                 # 1: the row has been in a panel
        Rb = R[nt:]                                  # border rows
        BQb = Rb[:, nt:].toarray().astype(dt)        # border rows on the border columns (current)
        BTin = np.zeros((nB, 0), dt)                 # border rows on Yin (current)
        Ga, Ab = np.zeros((0, 0), dt), np.zeros((0, nB), dt)     # rows left by the layer before: on Yin, border
        defs = np.zeros(0, np.int64)                 # rows split at the interface before (order of Yin)
        self.steps = []
        parts = dict(band=0.0, u_border=0.0, schur=0.0, dense=0.0, closing=0.0)
        for k, i in enumerate(seq):
            c0, c1 = off[i], off[i + 1]
            ni = c1 - c0
            last = k == nL - 1
            nx = None if last else seq[k + 1]
            n0, n1 = (off[nx], off[nx + 1]) if not last else (0, 0)
            r_in = len(defs)
            # columns by degree and, inside a degree, in REVERSE layer order (the highest Chebyshev coefficient
            # of a field first): the interior rows, which are largest there, take those pivots and the
            # boundary rows are kept for the low coefficients. In layer order the factor on the Yin columns
            # grows along the degrees (K 24: 70 at N 45, 2.6e4 at N 60; K 72, N 60: 1.9e9); reversed: ~50.
            cperm = np.lexsort((-np.arange(ni), cdeg[c0:c1]))
            gdeg, gw = np.unique(cdeg[c0:c1][cperm], return_counts=True)
            coff = np.r_[0, np.cumsum(gw)]
            didx = np.full(n, -1, np.int64)
            didx[defs] = np.arange(r_in)
            loc = np.arange(c0, c1)
            if np.any((state[loc] != 0) & (didx[loc] < 0)):
                raise _NotBlock
            sgnD = np.where(state[loc] == 0, 1.0, -1.0)
            if last:
                pulled = np.zeros(0, np.int64)
            else:                                    # rows of the next layer that reach into this one
                seg = ind[indptr[n0]:indptr[n1]]
                hit = (seg >= c0) & (seg < c1)
                pulled = np.unique(np.repeat(np.arange(n0, n1), np.diff(indptr[n0:n1 + 1]))[hit])
                pulled = pulled[state[pulled] == 0]
                if len(pulled):
                    cl = lay[R[pulled].indices]
                    if np.any((cl != i) & (cl != nx) & (cl != nL)):
                        raise _NotBlock
            prevown = defs[lay[defs] != i]           # split rows that belong to the layer before
            Rrows = np.concatenate([pulled, prevown])
            sgn = np.concatenate([sgnD, np.ones(len(pulled)), -np.ones(len(prevown))])
            deg = np.concatenate([rdeg[loc], rdeg[Rrows]])
            order_ = np.argsort(deg, kind='stable')
            mb = ni + len(Rrows)
            Rsub = R[Rrows][:, c0:c1][:, cperm].tocsc()

            def load(c, rows, i=i, order_=order_, ni=ni, cperm=cperm, coff=coff, sgn=sgn, Rsub=Rsub):
                br = order_[rows]
                out = np.zeros((len(rows), coff[c + 1] - coff[c]), dt)
                isD = br < ni
                if isD.any():
                    out[isD] = diag_cols(i, br[isD], cperm[coff[c]:coff[c + 1]]) * sgn[br[isD], None]
                if not isD.all():
                    q = np.flatnonzero(~isD)
                    out[q] = Rsub[:, coff[c]:coff[c + 1]].tocsr()[br[q] - ni].toarray() * sgn[br[q], None]
                return out

            F = _DegBand(deg[order_], gdeg, gw, B, load, dt)
            e2 = mb - ni
            # rows with a right-hand side: own rows not split before, and the rows pulled from the next layer
            act = np.concatenate([loc[sgnD > 0], pulled])            # global row
            actb = np.concatenate([np.flatnonzero(sgnD > 0), ni + np.arange(len(pulled))])   # band row
            # the rows split at this interface: own rows with entries in the next layer, and the pulled rows
            if last:
                Rn = np.zeros(0, np.int64)
            else:
                own = loc[sgnD > 0]
                sub = R[own]
                has = np.zeros(len(own), bool)
                q = (sub.indices >= n0) & (sub.indices < n1)
                has[np.repeat(np.arange(len(own)), np.diff(sub.indptr))[q]] = True
                Rn = np.concatenate([own[has], pulled])
                Rn = Rn[np.argsort(rdeg[Rn], kind='stable')]
            r_out = len(Rn)
            t = r_in + r_out + nB
            # T: the band rows on [Yin | Yout | border] (sparse), rows in the sorted order
            oidx = np.full(n, -1, np.int64)
            oidx[Rn] = np.arange(r_out)
            brow = np.full(n, -1, np.int64)                           # band row of a global row (active rows)
            brow[act] = actb
            dD = np.flatnonzero(sgnD < 0)
            tr = [dD, ni + len(pulled) + np.arange(len(prevown)), brow[Rn]]
            tc = [didx[loc[dD]], didx[prevown], r_in + np.arange(r_out)]
            tv = [np.ones(len(dD)), np.ones(len(prevown)), np.ones(r_out)]
            if nB:
                sub = R[act][:, nt:].tocoo()
                tr.append(actb[sub.row])
                tc.append(r_in + r_out + sub.col)
                tv.append(sub.data)
            T = sps.csr_matrix((np.concatenate(tv).astype(dt), (np.concatenate(tr), np.concatenate(tc))),
                               shape=(mb, t))[order_].tocsc()
            # Schur complement of the band rows on the border columns, and the border rows
            BP = Rb[:, c0:c1].toarray()[:, cperm].astype(dt) if nB else np.zeros((0, ni), dt)
            Lb = np.ascontiguousarray(F.rsolve_u(BP))
            BT = np.hstack([BTin, np.zeros((nB, r_out), dt), BQb])
            S2 = np.zeros((e2, t), dt)
            for a_ in range(0, t, 256):
                b_ = min(a_ + 256, t)
                Tc = T[:, a_:b_]
                if Tc.nnz == 0:
                    continue
                V = np.asfortranarray(Tc.toarray())
                F.fwd(V, F.first_step(int(Tc.indices.min())))
                S2[:, a_:b_] = V[ni:]
                if nB:
                    BT[:, a_:b_] -= Lb @ V[:ni]
                parts['schur'] = max(parts['schur'], float(np.abs(V[ni:]).max()) if e2 else 0.0)
                parts['u_border'] = max(parts['u_border'], float(np.abs(V[:ni]).max()))
            V = None
            M2 = np.vstack([np.hstack([Ga, np.zeros((len(Ga), r_out), dt), Ab]), S2])
            del S2
            if last:
                if len(M2) != r_in:
                    raise _NotBlock
                if r_in + nB:
                    Mc = np.vstack([M2, BT])
                    parts['border_rows'] = float(np.abs(BT).max()) if BT.size else 0.0
                    amax = max(float(np.abs(Mc).max()), 1.0)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', sla.LinAlgWarning)
                        lu = sla.lu_factor(Mc, overwrite_a=True, check_finite=False)
                    d = np.abs(np.diag(lu[0]))
                    if not np.all(np.isfinite(d)) or d.min() == 0.0:
                        raise _NotBlock
                    parts['closing'] = float(np.abs(lu[0]).max()) / amax
                    parts['band'] = max(parts['band'], F.umax)
                    self.final = lu
                else:
                    self.final = None
                self.steps.append((F, c0 + cperm, act, np.argsort(order_)[actb],
                                   mb, ni, r_in, r_out, T.tocsr(), Lb, None, None, None, None))
                break
            if len(M2) < r_in:
                raise _NotBlock
            if r_in:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', sla.LinAlgWarning)
                    lu2, piv = sla.lu_factor(np.array(M2[:, :r_in], order='F'), overwrite_a=True,
                                             check_finite=False)
                d = np.abs(np.diag(lu2))
                if not np.all(np.isfinite(d)) or d.min() == 0.0:
                    raise _NotBlock
                at2 = np.arange(len(M2))
                for a_, b_ in enumerate(piv):
                    if a_ != b_:
                        at2[a_], at2[b_] = at2[b_], at2[a_]
                M2r = M2[at2][:, r_in:]
                U2t = sla.solve_triangular(lu2[:r_in], M2r[:r_in], lower=True, unit_diagonal=True,
                                           check_finite=False)
                left = M2r[r_in:] - lu2[r_in:] @ U2t
                Lb2 = sla.solve_triangular(lu2[:r_in], BT[:, :r_in].T, trans='T', lower=False,
                                           check_finite=False).T if nB else np.zeros((0, r_in), dt)
                BTn = BT[:, r_in:] - Lb2 @ U2t
                parts['dense'] = max(parts['dense'], float(np.abs(U2t).max()) if U2t.size else 0.0,
                                     float(np.abs(left).max()) if left.size else 0.0)
                del M2r
            else:
                lu2 = at2 = U2t = Lb2 = None
                left, BTn = M2, BT
            del M2
            Ga, Ab = np.ascontiguousarray(left[:, :r_out]), np.ascontiguousarray(left[:, r_out:])
            BTin, BQb = np.ascontiguousarray(BTn[:, :r_out]), np.ascontiguousarray(BTn[:, r_out:])
            state[act] = 1
            defs = Rn
            parts['band'] = max(parts['band'], F.umax)
            self.steps.append((F, c0 + cperm, act, np.argsort(order_)[actb], mb, ni, r_in, r_out, T.tocsr(), Lb,
                               lu2, at2, U2t, Lb2))
        # largest entry of the factors (the scaled matrix has maximum 1), by stage: U of the band LU, U on the
        # border columns, the Schur complement, the dense stage on Yin; the closing LU relative to its input
        # (as _LayerLU). border_rows: the border rows before the closing LU; they are never pivots, so their
        # size is not bounded by the pivoting (the same in _LayerLU) and is not counted as growth.
        self.growth_parts = parts
        self.growth = max(v for k_, v in parts.items() if k_ != 'border_rows')

    def solve(self, rhs):
        dt = np.result_type(self.dt, rhs.dtype)
        g = rhs.astype(dt)
        nt, nB = self.nt, self.nB
        gb = g[nt:].copy()
        ga = np.zeros(0, dt)
        Y = []
        for F, cols, act, apos, mb, ni, r_in, r_out, T, Lb, lu2, at2, U2t, Lb2 in self.steps:
            v = np.zeros(mb, dt)
            v[apos] = g[act]
            F.fwd(v)
            y1 = v[:ni].copy()
            if nB:
                gb -= Lb @ y1
            z = np.concatenate([ga, v[ni:]])
            if lu2 is not None:
                z = z[at2]
                y2 = sla.solve_triangular(lu2[:r_in], z[:r_in], lower=True, unit_diagonal=True, check_finite=False)
                ga = z[r_in:] - lu2[r_in:] @ y2
                if nB:
                    gb -= Lb2 @ y2
            else:
                y2 = np.zeros(0, dt)
                ga = z                               # the last step: kept for the closing system
            Y.append((y1, y2))
        x = np.zeros(self.n, dt)
        sol = sla.lu_solve(self.final, np.concatenate([ga, gb]), check_finite=False) if self.final is not None \
            else np.zeros(0, dt)
        r_last = self.steps[-1][6]
        xb = sol[r_last:]
        x[nt:] = xb
        yout = np.zeros(0, dt)
        for j in range(len(self.steps) - 1, -1, -1):
            F, cols, act, apos, mb, ni, r_in, r_out, T, Lb, lu2, at2, U2t, Lb2 = self.steps[j]
            y1, y2 = Y[j]
            if j == len(self.steps) - 1:             # the closing system gave Yin of the last layer
                yin = sol[:r_in]
            elif U2t is not None:
                yin = sla.solve_triangular(lu2[:r_in], y2 - U2t @ np.concatenate([yout, xb]), lower=False,
                                           check_finite=False)
            else:
                yin = np.zeros(0, dt)
            u = T @ np.concatenate([yin, yout, xb])
            F.fwd(u)
            x[cols] = F.bwd(y1 - u[:ni])
            yout = yin
        return x

    def stored_gb(self):
        """Memory kept by the factors (GB) and, per step, the number of split rows over the layer size."""
        b = self.final[0].nbytes if self.final is not None else 0
        kept = []
        for F, cols, act, apos, mb, ni, r_in, r_out, T, Lb, lu2, at2, U2t, Lb2 in self.steps:
            b += F.nbytes() + Lb.nbytes + T.data.nbytes + T.indices.nbytes
            b += sum(a.nbytes for a in (lu2, U2t, Lb2) if a is not None)
            kept.append(r_out / max(ni, 1))
        return b / 1e9, kept


def _gmres(A, b, prec, x0=None, tol=1e-12, restart=30, maxiter=90):
    """Restarted GMRES, right-preconditioned (A prec(z) = r), Gram-Schmidt twice. Returns
    (x, iterations, true relative residual). Stops at tol, at maxiter, or when a restart no longer
    lowers the true residual (the attainable accuracy of an ill-conditioned system)."""
    nb = max(np.linalg.norm(b), 1e-300)
    dt = np.result_type(A.dtype, b.dtype)
    x = np.zeros(len(b), dt) if x0 is None else np.asarray(x0, dt).copy()
    r = b - A @ x
    res = np.linalg.norm(r) / nb
    it = 0
    while it < maxiter and res > tol:
        beta = np.linalg.norm(r)
        V, Z = [r / beta], []
        Hh = np.zeros((restart + 1, restart), dt)
        g = None
        for j in range(restart):
            z = prec(V[j])
            Z.append(z)
            w = A @ z
            for _ in range(2):
                for k in range(j + 1):
                    h = np.vdot(V[k], w)
                    Hh[k, j] += h
                    w = w - h * V[k]
            Hh[j + 1, j] = np.linalg.norm(w)
            it += 1
            e1 = np.zeros(j + 2, dt)
            e1[0] = beta
            g = np.linalg.lstsq(Hh[:j + 2, :j + 1], e1, rcond=None)[0]
            est = np.linalg.norm(Hh[:j + 2, :j + 1] @ g - e1) / nb
            if est < 0.5 * tol or Hh[j + 1, j] <= 1e-300 or it >= maxiter:
                break
            V.append(w / Hh[j + 1, j])
        xn = x + sum(gk * zk for gk, zk in zip(g, Z))
        rn = b - A @ xn
        resn = np.linalg.norm(rn) / nb
        if not np.isfinite(resn) or resn >= res:      # stagnation at the attainable accuracy
            break
        x, r, res = xn, rn, resn
    return x, it, res



def _true_fluid(model, i):
    """A true (inviscid) fluid layer: shear modulus identically zero. A viscous layer (EarthModel.viscous_core,
    mu = i omega eta) is welded to its neighbours however small |mu| is."""
    m = model.mu[i]
    c = getattr(m, 'coef', None)
    if c is not None:
        return bool(np.all(np.asarray(c) == 0))
    return abs(m(0.5 * (model.r0[i] + model.r1[i]))) == 0.0

class TidalSolver:
    """Tidal (or load) response of a layered, possibly flattened and rotating Earth for one order m
    and one forcing degree n_force (manual, chapters 'How the solver works' and 'API reference').

    Resolution: K spheroidal (KT toroidal) degrees per field, N Chebyshev polynomials per layer
    (fluid_N in the (weak-)fluid layers when given), n_theta Gauss-Legendre points; or explicit
    degrees=([spheroidal], [toroidal]). fluid_K > K: K (spheroidal and toroidal) degrees of U, V, W
    in the (weak-)fluid layers only; the potential P has the fluid_K degrees in every layer (a core
    density change of any degree has its potential everywhere). Extra fluid degrees meet a solid
    that does not deform at that degree: rigid wall (welded) or normal displacement continuous
    (slip), see _extra_rows.
    Forcing: forcing='tide' (potential -g a (r/a)^nf Ybar cos m lam), 'load' (unit surface mass
    density Ybar cos m lam) or 'none'.
    Dynamics: omega_f (rad/s) inertia of a non-rotating run; coriolis (rotation rate) and omega_t
    (signed tidal frequency) for a rotating run; dyn_fluid adds them in the liquid layers;
    omega_rot is the rotation rate of the reference state; consistent_reference computes Phi0
    from the model's own density; gravity=False switches self-gravitation off.
    Frame: frame='mantle' | 'surface' for the rigid rotation removed from the output; wahr_frame
    lets the regions rotate freely (rotating runs with m = 0, 1).
    Linear algebra: linear_solver='sparse' (SuperLU) | 'dense' (LAPACK).
    Centre (spheroidal degree 1 only): centre='fixed' pins the centre, U_1 = V_1 = 0 (the frame of
    static runs, where a translation is free); centre='free' lets it move, U_1 = V_1, dU_1/ds = 0,
    P_1 = 0, so that inertia determines the translation (needs omega_f or omega_t; manual,
    'Degree 1').
    Experimental (paper 2): lateral, all_degrees, fixed_density_layers, eulerian_interfaces,
    background.
    solve() returns results(): U, V, W, k per degree and h, l, kk for the forcing degree.
    """
    # band != None: no SuperLU fallback above this many unknowns (its fill-in at K ~100 is tens of GB)
    BAND_FALLBACK_MAX = int(os.environ.get('EARTHTIDES_BAND_FALLBACK_MAX', '40000'))
    # solver-10 (2026-11-11): accept the block path at relative residual <= 1e-9 (was 1e-10). Near the FCN GMRES
    # stalls at 1-4e-10 (its attainable accuracy); those solutions agree with SuperLU to <= 1.1e-8 of |h0| and
    # 2e-10 in the latitude terms. The environment variable EARTHTIDES_BLOCK_ACCEPT sets the threshold.
    BLOCK_ACCEPT = float(os.environ.get('EARTHTIDES_BLOCK_ACCEPT', '1e-9'))
    LAYERS_REFINE = 5        # linear_solver='layers': at most this many steps of iterative refinement
    LAYERS_BANDED = True     # 'layers' with band=B: banded panels (_BandLayerLU); False: dense (_LayerLU);
                             # 'always': _BandLayerLU also without a band (tests)


    def __init__(self, model, geometry=None, m=0, K=3, KT=None, N=24,
                 n_theta=None, n_force=2, omega_f=1.4053e-4, omega_rot=0.0,
                 background='level', tisserand=None, regions=None,
                 coriolis=0.0, omega_t=0.0, wahr_frame=False, dyn_fluid=False,
                 gravity=True, consistent_reference=False, frame='surface',
                 forcing='tide', deg_window=None, lateral=None, all_degrees=False,
                 degrees=None, fixed_density_layers=(), eulerian_interfaces=False,
                 linear_solver='block', centre='fixed', load_inertia=True, nonhydro=None,
                 fluid_N=None, fluid_K=None, layer_K=None, layer_N=None, factor_cache=None,
                 force_layers=None, p1_normal='fluid_true', assembly='hybrid', band=None,
                 solid_P=None, fluid_pressure=False, slip_nonlevel=True):
        self.model = model
        # slip_nonlevel: the normal condition at a slipping fluid-solid boundary with the term for a wall
        # that is not a level surface of Phi0 (_bnd; consistent_reference=True with a first-order or a
        # bumped map: the wall is off level at O(eps^2), and without the term the latitude terms of a
        # slipping core are off by O(eps)). Default True since version 1.1; False: the conditions of
        # Tromp & Mitrovica (1999, eq. 3.7), valid on level surfaces (version 1.0). On a level
        # wall (consistent_reference=False: Phi0 = W(s)) the term vanishes.
        self.slip_nonlevel = bool(slip_nonlevel)
        # fluid_pressure: in every (weak-)fluid layer Pi = lambda div u is an unknown of its own (fields
        # ('Q', n)), the stress is Pi I + 2 mu eps and the rows of Q are div u - Pi / lambda = 0: the bulk
        # modulus no longer shares a row with the viscous terms (better conditioned for thin Ekman layers)
        self.fluid_pressure = bool(fluid_pressure)
        self.geom = geometry if geometry is not None else SphereGeometry()
        self.m, self.N, self.nf = m, N, n_force
        self.fluid_N = fluid_N
        self.omega_f, self.omega_rot = omega_f, omega_rot
        self.background = background
        # rotating-frame dynamics in the solid layers only: Coriolis (rate
        # `coriolis`, rad/s) and inertia at the signed tidal frequency omega_t
        self.coriolis, self.omega_t = coriolis, omega_t
        # gravity=False: no pre-stress, no self-gravitation (pure elasticity)
        self.gravity = gravity
        self.G = G if gravity else 0.0
        # consistent_reference: Phi0 from a Poisson solve for the model density
        # plus the exact centrifugal potential (see refpot.py)
        # frame: 'surface' = best-fit rotation of the surface displacement (Wahr
        # 1981, eq. 4.3); 'mantle' = mean (Tisserand) rotation of the mantle volume,
        # the terrestrial frame of Mathews et al. (1995) and de Vries & Wahr (1991)
        self.frame = frame
        # forcing='tide': degree-n_force tidal potential (body force);
        # forcing='load': unit surface mass density Ybar_nf^m cos(m lam) [kg/m^2]
        # on the outer surface (normal traction -g sigma, flux jump 4 pi G sigma)
        # forcing='none': no right-hand side (the matrix for a user right-hand side, born.py)
        if forcing not in ('tide', 'load', 'none'):
            raise ValueError(forcing)
        self.forcing = forcing
        # force_layers: layers where the tidal body force acts (default: all). The force restricted
        # to the fluid outer core is the centrifugal force of a core wobble relative to the mantle
        # (Sasao et al. 1980, eq. 46-50: unit m_f = this forcing with phi = -1)
        self.force_layers = None if force_layers is None else tuple(force_layers)
        # fixed_density_layers: layers where the EULERIAN density is held fixed (static support of
        # a prescribed final density, born.StaticSupport): the terms div(rho u) grad Phi0 and the
        # volume source of Poisson's equation are dropped there (interface masses remain)
        self.fixed_density_layers = tuple(fixed_density_layers)
        # eulerian_interfaces: between two fixed-density layers the welded conditions are on the
        # EULERIAN traction (tau - rho0 g u.n n) and on n.grad phi1 alone: a deflected density jump
        # carries no load or surface mass there (phase boundaries crossed by flow; whole-mantle
        # flow in the viscous limit of born.StaticSupport). Default: Lagrangian traction and
        # flux with 4 pi G rho u.n (material interfaces).
        self.eulerian_interfaces = eulerian_interfaces
        # linear_solver: 'sparse' (SuperLU with COLAMD ordering; the matrix is block-banded by
        # layer, bordered by a few dense rows and columns, ~8 per cent non-zero) or 'dense' (LAPACK).
        # Both give residuals at round-off; they differ only in the poorly determined directions of
        # the matrix (high degrees in the fluid core, condition number up to ~1e14 at M2), which
        # change the Love numbers by < 1e-9. 'block': GMRES preconditioned by a block LU by layer
        # (dense LAPACK blocks, Accelerate; Schur complement for the border; _BlockLU, _gmres); no
        # fill-in beyond the layer blocks; singular layer blocks factored with a shift (it only
        # touches the preconditioner); falls back to 'sparse' only for a matrix that is not
        # block-tridiagonal by layer or when GMRES fails.
        if linear_solver not in ('sparse', 'dense', 'block', 'layers'):
            raise ValueError(linear_solver)
        self.linear_solver = linear_solver
        # assembly: 'hybrid' (dense layer blocks + sparse rest, _HybridMatrix; 2026-11-02) or 'sparse'
        # (everything CSR, _BlockMatrix: the state before, a run-time revert). Same matrix entry for
        # entry; the block path then works on the dense blocks without a CSR copy
        if assembly not in ('hybrid', 'sparse'):
            raise ValueError(assembly)
        self.assembly = assembly
        # band: drop the angular coupling between degrees |n - n'| > band (plan_banded_coupling_2026-11-02;
        # 12 keeps every Love number within 2e-11 of the full coupling at delta 1 km, psi1 worst). Layers
        # with a narrow band are stored and factored as LAPACK bands (memory ~K instead of ~K^2). None:
        # full coupling (the default). Needs the hybrid assembly.
        if band is not None and (assembly != 'hybrid' or int(band) < 1):
            raise ValueError('band needs assembly="hybrid" and band >= 1')
        self.band = None if band is None else int(band)
        self._band_info = None
        self._Hh = self._Hsp = None
        # factor_cache: a BlockCache shared by the solvers of neighbouring frequencies (sweeps)
        self.factor_cache = factor_cache
        # centre: 'fixed' (U_1 = V_1 = 0 at s = 0) or 'free' (U_1 = V_1, dU_1/ds = 0: the centre
        # moves, the translation follows from inertia; a static run would be singular)
        if centre not in ('fixed', 'free'):
            raise ValueError(centre)
        if centre == 'free' and omega_f == 0.0 and omega_t == 0.0:
            raise ValueError("centre='free' needs inertia (omega_f or omega_t): a static "
                             "translation is undetermined; use centre='fixed'")
        self.centre = centre
        # load_inertia: in a rotating run the load material must move to redistribute its mass; the
        # inertial and Coriolis forces of that motion add the tangential traction
        # -(d Lambda / dt) Omega x r (Sasao & Wahr 1981, eq. 4.14), which supplies the torque
        # d(c_M . Omega)/dt and decides the sign of the free-core-nutation resonance of the load
        # Love numbers. False: pressure and potential only (the non-rotating load problem).
        self.load_inertia = load_inertia
        self.refpot = None
        if consistent_reference:
            from .refpot import ReferencePotential
            self.refpot = ReferencePotential(model, self.geom, omega_rot,
                                             drho=getattr(nonhydro, 'drho', None))
        # lateral(i, s, theta) -> (f, f_s, f_theta): relative shear-modulus variation
        # mu = mu0(s) (1 + f) in layer i (zero layers: return 0) and its derivatives;
        # axisymmetric (m stays separate). all_degrees: keep both parities (needed when
        # the variation is not symmetric about the equator, e.g. f ~ cos theta).
        self.lateral = lateral
        # nonhydro: non-hydrostatic final state (earthtides.nonhydro.FlatState). An object with
        #   drho(i, s2, th, mp) -> (drho, drho_r, drho_t): density beyond rho0(s) at the physical
        #       points (d/dr, d/dtheta at fixed r), or None;
        #   tau(i, s2, th, mp) -> dict of jets rr, tt, pp, rt (axisymmetric stress beyond the
        #       background pressure -p0(s) I, with d/dr, d/dtheta), or None;
        #   slip(i_fluid, r_boundary, theta) -> (dp, d dp/dtheta) at fluid-solid boundaries (optional).
        # The operator then adds C(u) = div P1(tau) - D b + (grad u)^T b, the difference between the
        # general prestressed equation and the hydrostatic form used here for a state in
        # equilibrium (paper, Supplementary Material Sect. S8); zero for tau = -q I and for a rigid
        # rotation.
        self.nonhydro = nonhydro
        # p1_normal: normal used in the initial-stress traction P1(tau).n at interfaces (nonhydro).
        # 'fluid_true' (default since 2026-10-31): the true boundary normal on the fluid side
        # (tau = -q I at the boundary points, _isotropic; a welded viscous core too), the
        # background normal (geometry.base) on the solid side; 'background': the background normal on
        # both sides (the earlier form). On the fluid side tau = -q I and the normal part of
        # P1.n = q (D n - G^T n) is finite only through a cancellation between d_r u_r and the slope
        # times the Ekman shear d_r u_theta ~ du/delta; with n_bg a leftover q (n - n_bg)_t du/delta
        # remains, second order in the excess and growing as 1/delta
        if p1_normal not in ('fluid_true', 'background'):
            raise ValueError(p1_normal)
        self.p1_normal = p1_normal
        # resolution per layer: layer_K / layer_N (lists, one entry per layer; resolution_per_layer
        # builds them from a rule set), or fluid_K / fluid_N in the (weak-)fluid layers, else K / N
        fl0 = self.fluid_layers()
        nL = model.n_layers
        Kl = list(layer_K) if layer_K is not None else \
            [fluid_K if (fluid_K and fl0[i]) else K for i in range(nL)]
        Nl = list(layer_N) if layer_N is not None else \
            [fluid_N if (fluid_N and fl0[i]) else N for i in range(nL)]
        if len(Kl) != nL or len(Nl) != nL:
            raise ValueError('layer_K / layer_N need %d entries' % nL)
        if len(set(Kl)) > 1:
            if degrees is not None or all_degrees or deg_window is not None or KT not in (None, K) \
                    or lateral is not None or nonhydro is not None:
                raise NotImplementedError('K per layer with explicit degrees, all_degrees, deg_window, '
                                          'KT != K, lateral or nonhydro')
            if min(Kl) < 2 or Kl[-1] < 3:
                raise ValueError('K per layer: at least 2, and 3 in the outermost layer')
        K = Kl[-1]                      # the outermost layer sets the surface degrees (results)
        self.Kl, self.Nl = Kl, Nl
        if degrees is not None:         # explicit (spheroidal, toroidal) degree lists
            self.deg, degW = list(degrees[0]), list(degrees[1])
            KT = len(degW)
        elif all_degrees:
            self.deg = list(range(m, m + K))
            KT = K if KT is None else KT
            degW = list(range(max(m, 1), max(m, 1) + KT))
        elif deg_window is None:        # K degrees upward from m
            self.deg = [n for n in range(m, m + 2 * K + 2) if (n - n_force) % 2 == 0][:K]
            KT = K if KT is None else KT
            degW = [n for n in range(max(m, 1), max(m, 1) + 2 * KT + 2) if (n - n_force) % 2][:KT]
        else:                           # n_force +- 2 deg_window only (high-degree loads)
            w = deg_window
            self.deg = [n for n in range(n_force - 2 * w, n_force + 2 * w + 1, 2) if n >= m]
            degW = [n for n in range(n_force - 2 * w - 1, n_force + 2 * w + 2, 2) if n >= max(m, 1)]
        self.degV = [n for n in self.deg if n >= 1]
        self.degW = [] if (m == 0 and not coriolis and degrees is None) else degW
        self.K = len(self.deg)
        self.fluid_K = fluid_K

        def degsets(k):                 # K degrees upward from m (spheroidal) and toroidal
            if k == self.K:
                return self.deg, self.degW
            d = [n for n in range(m, m + 2 * k + 2) if (n - n_force) % 2 == 0][:k]
            dW = [n for n in range(max(m, 1), max(m, 1) + 2 * k + 2) if (n - n_force) % 2][:k] \
                if self.degW else []
            return d, dW
        # uniform K: every layer has exactly the set built above (explicit degrees, all_degrees and
        # deg_window included); per-layer K: K degrees upward from m in each layer
        self._degsets = {Kl[0]: (self.deg, self.degW)} if len(set(Kl)) == 1 else \
            {k: degsets(k) for k in sorted(set(Kl))}
        kmax = max(Kl)
        self._degF, self._degWF = self._degsets[kmax]
        # potential: the largest set, in every layer (solid_P None, the default). solid_P = M: a solid layer
        # carries only its own spheroidal degrees plus the next M of the largest set; above them the
        # solid has no motion, and the potential there is taken as a harmonic exterior (mantle,
        # ~ s^-(n+1)) or interior (inner core, ~ s^n) field, matched to the fluid at the CMB / ICB
        # (_top_rows, _bottom_rows). Welded (viscous) cores only.
        self.degP = list(self._degF)
        self.solid_P = None if solid_P is None else int(solid_P)
        fl_ = self.fluid_layers()
        if self.solid_P is not None:
            tf = [_true_fluid(model, i) for i in range(model.n_layers)]
            if any(a != b for a, b in zip(tf[:-1], tf[1:])) or self.solid_P < 0:
                raise ValueError('solid_P needs welded fluid-solid interfaces (a viscous core) and solid_P >= 0')

        def degP_of(i):
            if self.solid_P is None or fl_[i]:
                return tuple(self._degF)
            return tuple(self._degF[:min(len(self._degF), Kl[i] + self.solid_P)])
        self.fields = [('U', n) for n in self.deg] + [('V', n) for n in self.degV] \
            + [('W', n) for n in self.degW] + [('P', n) for n in self.degP]
        self.nfld = len(self.fields)
        self.tilt = (m in (0, 1) and 1 in self.degW) if tisserand is None else tisserand
        # Tisserand regions: contiguous groups of solid / (weak-)fluid layers;
        # each gets its own rigid-rotation inertial force and zero-rotation constraint
        if regions is None:
            fluid = self.fluid_layers()
            regions, cur = [], [0]
            for i in range(1, model.n_layers):
                if fluid[i] == fluid[i - 1]:
                    cur.append(i)
                else:
                    regions.append(cur)
                    cur = [i]
            regions.append(cur)
        # wahr_frame: solid regions rotate freely (their rigid rotation is set by
        # inertia + Coriolis, i.e. the forced nutation / rotation-rate change, whose
        # centrifugal and Coriolis effects stay in the body tide); only the static
        # weak-solid core keeps a Tisserand constraint. The rigid rotation is then
        # removed from the *surface* displacement only (Wahr 1981, eqs. 4.3, 4.6).
        self.wahr_frame, self.dyn_fluid = wahr_frame, dyn_fluid
        # layers with mu = 0 exactly are true fluids: slip at fluid-solid interfaces. (Until 2026-11-10 the test
        # was |mu| < 1 Pa at mid-layer, which turned a viscous core, |mu| = omega eta ~ rho Omega^2 delta^2, into a
        # slipping inviscid one below delta ~ 130 m for PREM and ~ 185 m for Hough's Earth.)
        self.true_fluid = [_true_fluid(model, i) for i in range(model.n_layers)]
        if wahr_frame:
            fl = self.fluid_layers()
            regions = [] if dyn_fluid else [r for r in regions if all(fl[i] for i in r)]
        self.regions = regions if self.tilt else []
        self.H_dtype = complex if getattr(model, 'is_complex', False) else float
        n_theta = n_theta or 2 * max(self._degF + self._degWF + [n_force]) + 24
        x, w = L.leggauss(n_theta)
        th = np.arccos(x)
        self.theta = th[None, None, :]
        self.w = w
        self.ang = {}
        for n in sorted(set(self._degF + self._degWF)):
            y = ybar(n, m, th, 3)
            ys = ybar(n, m, th, 2, over_sin=True) if m > 0 else [0 * th] * 3
            self.ang[n] = dict(Y=y, mYs=[m * v for v in ys])
        # test functions (times quadrature weights); vector tests are (theta, lambda)
        # per-layer degree sets and test functions (one shared basis object per K)
        bases = {}
        mixed = [self.fluid_pressure and f for f in fl_]
        if any(mx and tf for mx, tf in zip(mixed, self.true_fluid)):
            raise ValueError('fluid_pressure: viscous (welded) fluid layers only, not mu = 0')
        for i, k in enumerate(Kl):
            key = (k, degP_of(i), mixed[i])
            if key not in bases:
                bases[key] = self._basis(*self._degsets[k], degP=list(key[1]), Q=mixed[i])
        self.B = [bases[k, degP_of(i), mixed[i]] for i, k in enumerate(Kl)]
        self._mixed = mixed
        self.degP = list(self.B[-1].P)          # exterior coefficients c_n: the surface layer's set
        if self.solid_P is not None:            # S.fields: the surface layer's fields (love.py, results)
            self.fields, self.nfld = list(self.B[-1].fields), self.B[-1].nfld
        # the outermost layer's basis, for code outside the solver (surface values, born, nonhydro)
        Bs = self.B[-1]
        self.testU, self.testP, self.testV, self.testW = Bs.testU, Bs.testP, Bs.testV, Bs.testW

    def _basis(self, deg, degW, degP=None, Q=False):
        """Fields and test functions of one degree set: U, V, W of deg / degW, P of degP (default
        self.degP); Q: also the pressure-like field Pi of deg (fluid_pressure), after P."""
        w = self.w
        degP = list(self.degP if degP is None else degP)
        degV = [n for n in deg if n >= 1]
        b = _D(U=list(deg), V=degV, W=list(degW), P=degP, Q=list(deg) if Q else [])
        b.fields = [('U', n) for n in deg] + [('V', n) for n in degV] + [('W', n) for n in degW] \
            + [('P', n) for n in degP] + [('Q', n) for n in b.Q]
        b.nfld, b.nS = len(b.fields), len(deg) + len(degV) + len(degW)
        b.testU = np.array([self.ang[n]['Y'][0] * w for n in deg])
        b.testP = np.array([self.ang[n]['Y'][0] * w for n in degP])
        b.testQ = np.array([self.ang[n]['Y'][0] * w for n in b.Q]).reshape(-1, len(w))
        b.testV = [(self.ang[n]['Y'][1] * w, -self.ang[n]['mYs'][0] * w) for n in degV]
        b.testW = [(-self.ang[n]['mYs'][0] * w, self.ang[n]['Y'][1] * w) for n in degW]
        return b

    def fluid_layers(self):
        """(Weak-)fluid layers: mu below the model's weak_fluid_mu (default 1e8 Pa; lowered for
        the viscous-limit static models of prestress.FinalState, whose mantle mu is small)."""
        md = self.model
        cut = getattr(md, 'weak_fluid_mu', 1e8)
        return [abs(md.mu[i](0.5 * (md.r0[i] + md.r1[i]))) < cut for i in range(md.n_layers)]

    # ------------------------------------------------------------------
    def props(self, i, s):
        """Material, reference potential and geometry at s (Ns,) x theta."""
        md = self.model
        s2 = s[:, None]
        th = self.theta[0]
        if hasattr(self.geom, 'map_derivs'):
            m = mapping_from_R(*self.geom.map_derivs(i, s2, th))
        else:
            m = mapping(s2, th, *self.geom.eps(i, s2))
        r = m['R']
        p = _D(r=r, R_t=m['R_t'], s_r=m['s_r'], cot=1 / np.tan(th), isn=1 / np.sin(th),
               cos=np.cos(th) + 0 * r, sin=np.sin(th))
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.background == 'level':
                for name, f in (('rho', md.rho[i]), ('lam', md.lam[i]), ('mu', md.mu[i])):
                    d = f.deriv()(s2)
                    p[name] = f(s2) + 0 * r
                    p[name + '_r'] = d * m['s_r']
                    p[name + '_t'] = d * m['s_t']
                gs = md.gravity(i, s2)
                W1 = gs - 2.0 / 3.0 * self.omega_rot**2 * s2
                W2 = 4 * np.pi * self.G * md.rho[i](s2) - 2 * gs / s2 \
                    - 2.0 / 3.0 * self.omega_rot**2
                ph0 = chain(0.0, W1, W2, 0.0, 0.0, 0.0, m)
            else:   # spherical Earth on a (possibly distorted) mesh
                for name, f in (('rho', md.rho[i]), ('lam', md.lam[i]), ('mu', md.mu[i])):
                    p[name] = f(r)
                    p[name + '_r'] = f.deriv()(r)
                    p[name + '_t'] = 0 * r
                g = md.gravity(i, r)
                ph0 = dict(r=g, t=0 * r, rr=4 * np.pi * self.G * md.rho[i](r) - 2 * g / r,
                           rt=0 * r, tt=0 * r)
        if self.lateral is not None:
            f, f_s, f_t = (np.broadcast_to(v, r.shape) for v in self.lateral(i, s2, th))
            mu0, mu0_r, mu0_t = p['mu'], p['mu_r'], p['mu_t']
            p['mu'] = mu0 * (1 + f)
            p['mu_r'] = mu0_r * (1 + f) + mu0 * f_s * m['s_r']
            p['mu_t'] = mu0_t * (1 + f) + mu0 * (f_t + f_s * m['s_t'])
        if self.nonhydro is not None:
            d = self.nonhydro.drho(i, s2, th, m)
            if d is not None:
                p['rho'] = p['rho'] + d[0]
                p['rho_r'] = p['rho_r'] + d[1]
                p['rho_t'] = p['rho_t'] + d[2]
            p['tau'] = self.nonhydro.tau(i, s2, th, m)
        if self.refpot is not None and self.background == 'level':
            ph0 = self.refpot.phys_derivs(i, s, th, m)
        if not self.gravity:
            ph0 = {k: 0 * r for k in ('r', 't', 'rr', 'rt', 'tt')}
        p['ph0'] = ph0
        p['map'] = m
        p['s'], p['i'] = s2 + 0 * r, i
        p['fixed'] = 1.0 if i in self.fixed_density_layers else 0.0
        p['solid'] = 0.0 if (self.fluid_layers()[i] and not self.dyn_fluid) else 1.0
        p['mixed'] = bool(self.fluid_pressure and self.fluid_layers()[i])
        return p

    def _radial(self, i, x):
        """Chebyshev polynomials of layer i and their first two derivatives with respect to s."""
        T, T1, T2 = cheb_basis(x, self.Nl[i])
        sc = 2.0 / (self.model.r1[i] - self.model.r0[i])
        return T[:, :, None], (T1 * sc)[:, :, None], (T2 * sc * sc)[:, :, None]

    @staticmethod
    def _comp(A, ang, mp):
        """Chain-rule derivatives of radial (A, A', A'') x angular (b, b', b'')."""
        A0, A1, A2 = A
        b0, b1, b2 = ang[0], ang[1], ang[2]
        return _D(chain(A0 * b0, A1 * b0, A2 * b0, A0 * b1, A1 * b1, A0 * b2, mp))

    def _fields_at(self, i, x, p, A=None):
        """Per field: (u_r, u_theta, u_lambda, phi1) amplitude-derivative dicts. A: radial parts
        (value, d/ds, d2/ds2), default the Chebyshev polynomials of layer i (_radial)."""
        A = self._radial(i, x) if A is None else A
        mp = p['map']
        out = []
        for typ, n in self.B[i].fields:
            a = self.ang[n]
            Y, mYs = a['Y'], a['mYs']
            neg = [-v for v in mYs]
            if typ == 'U':
                out.append((self._comp(A, Y, mp), ZERO, ZERO, ZERO))
            elif typ == 'P':
                out.append((ZERO, ZERO, ZERO, self._comp(A, Y, mp)))
            elif typ == 'V':
                out.append((ZERO, self._comp(A, Y[1:], mp), self._comp(A, neg, mp), ZERO))
            elif typ == 'Q':                    # no displacement, no potential: Pi rides on the tuple
                f = _QField((ZERO, ZERO, ZERO, ZERO))
                f.q = self._comp(A, Y, mp)
                out.append(f)
            else:   # W
                out.append((ZERO, self._comp(A, neg, mp), self._comp(A, Y[1:], mp), ZERO))
        return out

    # ------------------------------------------------------------------
    def _stress(self, uA, uB, uC, p, jets=True, q=None):
        """Strain and stress amplitudes as jets (needs 2nd derivatives of u). In a layer with a pressure
        unknown (p.mixed) lambda D is replaced by Pi: q of a 'Q' trial field, zero for the others."""
        m = self.m
        if jets:
            mk = lambda d: (J(d.v, d.r, d.t), J(d.r, d.rr, d.rt), J(d.t, d.rt, d.tt))
        else:
            mk = lambda d: (J(d.v), J(d.r), J(d.t))
        A, A_r, A_t = mk(uA)
        B, B_r, B_t = mk(uB)
        Cc, C_r, C_t = mk(uC)
        sn = p.sin
        R = J(p.r, 1.0, 0.0)
        cot = J(p.cot, 0.0, -1.0 / sn**2)
        isn = J(p.isn, 0.0, -p.cos / sn**2)
        lam = J(p.lam, p.lam_r, p.lam_t)
        mu = J(p.mu, p.mu_r, p.mu_t)
        err = A_r
        ett = (B_t + A) / R
        epp = (m * Cc * isn + A + B * cot) / R
        ert = 0.5 * (A_t / R + B_r - B / R)
        erp = 0.5 * ((-m) * A * isn / R + C_r - Cc / R)
        etp = 0.5 * ((C_t - Cc * cot) / R - m * B * isn / R)
        D = err + ett + epp
        if p.get('mixed'):
            lD = J(0.0) if q is None else (J(q.v, q.r, q.t) if jets else J(q.v))
        else:
            lD = lam * D
        return _D(D=D, A=A, B=B, C=Cc, Ar=A_r, At=A_t, Br=B_r, Bt=B_t, Cr=C_r, Ct=C_t,
                  trr=lD + 2 * mu * err, ttt=lD + 2 * mu * ett, tpp=lD + 2 * mu * epp,
                  trt=2 * mu * ert, trp=2 * mu * erp, ttp=2 * mu * etp)

    def operator(self, uA, uB, uC, ph, p, q=None):
        """Residuals E_r, E_theta, E_lambda of the momentum equation and E_phi of Poisson's
        equation for one trial field (displacement amplitudes uA, uB, uC and potential ph, with
        their derivatives) at the points of props p: elasticity, the gravitational terms of the
        hydrostatic reference state, self-gravitation, inertia and the Coriolis force."""
        m, r, cot, isn = self.m, p.r, p.cot, p.isn
        Pi = q                                # (q is reused below for u . grad Phi0)
        e = self._stress(uA, uB, uC, p, q=q)
        trr, ttt, tpp, trt, trp, ttp = e.trr, e.ttt, e.tpp, e.trt, e.trp, e.ttp
        div_r = trr.r + trt.t / r + m * trp.v * isn / r \
            + (2 * trr.v - ttt.v - tpp.v + trt.v * cot) / r
        div_t = trt.r + ttt.t / r + m * ttp.v * isn / r \
            + ((ttt.v - tpp.v) * cot + 3 * trt.v) / r
        div_p = trp.r + ttp.t / r - m * tpp.v * isn / r \
            + (3 * trp.v + 2 * ttp.v * cot) / r
        ph0 = p.ph0
        R = J(r, 1.0, 0.0)
        gr = J(ph0['r'], ph0['rr'], ph0['rt'])
        gt = J(ph0['t'], ph0['rt'], ph0['tt']) / R
        q = e.A * gr + e.B * gt
        rho, w2, D = p.rho, self.omega_f**2, e.D.v
        Er = div_r - rho * q.r + rho * D * gr.v - rho * ph.r + rho * w2 * e.A.v
        Et = div_t - rho * q.t / r + rho * D * gt.v - rho * ph.t / r + rho * w2 * e.B.v
        Ep = div_p + rho * m * q.v * isn / r + rho * m * ph.v * isn / r + rho * w2 * e.C.v
        if self.coriolis or self.omega_t:
            # fields are Re[(A, B, -iC) exp(i(omega t + m lambda))]; the terms
            # rho omega^2 u - 2 i omega rho Omega x u become real in (A, B, C)
            rs_, wt = rho * p.solid, self.omega_t
            cw = 2.0 * wt * self.coriolis
            sn, cs = p.sin, p.cos
            Er = Er + rs_ * (wt**2 * e.A.v + cw * sn * e.C.v)
            Et = Et + rs_ * (wt**2 * e.B.v + cw * cs * e.C.v)
            Ep = Ep + rs_ * (wt**2 * e.C.v + cw * (sn * e.A.v + cs * e.B.v))
        if p.get('tau') is not None:
            Cr, Ct, Cp = self._nonhydro_terms(e, p, gr, gt)
            Er, Et, Ep = Er + Cr, Et + Ct, Ep + Cp
        Eg = ph.rr + 2 * ph.r / r + (ph.tt + cot * ph.t) / r**2 - (m * isn / r)**2 * ph.v \
            + 4 * np.pi * self.G * (rho * D + p.rho_r * e.A.v + p.rho_t / r * e.B.v)
        if p.get('fixed', 0.0):              # Eulerian density fixed: drop div(rho u) terms
            dru = rho * D + p.rho_r * e.A.v + p.rho_t / r * e.B.v
            Er = Er - dru * gr.v
            Et = Et - dru * gt.v
            Eg = Eg - 4 * np.pi * self.G * dru
        if p.get('mixed'):                    # fifth residual, the rows of Q: div u - Pi / lambda
            return Er, Et, Ep, Eg, D - (0.0 if Pi is None else Pi.v) / p.lam
        return Er, Et, Ep, Eg

    def _nonhydro_terms(self, e, p, gr, gt):
        """C(u) (see __init__, nonhydro) at the points of props p. Amplitudes: u_r, u_theta and
        scalars ~ cos(m lambda), u_lambda ~ sin(m lambda); d/dlambda of a cos amplitude X is
        -m X (sin), of a sin amplitude Y is +m Y (cos). tau, b, rho, Phi0 are axisymmetric."""
        m, r = self.m, p.r
        R = J(r, 1.0, 0.0)
        cot = J(p.cot, 0.0, -1.0 / p.sin**2)
        isn = J(p.isn, 0.0, -p.cos / p.sin**2)
        A, B, C = e.A, e.B, e.C                   # jets (value, d/dr, d/dtheta)
        Aj, Bj, Cj = A, B, C
        # displacement gradient G_ij = d_j u_i as jets (needs the second derivatives in e)
        G = {'rr': e.Ar, 'rt': (e.At - Bj) / R, 'rl': (-m * Aj * isn - Cj) / R,
             'tr': e.Br, 'tt': (e.Bt + Aj) / R, 'tl': (-m * Bj * isn - Cj * cot) / R,
             'lr': e.Cr, 'lt': e.Ct / R, 'll': (m * Cj * isn + Aj + Bj * cot) / R}
        T = p['tau']
        z = J(0.0)
        tau = {'rr': T['rr'], 'tt': T['tt'], 'll': T['pp'], 'rt': T['rt'], 'tr': T['rt'],
               'rl': z, 'lr': z, 'tl': z, 'lt': z}
        D = G['rr'] + G['tt'] + G['ll']
        om = {a + c: 0.5 * (G[a + c] - G[c + a]) for a in 'rtl' for c in 'rtl'}
        P = {}
        for a in 'rtl':
            for c in 'rtl':
                v = tau[a + c] * D
                for k in 'rtl':
                    v = v - tau[a + k] * G[c + k] + om[a + k] * tau[k + c] - tau[a + k] * om[k + c]
                P[a + c] = v
        # divergence d_j P_ij; lambda derivative from the parity (count of 'l' in ij)
        dl = lambda key: (m if key.count('l') % 2 else -m) * P[key].v
        s_ = p.sin
        Cr = P['rr'].r + P['rt'].t / r + dl('rl') / (r * s_) \
            + (2 * P['rr'].v - P['tt'].v - P['ll'].v + p.cot * P['rt'].v) / r
        Ct = P['tr'].r + P['tt'].t / r + dl('tl') / (r * s_) \
            + (2 * P['tr'].v + P['rt'].v + p.cot * (P['tt'].v - P['ll'].v)) / r
        Cp = P['lr'].r + P['lt'].t / r + dl('ll') / (r * s_) \
            + (2 * P['lr'].v + P['rl'].v + p.cot * (P['lt'].v + P['tl'].v)) / r
        # equilibrium form: with b = div tau, grad(u.b) - (u.grad) b + (density terms) reduces to
        # (grad u)^T b when the state is in equilibrium (curl b = curl delta(rho grad Phi0)); this
        # form is exactly zero for a rigid rotation whatever tau is (as the hydrostatic form), so
        # the O(eps x excess) equilibrium error of the data cannot feed the tilt-over mode
        # b = div tau from the jets of tau at THESE points (r, cot of this map): the identity
        # div P1 = omega b for a rigid rotation then holds exactly
        cot0 = p.cot
        br = T['rr'].r + T['rt'].t / r + (2 * T['rr'].v - T['tt'].v - T['pp'].v + cot0 * T['rt'].v) / r
        bt = T['rt'].r + T['tt'].t / r + ((T['tt'].v - T['pp'].v) * cot0 + 3 * T['rt'].v) / r
        Dv = D.v
        Cr = Cr - Dv * br + G['rr'].v * br + G['tr'].v * bt
        Ct = Ct - Dv * bt + G['rt'].v * br + G['tt'].v * bt
        Cp = Cp + G['rl'].v * br + G['tl'].v * bt
        return Cr, Ct, Cp

    def normal(self, p):
        """Unit normal (r, theta components) of the level surface at the points of props p."""
        nr, nt = p.r, -p.R_t
        h = np.hypot(nr, nt)
        return nr / h, nt / h

    def interface(self, uA, uB, uC, ph, p, eulerian=False, q=None):
        """n.tau (r, theta, lambda) and gravity flux n.grad phi1 + 4 pi G rho u.n; eulerian=True:
        n.(tau - rho0 g u.n I) and n.grad phi1 (see eulerian_interfaces)."""
        e = self._stress(uA, uB, uC, p, jets=False, q=q)
        nr, nt = self.normal(p)
        tr = e.trr.v * nr + e.trt.v * nt
        tt = e.trt.v * nr + e.ttt.v * nt
        tp = e.trp.v * nr + e.ttp.v * nt
        un = e.A.v * nr + e.B.v * nt
        if eulerian:
            gn = p.ph0['r'] * nr + p.ph0['t'] / p.r * nt
            return tr - p.rho * gn * un * nr, tt - p.rho * gn * un * nt, tp, ph.r * nr + ph.t / p.r * nt
        fl = ph.r * nr + ph.t / p.r * nt + 4 * np.pi * self.G * p.rho * un
        if p.get('tau') is not None:            # initial-stress traction P1(tau).n (nonhydro)
            # solid side: with the normal of the surface on which the state's own traction balance
            # tau.n was imposed (geometry.base, if any): tau jumps at interfaces, so tau.(n - n_bg)
            # would add a second-order term that breaks the exact force-freeness of a rigid rotation;
            # fluid side: the true normal (see p1_normal in __init__)
            nb_ = None if (self.p1_normal == 'fluid_true' and self._isotropic(p['tau'])) \
                else self._normal_bg(p)
            a_, b_, c_ = self._p1_traction(uA, uB, uC, p, *(nb_ if nb_ is not None else (nr, nt)))
            tr, tt, tp = tr + a_, tt + b_, tp + c_
        return tr, tt, tp, fl

    @staticmethod
    def _isotropic(T):
        """Is the initial stress tau = -q I at these points (the fluid side of a boundary, also
        a viscous core with welded walls)?"""
        rr = T['rr'].v
        sc = max(np.max(np.abs(rr)), 1e-300)
        return (np.max(np.abs(T['tt'].v - rr)) <= 1e-12 * sc and np.max(np.abs(T['pp'].v - rr)) <= 1e-12 * sc
                and np.max(np.abs(T['rt'].v)) <= 1e-12 * sc)

    def _normal_bg(self, p):
        """Unit normal of the background map (geometry.base) at the points of props p, or None."""
        base = getattr(self.geom, 'base', None)
        if base is None or p.get('s') is None:
            return None
        th = np.arccos(p.cos)
        if hasattr(base, 'map_derivs'):
            mp = mapping_from_R(*base.map_derivs(p['i'], p['s'], th))
        else:
            mp = mapping(p['s'], th, *base.eps(p['i'], p['s']))
        h = np.hypot(mp['R'], mp['R_t'])
        return mp['R'] / h, -mp['R_t'] / h

    def _p1_traction(self, uA, uB, uC, p, nr, nt):
        """P1(tau).n at boundary points (values only), P1 = tau D - tau G^T + omega tau - tau omega."""
        m, r = self.m, p.r
        isn, cot = p.isn, p.cot
        A, B, C = uA.v, uB.v, uC.v
        G = {'rr': uA.r, 'rt': (uA.t - B) / r, 'rl': (-m * A * isn - C) / r,
             'tr': uB.r, 'tt': (uB.t + A) / r, 'tl': (-m * B * isn - C * cot) / r,
             'lr': uC.r, 'lt': uC.t / r, 'll': (m * C * isn + A + B * cot) / r}
        T = p['tau']
        tau = {'rr': T['rr'].v, 'tt': T['tt'].v, 'll': T['pp'].v, 'rt': T['rt'].v, 'tr': T['rt'].v,
               'rl': 0.0, 'lr': 0.0, 'tl': 0.0, 'lt': 0.0}
        D = G['rr'] + G['tt'] + G['ll']
        om = {a + c: 0.5 * (G[a + c] - G[c + a]) for a in 'rtl' for c in 'rtl'}
        def P(a, c):
            v = tau[a + c] * D
            for k in 'rtl':
                v = v - tau[a + k] * G[c + k] + om[a + k] * tau[k + c] - tau[a + k] * om[k + c]
            return v
        return (P('r', 'r') * nr + P('r', 't') * nt, P('t', 'r') * nr + P('t', 't') * nt,
                P('l', 'r') * nr + P('l', 't') * nt)

    def _band_tests(self, Bi, n):
        """(indices into Bi.fields, basis of those test functions) of the test fields within
        self.band degrees of n, in the field order of Bi (cached per basis and degree)."""
        cache = Bi.__dict__.setdefault('_band_sub', {})
        if n not in cache:
            B = self.band
            keep = [k for k, (_, d) in enumerate(Bi.fields) if abs(d - n) <= B]
            sub = _D(testU=np.array([t for t, d in zip(Bi.testU, Bi.U) if abs(d - n) <= B]).reshape(-1, len(self.w)),
                     testV=[t for t, d in zip(Bi.testV, Bi.V) if abs(d - n) <= B],
                     testW=[t for t, d in zip(Bi.testW, Bi.W) if abs(d - n) <= B],
                     testP=np.array([t for t, d in zip(Bi.testP, Bi.P) if abs(d - n) <= B]).reshape(-1, len(self.w)),
                     testQ=np.array([t for t, d in zip(Bi.testQ, Bi.Q) if abs(d - n) <= B]).reshape(-1, len(self.w)))
            cache[n] = (np.array(keep), sub)
        return cache[n]

    def _project(self, Er, Et, Ep, Eg, B=None, EQ=None):
        """Residual arrays (..., Nth) -> stacked equations (..., nfld) on the test functions of the
        basis B (default: the outermost layer's). EQ: residual of the Q rows (fluid_pressure), zero if None."""
        B = self.B[-1] if B is None else B
        if EQ is None:
            Er, Et, Ep, Eg = np.broadcast_arrays(Er, Et, Ep, Eg)
            EQ = 0 * Er
        else:
            Er, Et, Ep, Eg, EQ = np.broadcast_arrays(Er, Et, Ep, Eg, EQ)
        cols = [np.einsum('...j,kj->...k', Er, B.testU)]
        for tests in (B.testV, B.testW):
            if tests:
                cols.append(np.stack([Et @ a + Ep @ b for a, b in tests], axis=-1))
        cols.append(np.einsum('...j,kj->...k', Eg, B.testP))
        tQ = B['testQ'] if 'testQ' in B else B.__dict__.get('testQ', ())    # _D: keyword or attribute
        if len(tQ):
            cols.append(np.einsum('...j,kj->...k', EQ, tQ))
        return np.concatenate(cols, axis=-1)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _bnd(self, i, x, p):
        """Boundary quantities of every trial field of layer i at end x = +-1."""
        nr, nt = self.normal(p)
        out = []
        if self.slip_nonlevel:
            md = self.model
            ifl = i if self.true_fluid[i] else (i - 1 if x < 0 else i + 1)     # the fluid side of this wall
            rho_f = float(md.rho[ifl](md.r0[i] if x < 0 else md.r1[i]))
            dpw_t = -rho_f * (p.ph0['r'] * p.R_t + p.ph0['t'])
        for uA, uB, uC, ph in self._fields_at(i, np.array([x]), p):
            e = self._stress(uA, uB, uC, p, jets=False)
            tr = e.trr.v * nr + e.trt.v * nt
            tt = e.trt.v * nr + e.ttt.v * nt
            tp = e.trp.v * nr + e.ttp.v * nt
            tnn = tr * nr + tt * nt
            un = e.A.v * nr + e.B.v * nt
            fl = ph.r * nr + ph.t / p.r * nt + 4 * np.pi * self.G * p.rho * un
            if self.true_fluid[i]:            # fluid: tangential momentum equation
                _, Et, Ep, _ = self.operator(uA, uB, uC, ph, p)
            else:                              # solid: tangential traction
                Et, Ep = tt - tnn * nt, tp
            if p.get('slip') is not None:
                # nonhydro, slipping boundary (born.slip_terms, notes_loading 6.3): the fluid pressure
                # is sampled where the solid has slid to (u.grad_S dp on both sides of n.tau.n), and
                # the solid's tangential traction balances tau0.eps.n, tau0 = tau + dp I
                # on the flattened boundary: slip along the tangent dR/dtheta|_s = R_t r + R theta
                # (dp_t is d/dtheta along the boundary), and the boundary normal (nr, nt)
                dp, dp_t = p['slip']
                if not self.slip_nonlevel:        # (with slip_nonlevel: the complete term below)
                    tnn = tnn + (e.B.v + e.A.v * p.R_t / p.r) * dp_t / np.hypot(p.r, p.R_t)
                if not self.true_fluid[i]:
                    T, mu, lam = p['tau'], p.mu, p.lam
                    D = e.D.v
                    err = (e.trr.v - lam * D) / (2 * mu)
                    ett = (e.ttt.v - lam * D) / (2 * mu)
                    ert, erp, etp = e.trt.v / (2 * mu), e.trp.v / (2 * mu), e.ttp.v / (2 * mu)
                    vr, vt, vp = err * nr + ert * nt, ert * nr + ett * nt, erp * nr + etp * nt
                    t0 = (T['rr'].v + dp, T['tt'].v + dp, T['pp'].v + dp, T['rt'].v)
                    wr = t0[0] * vr + t0[3] * vt
                    wt = t0[3] * vr + t0[1] * vt
                    wp = t0[2] * vp
                    wn = wr * nr + wt * nt
                    Et = Et - (wt - wn * nt)
                    Ep = Ep - wp
            if self.slip_nonlevel:
                # wall off the level surface of Phi0: the normal term with dp the pressure of the fluid at
                # rest along the wall, d(dp)/dtheta = -rho_fluid dPhi0/dtheta along the wall (Phi0 and its
                # derivative along the wall are continuous: either side's ph0 serves). Hydrostatic state:
                # the wall of the first-order map, off level at O(eps^2). Non-hydrostatic state: Phi0
                # contains its loads and the map its deflections, so this is the whole slope of its wall and
                # replaces the state's own dp_t above (which lacks the O(eps^2) of the background; both
                # together would count the slope twice). The solid's tangential term (tau0 . eps . n) is
                # second order in the hydrostatic state and left out there.
                tnn = tnn + (e.B.v + e.A.v * p.R_t / p.r) * dpw_t / np.hypot(p.r, p.R_t)
            out.append(dict(un=un, tnn=tnn, Et=Et, Ep=Ep, fl=fl, phi=ph.v))
        return out

    def _forced(self, i):
        """Does the tidal body force act in layer i?"""
        return self.forcing == 'tide' and (self.force_layers is None or i in self.force_layers)

    def _tidal_force_rows(self, p):
        """Projected tidal body force rho grad(phiT) at the points of props p."""
        md, nfo, m = self.model, self.nf, self.m
        a, ga = md.a, md.g_surface
        Yf = ybar(nfo, m, self.theta[0, 0], 1)
        rr = p.r
        phT = -ga * a * (rr / a)**nfo * Yf[0]
        return self._project(p.rho * nfo * phT / rr,
                             p.rho * (-ga * (rr / a)**(nfo - 1) * Yf[1]),
                             p.rho * (-m) * phT * p.isn / rr, 0 * rr, self.B[p['i']])

    def _fs_interface(self, H, f, iL, rows_b, rows_t, col):
        """Rows for a fluid-solid interface between layer iL (below) and iL+1.

        Bottom rows of the upper layer: normal displacement and phi1 continuous,
        tangential condition of the upper layer. Top rows of the lower layer:
        normal traction and gravity flux continuous, tangential condition of the
        lower layer. Tangential condition: zero tangential traction (solid) or the
        tangential momentum equation at the boundary node (fluid). Valid on level
        surfaces (Tromp & Mitrovica 1999, eq. 3.7).
        """
        md = self.model
        iU, rI = iL + 1, md.r1[iL]
        NL, NU = self.Nl[iL], self.Nl[iU]
        pL, pU = self.props(iL, np.array([rI])), self.props(iU, np.array([rI]))
        if self.nonhydro is not None and hasattr(self.nonhydro, 'slip'):
            ifl = iL if self.true_fluid[iL] else iU
            sl = self.nonhydro.slip(ifl, rI, self.theta[0])
            pL['slip'] = pU['slip'] = sl
        BL, BU = self._bnd(iL, 1.0, pL), self._bnd(iU, -1.0, pU)
        GL, GU = self.B[iL], self.B[iU]               # rows_t on the tests of iL, rows_b on iU's
        # U degrees of the lower layer that the upper layer lacks: normal displacement continuous
        # (by projection) instead of normal traction (_extra_rows)
        xr = self._extra_rows(iL)
        zero = 0.0
        for j, b in enumerate(BL):
            H[rows_b, col(iL, j):col(iL, j) + NL] = self._project(b['un'], zero, zero, b['phi'], GU)[:, 0, :].T
            Q = self._project(b['tnn'], b['Et'], b['Ep'], b['fl'], GL)[:, 0, :].T
            if xr:
                Q[xr] = self._project(b['un'], zero, zero, 0 * b['phi'], GL)[:, 0, :].T[xr]
            H[rows_t, col(iL, j):col(iL, j) + NL] = Q
        for j, b in enumerate(BU):
            H[rows_b, col(iU, j):col(iU, j) + NU] = self._project(-b['un'], b['Et'], b['Ep'], -b['phi'], GU)[:, 0, :].T
            Q = self._project(-b['tnn'], zero, zero, -b['fl'], GL)[:, 0, :].T
            if xr:
                Q[xr] = self._project(-b['un'], zero, zero, 0 * b['phi'], GL)[:, 0, :].T[xr]
            H[rows_t, col(iU, j):col(iU, j) + NU] = Q
        # tidal force in the fluid tangential rows
        for lay, pp, rows in ((iL, pL, rows_t), (iU, pU, rows_b)):
            if self.true_fluid[lay] and self._forced(lay):
                nS0, nS1 = len(self.B[lay].U), self.B[lay].nS
                fr = self._tidal_force_rows(pp)[0]
                f[rows.start + nS0: rows.start + nS1] = fr[nS0:nS1]

    def _extra_rows(self, i, welded=False):
        """Top-row slots of layer i whose traction condition has no partner in layer i + 1 (fluid
        degrees the solid above lacks): U slots (slip) or U, V, W slots (welded). Below the solid
        (inner-core boundary) the bottom rows of the fluid already carry the displacement
        conditions, so only the top of a fluid layer needs this."""
        if i + 1 >= self.model.n_layers:
            return []
        lo, up = self.B[i], self.B[i + 1]
        if lo is up:
            return []
        types = ('U', 'V', 'W') if welded else ('U',)
        return [r for r, fld in enumerate(lo.fields) if fld[0] in types and fld not in up.fields]

    # ------------------------------------------------------------------
    # Assembly of the linear system (manual, 'Assembling the linear system'). Unknowns: per
    # layer i and field j the N Chebyshev coefficients (columns col(i, j) .. + N), then the K
    # exterior coefficients c_n, then one rigid-rotation force per Tisserand region. Rows per
    # layer: nfld bottom rows, nfld * (N - 2) interior rows, nfld top rows.
    def solve(self):
        """Assemble the system, solve it and return results()."""
        H, f = self._allocate()
        for i in range(self.model.n_layers):
            self._interior_rows(H, f, i)
            self._bottom_rows(H, i)
            self._top_rows(H, f, i)
        for ir, lays in enumerate(self.regions):
            H[self._c_alpha + ir, :self._ntot] = self._tisserand_row(lays)
        if isinstance(H, _HybridMatrix):
            self._Hh, self._Hsp = H.finish(), None
            self._band_info = H.band_info()
            x = self._solve_linear(H, f)
        else:
            self._Hh, self._Hsp = None, H.tocsr()
            x = self._solve_linear(self._Hsp, f)
        self.coef, self.f = x, f
        return self.results()

    def _allocate(self):
        """Empty matrix and right-hand side; index helpers of the layout."""
        md, K = self.model, len(self.degP)
        self._nbl = [B.nfld * n for B, n in zip(self.B, self.Nl)]   # unknowns per layer
        self._off = np.concatenate([[0], np.cumsum(self._nbl)]).astype(int).tolist()
        self._ntot = self._off[-1]                               # all layer unknowns
        # unknowns per layer when N is uniform (born.py, nonhydro.py index with it); None otherwise
        self._nb = self._nbl[0] if len(set(self._nbl)) == 1 else None
        self._nS = self.B[-1].nS                                 # traction slots at the surface
        self._c_alpha = self._ntot + K                           # first rigid-rotation column
        ncol = self._c_alpha + len(self.regions)
        dt = complex if getattr(md, 'is_complex', False) else float
        if self.assembly == 'hybrid':
            band = None
            if self.band is not None:
                rd, cd = np.empty(self._ntot, int), np.empty(self._ntot, int)
                for i, (Bi, N) in enumerate(zip(self.B, self.Nl)):
                    n = np.array([d for _, d in Bi.fields])
                    a, b = self._off[i], self._off[i + 1]
                    rd[a:b] = n[np.arange(b - a) % len(n)]     # rows: bottom, interior nodes, top; by field
                    cd[a:b] = np.repeat(n, N)                  # columns: N coefficients per field
                band = (self.band, rd, cd)
            return _HybridMatrix((ncol, ncol), dt, self._off, band), np.zeros(ncol, dt)
        return _BlockMatrix((ncol, ncol), dt), np.zeros(ncol, dt)

    @property
    def Hsp(self):
        """System matrix as CSR (unscaled). With the hybrid assembly it is built on first use (the
        solve does not need it), at the memory cost of the sparse assembly."""
        if self._Hsp is None and self._Hh is not None:
            self._Hsp = self._Hh.tocsr()
        return self._Hsp

    @Hsp.setter
    def Hsp(self, A):
        self._Hsp = A

    @property
    def H(self):
        """Dense copy of the system matrix (born.py, diagnostics); the solve itself uses the sparse
        self.Hsp. Refused above EARTHTIDES_MAX_GB (default 8)."""
        _check_dense(self.Hsp.shape[0], self.Hsp.dtype)
        return self.Hsp.toarray()

    def _col(self, i, j):
        """First column of field j of layer i."""
        return self._off[i] + j * self.Nl[i]

    def layer_coef(self, i, j, x=None):
        """Chebyshev coefficients of field j in layer i (of the solution, or of the vector x)."""
        c0 = self._col(i, j)
        return (self.coef if x is None else x)[c0:c0 + self.Nl[i]]

    def _interior_rows(self, H, f, i):
        """Collocation at the N - 2 interior Gauss-Lobatto nodes of layer i: the projected
        residual of every trial field, the tidal force and the rigid-rotation force."""
        md, N, nfld, m = self.model, self.Nl[i], self.B[i].nfld, self.m
        col, row0 = self._col, self._off[i]
        Bi = self.B[i]
        r0, r1 = md.r0[i], md.r1[i]
        xin = -np.cos(np.pi * np.arange(1, N - 1) / (N - 1))
        s = r0 + (xin + 1) * (r1 - r0) / 2
        p = self.props(i, s)
        # the operator is linear in the radial part (A, A', A'') at each node: evaluate it on the
        # three unit jets and combine with the polynomials, sum_d T_k^(d)(s) M_d(s) (N / 3 times
        # less work than evaluating every polynomial; equal to round-off)
        T = [a[:, :, 0] for a in self._radial(i, xin)]               # 3 x (N, Ns)
        unit = np.eye(3)[:, :, None, None] * np.ones((1, 1, len(xin), 1))
        A3 = (unit[0], unit[1], unit[2])                              # each (3, Ns, 1)
        node_rows = row0 + nfld * (1 + np.arange(N - 2))
        for j, fl in enumerate(self._fields_at(i, xin, p, A3)):
            # band: only the test functions within band of the trial degree (the others are dropped
            # anyway; plan_banded_coupling_2026-11-02, step 4)
            idx, Bt = (None, Bi) if self.band is None else self._band_tests(Bi, Bi.fields[j][1])
            E = self.operator(*fl, p, q=getattr(fl, 'q', None))
            M = self._project(*E[:4], Bt, EQ=E[4] if len(E) > 4 else None)    # (3, Ns, ntest): d = value, d/ds, d2/ds2
            Q = T[0][:, :, None] * M[0] + T[1][:, :, None] * M[1] + T[2][:, :, None] * M[2]
            if idx is None:
                # rows of node k are row0 + nfld (1 + k) ..: all N - 2 nodes form one contiguous block
                H[row0 + nfld:row0 + nfld * (N - 1), col(i, j):col(i, j) + N] = \
                    Q.transpose(1, 2, 0).reshape((N - 2) * nfld, N)
            else:
                H[(node_rows[:, None] + idx[None, :]).ravel(), col(i, j):col(i, j) + N] = \
                    Q.transpose(1, 2, 0).reshape((N - 2) * len(idx), N)
        Qf = self._tidal_force_rows(p)
        reg = [ir for ir, lays in enumerate(self.regions) if i in lays]
        if reg:
            rr = p.r
            if m == 1:      # rigid rotation about the y axis
                Qa = self._project(0 * rr, p.rho * rr, -p.rho * rr * p.cos, 0 * rr, Bi)
            else:           # m = 0: rigid rotation about the spin axis
                Qa = self._project(0 * rr, 0 * rr, p.rho * rr * p.sin, 0 * rr, Bi)
        for k in range(N - 2):
            rows = slice(row0 + nfld * (1 + k), row0 + nfld * (2 + k))
            if self._forced(i):
                f[rows] = Qf[k]
            if reg:
                H[rows, self._c_alpha + reg[0]] = Qa[k]

    def _harmonic_flux(self, p, n, s0, outward):
        """n.grad of the harmonic continuation of a unit potential degree n beyond the level surface s0
        (solid_P): exterior (s0 / s)^(n+1) above (outward), interior (s / s0)^n below; at the points of
        props p on s0, as the flux in interface(): phi.r n_r + phi.t / r n_t."""
        e = -(n + 1) if outward else n
        A = (np.ones((1, 1, 1)), np.full((1, 1, 1), e / s0), np.full((1, 1, 1), e * (e - 1) / s0**2))
        ph = self._comp(A, self.ang[n]['Y'], p['map'])
        nr, nt = self.normal(p)
        return ph.r * nr + ph.t / p.r * nt

    def _bottom_rows(self, H, i):
        """Bottom rows of layer i: regularity at the centre (U = V = W = 0; P' = 0, but P = 0 for
        degree 1, where the regular potential grows as s) or continuity of every field with layer
        i - 1 (welded interface); a liquid-solid interface is filled by _fs_interface from the top
        rows of layer i - 1. For degree 1, centre='fixed' (U = V = 0) also fixes the frame: the
        centre does not move (exact for static runs, where a translation is free); centre='free'
        imposes U = V and dU/ds = 0, the regular behaviour U, V = const + O(s^2), and leaves the
        translation to inertia (manual, 'Degree 1')."""
        md, N, nfld = self.model, self.Nl[i], self.B[i].nfld
        col, rb = self._col, self._off[i]
        flds = self.B[i].fields
        if i == 0:
            T, T1, _ = cheb_basis(-1.0, N)
            sc = 2.0 / (md.r1[i] - md.r0[i])
            free = self.centre == 'free' and ('U', 1) in flds and ('V', 1) in flds
            jU, jV = (flds.index(('U', 1)), flds.index(('V', 1))) if free else (None, None)
            for j, (typ, n) in enumerate(flds):
                if typ == 'Q':
                    H[rb + j, col(i, j):col(i, j) + N] = _unit(N, N - 1)
                    continue
                if j != jV:                     # (row jV of a free centre is dU_1/ds = 0, below)
                    H[rb + j, col(i, j):col(i, j) + N] = (T1[:, 0] * sc) if (typ == 'P' and n != 1) else T[:, 0]
            if free:
                H[rb + jU, col(i, jV):col(i, jV) + N] = -T[:, 0]          # U_1 - V_1 = 0
                H[rb + jV, col(i, jU):col(i, jU) + N] = T1[:, 0] * sc     # dU_1/ds = 0
        elif self.true_fluid[i - 1] != self.true_fluid[i]:
            pass                       # filled by _fs_interface (layer i-1)
        else:
            Np = self.Nl[i - 1]
            Tm, _, _ = cheb_basis(-1.0, N)
            Tp, _, _ = cheb_basis(1.0, Np)
            below = self.B[i - 1].fields
            Fb = FI = None
            for j, fld in enumerate(flds):
                if fld[0] == 'Q':
                    # Pi has no boundary condition and two coefficients less than the displacement (the
                    # P_N / P_(N-2) pair): its 'bottom' row sets the highest coefficient to zero, its
                    # 'top' row the one below (_top_rows)
                    H[rb + j, col(i, j):col(i, j) + N] = _unit(N, N - 1)
                    continue
                if fld[0] == 'P' and fld not in below:
                    # solid_P: no potential of this degree in the solid below. Flux continuity with its
                    # interior harmonic field P_n(s0) (s / s0)^n instead of P_n = 0 at the wall
                    if Fb is None:
                        s0 = md.r0[i]
                        pb, pI = self.props(i, np.array([s0])), self.props(i - 1, np.array([s0]))
                        Fb = [self.interface(*fl, pb)[3] for fl in self._fields_at(i, np.array([-1.0]), pb)]
                        FI = [self.interface(*fl, pI)[3] for fl in self._fields_at(i - 1, np.array([1.0]), pI)]
                    tk = self.B[i].testP[self.B[i].P.index(fld[1])]
                    ext = (self._harmonic_flux(pI, fld[1], md.r0[i], False) @ tk)[0, 0] * Tm[:, 0]
                    for jj, F in enumerate(Fb):
                        v = np.broadcast_to(F, (N,) + np.shape(F)[-2:])[:, 0, :] @ tk
                        H[rb + j, col(i, jj):col(i, jj) + N] = v - ext if jj == j else v
                    for jj, F in enumerate(FI):
                        v = np.broadcast_to(F, (Np,) + np.shape(F)[-2:])[:, 0, :] @ tk
                        H[rb + j, col(i - 1, jj):col(i - 1, jj) + Np] = -v
                    continue
                # continuity with the same field below; a fluid degree the solid below lacks is
                # zero at the wall (the solid does not deform at that degree)
                if fld in below:
                    jb = below.index(fld)
                    H[rb + j, col(i - 1, jb):col(i - 1, jb) + Np] = Tp[:, 0]
                H[rb + j, col(i, j):col(i, j) + N] = -Tm[:, 0]

    def _top_rows(self, H, f, i):
        """Top rows of layer i: continuity of traction and gravity flux with layer i + 1 (welded),
        the liquid-solid conditions, or the free surface (_surface_rows)."""
        md, N, Bi = self.model, self.Nl[i], self.B[i]
        nfld = Bi.nfld
        col = self._col
        rt = self._off[i + 1] - nfld
        last = (i == md.n_layers - 1)
        if not last and self.true_fluid[i] != self.true_fluid[i + 1]:
            self._fs_interface(H, f, i, slice(self._off[i + 1], self._off[i + 1] + self.B[i + 1].nfld),
                               slice(rt, rt + nfld), col)
            return
        nrow = Bi.nS if last else nfld
        # welded top of a fluid layer below a solid with fewer degrees: rigid wall for the extra
        # U, V, W degrees instead of traction continuity (_extra_rows)
        xr = [] if last else self._extra_rows(i, welded=True)
        eul = (self.eulerian_interfaces and not last and i in self.fixed_density_layers
               and i + 1 in self.fixed_density_layers)
        r1 = md.r1[i]
        pt = self.props(i, np.array([r1]))
        Ft = [self.interface(*fl, pt, eul, q=getattr(fl, 'q', None))
              for fl in self._fields_at(i, np.array([1.0]), pt)]
        Tp1 = cheb_basis(1.0, N)[0][:, 0] if xr else None
        # solid_P: potential degrees the solid above lacks: the flux of their exterior harmonic field
        # P_n(s1) (s1 / s)^(n+1) is the solid side of the flux row
        hx = {} if last else {j: self._project(0, 0, 0, self._harmonic_flux(
            self.props(i + 1, np.array([r1])), n, r1, True), Bi)[0, 0][:nrow]
            for j, (typ, n) in enumerate(Bi.fields) if typ == 'P' and (typ, n) not in self.B[i + 1].fields}
        for j, F in enumerate(Ft):
            Q = self._project(*F, Bi)[:, 0, :].T[:nrow]
            if Bi.fields[j][0] == 'Q' and j < nrow:
                Q[j] = _unit(N, N - 2)
            if j in hx:
                Q = Q - hx[j][:, None] * cheb_basis(1.0, N)[0][:, 0][None, :]
            if xr:
                Q[xr] = 0.0
                if j in xr:
                    Q[j] = Tp1                      # field j of degree the solid lacks: zero at the wall
            H[rt:rt + nrow, col(i, j):col(i, j) + N] = Q
        if not last:
            pn = self.props(i + 1, np.array([r1]))
            Fn = [self.interface(*fl, pn, eul, q=getattr(fl, 'q', None))
                  for fl in self._fields_at(i + 1, np.array([-1.0]), pn)]
            for j, F in enumerate(Fn):
                Q = -self._project(*F, Bi)[:, 0, :].T
                if xr:
                    Q[xr] = 0.0
                H[rt:rt + nfld, col(i + 1, j):col(i + 1, j) + self.Nl[i + 1]] = Q
        else:
            self._surface_rows(H, f, i, rt, pt, Ft)

    def _surface_rows(self, H, f, i, rt, pt, Ft):
        """Free surface: zero traction (or the weight of a load), phi1 continuous with the
        exterior potential sum c_n (a/r)^(n+1) Ybar_n, and continuity of the gravity flux (rows
        after the layers), with the surface mass of a load."""
        md, N, Bs = self.model, self.Nl[i], self.B[i]
        K, testP = len(self.degP), Bs.testP
        col, nS, nt0 = self._col, Bs.nS, self._ntot
        a, nfo, m = md.a, self.nf, self.m
        r = pt.r
        nr, nt = self.normal(pt)
        Tp, _, _ = cheb_basis(1.0, N)
        jP0 = nS
        for jj, n in enumerate(self.degP):         # phi1 continuity
            phin = Tp[:, 0][:, None] * self.ang[n]['Y'][0][None, :]
            H[rt + jP0:rt + jP0 + K, col(i, jP0 + jj):col(i, jP0 + jj) + N] = (phin @ testP.T).T
        for jj, n in enumerate(self.degP):         # exterior c_n
            E = (a / r)**(n + 1) * self.ang[n]['Y'][0]
            Er_ = -(n + 1) / r * E
            Et_ = (a / r)**(n + 1) * self.ang[n]['Y'][1] / r
            cc = nt0 + jj
            H[rt + jP0:rt + jP0 + K, cc] = -(E[0] @ testP.T)
            H[nt0:nt0 + K, cc] = -((Er_ * nr + Et_ * nt)[0] @ testP.T)
        for j, F in enumerate(Ft):                 # flux continuity
            fl = np.broadcast_to(F[3], (N,) + np.shape(F[3])[-2:])
            H[nt0:nt0 + K, col(i, j):col(i, j) + N] = \
                np.einsum('nj,kj->kn', fl[:, 0, :], testP)
        if self.forcing == 'load':                 # surface mass sigma = Ybar_nf^m
            sig = ybar(nfo, m, self.theta[0, 0], 0)[0][None, :]
            gn = pt.ph0['r'] * nr + pt.ph0['t'] / r * nt
            tl = 0 * r
            if self.load_inertia and self.coriolis and self.omega_t:
                # -(dLambda/dt) Omega x r, eastward; amplitude in the (A, B, -iC) convention
                tl = self.omega_t * self.coriolis * r * pt.sin * sig
            Ql = self._project(-gn * sig * nr, -gn * sig * nt, tl, 0 * r, Bs)[0]
            f[rt:rt + nS] = Ql[:nS]
            f[nt0:nt0 + K] = -4 * np.pi * self.G * (sig[0] @ testP.T)

    def _solve_linear(self, H, f):
        """Scale rows, then columns, to unit maximum (the Tisserand rows are ~1e30 times larger
        and must not dominate the column scaling) and solve: sparse LU (SuperLU, COLAMD) or
        dense LAPACK (manual, 'Solving the system'). H: the hybrid matrix of the assembly (block path
        on its dense blocks; SuperLU / LAPACK through its CSR) or a CSR / dense matrix."""
        if self.linear_solver == 'layers':
            # direct LU by layers with pivoting across the interfaces (_LayerLU): no GMRES, no fallback;
            # iterative refinement with the same factors (the residual before and after in S._layer_info)
            if isinstance(H, _HybridMatrix):
                # from the hybrid matrix itself: no CSR of the whole system; the rows of a layer block go
                # (scaled) straight into the panel, the residual uses the unscaled blocks (_ScaledHybrid)
                rs, cs, irs, ics, Rc = self._scale_hybrid(H)
                off = H.off
                b = f / rs
                dtF = np.result_type(H.dtype, b.dtype)

                def diag_rows(L, rows):
                    i0, i1 = off[L], off[L + 1]
                    B = H.blocks[L]
                    if not isinstance(B, _BandBlock):
                        return (B[rows] * irs[i0:i1][rows, None]) * ics[None, i0:i1]
                    wr, wc = irs[i0:i1][B.rperm], ics[i0:i1][B.cperm]
                    out = np.empty((len(rows), i1 - i0), dtF)
                    order = np.argsort(B.rpos[rows])             # runs of consecutive rows in degree order
                    ps = B.rpos[rows][order]
                    cut = np.flatnonzero(np.diff(ps) != 1) + 1
                    for a_, b_ in zip(np.r_[0, cut], np.r_[cut, len(ps)]):
                        for c_ in range(a_, b_, 512):
                            e_ = min(c_ + 512, b_)
                            out[order[c_:e_]] = B.dense(ps[c_], ps[e_ - 1] + 1, 0, B.n, wr, wc, dtF)[:, B.cpos]
                    return out

                def diag_cols(L, rows, cols):
                    i0, i1 = off[L], off[L + 1]
                    B = H.blocks[L]
                    if not isinstance(B, _BandBlock):
                        out = B[np.ix_(rows, cols)].astype(dtF)
                    else:
                        pr, pc = B.rpos[rows], B.cpos[cols]
                        T = B.ku + pr[:, None] - pc[None, :]
                        ok = (T >= 0) & (T < B.ab.shape[0])
                        out = np.zeros(T.shape, dtF)
                        out[ok] = B.ab[T[ok], np.broadcast_to(pc[None, :], T.shape)[ok]]
                    out *= irs[i0:i1][rows, None]
                    out *= ics[i0:i1][None, cols]
                    return out

                A = _ScaledHybrid(H, rs, cs)
                banded = self.LAYERS_BANDED == 'always' or (self.LAYERS_BANDED and H.band is not None)
                if banded:                           # banded panels (_BandLayerLU): memory ~K
                    bd = H.band if H.band is not None else (None, None, None)
                    F = _BandLayerLU(Rc, off, H.nL, H.nt, dtF, diag_cols, rdeg=bd[1], cdeg=bd[2], B=bd[0])
                else:
                    F = _LayerLU(Rc, off, H.nL, H.nt, dtF, diag_rows=diag_rows)
                mgb = (Rc.data.nbytes + Rc.indices.nbytes + Rc.indptr.nbytes) / 1e9
            else:
                A, rs, cs = self._scale_csr(sps.csr_matrix(H))
                b = f / rs
                F = _LayerLU(A, self._off, self.model.n_layers, self._ntot, np.result_type(A.dtype, b.dtype))
                mgb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e9
            y = F.solve(b)
            nb = max(np.linalg.norm(b), 1e-300)
            r = b - A @ y
            res0 = res = np.linalg.norm(r) / nb
            steps = 0
            while steps < self.LAYERS_REFINE and res > 1e-13:   # iterative refinement with the same factors
                y1 = y + F.solve(r)
                r1 = b - A @ y1
                res1 = np.linalg.norm(r1) / nb
                if not res1 < 0.5 * res:
                    if res1 < res:
                        y, r, res = y1, r1, res1
                        steps += 1
                    break
                y, r, res = y1, r1, res1
                steps += 1
            gbF, kept = F.stored_gb()
            self._layer_info = dict(residual=res, residual_first=res0, refinements=steps, growth=F.growth,
                                    factors_gb=gbF, interface_rows=kept, scaled_copy_gb=mgb)
            self._layer_info['banded'] = isinstance(F, _BandLayerLU)
            if isinstance(F, _BandLayerLU):
                self._layer_info['growth_parts'] = F.growth_parts
            self._solver_used = 'layers'
            return y / cs
        if isinstance(H, _HybridMatrix):
            if self.linear_solver == 'block' and not self.regions:
                try:
                    y = self._solve_hybrid(H, f)
                    self._solver_used = 'block'
                    return y
                except _NotBlock:
                    failed = True
            else:
                failed = False
            if self.band is not None and H.shape[0] > self.BAND_FALLBACK_MAX:
                # SuperLU fill-in of a large banded system would exhaust the memory
                raise RuntimeError('banded system with %d unknowns: block path failed (GMRES residual %s) and '
                                   'the SuperLU fallback is refused above BAND_FALLBACK_MAX = %d; near the FCN? Try '
                                   'linear_solver="layers"' % (H.shape[0], getattr(self, '_gmres_res', '?'),
                                                                         self.BAND_FALLBACK_MAX))
            H = self.Hsp if H is self._Hh else H.tocsr()
            if self.linear_solver != 'dense':
                Hc, rs, cs = self._scale_csr(H)
                self._solver_used = ('sparse (block path failed)' if failed else
                                     'sparse (rotation regions)' if self.linear_solver == 'block' else 'sparse')
                A = Hc.tocsc()
                del Hc
                return spla.splu(A, permc_spec='COLAMD').solve(f / rs) / cs
        if self.linear_solver in ('sparse', 'block'):
            # one copy, scaled in place (each sps product would copy the matrix again)
            Hc = sps.csr_matrix(H, copy=True)
            Hc.sum_duplicates()
            a = np.abs(Hc.data)
            nz = np.diff(Hc.indptr)
            if np.any(nz == 0):
                raise RuntimeError('empty row %s' % np.where(nz == 0)[0][:10])
            rs = np.maximum.reduceat(a, Hc.indptr[:-1])
            del a
            Hc.data *= np.repeat(1.0 / rs, nz)          # = diags(1 / rs) @ H, bit for bit
            cs = np.zeros(Hc.shape[1])
            np.maximum.at(cs, Hc.indices, np.abs(Hc.data))
            cs[cs == 0] = 1.0
            Hc.data *= (1.0 / cs)[Hc.indices]
            # rotation regions (static runs with Tisserand constraints): SuperLU by rule, not by
            # fallback. The block path solves them (equilibrated border), but the rigid-rotation
            # component W stays ~1e-12..2e-10 off (exact 0 on a static sphere; SuperLU 1e-15);
            # so they are routed to SuperLU
            self._solver_used = 'sparse'
            if self.linear_solver == 'block' and not self.regions:
                try:
                    y = self._block_path(Hc, f / rs, rs, cs) / cs
                    self._solver_used = 'block'
                    return y
                except _NotBlock:
                    self._solver_used = 'sparse (block path failed)'
            elif self.linear_solver == 'block':
                self._solver_used = 'sparse (rotation regions)'
            A = Hc.tocsc()
            del Hc
            return spla.splu(A, permc_spec='COLAMD').solve(f / rs) / cs
        if sps.issparse(H):
            _check_dense(H.shape[0], H.dtype)
            H = H.toarray()
        rs = np.abs(H).max(axis=1)
        if np.any(rs == 0):
            raise RuntimeError('empty row %s' % np.where(rs == 0)[0][:10])
        Hs = H / rs[:, None]
        cs = np.abs(Hs).max(axis=0)
        cs[cs == 0] = 1.0
        return sla.solve(Hs / cs, f / rs) / cs

    @staticmethod
    def _scale_csr(H):
        """Row- then column-scaled copy of a CSR matrix (as in _solve_linear): (Hc, rs, cs)."""
        Hc = sps.csr_matrix(H, copy=True)
        Hc.sum_duplicates()
        a = np.abs(Hc.data)
        nz = np.diff(Hc.indptr)
        if np.any(nz == 0):
            raise RuntimeError('empty row %s' % np.where(nz == 0)[0][:10])
        rs = np.maximum.reduceat(a, Hc.indptr[:-1])
        del a
        Hc.data *= np.repeat(1.0 / rs, nz)
        cs = np.zeros(Hc.shape[1])
        np.maximum.at(cs, Hc.indices, np.abs(Hc.data))
        cs[cs == 0] = 1.0
        Hc.data *= (1.0 / cs)[Hc.indices]
        return Hc, rs, cs

    @staticmethod
    def _scale_hybrid(Hh):
        """Row and column scales of the hybrid matrix (as _scale_csr on its CSR, computed block by block) and
        the scaled copy of its part outside the layer blocks: (rs, cs, 1 / rs, 1 / cs, Rc)."""
        R, off, n = Hh.rest, Hh.off, Hh.shape[0]
        nz = np.diff(R.indptr)
        a = np.abs(R.data)
        rs = np.zeros(n)
        k = nz > 0
        if a.size:
            rs[k] = np.maximum.reduceat(a, R.indptr[:-1][k])
        for B, i0, i1 in zip(Hh.blocks, off[:-1], off[1:]):
            if isinstance(B, _BandBlock):
                rs[i0:i1] = np.maximum(rs[i0:i1], B.row_absmax())
            elif B.size:
                rs[i0:i1] = np.maximum(rs[i0:i1], np.abs(B).max(axis=1))
        if np.any(rs == 0):
            raise RuntimeError('empty row %s' % np.where(rs == 0)[0][:10])
        irs = 1.0 / rs
        Rc = R.copy()
        Rc.data *= np.repeat(irs, nz)
        cs = np.zeros(n)
        np.maximum.at(cs, Rc.indices, np.abs(Rc.data))
        for B, i0, i1 in zip(Hh.blocks, off[:-1], off[1:]):
            if isinstance(B, _BandBlock):
                cs[i0:i1] = np.maximum(cs[i0:i1], B.col_absmax(irs[i0:i1]))
            elif B.size:
                cs[i0:i1] = np.maximum(cs[i0:i1], np.abs(B * irs[i0:i1, None]).max(axis=0))
        cs[cs == 0] = 1.0
        ics = 1.0 / cs
        Rc.data *= ics[Rc.indices]
        return rs, cs, irs, ics, Rc

    def _solve_hybrid(self, Hh, f):
        """Block path on the hybrid matrix: the same row and column scaling as the CSR path (entry
        for entry the same operations, x * (1 / rs) * (1 / cs), so the scaled entries are bit-identical),
        computed block by block; the blocks themselves stay unscaled (S.Hsp, diagnostics), the block
        LU factors a scaled copy of each in place, and GMRES multiplies with the unscaled blocks
        (_ScaledHybrid). Raises _NotBlock as _block_path."""
        off = Hh.off
        rs, cs, irs, ics, Rc = self._scale_hybrid(Hh)
        b = f / rs
        dt = np.result_type(Hh.dtype, b.dtype)

        def get_block(i):                            # scaled copy, column-major (getrf in place)
            i0, i1 = off[i], off[i + 1]
            if isinstance(Hh.blocks[i], _BandBlock):  # (factored from the band itself: _BandLU)
                return Hh.blocks[i], irs[i0:i1], ics[i0:i1], dt
            S = np.multiply(Hh.blocks[i], irs[i0:i1, None], order='F', dtype=dt)
            S *= ics[None, i0:i1]
            return S

        A = _ScaledHybrid(Hh, rs, cs)
        y = self._block_path(A, b, rs, cs, factor=lambda: _BlockLU(Rc, off, Hh.nL, Hh.nt, dt,
                                                                    get_block=get_block))
        return y / cs

    def _block_path(self, A, b, rs, cs, factor=None):
        """Solve the scaled system A y = b with GMRES, right-preconditioned by a block LU of the
        layers (_BlockLU): from the factor cache of a neighbouring frequency if one is given (no
        factorization; the previous solution is the starting guess), else a fresh factorization of
        this matrix (a singular layer block is factored with a small diagonal shift: the shift only
        affects the preconditioner, GMRES solves the unshifted system). Raises _NotBlock when the
        matrix is not block-tridiagonal by layer or GMRES does not converge (caller: SuperLU)."""
        cache = self.factor_cache
        key = (A.shape, tuple(self._off), A.dtype)
        x0 = None
        if cache is not None and cache.F is not None and cache.key == key:
            F0, rs0, cs0 = cache.F, cache.rs, cache.cs
            sr, sc_ = rs / rs0, cs / cs0
            prec = lambda r: sc_ * F0.apply(sr * r)          # A^-1 ~ C C0^-1 A0^-1 R0 R^-1
            y0 = cache.x * cs if cache.x is not None else None
            y, it, res = _gmres(A, b, prec, y0, maxiter=cache.max_reuse_iter)
            if res <= self.BLOCK_ACCEPT:
                cache.x, cache.reused = y / cs, cache.reused + 1
                self._block_info = dict(residual=res, iterations=it, reused=True, shifted=F0.shifted)
                return y
            x0 = y if np.isfinite(res) else None             # refactor, continue from there
        F = factor() if factor is not None else \
            _BlockLU(A, self._off, self.model.n_layers, self._ntot, np.result_type(A.dtype, b.dtype))
        y, it, res = _gmres(A, b, F.apply, x0)
        self._gmres_res = res
        if not np.isfinite(res) or res > self.BLOCK_ACCEPT:
            raise _NotBlock
        if cache is not None:
            cache.F, cache.rs, cache.cs, cache.key, cache.x = F, rs, cs, key, y / cs
            cache.factored += 1
        self._block_info = dict(residual=res, iterations=it, reused=False, shifted=F.shifted)
        return y

    def _tisserand_row(self, layers):
        """int rho r (u_th - u_lam cos th) dV = y-component of int rho r x u dV / pi."""
        md = self.model
        row = np.zeros(self._ntot, self.H_dtype)
        for i in layers:
            N = self.Nl[i]
            xg, wg = L.leggauss(N + 4)
            r0, r1 = md.r0[i], md.r1[i]
            s = r0 + (xg + 1) * (r1 - r0) / 2
            p = self.props(i, s)
            # dV = R^2 R_s ds sin th dth dlam,  R_s = 1/s_r,  ds = (r1-r0)/2 dx
            wv = p.rho * p.r**3 / p.s_r * (wg * (r1 - r0) / 2)[:, None] * self.w[None, :]
            for j, (uA, uB, uC, _) in enumerate(self._fields_at(i, xg, p)):
                w_ = uB.v - uC.v * p.cos if self.m == 1 else uC.v * p.sin
                integrand = np.broadcast_to(w_, (N,) + p.r.shape)
                row[self._col(i, j):self._col(i, j) + N] = np.sum(integrand * wv, axis=(1, 2))
        return row

    def inertia_product(self, layers):
        """Change of the product of inertia I_13 = -int rho x z dV of the material in `layers`
        (Lagrangian, first order: -int rho0 u . grad(x z) dV) per unit amplitude of the solution
        (m = 1, the cos(lambda) part); with the reference moment A_eq = int rho (y^2 + z^2) dV of
        the same layers. The lambda integrals (pi) are included."""
        if self.m != 1:
            raise ValueError('inertia_product needs m = 1')
        md = self.model
        dI = A_eq = 0.0
        for i in layers:
            N = self.Nl[i]
            xg, wg = L.leggauss(N + 4)
            r0, r1 = md.r0[i], md.r1[i]
            s = r0 + (xg + 1) * (r1 - r0) / 2
            p = self.props(i, s)
            wv = p.rho * p.r**2 / p.s_r * (wg * (r1 - r0) / 2)[:, None] * self.w[None, :]
            A_eq += np.pi * np.sum(wv * p.r**2 * (p.sin**2 + 2 * p.cos**2))
            for j, (uA, uB, uC, _) in enumerate(self._fields_at(i, xg, p)):
                # z u_x + x u_z, lambda-integrated: pi r (2 u_r sin cos + u_th cos 2th - u_lam cos)
                w_ = 2 * uA.v * p.sin * p.cos + uB.v * (p.cos**2 - p.sin**2) - uC.v * p.cos
                c = self.layer_coef(i, j)
                integrand = np.tensordot(c, np.broadcast_to(w_, (N,) + p.r.shape), axes=(0, 0))
                dI -= np.pi * np.sum(integrand * p.r * wv)
        return dI, A_eq

    # ------------------------------------------------------------------
    def surface_coefficients(self):
        """Values at the surface (s = a) of every field (dict keyed by (type, degree)) and the
        exterior coefficients c_n of phi1."""
        md = self.model
        i = md.n_layers - 1
        Tp, _, _ = cheb_basis(1.0, self.Nl[i])
        vals = {}
        for j, fld in enumerate(self.fields):
            c = self.layer_coef(i, j)
            v = c @ Tp[:, 0]
            vals[fld] = complex(v) if np.iscomplexobj(v) else float(v)
        cext = {n: self.coef[self._ntot + jj] for jj, n in enumerate(self.degP)}
        return vals, cext

    def _tilt_potential(self, n=2):
        """Degree-n exterior coefficient of -(xi x x).grad(phi0): the potential change
        of the flattened Earth tilted rigidly by the removed rotation xi (Wahr 1981,
        eq. 4.5). For m = 1 and a zonal phi0 = sum c_n (a/r)^(n+1) P_n it is
        sqrt(n (n+1) / 2) xi c_n (dP_n/dtheta = -sqrt(n(n+1)/2) Ybar_n^1): sqrt(3) xi c2 for
        degree 2 and sqrt(10) xi c4 for degree 4, which, with xi ~ 1/(sigma - 1), dominates the
        degree-4 coefficient near K1. Degrees above 2 need the consistent reference potential."""
        from .model import G as Gc
        th = np.linspace(0.02, np.pi - 0.02, 401)
        self.surface_field(th, remove_rigid=True)          # sets self._xi
        md = self.model
        if self.refpot is not None:
            rp = self.refpot
            if n not in rp.deg:
                return 0.0
            cn = rp.coef[md.n_layers * rp.nb + rp.deg.index(n)]
        elif n == 2:                                        # Clairaut: J2 = (2f - m)/3
            fl = getattr(self.geom, 'flattening_surface', 0.0)
            mq = getattr(self.geom, 'm', 0.0)
            cn = Gc * md.M * (2 * fl - mq) / 3 / md.a
        else:
            return 0.0
        return np.sqrt(n * (n + 1) / 2.0) * self._xi * cn

    def results(self):
        """Love numbers: U, V, W (surface values / a) and k per degree, and h, l, kk for the
        forcing degree; alpha = rigid-rotation forces of static runs. For wahr_frame with m = 1
        the potential of the rigidly tilted flattened Earth is removed from k (Wahr 1981, 4.5)."""
        vals, cext = self.surface_coefficients()
        a, ga = self.model.a, self.model.g_surface
        out = dict(U={n: vals[('U', n)] / a for n in self.deg},
                   V={n: vals[('V', n)] / a for n in self.degV},
                   W={n: vals[('W', n)] / a for n in self.degW},
                   k={n: -cext[n] / (ga * a) for n in self.deg})
        if self.wahr_frame and self.m == 1:
            for n in out['k']:
                if n >= 2:
                    out['k'][n] = out['k'][n] - self._tilt_potential(n) / (-ga * a)
        nf = self.nf
        out['h'], out['l'], out['kk'] = out['U'][nf], out['V'].get(nf, 0.0), out['k'][nf]
        if self.regions:
            out['alpha'] = [complex(v) for v in self.coef[-len(self.regions):]]
        return out

    def load_love(self):
        """Load Love numbers (Farrell 1972) of a solve with forcing='load'.

        Normalisation by Phi_L = 4 pi G a / (2 nf + 1) (geodetic sign, a the mean
        radius), the load potential on a sphere. For every degree n of the response:
        h'_n = g U_n(a) / Phi_L, l'_n = g V_n(a) / Phi_L, k'_n = potential of the
        deformation / Phi_L, i.e. the exterior coefficient minus that of the load
        itself, which is computed on the actual (flattened) surface: on an
        ellipsoid the load's own degree-n potential differs from Phi_L by O(n f),
        large compared with k'. On a sphere only n = nf is nonzero.
        """
        from .model import G as Gc
        vals, cext = self.surface_coefficients()
        md = self.model
        a, g = md.a, md.g_surface
        PhiL = 4 * np.pi * Gc * a / (2 * self.nf + 1)
        cL = self.load_potential()
        out = dict(h={n: g * vals[('U', n)] / PhiL for n in self.deg},
                   l={n: g * vals[('V', n)] / PhiL for n in self.degV},
                   k={n: -(cext[n] - cL[n]) / PhiL for n in self.deg})
        if self.wahr_frame and self.m == 1:
            # as for the body tide (results): remove the potential of the rigidly tilted,
            # flattened Earth, so that k' is the potential of the deformation alone
            for n in out['k']:
                if n >= 2:
                    out['k'][n] = out['k'][n] + self._tilt_potential(n) / PhiL
        out['W'] = {n: g * vals[('W', n)] / PhiL for n in self.degW}
        return out

    def load_potential(self):
        """Exterior coefficients c_n (phi = sum c_n (a/r)^(n+1) Ybar_n^m cos m lam, code
        sign) of the unit surface density Ybar_nf^m cos m lam on the outer surface."""
        from .model import G as Gc
        md = self.model
        th = self.theta[0, 0]
        p = self.props(md.n_layers - 1, np.array([md.a]))
        R, Rt = p.r[0], p.R_t[0]
        dA = R * np.hypot(R, Rt)                      # dA / (dlambda dmu)
        sig = ybar(self.nf, self.m, th, 0)[0]
        pim = 2 * np.pi if self.m == 0 else np.pi
        a = md.a
        return {n: -Gc * a * pim * np.sum(self.w * (R / a)**n * dA / a**2 * self.ang[n]['Y'][0] * sig)
                for n in self.deg}

    def surface_field(self, theta, remove_rigid=None):
        """Amplitudes of u_r, u_theta (x cos m lam) and u_lambda (x sin m lam)
        at the deformed-surface points (R(a,theta), theta). With remove_rigid
        (default: wahr_frame) the degree-1 toroidal term, the rigid rotation of
        the surface, is dropped (Wahr's separation of nutation / rotation rate)."""
        vals, _ = self.surface_coefficients()
        if remove_rigid is None:
            remove_rigid = self.wahr_frame
        th = np.asarray(theta, float)
        m = self.m
        ur = sum(vals[('U', n)] * ybar(n, m, th, 0)[0] for n in self.deg)
        ut, up = 0 * th, 0 * th
        for n in self.degV:
            y1 = ybar(n, m, th, 1)[1]
            ys = m * ybar(n, m, th, 0, over_sin=True)[0] if m else 0 * th
            ut = ut + vals[('V', n)] * y1
            up = up - vals[('V', n)] * ys
        for n in self.degW:
            y1 = ybar(n, m, th, 1)[1]
            ys = m * ybar(n, m, th, 0, over_sin=True)[0] if m else 0 * th
            ut = ut - vals[('W', n)] * ys
            up = up + vals[('W', n)] * y1
        if remove_rigid and m in (0, 1):
            if self.frame == 'mantle':
                ut, up = self._remove_mantle_rotation(th, ut, up)
            else:
                ut, up = self._remove_rigid_rotation(th, ut, up)
        return ur, ut, up

    def _rigid_pattern(self, i, s, th):
        """Rigid rotation (theta, lambda amplitudes) at (s, theta) of layer i."""
        if hasattr(self.geom, 'map_derivs'):
            R = self.geom.map_derivs(i, s, th)[0]
        else:
            R = mapping(s, th, *self.geom.eps(i, np.asarray(s, float)))['R']
        if self.m == 1:
            return R, -R * np.cos(th)
        return 0 * R, R * np.sin(th)

    def _remove_mantle_rotation(self, th, ut, up):
        """Subtract the mean rotation of the mantle volume (Tisserand frame of the
        mantle): xi = int rho r x u dV / int rho r x (rigid) dV over the mantle."""
        from numpy.polynomial import legendre as L
        md = self.model
        bt, bp = self._rigid_pattern(md.n_layers - 1, md.a, th)
        if getattr(self, '_xi_coef', None) is self.coef:    # same solution: xi computed already
            return ut - self._xi * bt, up - self._xi * bp
        fl = self.fluid_layers()
        mantle = [i for i in range(md.n_layers) if i > 0 and not fl[i] and fl[i - 1] or
                  (i > 0 and not fl[i] and any(fl[:i]))]
        num = den = 0.0
        for i in mantle:
            N = self.Nl[i]
            xg, wg = L.leggauss(N + 4)
            r0, r1 = md.r0[i], md.r1[i]
            s = r0 + (xg + 1) * (r1 - r0) / 2
            p = self.props(i, s)
            wv = p.rho * p.r**3 / p.s_r * (wg * (r1 - r0) / 2)[:, None] * self.w[None, :]
            if self.m == 1:
                proj = lambda b_t, b_p: b_t - b_p * p.cos
            else:
                proj = lambda b_t, b_p: b_p * p.sin
            # tangential displacement of the solution in this layer: only V and W have one, and only
            # its value is needed (the values of _fields_at: radial polynomial x angular function)
            T = self._radial(i, xg)[0][:, :, 0]                       # (N, Ns)
            ut_l, up_l = 0.0, 0.0
            for j, (typ, n) in enumerate(self.B[i].fields):
                if typ not in ('V', 'W'):
                    continue
                Rj = (self.layer_coef(i, j) @ T)[:, None]
                d1, ms = self.ang[n]['Y'][1], -self.ang[n]['mYs'][0]
                if typ == 'V':
                    ut_l, up_l = ut_l + Rj * d1, up_l + Rj * ms
                else:
                    ut_l, up_l = ut_l + Rj * ms, up_l + Rj * d1
            rt_, rp_ = self._rigid_pattern(i, s[:, None], self.theta[0])
            num = num + np.sum(proj(ut_l, up_l) * wv)
            den = den + np.sum(proj(rt_, rp_) * wv)
        xi = num / den
        self._xi, self._xi_coef = xi, self.coef
        return ut - xi * bt, up - xi * bp

    def _remove_rigid_rotation(self, th, ut, up):
        """Subtract the rigid rotation that best fits the surface displacement
        (Wahr 1981, eqs. 4.3, 4.6). On an ellipsoid the rotation is proportional
        to the physical radius R(a, theta), so it is not a pure W_1 field."""
        md = self.model
        i = md.n_layers - 1
        if hasattr(self.geom, 'map_derivs'):
            R = self.geom.map_derivs(i, md.a, th)[0]
        else:
            R = mapping(md.a, th, *self.geom.eps(i, np.array(md.a)))['R']
        if self.m == 1:                     # rotation about an equatorial axis
            bt, bp = R, -R * np.cos(th)
        else:                               # m = 0: spin about the polar axis
            bt, bp = 0 * th, R * np.sin(th)
        w = np.sin(th) * np.gradient(th)    # surface weight
        xi = np.sum(w * (ut * bt + up * bp)) / np.sum(w * (bt * bt + bp * bp))
        self._xi = xi
        return ut - xi * bt, up - xi * bp


def viscous_core_resolution(K_core, delta, N_bulk=30, N_wall=None, N_ekman=16, K=6, N=24):
    """Rule set (for resolution_per_layer) for a model made by EarthModel.viscous_core(delta=delta [m]):
    core K_core; N per kind of fluid sub-layer: N_bulk in the interior, N_wall in the ~40 km layer next to each
    solid boundary (default 30, and 36 below delta = 500 m), N_ekman in the 5-delta layer inside it; K and N
    in the solid layers. Measured on PREM (scripts/sublayer_N_prem.py, level figure, O1 and next to the FCN,
    against N 44 everywhere): the 5-delta layer gives the same Love numbers with N 16 as with 30 (K 12 at
    delta 1 km and 350 m, K 48 at 250 m); the 40 km layer needs 24 at 1 km but with 30 at 350 m leaves 1.4e-5
    in h0 (36: 2.5e-7); the interior: 30 -> 1e-6, 24 -> 6e-6 (1 km) or 1.7e-5 (350 m), 16 -> 2e-4. Time per
    solve: ~10 % less than N 30 everywhere (K 48, banded layer solver); the gain is accuracy, not speed."""
    if N_wall is None:
        N_wall = 30 if delta >= 500.0 else 36
    split = max(5.0, 10.0 * delta / 1e3)                   # km: thinner = 5-delta layer, thicker = 40 km layer
    return {'K': int(K), 'N': int(N),
            'rules': [{'fluid': True, 'K': int(K_core), 'N': int(N_bulk)},
                      {'fluid': True, 'thickness_km': [split, 100.0], 'N': int(N_wall)},
                      {'fluid': True, 'thickness_km': [0.0, split], 'N': int(N_ekman)}]}


def resolution_per_layer(model, spec):
    """(layer_K, layer_N) for TidalSolver from a rule set (dict, e.g. read from a JSON file):

        {"K": 6, "N": 24,                                  defaults
         "rules": [{"depth_km": [0, 220], "K": 4, "N": 16},  midpoint depth of the layer in range
                   {"fluid": true, "K": 12, "N": 30},        (weak-)fluid layers (core, Ekman sub-layers)
                   {"fluid": true, "thickness_km": [0, 50], "N": 16}]}   the thin Ekman sub-layers only

    A rule applies when all its selectors match; later rules override earlier ones, and a rule may
    set K, N or both. Selecting by depth, thickness and type, not by layer index, keeps a rule set
    valid when viscous_core splits the core into sub-layers (40 km and 5 delta next to each wall)."""
    bad = set(spec) - {'K', 'N', 'rules', 'comment'}
    if bad:
        raise ValueError(f'unknown resolution keys {sorted(bad)}')
    K, N = int(spec.get('K', 4)), int(spec.get('N', 24))
    cut = getattr(model, 'weak_fluid_mu', 1e8)
    Kl, Nl = [K] * model.n_layers, [N] * model.n_layers
    for rule in spec.get('rules', []):
        bad = set(rule) - {'depth_km', 'thickness_km', 'fluid', 'K', 'N', 'comment'}
        if bad:
            raise ValueError(f'unknown resolution rule keys {sorted(bad)}')
        for i in range(model.n_layers):
            rm = 0.5 * (model.r0[i] + model.r1[i])
            depth = (model.a - rm) / 1e3
            ok = True
            if 'depth_km' in rule:
                d0, d1 = rule['depth_km']
                ok &= d0 <= depth <= d1
            if 'thickness_km' in rule:
                t0, t1 = rule['thickness_km']
                ok &= t0 <= (model.r1[i] - model.r0[i]) / 1e3 <= t1
            if 'fluid' in rule:
                ok &= (abs(model.mu[i](rm)) < cut) == bool(rule['fluid'])
            if ok:
                Kl[i] = int(rule.get('K', Kl[i]))
                Nl[i] = int(rule.get('N', Nl[i]))
    return Kl, Nl


def ZonalSolver(model, geometry=None, **kw):
    """m = 0 solver (kept for the earlier scripts and tests)."""
    return TidalSolver(model, geometry, m=0, **kw)
