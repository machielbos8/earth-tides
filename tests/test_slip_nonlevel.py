# -*- coding: utf-8 -*-
#
# Slipping fluid-solid boundary on a wall that is not a level surface (TidalSolver slip_nonlevel).
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
# 14/11/2026 Machiel Bos
#===============================================================================
"""Hough's Earth (homogeneous Maclaurin spheroid, slipping fluid core, deformable mantle), 0.96 cycles per
sidereal day. The exact spheroid's wall is a level surface of the reference potential; a wall with a degree-4
bump of 3 eps^2 (210 m), or the wall of the first-order map, is not (same density on both sides: the bump moves
no mass). Known answers (lab notes 2026-11-14; viscous core delta 250 m, K 36, on the bumped wall: h2 -9.364e-4,
on the exact spheroid -9.307e-4):
  - on the exact spheroid the term vanishes: slip_nonlevel changes nothing;
  - on the bumped wall, with the term h2 stays within 1e-5 of the exact spheroid's (viscous: 5.7e-6), without it
    h2 moves by more than 1e-4;
  - on the first-order map the term brings h2 closer to the exact spheroid's (less than half the distance)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from earthtides.geometry import ExactSpheroidGeometry, ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver, resolution_per_layer
from earthtides.iers import iers_fit

_p = Path(__file__).resolve().parents[1] / 'examples' / 'ex5' / 'hough1895.py'    # not src/: CI tests the wheel
_spec = importlib.util.spec_from_file_location('hough1895', _p)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

S96 = 0.96


class Bumped(ExactSpheroidGeometry):
    """Exact similar spheroids with a degree-4 bump on every coordinate surface: R = s f(theta) (1 + b P4)."""

    def __init__(self, flattening, b):
        super().__init__(flattening)
        self.b = b

    def map_derivs(self, i, s, theta):
        R, R_s, R_ss, R_t, R_st, R_tt = super().map_derivs(i, s, theta)
        c, sn = np.cos(theta), np.sin(theta)
        P4 = (35 * c ** 4 - 30 * c * c + 3) / 8
        dP4 = (140 * c ** 3 - 60 * c) / 8
        ddP4 = (420 * c * c - 60) / 8
        g = 1 + self.b * P4
        g1 = -self.b * dP4 * sn
        g2 = self.b * (ddP4 * sn * sn - dP4 * c)
        return (R * g, R_s * g, R_ss * g, R_t * g + R * g1, R_st * g + R_s * g1, R_tt * g + 2 * R_t * g1 + R * g2)


class _Eps:
    def __init__(self, fl):
        self.fl = fl

    def eps(self, i, s):
        return np.full_like(s, self.fl), 0 * s, 0 * s


@pytest.fixture(scope='module')
def h2(tmp_path_factory):
    md = H.make(vs_m=6.0, vp_c=1000.0, path=tmp_path_factory.mktemp('hough') / 'HOUGH.dat')
    fl = H.maclaurin_flattening(OMEGA_EARTH)
    cl = ClairautGeometry(md)
    geo = {'exact': (ExactSpheroidGeometry(fl), _Eps(fl)), 'bump': (Bumped(fl, 3 * fl ** 2), _Eps(fl)),
           'first': (cl, cl)}
    w = S96 * OMEGA_EARTH
    Kl, Nl = resolution_per_layer(md, {'K': 6, 'N': 16, 'rules': [{'fluid': True, 'K': 6, 'N': 24}]})
    out = {}
    for name, (g, gf) in geo.items():
        for on in (False, True):
            S = TidalSolver(md, g, m=1, layer_K=Kl, layer_N=Nl, omega_f=0.0, coriolis=OMEGA_EARTH, omega_t=w,
                            omega_rot=OMEGA_EARTH, wahr_frame=True, dyn_fluid=True, consistent_reference=True,
                            frame='mantle', slip_nonlevel=on)
            S.solve()
            assert any(S.true_fluid)
            out[name, on] = complex(iers_fit(S, gf)['h2']).real
    return out


def test_level_wall_unchanged(h2):
    assert abs(h2['exact', True] - h2['exact', False]) < 1e-9
    assert abs(h2['exact', False] - (-9.307e-4)) < 2e-7


def test_bumped_wall(h2):
    assert abs(h2['bump', False] - h2['exact', False]) > 1e-4          # without the term: -8.136e-4
    assert abs(h2['bump', True] - h2['exact', False]) < 1e-5           # with it: -9.364e-4, the viscous value
    assert abs(h2['bump', True] - (-9.364e-4)) < 3e-7


def test_first_order_map(h2):
    # far from the resonance the first-order map differs from the exact spheroid by its own O(eps^2) as well:
    # without the term -9.416e-4, with it -9.277e-4 (exact -9.307e-4). The sharp statement is the resonant
    # share (0.0060 -> 0.0020, scripts/hough_share_test.py --slip-term), too slow for a test.
    off, on = (abs(h2['first', k] - h2['exact', False]) for k in (False, True))
    assert on < 0.5 * off
    assert on < 5e-6
