# -*- coding: utf-8 -*-
#
# earthtides.love: the convention options reproduce Table 3 of paper 1
# (elastic, hydrostatic PREM, changes in units of 1e-5, printed to 0.1); the
# JSON round trip; etide reads a table and refuses one in other conventions.
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
"""earthtides.love: the convention options reproduce Table 3 of paper 1 (elastic, hydrostatic PREM,
changes in units of 1e-5, printed to 0.1); the JSON round trip; etide reads a table and refuses
one in other conventions."""
import json

import numpy as np
import pytest

from earthtides import love

O1 = 0.92700 * 1.002737909          # cycles per day
M2 = 2 * 0.9661368 * 1.002737909
KEYS = ('h0', 'h2', 'l0', 'l2', 'l1', 'lp')
# Table 3 (results/convention_table.txt): baseline values and changes [1e-5]
BASE = {1: (0.59626, -0.00050, 0.08305, 0.00009, 0.00048, -0.00030),
        2: (0.60244, -0.00049, 0.08361, 0.00009, 0.00099)}
ROWS = {1: {'frame': (0.0, 0.0, 0.0, -0.0, 0.1, 34.9),
            'geodetic': (-123.2, 530.2, -3.5, 74.1, -27.7, 32.2),
            'normal': (18.1, -74.1, 16.6, 132.5, -99.4, 38.4),
            'normal+geodetic': (-104.9, 456.1, 12.9, 206.8, -127.4, 70.5),
            'mean': (-0.0, -60.4, 0.3, -111.5, 1.1, 0.1),
            'norm': (265.5, -0.2, 37.0, 0.0, 0.2, -0.1)},
        2: {'geodetic': (-124.6, 533.7, -17.3, 74.3, -56.3),
            'normal': (-37.6, -75.3, 67.0, 133.9, -200.9),
            'normal+geodetic': (-162.0, 458.7, 49.2, 207.2, -256.1),
            'mean': (-0.0, -61.1, 0.1, -115.0, -0.1),
            'norm': (268.3, -0.2, 37.2, 0.0, 0.4),
            'centrifugal': (-80.5, -0.5, -13.1, 0.2, -0.3)}}
VARIANTS = {'geodetic': dict(latitude='geodetic'), 'normal': dict(direction='normal'),
            'normal+geodetic': dict(direction='normal', latitude='geodetic'),
            'mean': dict(surface='mean'), 'norm': dict(norm='mean')}


def _params(m, f, **kw):
    o = love.Options(anelastic=False, **kw)
    S, r = love.solve_wave(o, m, f)
    return S, r


@pytest.mark.parametrize('m, f', [(1, O1), (2, M2)])
def test_conventions_table3(m, f):
    keys = KEYS[:len(BASE[m])]
    S, r = _params(m, f)
    base = love.parameters(S, r, love.Options(anelastic=False))
    assert all(abs(base[k].real - v) < 6e-6 for k, v in zip(keys, BASE[m]))
    for name, row in ROWS[m].items():
        if name in VARIANTS:
            p = love.parameters(S, r, love.Options(anelastic=False, **VARIANTS[name]))
        elif name == 'frame':
            S2, r2 = _params(m, f, frame='surface')
            p = love.parameters(S2, r2, love.Options(anelastic=False, frame='surface'))
        else:                                         # no centrifugal force in the reference
            S2, r2 = _params(m, f, centrifugal=False)
            p = love.parameters(S2, r2, love.Options(anelastic=False, centrifugal=False))
        got = [1e5 * (p[k].real - base[k].real) for k in keys]
        assert np.allclose(got, row, atol=0.051), (name, np.round(got, 1), row)


def test_json_roundtrip_and_etide(tmp_path):
    """A table written by love_table is read back exactly, and etide accepts it only in the IERS
    conventions (here: three semidiurnal frequencies, which etide must refuse as incomplete)."""
    from earthtides import etide
    o = love.Options()
    t = love.love_table(o, [(2, None, M2, 2)], progress=False)
    path = tmp_path / 't.json'
    love.write(t, path)
    back = love.read(path)
    assert back['metadata']['iers_conventions'] and back['metadata']['options']['K'] == 4
    assert back['entries'][0]['h0'] == complex(*t['entries'][0]['h0'])
    cat = etide.Tamura()
    with pytest.raises(ValueError, match='not in the table'):
        etide.LoveModel(cat, str(path))
    t['metadata']['iers_conventions'] = False
    json.dump(t, open(path, 'w'))
    with pytest.raises(ValueError, match='IERS conventions'):
        etide.LoveModel(cat, str(path))


def test_groups_table():
    """--groups: every Tamura wave is covered, the waves of a group share the Love numbers of the
    group's largest wave, and etide accepts the table."""
    from earthtides import etide
    groups = [('lp', 0.0, 0.501369), ('d', 0.501370, 1.470243), ('sd', 1.470244, 7.0)]
    t = love.group_table(love.Options(), groups, progress=False)
    cat = etide.Tamura()
    assert len(t['entries']) == len(cat.n)
    by = {}
    for e in t['entries']:
        if 'group' in e:
            by.setdefault(e['group'], set()).add(tuple(e['h0']))
    assert set(by) == {'lp', 'd', 'sd'} and all(len(v) == 1 for v in by.values())
    reps = {g['name']: g['representative_f_cpd'] for g in t['metadata']['groups']}
    assert abs(reps['sd'] - 1.9322736) < 1e-5 and abs(reps['d'] - 1.0027379) < 1e-5


def test_default_groups():
    """The default wave groups (data/groups_default.txt, tools/design_groups.py) cover every
    degree-2 wave exactly once and keep the displacement of the IERS model within 0.05 mm of its
    per-wave value (two stations, 60 days; the full check over a station grid and a year with two
    Earth models is tools/verify_groups.py, max 0.03 mm)."""
    import sys
    from pathlib import Path
    from earthtides import etide
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    from verify_groups import grouped_copy
    groups = love.read_groups(love.DEFAULT_GROUPS)
    cat = etide.Tamura()
    for n, f in zip(cat.n, cat.f_cpd):
        if n == 2 and abs(f) > 1e-9:
            assert sum(f1 - 1e-7 <= f <= f2 + 1e-7 for _, f1, f2 in groups) == 1, f
    iers = etide.LoveModel(cat, 'iers')
    G = grouped_copy(cat, iers, groups)
    mjd = 61041.0 + np.arange(0, 60, 1 / 24)
    for lat, lon in ((57.4, 11.9), (-42.8, 147.4)):
        a = np.column_stack(etide.displacement(mjd, lat, lon, 0.0, iers, cat=cat))
        b = np.column_stack(etide.displacement(mjd, lat, lon, 0.0, G, cat=cat))
        assert np.abs(a - b).max() < 5e-5
