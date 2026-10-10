# -*- coding: utf-8 -*-
#
# Golden values: the current code must reproduce the numbers of tag round-1
# (tests/golden/round3.json, written by tools/make_golden.py from
# tests/golden/cases.py).
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
"""Golden values: the current code must reproduce the numbers of tag round-1
(tests/golden/round3.json, written by tools/make_golden.py from tests/golden/cases.py).

Tolerance 1e-8 of a scale: for Love numbers, load Love numbers and etide series (metres) the scale
is max(|value|, 1), i.e. 1e-8 absolute for small parameters such as h(2) ~ 5e-4; for the surface
fields u_r, u_theta, u_lambda of a tide it is the largest of the three. Why absolute for small
parameters: the system matrix has condition numbers up to ~1e14 (M2: high degrees of the flow in
the fluid core are held only by inertia), so equally valid solutions (dense LAPACK vs sparse
SuperLU, both with residuals at round-off) differ by up to ~2e-10 in the Love numbers and 1e-9 of
the surface field, a large relative change of a parameter of 1e-4. The paper prints 5-6 decimals;
1e-8 is ten times the spread between equally valid solutions and far below the printed digits.
A failure means a change of results, which must be explained; regenerate the golden file only on
purpose.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'golden'))
import cases  # noqa: E402

GOLDEN = json.load(open(HERE / 'golden' / 'round3.json'))['values']
TOL = 1e-8


@pytest.mark.parametrize('case', cases.CASES, ids=lambda c: c.__name__)
def test_golden(case):
    ref = GOLDEN[case.__name__]
    got = case()
    assert set(got) == set(ref), set(got) ^ set(ref)
    bad = []
    for key, v in got.items():
        a = np.atleast_1d(np.asarray(v, complex))
        b = np.atleast_1d(cases.decode(ref[key]))
        if '.u_' in key:                         # surface fields of one tide: common scale
            tide = key.split('.u_')[0]
            scale = max(np.abs(np.atleast_1d(cases.decode(ref[tide + c]))).max()
                        for c in ('.u_r', '.u_theta', '.u_lambda'))
        else:
            scale = max(np.abs(b).max(), 1.0)
        err = np.abs(a - b).max() / scale
        if not err <= TOL:
            bad.append(f'{key}: difference {err:.2e} of the scale {scale:.3g}')
    assert not bad, '\n'.join(bad)
