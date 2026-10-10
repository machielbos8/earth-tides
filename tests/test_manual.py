# -*- coding: utf-8 -*-
#
# Every Python code block of the user manual runs (in order, in one namespace), so that the
# examples in the manual cannot go stale.
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
# 9/10/2026 Machiel Bos
#===============================================================================
import re
import warnings
from pathlib import Path

MANUAL = Path(__file__).resolve().parents[1] / 'docs' / 'manual' / 'earthtides_manual_v1.0.md'


def test_manual_python_blocks_run():
    blocks = re.findall(r'```python\n(.*?)```', MANUAL.read_text(), re.S)
    assert len(blocks) >= 5
    ns = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k, code in enumerate(blocks):
            if code.startswith('# excerpt'):            # a fragment of the source, not runnable
                continue
            try:
                exec(compile(code, f'manual block {k}', 'exec'), ns)
            except Exception as e:                       # report which block failed
                raise AssertionError(f'manual python block {k} failed: {e!r}\n{code}') from e
