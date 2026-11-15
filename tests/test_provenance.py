# -*- coding: utf-8 -*-
#
# Tests of scripts/provenance.py: sidecar and ledger line written, failure recorded.
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
# 31/10/2026 Machiel Bos
#===============================================================================
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import provenance as pv  # noqa: E402


def test_run_writes_sidecar_and_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, 'LEDGER', str(tmp_path / 'LEDGER.jsonl'))
    inp = tmp_path / 'model.dat'
    inp.write_text('x')
    out = tmp_path / 'r.txt'
    with pv.Run('unit test', params=dict(K=6, z=1 + 2j), inputs=[str(inp)]) as run:
        run.write_text(str(out), 'hello')
        run.note(fcn=457.3)
    meta = json.loads((tmp_path / 'r.txt.meta.json').read_text())
    assert meta['status'] == 'ok' and meta['params']['K'] == 6 and meta['params']['z'] == [1.0, 2.0]
    assert meta['outputs'][0]['sha256'] == pv.sha256(str(out))
    assert meta['inputs'][0]['sha256'] == pv.sha256(str(inp))
    assert meta['params']['_results']['fcn'] == 457.3
    lines = (tmp_path / 'LEDGER.jsonl').read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])['id'] == meta['id']


def test_failed_run_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, 'LEDGER', str(tmp_path / 'LEDGER.jsonl'))
    with pytest.raises(ValueError):
        with pv.Run('failing'):
            raise ValueError('boom')
    rec = json.loads((tmp_path / 'LEDGER.jsonl').read_text())
    assert rec['status'].startswith('failed: ValueError')
