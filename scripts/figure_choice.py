# -*- coding: utf-8 -*-
#
# Which figure a script runs on: environment variable EARTHTIDES_FIGURE (first-order | level).
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
"""For scripts with positional arguments, where a --figure option would shift them, and for their worker
processes (the environment is inherited): EARTHTIDES_FIGURE=level python scripts/X.py ...
FIGURE: 'first-order' (default of the scripts: every result up to 2026-11-14) or 'level'
(geometry.LevelGeometry). SUF: '' or '_level', appended to every output name, so that a level-figure
result cannot overwrite or be taken for a first-order one. base(model, **kw): the geometry."""
import os

FIGURE = os.environ.get('EARTHTIDES_FIGURE', 'first-order')
if FIGURE not in ('first-order', 'level'):
    raise SystemExit(f"EARTHTIDES_FIGURE must be 'first-order' or 'level', not {FIGURE!r}")
SUF = '' if FIGURE == 'first-order' else '_' + FIGURE


def base(model, **kw):
    from earthtides.geometry import figure
    return figure(model, FIGURE, **kw)


def Clairaut(model, omega=None, scale=1.0, **kw):
    """Drop-in for ClairautGeometry(model, omega=, scale=) in scripts written for the first-order figure:
    after `from earthtides.geometry import ClairautGeometry` add
        import figure_choice; ClairautGeometry = figure_choice.Clairaut
    Default (EARTHTIDES_FIGURE unset): ClairautGeometry itself, numbers unchanged. 'level': the level figure
    for that rotation rate (geometry.figure). Do NOT use it in tests of the first-order map itself."""
    from earthtides.geometry import figure, OMEGA_EARTH
    return figure(model, FIGURE, omega=OMEGA_EARTH if omega is None else omega, scale=scale, **kw)


def tagged(path):
    """path with _level before the extension when FIGURE is 'level' (inputs written by a level-figure run)."""
    import os as _os
    root, ext = _os.path.splitext(str(path))
    return root + SUF + ext if SUF and SUF not in _os.path.basename(root) else str(path)
