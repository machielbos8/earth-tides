# -*- coding: utf-8 -*-
#
# earthtides-examples: copy the worked examples and the PDF user manual into a new directory, so
# that a new user has something to run and to read right after installation.
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
# 9/10/2026 Machiel Bos
#===============================================================================
"""earthtides-examples [DIR]: copy the examples and the manual to DIR (default earthtides-examples)."""
import shutil
import sys
from pathlib import Path

EXAMPLES = (
    ('ex1', 'The pumped swing: trial function, residual and projection in one variable'),
    ('ex2', 'The loaded elliptical drum: collocation and projection on level curves'),
    ('ex3', 'Homogeneous spheres: exact Bessel solution (Bos & Scherneck 2013)'),
    ('ex4', 'Love numbers of a flattened homogeneous Earth (Greff-Lefftz et al. 2005)'),
    ('ex5', 'Free core nutation of a rigid shell with a fluid core (Hough 1895)'),
    ('ex6', 'Ekman damping of the free core nutation: a viscous, no-slip core (Greenspan 1968)'),
    ('ex7', 'A non-hydrostatic Earth: mantle loads, their supporting stress and the free core nutation'),
    ('ex8', 'The viscous core and the limit of vanishing viscosity'),
    ('ex9', 'The level figure against the first-order figure'),
)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        return
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('earthtides-examples')
    if dest.exists():
        sys.exit(f'Destination already exists: {dest.resolve()}\n'
                 'Remove it first or choose a different name.')
    here = Path(__file__).resolve().parent
    src = here / 'examples'
    if not src.is_dir():
        sys.exit('Examples directory not found in the installed package.')
    shutil.copytree(src, dest, symlinks=False,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.png'))
    manuals = sorted((here / 'docs').glob('earthtides_manual_*.pdf'))
    for m in manuals:
        shutil.copy(m, dest / m.name)
    print(f'Examples copied to: {dest.resolve()}')
    for name, topic in EXAMPLES:
        print(f'  {name}  {topic}')
    if manuals:
        print(f'Start with the manual: {manuals[-1].name}')
    print(f'Run an example from its directory, e.g.:  cd {dest}/ex1 && python swing.py')


if __name__ == '__main__':
    main()
