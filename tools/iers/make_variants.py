# -*- coding: utf-8 -*-
#
# Build the IERS DEHANTTIDEINEL binaries used by scripts/body_tide_check.py
# and scripts/etide_vs_iers.py, and HARDISP, into a work directory.
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
"""Build the IERS DEHANTTIDEINEL binaries used by scripts/body_tide_check.py and
scripts/etide_vs_iers.py, and HARDISP, into a work directory.

The IERS Conventions routines are not redistributed here (their licence restricts modified
copies); point this script at a copy of the sources: dehanttideinel.f, st1idiu.f, st1isem.f,
st1l1.f, step2diu.f, step2lon.f, sprod.f, norm8.f, zero_vec8.f (e.g. from the IERS Conventions
software collection, or the GAMIT tree kf/track, whose dehanttideinel.f takes T in Julian centuries
TT and FHR in UTC hours as inputs) and the HARDISP sources (*.F).

Variants (each keeps one part of the model; XCORSTA is zeroed after every accumulation loop so
that a disabled step does not re-add the previous correction):
  dehant     complete routine           dehant_d2  degree 2, nominal h, l only
  dehant_n3  without degree 3           dehant_l1  degree 2 + l(1) terms (ST1L1)
  dehant_op  degree 2 + out-of-phase    dehant_s2  degree 2 + STEP2DIU    dehant_lo  + STEP2LON
Usage: python tools/iers/make_variants.py IERS_SRC HARDISP_SRC WORKDIR
Then put de421.bsp (skyfield) into WORKDIR and run the scripts with WORKDIR as argument.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

src, hsrc, work = (Path(a) for a in sys.argv[1:4])
work.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
helpers = ['st1idiu.f', 'st1isem.f', 'st1l1.f', 'step2diu.f', 'step2lon.f', 'sprod.f', 'norm8.f',
           'zero_vec8.f']
for f in ['dehanttideinel.f'] + helpers:
    shutil.copy(src / f, work / f)
shutil.copy(here / 'driver.f', work / 'driver.f')
base = (work / 'dehanttideinel.f').read_text()


def variant(name, keep=(), deg3=True):
    s = base.replace('SUBROUTINE DEHANTTIDEINEL(', f'SUBROUTINE DEHANT{name.upper()}(')
    if not deg3:
        s = s.replace('H3/0.292D0/,L3/0.015D0/', 'H3/0.0D0/,L3/0.0D0/')
    for call in ('ST1IDIU', 'ST1ISEM', 'ST1L1', 'STEP2DIU', 'STEP2LON'):
        if call not in keep:
            s = re.sub(rf'^( *)CALL {call}', rf'*\1CALL {call}', s, flags=re.M)
    for lab in ('11', '12', '13', '14'):
        s = re.sub(rf'^({lab} +CONTINUE *)$', r'\1\n      CALL ZERO_VEC8(XCORSTA)', s, flags=re.M)
    (work / f'dehant_{name}.f').write_text(s)
    (work / f'driver_{name}.f').write_text(
        (work / 'driver.f').read_text().replace('CALL DEHANTTIDEINEL(', f'CALL DEHANT{name.upper()}('))
    subprocess.run(['gfortran', '-O2', '-std=legacy', '-o', str(work / f'dehant_{name}'),
                    f'driver_{name}.f', f'dehant_{name}.f'] + helpers, cwd=work, check=True)


subprocess.run(['gfortran', '-O2', '-std=legacy', '-o', str(work / 'dehant'), 'driver.f',
                'dehanttideinel.f'] + helpers, cwd=work, check=True)
variant('d2', deg3=False)
variant('n3', keep=('ST1IDIU', 'ST1ISEM', 'ST1L1', 'STEP2DIU', 'STEP2LON'), deg3=False)
variant('l1', keep=('ST1L1',), deg3=False)
variant('op', keep=('ST1IDIU', 'ST1ISEM'), deg3=False)
variant('s2', keep=('STEP2DIU',), deg3=False)
variant('lo', keep=('STEP2LON',), deg3=False)
hw = work / 'hardisp_src'
hw.mkdir(exist_ok=True)
for f in hsrc.glob('*.F'):
    shutil.copy(f, hw / f.name)
subprocess.run('gfortran -O2 -std=legacy -o ../hardisp *.F', shell=True, cwd=hw, check=True)
print('built in', work)
