# -*- coding: utf-8 -*-
#
# Latitude term delta+ of the diurnal gravimetric factor across the band, against DDW99 Table 9.
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
"""Elastic hydrostatic PREM: delta0, delta+ and the inertia part, and the resonant share
(delta+ - delta+(O1)) / (delta0 - delta0(O1)). Usage: python scripts/scan_delta_plus.py"""
import sys, warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np
from earthtides.model import EarthModel
from earthtides.love import model_path
from earthtides.geometry import ClairautGeometry, OMEGA_EARTH
from earthtides.solver import TidalSolver
from earthtides.gravimetry import surface_gravity
md=EarthModel(str(model_path('PREM'))); geo=ClairautGeometry(md)
def run(sig):
    w=sig*OMEGA_EARTH
    S=TidalSolver(md,geo,m=1,K=4,N=24,omega_f=0.0,coriolis=OMEGA_EARTH,omega_t=w,omega_rot=OMEGA_EARTH,
                  wahr_frame=True,dyn_fluid=True,consistent_reference=True,frame='mantle'); S.solve()
    th,dg,dgT=surface_gravity(S); th2,dg0,_=surface_gravity(S,inertia=False)
    f=np.sqrt(3)/(2*np.sqrt(2))*(7*np.cos(th)**2-3); G=np.column_stack([dgT,dgT*f])
    fit=lambda y: np.linalg.lstsq(G,y,rcond=None)[0].real
    return fit(dg), fit(dg-dg0)
DDW={0.92700:(1.15279,0.00004),0.99454:(1.14777,-0.00013),1.00273:(1.23735,0.00273),1.00546:(1.16775,0.00051)}
ref=None
for s in [0.92700,0.96,0.99454,0.9998,1.0002,1.0015,1.00273,1.00546,1.0730]:
    (d0,dp),(a0,ap)=run(s)
    if ref is None: ref=(d0,dp)
    rr=(dp-ref[1])/(d0-ref[0]) if abs(d0-ref[0])>1e-6 else float('nan')
    extra=f"  DDW d0 {DDW[s][0]:.5f} d+ {DDW[s][1]:+.5f}" if s in DDW else ''
    print(f"sigma {s:.5f}  d0 {d0:.5f}  d+ {dp:+.6f}  (inertia part d0 {a0:+.5f} d+ {ap:+.6f})  resonant d+/d0 {rr:+.4f}{extra}")
