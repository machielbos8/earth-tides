## Example 7: a non-hydrostatic Earth

Hydrostatic PREM puts the free core nutation (FCN) at 459 sidereal days (anelastic, level figure);
VLBI observes about 431. The Earth is not hydrostatic: its J2 exceeds the hydrostatic value by about
one per cent, and the observed period needs a core some 5 per cent more flattened. A non-hydrostatic
shape must be held up by deviatoric stress.

### What the script does

Two zonal density loads in the mantle, δρ = a_k ρ0(s) P2(cos θ) in the lower and the upper mantle, are
carried by the self-gravitating, rotating, flattened Earth (elastic support). Their amplitudes follow
from two targets: the excess J2 of Chambat et al. (2010) and an excess dynamical ellipticity e_f of the
core. The deformed Earth, with its deviatoric stress, becomes the reference state of the tidal
calculation (manual, Sect. "A non-hydrostatic Earth"). For the hydrostatic Earth and two states, the
excess e_f of Mathews et al. (2002) and the excess that gives the observed period, the script prints
the excess flattening of the core–mantle boundary, the FCN period (a pole fit of k just above the
resonance) and the Love numbers at O1, ψ1 and M2.

### What to look for

The excess of Mathews et al. (2002) gives 437 days; 486 m of excess flattening of the core–mantle
boundary gives the observed 431.4 days. Away from the resonance (O1, M2) the non-hydrostatic Earth
changes the Love numbers by about 1e-4; at ψ1, next to the resonance, h0 changes by 0.12.

    python nonhydrostatic.py        (about 30 seconds; compare with nonhydrostatic.out)
