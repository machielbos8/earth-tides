## Example 9: the level figure against the first-order figure

Up to version 1.0 the coordinate surfaces were the first-order (Clairaut) figure of the flattened Earth.
Since version 1.1 every coordinate surface is, by default, a level surface of the model's own reference
potential (manual, Sect. "The level figure").

### What the script does

For elastic, hydrostatic PREM it prints, on both figures, the inverse flattening of the surface, the
equatorial minus polar radius of the core–mantle boundary, the period of the free core nutation and
h0 at O1 and ψ1.

### What to look for

The figures differ by metres: the equatorial minus polar radius of the core–mantle boundary differs by
14 m. Away from the resonance (O1) h0 changes by 2e-5; the free core nutation is 1.1 days shorter on the
level figure (456.2 against 457.3 sidereal days), and h0 at ψ1, next to it, changes by 5e-3.

    python figures.py               (about 10 seconds; compare with figures.out)
