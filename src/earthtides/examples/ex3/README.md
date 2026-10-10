## Example 3: Homogeneous spheres

The first rung of the verification ladder (paper, Sect. 4.1): Love numbers of the two homogeneous
Earth models of Bos & Scherneck (2013, Sect. 1.7), after Alterman, Jarosch & Pekeris (1959).
G = 6.673e-11, a = 6371 km.

* **Model b**: homogeneous sphere, rho 5517 kg/m^3, mu 146 GPa, lambda 347 GPa. Gravity is
  proportional to r, so the spherical-Bessel solution is exact. Their Table 1.3 gives the body-tide
  (h, n l, n k) and load (h', n l', n k') Love numbers at the M2 frequency. We reproduce all six to
  the five digits given, at degrees 2, 10 and 100.
* **Model a**: homogeneous mantle (rho 4460, mu 174 GPa, lambda 231 GPa) over a fluid core of
  radius 0.55 a (rho 11020, lambda 950 GPa), at forcing periods of 6, 12 and 24 h. Gravity is no
  longer proportional to r, but B&S assumed it is, layer by layer, to keep the Bessel solutions.
  Their Table 1.4 is therefore approximate, off by 4 % in h and 15 % in l. The reference is
  astyanax `computelove`, which solves the radial equations with exact gravity. Agreement is 1e-9.

Run:

    python homogeneous.py      # < 2 s; writes modela.dat, modelb.dat and homogeneous.out

Expected output: [homogeneous.out](homogeneous.out).
