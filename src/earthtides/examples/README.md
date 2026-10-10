# earth-tides examples

Each example is self-contained: a README that explains the physics and the steps, a script, the
Earth model it uses (written by the script or included) and the expected output (`*.out`). Run a
script from its own directory and compare its output with the `.out` file:

    cd ex1
    python swing.py

The examples form a ladder of increasing complexity; the manual (`earthtides_manual_v1.0.pdf`,
chapter Examples) explains each of them.

| | Topic | Runtime |
|---|---|---|
| ex1 | The pumped swing: trial function, residual and projection in one variable | 1 s |
| ex2 | The loaded elliptical drum: Chebyshev collocation and Galerkin projection on level curves | 2 s |
| ex3 | Homogeneous spheres: exact Bessel solution and a fluid-core sphere (Bos & Scherneck 2013) | 2 s |
| ex4 | Analytical Love numbers of a flattened homogeneous Earth (Greff-Lefftz et al. 2005) | 6 min |
| ex5 | Free core nutation of a rigid shell with a fluid core (Hough 1895) | 20 s |
