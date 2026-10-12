#!/usr/bin/env python3
"""Check haynes_curves.py against the Sage code it was ported from.

    python check_haynes_curves.py

Needs sympy (``pip install sympy``), not Sage. It does three things:

1. Re-runs the algorithm of ``haynes_thesis_info`` from
   ``haynes-project-code_goins2.sage`` in sympy -- the same resultants for
   phi and psi, the same common-factor reduction, the same Plücker
   formulas -- and checks the numpy port gives the same numbers.
2. Checks those numbers against the outputs recorded in the Sage
   docstrings, and the defining polynomials against the phi printed there.
3. Checks the facts the Movement II animation relies on: each reduced dual
   line really is tangent at its point (cusps included), and TO_DELTOID
   carries the tricuspidal quartic exactly onto the deltoid.
"""

from __future__ import annotations

import sys

import numpy as np
import sympy as sp

import haynes_curves as hc

t = sp.symbols("t")
x1, x2, x3, u1, u2, u3 = sp.symbols("x1 x2 x3 u1 u2 u3")

# Plücker data printed in the Sage docstrings of haynes_thesis_info:
# (n, nu, nodes, cusps, bitangents, flexes). "Tricuspidal Quartic" has no
# docstring entry; it is the same curve as "Unicuspidal Quartic".
SAGE_DOCSTRING = {
    "Conic Section": (2, 2, 0, 0, 0, 0),
    "Cuspidal Cubic": (3, 3, 0, 1, 0, 1),
    "Trident Curve": (3, 4, 1, 0, 0, 3),
    "Tricuspidal Quartic": (4, 5, 2, 1, 2, 4),
    "Bicuspidal Quartic": (4, 4, 1, 2, 1, 2),
    "Unicuspidal Quartic": (4, 5, 2, 1, 2, 4),
    "Trinodal Quartic": (4, 6, 3, 0, 4, 6),
    "Chapter 4 Example": (4, 3, 0, 3, 1, 0),
    "Chapter 5 Example": (4, 4, 1, 2, 1, 2),
    "Chapter 6 Example": (4, 5, 2, 1, 2, 4),
    "Chapter 8 Example": (3, 3, 0, 1, 0, 1),
}

# phi exactly as printed in the Sage docstrings (up to a constant factor).
SAGE_PHI = {
    "Conic Section": "x1^2 + x2^2 - x3^2",
    "Cuspidal Cubic": "x1^3 + x1*x2^2 - x2^2*x3",
    "Trident Curve": "x1^2*x2 + x2^3 + x1^2*x3 - x2^2*x3",
    "Unicuspidal Quartic": "x1^2*x2^2 + 529/128*x2^4 + 32/3*x1^3*x3 + 40*x1*x2^2*x3 "
                           "- 32*x1^2*x3^2 + 55*x2^2*x3^2 + 32*x1*x3^3 - 32/3*x3^4",
    "Bicuspidal Quartic": "x1^2*x2^2 - 2*x1^2*x2*x3 + x1*x2^2*x3 + x1^2*x3^2 "
                          "- 2*x1*x2*x3^2 + x2^2*x3^2",
    "Trinodal Quartic": "x1^4 + 8*x1^2*x2^2 + 16*x2^4 - 18/5*x1^3*x3 + 216/5*x1*x2^2*x3 "
                        "+ 17/5*x1^2*x3^2 + 68/5*x2^2*x3^2 - 4/5*x3^4",
    "Chapter 4 Example": "x1^2*x2^2 - 2*x1^2*x2*x3 - 2*x1*x2^2*x3 + x1^2*x3^2 "
                         "- 2*x1*x2*x3^2 + x2^2*x3^2",
    "Chapter 5 Example": "x1^2*x2^2 - 2*x1^2*x2*x3 - 2*x1*x2^2*x3 + x1^2*x3^2 "
                         "+ x1*x2*x3^2 + x2^2*x3^2",
    "Chapter 6 Example": "-x1^2*x2^2 - 1/144*x1^3*x3 - 47/48*x1^2*x2*x3 + 93/32*x1*x2^2*x3 "
                         "+ 9/64*x2^3*x3 - 7679/36864*x1^2*x3^2 + 8735/3072*x1*x2*x3^2 "
                         "- 1943/1024*x2^2*x3^2 + 2655/4096*x1*x3^3 - 4005/2048*x2*x3^3 "
                         "- 7575/16384*x3^4",
    "Chapter 8 Example": "x1^2*x2 + 6*x1*x2^2 + 36*x2^3 - x1^2*x3 + 12*x1*x2*x3 - 36*x2^2*x3",
}

failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + message)
    if not ok:
        failures.append(message)


def to_sympy(p: hc.QPoly):
    return sum(sp.Rational(c.numerator, c.denominator) * t**i for i, c in enumerate(p.c))


def strip_power(poly, gens):
    """phi0 / gcd(phi0, x3^100), as in the Sage code."""
    P = sp.Poly(sp.expand(poly), *gens)
    k = min(m[-1] for m in P.monoms())
    return sp.expand(P.as_expr() / gens[-1] ** k)


def sage_algorithm(name: str):
    """haynes_thesis_info(name, "Curve Data"), redone in sympy."""
    f = [to_sympy(p) for p in hc.PARAMETRIZATIONS[name]]
    g = sp.gcd(sp.gcd(f[0], f[1]), f[2])
    ff = [sp.cancel(e / g) for e in f]
    phi = strip_power(sp.resultant(ff[2] * x1 - ff[0] * x3, ff[2] * x2 - ff[1] * x3, t),
                      (x1, x2, x3))
    d = [sp.expand(f[1] * sp.diff(f[2], t) - f[2] * sp.diff(f[1], t)),
         sp.expand(f[2] * sp.diff(f[0], t) - f[0] * sp.diff(f[2], t)),
         sp.expand(f[0] * sp.diff(f[1], t) - f[1] * sp.diff(f[0], t))]
    gd = sp.gcd(sp.gcd(d[0], d[1]), d[2])
    rd = [sp.cancel(e / gd) for e in d]
    psi = strip_power(sp.resultant(rd[2] * u1 - rd[0] * u3, rd[2] * u2 - rd[1] * u3, t),
                      (u1, u2, u3))
    n = sp.Poly(phi, x1, x2, x3).total_degree()
    nu = sp.Poly(psi, u1, u2, u3).total_degree()
    plucker = (n, nu, (n - 1) * (n - 6) // 2 + nu, 2 * n - nu - 2,
               (nu - 1) * (nu - 6) // 2 + n, 2 * nu - n - 2)
    return phi, plucker


def proportional(a, b, gens) -> bool:
    A, B = sp.Poly(a, *gens), sp.Poly(b, *gens)
    if set(A.monoms()) != set(B.monoms()):
        return False
    m0 = A.monoms()[0]
    ratio = A.coeff_monomial(m0) / B.coeff_monomial(m0)
    return sp.expand(A.as_expr() - ratio * B.as_expr()) == 0


def main() -> int:
    print("1-2. Plücker numbers and phi, port vs. Sage algorithm vs. Sage docstrings")
    for name in hc.CURVE_NAMES:
        print(f" {name}")
        phi, sage_numbers = sage_algorithm(name)
        p = hc.plucker_numbers(name)
        port = (p["degree"], p["class"], p["nodes"], p["cusps"], p["bitangents"], p["flexes"])
        check(port == sage_numbers, f"port {port} == Sage algorithm {sage_numbers}")
        check(port == SAGE_DOCSTRING[name], f"port {port} == Sage docstring {SAGE_DOCSTRING[name]}")
        f = [to_sympy(q) for q in hc.PARAMETRIZATIONS[name]]
        on_curve = sp.expand(phi.subs({x1: f[0], x2: f[1], x3: f[2]}, simultaneous=True)) == 0
        check(on_curve, "parametrization satisfies phi exactly")
        if name in SAGE_PHI:
            printed = sp.sympify(SAGE_PHI[name].replace("^", "**"))
            check(proportional(phi, printed, (x1, x2, x3)), "phi matches the Sage docstring")

    print(" 'Tricuspidal Quartic' key vs 'Unicuspidal Quartic' key")
    check(hc.PARAMETRIZATIONS["Tricuspidal Quartic"] == hc.PARAMETRIZATIONS["Unicuspidal Quartic"],
          "same parametrization (so the three-cusped curve is 'Chapter 4 Example')")

    print("3. Facts the animation relies on (tricuspidal quartic = Chapter 4 Example)")
    name = hc.TRICUSPIDAL
    phi, _ = sage_algorithm(name)
    X, Y, Z = sp.symbols("X Y Z")
    check(proportional(phi, hc.TRICUSPIDAL_F(x1, x2, x3), (x1, x2, x3)),
          "TRICUSPIDAL_F (from plot_tricuspidal_quartic) is phi for Chapter 4")

    f = [to_sympy(q) for q in hc.parametrization(name)]
    ell = [to_sympy(q) for q in hc.dual_parametrization(name)]
    incid = sp.expand(sum(a * b for a, b in zip(ell, f)))
    touch = sp.expand(sum(a * sp.diff(b, t) for a, b in zip(ell, f)))
    check(incid == 0 and touch == 0, "reduced dual line passes through, and is tangent at, every point")
    print("     reduced dual parametrization:", [sp.factor(e) for e in ell])

    phis = np.linspace(-np.pi / 2, np.pi / 2, 20001)
    pts = hc.points(name, phis)
    lines = hc.tangent_lines(name, phis)
    scale = np.linalg.norm(pts, axis=1) * np.linalg.norm(lines, axis=1)
    check(np.max(np.abs(np.sum(pts * lines, axis=1)) / scale) < 1e-12,
          "numeric points() lie on numeric tangent_lines(), t = infinity included")

    D = hc.transform_points(hc.TO_DELTOID, pts)
    xy = hc.dehomogenize(D)
    resid = np.max(np.abs(hc.deltoid_equation(xy[:, 0], xy[:, 1])))
    check(resid < 1e-9, f"TO_DELTOID image lies on the deltoid (max residual {resid:.1e})")
    cusps = hc.dehomogenize(hc.transform_points(hc.TO_DELTOID, hc.points(name, hc.TRICUSPIDAL_CUSP_PHIS)))
    targets = {(3.0, 0.0), (-1.5, 1.5 * 3**0.5), (-1.5, -1.5 * 3**0.5)}
    check(all(min(np.hypot(c[0] - a, c[1] - b) for a, b in targets) < 1e-12 for c in cusps),
          "cusps t = 0, 1, infinity go to the deltoid's three cusps")
    L = hc.transform_lines(hc.TO_DELTOID, lines)
    check(np.max(np.abs(np.sum(D * L, axis=1)) / (np.linalg.norm(D, axis=1) * np.linalg.norm(L, axis=1))) < 1e-12,
          "transformed lines stay tangent to the transformed curve")
    u = np.linspace(0, 2 * np.pi, 2001)
    dx, dy = 2 * np.cos(u) + np.cos(2 * u), 2 * np.sin(u) - np.sin(2 * u)
    gap = max(np.min(np.hypot(xy[:, 0] - a, xy[:, 1] - b)) for a, b in zip(dx, dy))
    check(gap < 1e-2, f"image covers the whole deltoid 2cos u + cos 2u, 2sin u - sin 2u (gap {gap:.1e})")

    s = sp.symbols("s")
    bit = sp.factor(hc.TRICUSPIDAL_F(s, 1, -s - 1))
    check(sp.simplify(bit - (s**2 + s + 1)**2) == 0,
          "X + Y + Z = 0 is the bitangent, touching at the complex roots of X^2 + X + 1")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
