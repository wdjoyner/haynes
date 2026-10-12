"""Rational plane curves from Euphemia Haynes's 1943 thesis, in plain Python.

Ported from ``haynes-project-code_goins2.sage`` (Sage code by Edray Goins
and David Joyner) so that the Manim scenes do not need Sage. Only the
Python standard library (``fractions``) and numpy are used.

What is ported, function by function, from the Sage file:

* ``haynes_thesis_info`` -> :data:`PARAMETRIZATIONS` (the same eleven
  parametrizations ``t -> [f1(t) : f2(t) : f3(t)]``, coefficient for
  coefficient), :func:`dual_parametrization` (the same formula,
  ``[f2 f3' - f3 f2', f3 f1' - f1 f3', f1 f2' - f2 f1']``), the common-factor
  reduction the Sage code does before taking resultants (``ff = f/gcd``,
  ``ru = rho/gcd``), and the Plücker numbers with the same formulas.
* ``plot_tricuspidal_quartic`` -> :data:`TRICUSPIDAL_F`, the defining
  quartic of the tricuspidal curve (= "Chapter 4 Example").

The Sage code finds the degree ``n`` and class ``nu`` from the resultant
polynomials phi and psi. Here they are read off as the degrees of the
reduced parametrizations, which agree with the resultants because each
parametrization is generically one-to-one. ``check_haynes_curves.py``
confirms that, and every other number here, against an independent sympy
re-run of the Sage algorithm and against the outputs recorded in the Sage
docstrings.

Note on names: in the Sage file the key ``"Tricuspidal Quartic"`` holds the
same parametrization as ``"Unicuspidal Quartic"`` (one cusp, class 5). The
three-cusped quartic of Haynes's Chapters 4 and 7 is ``"Chapter 4
Example"``; :data:`TRICUSPIDAL` names it.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Exact one-variable polynomials over Q (just enough to port the Sage code)
# ---------------------------------------------------------------------------

class QPoly:
    """A polynomial in t with rational coefficients, lowest degree first."""

    __slots__ = ("c",)

    def __init__(self, coeffs: Sequence = (0,)):
        c = [Fraction(x) for x in coeffs] or [Fraction(0)]
        while len(c) > 1 and c[-1] == 0:
            c.pop()
        self.c = tuple(c)

    @classmethod
    def t(cls) -> "QPoly":
        return cls((0, 1))

    @staticmethod
    def _lift(x) -> "QPoly":
        return x if isinstance(x, QPoly) else QPoly((x,))

    @property
    def degree(self) -> int:
        return -1 if self.is_zero() else len(self.c) - 1

    def is_zero(self) -> bool:
        return len(self.c) == 1 and self.c[0] == 0

    def __add__(self, other):
        o = self._lift(other)
        n = max(len(self.c), len(o.c))
        a = self.c + (Fraction(0),) * (n - len(self.c))
        b = o.c + (Fraction(0),) * (n - len(o.c))
        return QPoly([x + y for x, y in zip(a, b)])

    __radd__ = __add__

    def __neg__(self):
        return QPoly([-x for x in self.c])

    def __sub__(self, other):
        return self + (-self._lift(other))

    def __rsub__(self, other):
        return self._lift(other) - self

    def __mul__(self, other):
        o = self._lift(other)
        out = [Fraction(0)] * (len(self.c) + len(o.c) - 1)
        for i, x in enumerate(self.c):
            if x:
                for j, y in enumerate(o.c):
                    out[i + j] += x * y
        return QPoly(out)

    __rmul__ = __mul__

    def __pow__(self, k: int):
        out = QPoly((1,))
        for _ in range(k):
            out = out * self
        return out

    def __eq__(self, other):
        return isinstance(other, QPoly) and self.c == other.c

    def __hash__(self):
        return hash(self.c)

    def __repr__(self):
        return f"QPoly({[str(x) for x in self.c]})"

    def diff(self) -> "QPoly":
        return QPoly([i * x for i, x in enumerate(self.c)][1:] or [0])

    def divmod(self, d: "QPoly") -> tuple["QPoly", "QPoly"]:
        if d.is_zero():
            raise ZeroDivisionError("polynomial division by zero")
        r = list(self.c)
        q = [Fraction(0)] * max(len(r) - len(d.c) + 1, 1)
        while len(r) >= len(d.c) and any(r):
            k = len(r) - len(d.c)
            f = r[-1] / d.c[-1]
            q[k] = f
            for i, y in enumerate(d.c):
                r[i + k] -= f * y
            r.pop()
        return QPoly(q), QPoly(r or [0])

    def monic(self) -> "QPoly":
        return QPoly([x / self.c[-1] for x in self.c])

    def as_float(self) -> np.ndarray:
        return np.array([float(x) for x in self.c])


def qgcd(*polys: QPoly) -> QPoly:
    """Monic gcd over Q (Sage's ``gcd([f1, f2, f3])``, up to a constant)."""
    g = QPoly((0,))
    for p in polys:
        a, b = g, p
        while not b.is_zero():
            a, b = b, a.divmod(b)[1]
        g = a
    return g.monic() if not g.is_zero() else g


# ---------------------------------------------------------------------------
# The thesis curves (haynes_thesis_info in the Sage file)
# ---------------------------------------------------------------------------

def _parametrizations() -> dict[str, tuple[QPoly, QPoly, QPoly]]:
    t = QPoly.t()
    F = Fraction
    out: dict[str, tuple[QPoly, QPoly, QPoly]] = {}
    out["Conic Section"] = (1 - t**2, 2 * t, 1 + t**2)
    out["Cuspidal Cubic"] = (t**2, t**3, 1 + t**2)
    out["Trident Curve"] = (t * (1 - t**2), 1 - t**2, 1 + t**2)
    # Sage key "Tricuspidal Quartic": the same curve as "Unicuspidal Quartic".
    out["Tricuspidal Quartic"] = (3 - 6 * t**2 - t**4, 8 * t**3, 3 * (1 + 2 * t**2))
    alpha = -1
    out["Bicuspidal Quartic"] = (t**2 - alpha * t + 1, t**2, t**2 * (t**2 - alpha * t + 1))
    out["Unicuspidal Quartic"] = (-t**4 - 6 * t**2 + 3, 8 * t**3, 6 * t**2 + 3)
    r = F(1, 2)
    out["Trinodal Quartic"] = (
        2 * (-r * t**4 - 6 * t**2 + (r + 2)),
        -2 * t * ((r + 3) * t**2 + (r - 1)),
        (r + 2) * (t**2 + 1)**2,
    )
    out["Chapter 4 Example"] = (t**2 * (t - 1)**2, (t - 1)**2, t**2)
    alpha = 1
    out["Chapter 5 Example"] = (t**2 - alpha * t + 1, t**2 * (t**2 - alpha * t + 1), t**2)
    alpha, beta = F(1, 3), F(2, 3)
    out["Chapter 6 Example"] = (
        3 * ((t - 1)**2 * (t - alpha) + 32 * t**2),
        t * (t - 1)**2 * (t - beta) - 32 * t**2,
        64 * t**2,
    )
    a, b, c = 1, F(1, 2), 1
    out["Chapter 8 Example"] = (
        2 * (t - 1)**2 * (a**2 + 2 * b + c),
        t * (t - 1)**2,
        t * (a * t**2 + 2 * b * t + c),
    )
    return out


PARAMETRIZATIONS = _parametrizations()
CURVE_NAMES = tuple(PARAMETRIZATIONS)

#: The three-cusped (tricuspidal) quartic of Haynes's Chapters 4 and 7.
TRICUSPIDAL = "Chapter 4 Example"


def _reduce(polys: Sequence[QPoly]) -> tuple[QPoly, QPoly, QPoly]:
    g = qgcd(*polys)
    return tuple(p.divmod(g)[0] for p in polys)  # type: ignore[return-value]


def parametrization(name: str, reduced: bool = True) -> tuple[QPoly, QPoly, QPoly]:
    """``[f1, f2, f3]``; *reduced* divides out their common factor (Sage's ``ff``)."""
    f = PARAMETRIZATIONS[name]
    return _reduce(f) if reduced else f


def dual_parametrization(name: str, reduced: bool = True) -> tuple[QPoly, QPoly, QPoly]:
    """Tangent line at the point with parameter t, as line coordinates [u1:u2:u3].

    The Sage formula ``[f2 f3' - f3 f2', f3 f1' - f1 f3', f1 f2' - f2 f1']``
    (the cross product of the point and its derivative). All three vanish
    at a cusp; *reduced* divides out the common factor (Sage's ``ru``) so
    the tangent line is defined there too.
    """
    f1, f2, f3 = PARAMETRIZATIONS[name]
    d = (f2 * f3.diff() - f3 * f2.diff(),
         f3 * f1.diff() - f1 * f3.diff(),
         f1 * f2.diff() - f2 * f1.diff())
    return _reduce(d) if reduced else d


def plucker_numbers(name: str) -> dict[str, int]:
    """Degree, class, and the four Plücker counts, as in haynes_thesis_info."""
    n = max(p.degree for p in parametrization(name))
    nu = max(p.degree for p in dual_parametrization(name))
    return {
        "degree": n,
        "class": nu,
        "nodes": (n - 1) * (n - 6) // 2 + nu,
        "cusps": 2 * n - nu - 2,
        "bitangents": (nu - 1) * (nu - 6) // 2 + n,
        "flexes": 2 * nu - n - 2,
    }


def TRICUSPIDAL_F(X, Y, Z):
    """Defining quartic of the tricuspidal curve (``plot_tricuspidal_quartic``)."""
    return (X**2 * Y**2 - 2 * X**2 * Y * Z - 2 * X * Y**2 * Z + X**2 * Z**2
            - 2 * X * Y * Z**2 + Y**2 * Z**2)


# ---------------------------------------------------------------------------
# Numerical evaluation, including the parameter value t = infinity
# ---------------------------------------------------------------------------

def _homogeneous_eval(polys: Sequence[QPoly], s, w) -> np.ndarray:
    """Evaluate polynomials at t = s/w homogeneously (w = 0 is t = infinity)."""
    d = max(p.degree for p in polys)
    s = np.asarray(s, dtype=float)
    w = np.asarray(w, dtype=float)
    out = []
    for p in polys:
        c = p.as_float()
        out.append(sum(ci * s**i * w**(d - i) for i, ci in enumerate(c)))
    return np.stack(out, axis=-1)


def points(name: str, phi) -> np.ndarray:
    """Homogeneous points at t = tan(phi); phi in [-pi/2, pi/2] covers the curve once."""
    phi = np.asarray(phi, dtype=float)
    return _homogeneous_eval(parametrization(name), np.sin(phi), np.cos(phi))


def tangent_lines(name: str, phi) -> np.ndarray:
    """Homogeneous tangent lines [u1:u2:u3] at t = tan(phi) (reduced, so cusps are fine)."""
    phi = np.asarray(phi, dtype=float)
    return _homogeneous_eval(dual_parametrization(name), np.sin(phi), np.cos(phi))


# ---------------------------------------------------------------------------
# The tricuspidal quartic as the familiar deltoid
# ---------------------------------------------------------------------------

_R3 = 3.0 ** 0.5

#: Projective change of coordinates taking Haynes's tricuspidal quartic
#: (cusps at [1:0:0], [0:1:0], [0:0:1]; bitangent X + Y + Z = 0, touching at
#: the complex roots of X^2 + X + 1) to the deltoid with cusps
#: (3, 0), (-3/2, 3*sqrt(3)/2), (-3/2, -3*sqrt(3)/2): each cusp goes to a
#: cusp and the bitangent goes to the line at infinity. The image is
#: exactly (x^2 + y^2)^2 + 18 (x^2 + y^2) - 27 = 8 (x^3 - 3 x y^2), the
#: curve traced by x = 2 cos u + cos 2u, y = 2 sin u - sin 2u.
TO_DELTOID = np.array([
    [3.0, -1.5, -1.5],
    [0.0, 1.5 * _R3, -1.5 * _R3],
    [1.0, 1.0, 1.0],
])

#: Parameter values (phi, with t = tan(phi)) of the three cusps: t = 0, 1, infinity.
TRICUSPIDAL_CUSP_PHIS = (0.0, np.pi / 4, np.pi / 2)


def deltoid_equation(x, y):
    """Zero exactly on the deltoid that TO_DELTOID produces."""
    r2 = x**2 + y**2
    return r2**2 + 18 * r2 - 27 - 8 * (x**3 - 3 * x * y**2)


def transform_points(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply a projective map to homogeneous points (rows)."""
    return P @ np.asarray(M).T


def transform_lines(M: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Apply the same map to homogeneous lines: lines move by the inverse transpose."""
    return L @ np.linalg.inv(np.asarray(M))


def dehomogenize(P: np.ndarray) -> np.ndarray:
    """[x:y:z] -> (x/z, y/z)."""
    P = np.asarray(P, dtype=float)
    return P[..., :2] / P[..., 2:3]


def pole(L: np.ndarray, center: Sequence[float], radius: float) -> np.ndarray:
    """Pole of the line(s) ax + by + c = 0 in the circle (center, radius).

    Polar reciprocity is how a dual curve is drawn in an ordinary picture:
    each tangent line becomes a point. Lines through *center* go to
    infinity. Returned relative to *center*.
    """
    L = np.asarray(L, dtype=float)
    a, b, c = L[..., 0], L[..., 1], L[..., 2]
    cc = c + a * center[0] + b * center[1]
    return -radius**2 * np.stack([a, b], axis=-1) / cc[..., None]
