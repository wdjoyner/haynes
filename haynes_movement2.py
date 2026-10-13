#!/usr/bin/env python3
"""Euphemia Haynes — Mathematics as an Open Door: Lesson II.

(File and class names say "movement", from before the on-screen word became
"Lesson". The whole film, both lessons with one set of credits, is
``HaynesFilm`` in haynes_film.py.)

What her Ph.D. work looked like (see MOVEMENT2_PLAN.md). Built so far:

* Scene 11, duality: a tangent line rolls around the three-cuspided
  (tricuspidal) quartic of Haynes's thesis while its dual curve draws
  itself; then the curve deforms and its Plücker counts stay fixed.
* Closing, first part: the Simson lines of Movement I's 3-4-5 triangle
  sweep out the same three-cuspided quartic.

The curves come from ``haynes_curves.py``, a numpy port of the Sage code by
Edray Goins and David Joyner (``check_haynes_curves.py`` verifies the port).
The text helpers, narration timing and colors are imported from Movement I.

Generate narration first (or ``--placeholder`` for silent preview clips)::

    python generate_movement2_audio.py --voice af_heart --speed 0.94

Low-resolution preview of the whole movement, or of one part::

    manim -pql haynes_movement2.py HaynesMovementTwo
    manim -pql haynes_movement2.py DualityScene
    manim -pql haynes_movement2.py SimsonClosing
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from manim import *

import haynes_curves as hc
import haynes_movement1 as lesson1
from haynes_movement1 import (CHALK, DOOR_LIGHT, DOOR_OPEN_ANGLE, CurveText, NarratedScene,
                              SwingingDoor, write_along)

HERE = Path(__file__).resolve().parent
AUDIO_DIR = HERE / "audio" / "movement2"
TEXT_PATH = HERE / "haynes_movement2_text.json"


def _load_text(path: Path = TEXT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; it holds every word in Movement II.")
    return json.loads(path.read_text(encoding="utf-8"))


TEXT = _load_text()

BACKGROUND = "#10131a"          # same as Movement I
LINE_GOLD = DOOR_LIGHT          # tangent lines, and the dual curve they make
CUSP_COLOR = "#e8896b"          # cusps, and the flexes they become
NODE_COLOR = "#8fc1e3"          # the bitangent, and the isolated node it becomes


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------

def _p3(xy) -> np.ndarray:
    return np.array([float(xy[0]), float(xy[1]), 0.0])


def clip_line(point, direction, box, half_length=None):
    """The part of a line inside a box (xmin, xmax, ymin, ymax), as two points.

    With *half_length*, the line is first cut to that distance either side
    of *point*. Returns None if nothing is left.
    """
    p = np.asarray(point, dtype=float)[:2]
    d = np.asarray(direction, dtype=float)[:2]
    d = d / np.linalg.norm(d)
    lo, hi = (-half_length, half_length) if half_length else (-1e6, 1e6)
    xmin, xmax, ymin, ymax = box
    for k, (a, b) in enumerate(((xmin, xmax), (ymin, ymax))):
        if abs(d[k]) < 1e-12:
            if not a <= p[k] <= b:
                return None
            continue
        s0, s1 = sorted(((a - p[k]) / d[k], (b - p[k]) / d[k]))
        lo, hi = max(lo, s0), min(hi, s1)
    if hi <= lo:
        return None
    return _p3(p + lo * d), _p3(p + hi * d)


def runs_inside(points: np.ndarray, box, max_jump: float = 1.5) -> list[np.ndarray]:
    """Split a sampled curve into the unbroken pieces lying inside a box.

    A curve through infinity (as a dual curve may be) leaves the box and
    comes back from the other side; the pieces are drawn separately so no
    stray segment joins them.
    """
    xmin, xmax, ymin, ymax = box
    inside = ((points[:, 0] >= xmin) & (points[:, 0] <= xmax)
              & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
              & np.isfinite(points).all(axis=1))
    runs, current = [], []
    for i, ok in enumerate(inside):
        if ok and (not current or np.linalg.norm(points[i] - current[-1]) < max_jump):
            current.append(points[i])
        else:
            if len(current) > 1:
                runs.append(np.array(current))
            current = [points[i]] if ok else []
    if len(current) > 1:
        runs.append(np.array(current))
    return runs


def polyline_group(runs, color, width, opacity=1.0) -> VGroup:
    group = VGroup()
    for run in runs:
        pts = np.column_stack([run[:, 0], run[:, 1], np.zeros(len(run))])
        group.add(VMobject().set_points_as_corners(pts)
                  .set_stroke(color, width=width, opacity=opacity))
    return group


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------
# Scene 11 geometry: the three-cuspided quartic and its dual
# ---------------------------------------------------------------------------

class TricuspidalPicture:
    """Screen geometry of Haynes's three-cuspided quartic and its dual.

    The curve is ``"Chapter 4 Example"`` from the Sage code, moved by
    ``hc.TO_DELTOID`` into the familiar deltoid shape, then by a projective
    deformation ``T`` (identity at first), then rotated, scaled and placed
    in the left panel. Each tangent line is shown in the right panel as its
    pole in a fixed circle (polar reciprocity), which turns the family of
    tangent lines into the dual curve.

    Points along the curve are indexed by ``u`` in [0, 1), proportional to
    arc length on the undeformed deltoid, so the tangent line rolls at an
    even speed. The cusps are at u = 0, 1/3, 2/3.
    """

    SAMPLES = 1500

    LEFT_CENTER = np.array([-3.55, -0.45])
    CURVE_SCALE = 0.85
    CURVE_ROTATION = PI / 2            # one cusp pointing straight up
    LEFT_BOX = (-6.95, -0.25, -2.75, 2.65)

    # Polar reciprocity, in the deltoid's own coordinates (cusp radius 3):
    # a circle centered below the curve. Its center lies on none of the
    # cusp tangents, so all three flexes of the dual stay in view.
    POLE_CENTER = (0.0, -4.0)
    POLE_RADIUS = 2.6
    RIGHT_CENTER = np.array([3.75, -0.65])
    DUAL_SCALE = 0.75
    RIGHT_BOX = (0.25, 6.95, -2.75, 2.65)

    # Keyframes of the deformation (projective maps of the deltoid's plane).
    KEYFRAMES = [
        np.eye(3),
        np.array([[1.10, 0.18, 0.0], [0.0, 0.92, 0.0], [0.04, 0.045, 1.0]]),
        np.array([[0.94, -0.12, 0.0], [0.08, 1.06, 0.0], [-0.03, 0.02, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.12, 0.0], [0.0, -0.06, 1.0]]),
        np.eye(3),
    ]

    def __init__(self):
        name = hc.TRICUSPIDAL
        self.plucker = hc.plucker_numbers(name)

        # phi parametrizes the curve with t = tan(phi); phi in [-pi/2, pi/2].
        phis = np.linspace(-PI / 2, PI / 2, 8001)
        P = hc.transform_points(hc.TO_DELTOID, hc.points(name, phis))
        xy = hc.dehomogenize(P)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        self.u_grid = np.linspace(0.0, 1.0, self.SAMPLES + 1)
        self.phi_grid = np.interp(self.u_grid, arc / arc[-1], phis)
        self.P0 = hc.transform_points(hc.TO_DELTOID, hc.points(name, self.phi_grid))
        self.L0 = hc.transform_lines(hc.TO_DELTOID, hc.tangent_lines(name, self.phi_grid))
        # The cusps t = infinity, 0, 1 (phi = -pi/2, 0, pi/4); by symmetry
        # they fall at u = 0, 1/3, 2/3.
        self.cusp_u = [0.0] + [float(np.interp(p, phis, arc / arc[-1]))
                               for p in hc.TRICUSPIDAL_CUSP_PHIS[:2]]
        # The bitangent: the line at infinity of the deltoid's plane.
        self.bitangent = hc.transform_lines(
            hc.TO_DELTOID, np.array([[1.0, 1.0, 1.0]]))[0]   # X + Y + Z = 0
        c, s = math.cos(self.CURVE_ROTATION), math.sin(self.CURVE_ROTATION)
        self._rot = np.array([[c, -s], [s, c]])

    # -- deformation -------------------------------------------------------

    def deformation(self, s: float) -> np.ndarray:
        """Projective map at time s in [0, len(KEYFRAMES) - 1]."""
        k = len(self.KEYFRAMES) - 1
        s = min(max(s, 0.0), float(k))
        i = min(int(s), k - 1)
        w = smoothstep(s - i)
        return (1 - w) * self.KEYFRAMES[i] + w * self.KEYFRAMES[i + 1]

    # -- evaluation at parameter u -----------------------------------------

    def _rows(self, A: np.ndarray, u):
        """Linear interpolation of homogeneous rows on the u grid (u mod 1)."""
        u = np.atleast_1d(np.asarray(u, dtype=float)) % 1.0
        x = u * self.SAMPLES
        i = np.minimum(x.astype(int), self.SAMPLES - 1)
        w = (x - i)[:, None]
        a, b = A[i], A[i + 1]
        # Homogeneous rows may flip sign between grid points; align them.
        b = np.where((np.sum(a * b, axis=1) < 0)[:, None], -b, b)
        return (1 - w) * a + w * b

    def curve_screen(self, T, u=None) -> np.ndarray:
        P = self.P0 if u is None else self._rows(self.P0, u)
        xy = hc.dehomogenize(hc.transform_points(T, P))
        return self.LEFT_CENTER + self.CURVE_SCALE * xy @ self._rot.T

    def lines_model(self, T, u=None) -> np.ndarray:
        L = self.L0 if u is None else self._rows(self.L0, u)
        return hc.transform_lines(T, L)

    def tangent_screen(self, T, u: float):
        """(point of tangency, direction) of the tangent line, on screen."""
        p = self.curve_screen(T, u)[0]
        a, b, _ = self.lines_model(T, u)[0]
        d = self._rot @ np.array([-b, a])
        return p, d / np.linalg.norm(d)

    def dual_screen(self, T, u=None, lines=None) -> np.ndarray:
        L = self.lines_model(T, u) if lines is None else hc.transform_lines(T, lines)
        q = hc.pole(L, self.POLE_CENTER, self.POLE_RADIUS)
        return self.RIGHT_CENTER + self.DUAL_SCALE * q

    def node_screen(self, T) -> np.ndarray:
        """The isolated point of the dual: the dual of the bitangent."""
        return self.dual_screen(T, lines=self.bitangent[None, :])[0]

    def dual_text_path(self, T, samples: int = 240, smoothing: float = 0.15):
        """A function a -> point riding the dual's in-view stretch, for CurveText.

        Follows the dual from the flex at u = 2/3 through the one at u = 0
        to the one at u = 1/3, a stretch that never passes through infinity
        during the deformation. The path is resampled evenly by arc length
        and smoothed (a Gaussian average over *smoothing* of its length) so
        letters don't bunch up where the dual bends sharply; it still moves
        with the dual as it deforms.
        """
        raw = self.dual_screen(T, (2 / 3 + np.linspace(0.0, 1.0, 1200) * 2 / 3) % 1.0)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))])
        even = np.linspace(0.0, arc[-1], samples)
        pts = np.column_stack([np.interp(even, arc, raw[:, j]) for j in range(2)])
        sigma = max(smoothing * samples, 1e-6)
        k = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
        w = np.exp(-0.5 * (k / sigma) ** 2)
        w /= w.sum()
        h = len(k) // 2
        # Pad by point reflection through each end, so the ends stay put.
        padded = np.vstack([2 * pts[0] - pts[h:0:-1], pts, 2 * pts[-1] - pts[-2:-h - 2:-1]])
        smooth = np.column_stack([np.convolve(padded[:, j], w, mode="valid") for j in range(2)])

        def f(x: float):
            i = float(np.clip(x, 0.0, 1.0)) * (samples - 1)
            i0 = min(int(i), samples - 2)
            return _p3(smooth[i0] + (i - i0) * (smooth[i0 + 1] - smooth[i0]))
        return f


# ---------------------------------------------------------------------------
# Closing geometry: Simson lines of the 3-4-5 triangle
# ---------------------------------------------------------------------------

class SimsonPicture:
    """The right triangle of Movement I, its Simson lines and their envelope.

    Unit coordinates as in Movement I: A = (-2, -1.5), B = (2, -1.5),
    C = (2, 1.5), right angle at B, circumcircle centered at the origin
    with radius 5/2. The envelope of the Simson lines is a deltoid centered
    at the nine-point center (here halfway between the origin and B).
    """

    A = np.array([-2.0, -1.5])
    B = np.array([2.0, -1.5])
    C = np.array([2.0, 1.5])
    O = np.array([0.0, 0.0])
    R = 2.5

    # Placement in Movement I (haynes_movement1.scene_answer_to_reason) ...
    M1_SCALE = 0.55
    M1_SHIFT = np.array([-1.2, -0.35])
    # ... and here, sized so the deltoid fills the frame.
    SCALE = 0.82
    NINE_POINT_CENTER_SCREEN = np.array([0.0, -0.3])

    def __init__(self):
        H = self.A + self.B + self.C - 2 * self.O          # orthocenter (= B here)
        self.N = (self.O + H) / 2                          # nine-point center
        self.shift = self.NINE_POINT_CENTER_SCREEN - self.SCALE * self.N
        self.sides = [(self.A, self.B), (self.B, self.C), (self.C, self.A)]
        self.thetas = np.linspace(0, TAU, 2401)
        self.envelope = self._envelope(self.thetas)

    def screen(self, p, m1: bool = False) -> np.ndarray:
        s, sh = (self.M1_SCALE, self.M1_SHIFT) if m1 else (self.SCALE, self.shift)
        return _p3(s * np.asarray(p, dtype=float) + sh)

    def point_on_circle(self, theta: float) -> np.ndarray:
        return self.O + self.R * np.array([math.cos(theta), math.sin(theta)])

    @staticmethod
    def foot(p, a, b) -> np.ndarray:
        """Foot of the perpendicular from p to the line through a and b."""
        d = b - a
        return a + np.dot(p - a, d) / np.dot(d, d) * d

    def feet(self, theta: float) -> list[np.ndarray]:
        p = self.point_on_circle(theta)
        return [self.foot(p, a, b) for a, b in self.sides]

    def simson_line(self, theta: float) -> np.ndarray:
        """Homogeneous coordinates (a, b, c) of the Simson line, unit normal."""
        f = self.feet(theta)
        # Through the two feet farthest apart (at a vertex two feet coincide).
        pairs = [(0, 1), (1, 2), (0, 2)]
        i, j = max(pairs, key=lambda ij: np.linalg.norm(f[ij[0]] - f[ij[1]]))
        L = np.cross([*f[i], 1.0], [*f[j], 1.0])
        L = L / np.linalg.norm(L[:2])
        # Fix the sign so the family varies continuously with theta.
        ref = np.array([math.cos(-theta / 2), math.sin(-theta / 2)])
        return L if np.dot(L[:2], ref) >= 0 else -L

    def _envelope(self, thetas) -> np.ndarray:
        """Envelope of the Simson lines, as each line meets its neighbor.

        The point where the line L(theta) touches the envelope is
        L(theta) x L'(theta) -- the same cross product that turns a curve
        into its dual in haynes_curves.
        """
        h = 1e-5
        pts = []
        for th in thetas:
            L = self.simson_line(th)
            dL = (self.simson_line(th + h) - self.simson_line(th - h)) / (2 * h)
            pts.append(hc.dehomogenize(np.cross(L, dL)))
        return np.array(pts)

    def envelope_screen(self) -> np.ndarray:
        return self.SCALE * self.envelope + self.shift

    def deltoid_similarity(self):
        """(scale, rotation, center) taking the model deltoid (cusp radius 3,
        cusp at angle 0) onto the Simson envelope, in screen coordinates."""
        rel = self.envelope - self.N
        k = int(np.argmax(np.linalg.norm(rel, axis=1)))
        cusp_angle = math.atan2(rel[k, 1], rel[k, 0])
        radius = float(np.linalg.norm(rel[k]))            # = 3R/2
        return self.SCALE * radius / 3.0, cusp_angle, self.NINE_POINT_CENTER_SCREEN


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

class MovementTwoScene(NarratedScene):
    """Lesson I's narrated scene, pointed at Lesson II's audio and text."""

    audio_dir = AUDIO_DIR
    text = TEXT
    audio_script = "generate_movement2_audio.py"

    def setup(self):
        super().setup()
        self.camera.background_color = BACKGROUND

    def finish_clip(self, clip_id: str, minimum_pause: float = 0.4):
        """Wait out the rest of a narration clip."""
        end = self._clip_starts[clip_id] + self.narration_duration(clip_id)
        self.wait(max(end - self.renderer.time, minimum_pause))

    def lesson_two(self):
        """Lesson II, title and scenes built so far (the film reuses this)."""
        self.camera.background_color = BACKGROUND
        self.mark_chapter("lesson2")
        self.lesson_two_title()
        self.scene_duality()
        self.mark_chapter("simson")
        circle = self.scene_simson_closing()
        self.mark_chapter("door")
        self.scene_limit_and_door(circle)

    def lesson_two_title(self):
        """LESSON II and its subtitle, styled like Lesson I's title."""
        words = self.text["title"]
        title = VGroup(
            Text(words["lesson"], weight=BOLD, font_size=46),
            Text(words["subtitle"], font_size=32),
        ).arrange(DOWN, buff=0.28)
        self.play(FadeIn(title, shift=0.18 * UP), run_time=1.4)
        self.wait(1.8)
        self.play(FadeOut(title, shift=0.16 * UP), run_time=1.0)
        self.wait(0.3)

    # ------------------------------------------------------------------
    # Scene 11: duality
    # ------------------------------------------------------------------

    def scene_duality(self):
        clip = "m2_11_duality"
        words = self.text["duality"]
        cue = words["cues"]
        geo = TricuspidalPicture()
        self.start_narration(clip)

        # Sentence 1-2: Haynes's equation, in her coordinates.
        equation = MathTex(words["equation"], color=CHALK).scale(0.9)
        caption = Text(words["caption"], font_size=26, slant=ITALIC, color=CHALK)
        intro = VGroup(equation, caption).arrange(DOWN, buff=0.35).move_to(0.3 * UP)
        self.play(Write(equation), run_time=2.6)
        self.play(FadeIn(caption, shift=0.1 * UP), run_time=1.0)

        morph = ValueTracker(0.0)           # deformation time
        roll = ValueTracker(1 / 6)          # point of tangency (u)
        T = lambda: geo.deformation(morph.get_value())

        def curve_mob():
            return polyline_group(runs_inside(geo.curve_screen(T()), geo.LEFT_BOX, 9),
                                  CHALK, 4)

        def cusp_dots():
            return VGroup(*[Dot(_p3(p), radius=0.07, color=CUSP_COLOR)
                            for p in geo.curve_screen(T(), geo.cusp_u)])

        curve_title = Text(words["curve_title"], font_size=28, weight=BOLD, color=CHALK)
        curve_title.move_to([geo.LEFT_CENTER[0], 3.05, 0])
        dual_title = Text(words["dual_title"], font_size=28, weight=BOLD, color=LINE_GOLD)
        dual_title.move_to([geo.RIGHT_CENTER[0], 3.05, 0])

        self.wait_for(clip, cue["curve"])
        curve_static = curve_mob()
        self.play(
            intro.animate.scale(0.62).move_to([geo.LEFT_CENTER[0], -3.3, 0]),
            Create(curve_static), FadeIn(curve_title),
            run_time=2.6,
        )
        cusps = cusp_dots()
        self.play(LaggedStart(*[GrowFromCenter(d) for d in cusps], lag_ratio=0.3),
                  run_time=1.0)
        curve = always_redraw(curve_mob)
        self.remove(curve_static, cusps, *cusps)
        cusps = always_redraw(cusp_dots)
        self.add(curve, cusps)

        # Sentence 3: the tangent line, and its point in the dual picture.
        def tangent_mob():
            p, d = geo.tangent_screen(T(), roll.get_value())
            ends = clip_line(p, d, geo.LEFT_BOX, half_length=3.2)
            line = Line(*ends) if ends else VMobject()
            line.set_stroke(LINE_GOLD, width=3.5)
            return VGroup(line, Dot(_p3(p), radius=0.07, color=LINE_GOLD))

        def dual_dot():
            q = geo.dual_screen(T(), roll.get_value())[0]
            return Dot(_p3(q), radius=0.08, color=LINE_GOLD)

        def link():
            p, _ = geo.tangent_screen(T(), roll.get_value())
            q = geo.dual_screen(T(), roll.get_value())[0]
            return DashedVMobject(
                ArcBetweenPoints(_p3(p), _p3(q), angle=-0.5), num_dashes=40,
            ).set_stroke(LINE_GOLD, width=1.6, opacity=0.4)

        trail_start = roll.get_value()

        def trail():
            u_now = roll.get_value()
            if u_now <= trail_start:
                return VGroup()
            us = np.arange(trail_start, u_now, 1 / geo.SAMPLES)
            us = np.append(us, u_now)
            return polyline_group(
                runs_inside(geo.dual_screen(T(), us), geo.RIGHT_BOX), LINE_GOLD, 4)

        relation_arc = ArcBetweenPoints([-1.55, 3.0, 0], [1.55, 3.0, 0], angle=-0.9)
        relation_arc.set_stroke(CHALK, width=2, opacity=0.4)
        relation = CurveText(words["relation"], font_size=30, weight=BOLD, color=CHALK)
        relation.place_on(relation_arc, 0.08, 0.92, normal_offset=0.24)

        self.wait_for(clip, cue["roll"])
        tangent = always_redraw(tangent_mob)
        self.play(FadeIn(tangent), FadeIn(dual_title), run_time=0.9)
        moving_dot = always_redraw(dual_dot)
        link_arc = always_redraw(link)
        trail_mob = always_redraw(trail)
        self.play(FadeIn(moving_dot), FadeIn(link_arc), Create(relation_arc),
                  write_along(relation), run_time=1.6)
        self.add(trail_mob)

        # Sentences 3-4: one full turn, finishing as "Each cusp ..." begins.
        turn = max(self.time_until(clip, cue["cusp"]) - 0.2, 5.0)
        self.play(roll.animate.set_value(trail_start + 1.0), run_time=turn, rate_func=linear)

        # Sentence 5: each cusp <-> a flex of the dual.
        dual_full = always_redraw(lambda: polyline_group(
            runs_inside(geo.dual_screen(T()), geo.RIGHT_BOX), LINE_GOLD, 4))
        self.remove(trail_mob)
        self.add(dual_full)

        def flex_dots():
            return VGroup(*[Dot(_p3(q), radius=0.075, color=CUSP_COLOR)
                            for q in geo.dual_screen(T(), geo.cusp_u)])

        flexes_static = flex_dots()
        pairs = []
        for cusp_pt, flex in zip(geo.curve_screen(T(), geo.cusp_u), flexes_static):
            pairs.append(DashedVMobject(
                ArcBetweenPoints(_p3(cusp_pt), flex.get_center(), angle=-0.45),
                num_dashes=36).set_stroke(CUSP_COLOR, width=2, opacity=0.8))
        self.play(FadeOut(link_arc), run_time=0.3)
        for arc, flex in zip(pairs, flexes_static):
            self.play(Create(arc), GrowFromCenter(flex), run_time=0.75)
        self.play(*[FadeOut(a) for a in pairs], run_time=0.6)
        self.remove(flexes_static, *flexes_static)
        flexes = always_redraw(flex_dots)
        self.add(flexes)

        # Sentence 6: the bitangent becomes the isolated node.
        self.wait_for(clip, cue["node"])
        node = always_redraw(lambda: Dot(_p3(geo.node_screen(T())), radius=0.09,
                                         color=NODE_COLOR))
        self.play(GrowFromCenter(node), run_time=0.6)
        self.play(Flash(node, color=NODE_COLOR, line_length=0.25, flash_radius=0.3),
                  run_time=0.9)

        p = geo.plucker
        left_counts = VGroup(*[
            Text(s.format(**p), font_size=24, color=col)
            for s, col in zip(words["curve_counts"], (CHALK, CUSP_COLOR, NODE_COLOR))
        ]).arrange(RIGHT, buff=0.5).move_to([geo.LEFT_CENTER[0], -3.3, 0])
        right_counts = VGroup(*[
            Text(s.format(**p), font_size=24, color=col)
            for s, col in zip(words["dual_counts"], (LINE_GOLD, CUSP_COLOR, NODE_COLOR))
        ]).arrange(RIGHT, buff=0.5).move_to([geo.RIGHT_CENTER[0], -3.3, 0])
        self.play(FadeOut(intro), Write(left_counts), Write(right_counts), run_time=1.6)

        # Sentences 7-9: the curve changes; the counts do not.
        invariance = CurveText(words["invariance"], font_size=26, weight=BOLD, color=CHALK)
        ride = lambda m: m.place_on(geo.dual_text_path(T()), 0.12, 0.88, normal_offset=0.55)
        ride(invariance)
        self.wait_for(clip, cue["change"])
        self.play(write_along(invariance), run_time=1.2)
        invariance.add_updater(ride)

        last = len(geo.KEYFRAMES) - 1
        to_counts = max(self.time_until(clip, cue["counts"]), 2.0)
        self.play(morph.animate.set_value(2.0), run_time=to_counts, rate_func=linear)
        # The counts pulse while the curve keeps changing.
        to_end = max(self.time_until(clip, cue["invariance"]) + 1.0, 3.0)
        pulse = min(2.0, to_end / 2)
        mid = 2.0 + (last - 2.0) * pulse / to_end
        self.play(
            morph.animate(rate_func=linear).set_value(mid),
            LaggedStart(*[Indicate(m, scale_factor=1.15, color=m.get_color())
                          for m in (*left_counts, *right_counts)], lag_ratio=0.15),
            run_time=pulse,
        )
        self.play(morph.animate.set_value(float(last)), run_time=to_end - pulse,
                  rate_func=linear)

        # Sentence 10: RELATION -> STRUCTURE on the arc.
        invariance.clear_updaters()
        structure = CurveText(words["structure"], font_size=30, weight=BOLD, color=CHALK)
        structure.place_on(relation_arc, 0.04, 0.96, normal_offset=0.24)
        self.play(FadeOut(relation), FadeOut(invariance), run_time=0.6)
        self.play(write_along(structure), run_time=1.4)
        self.finish_clip(clip)

        self.play(FadeOut(Group(*self.mobjects)), run_time=1.2)
        self.clear()

    # ------------------------------------------------------------------
    # Closing: Simson lines sweep out the three-cuspided quartic
    # ------------------------------------------------------------------

    def scene_simson_closing(self):
        clip = "m2_closing_simson"
        words = self.text["simson"]
        cue = words["cues"]
        geo = SimsonPicture()
        self.start_narration(clip)

        # The triangle exactly as Movement I left it, then enlarged.
        s1, sh1 = geo.M1_SCALE, geo.M1_SHIFT
        A, B, C = (geo.screen(v, m1=True) for v in (geo.A, geo.B, geo.C))
        tri = Polygon(A, B, C).set_stroke(WHITE, width=4)
        labels = VGroup(
            MathTex(words["side_labels"][0]).scale(0.7).move_to(B + np.array([-0.45, 0.17, 0])),
            MathTex(words["side_labels"][1]).scale(0.7).move_to((B + C) / 2 + np.array([-0.2, 0.25, 0])),
            MathTex(words["side_labels"][2]).scale(0.7).move_to((A + C) / 2 + np.array([0.14, -0.18, 0])),
        )
        self.play(Create(tri), LaggedStart(*[FadeIn(x) for x in labels], lag_ratio=0.15),
                  run_time=2.0)
        k = geo.SCALE / s1
        move = geo.shift - k * sh1
        self.play(VGroup(tri, labels).animate.scale(k, about_point=ORIGIN).shift(_p3(move)),
                  run_time=1.8)

        # Sentence 2: a point on the circumcircle and its three perpendiculars.
        theta = ValueTracker(2.2)
        circle = Circle(radius=geo.SCALE * geo.R, color=CHALK).move_to(geo.screen(geo.O))
        circle.set_stroke(width=2.5, opacity=0.8)
        side_lines = VGroup(*[
            Line(geo.screen(a + 12 * (a - b) / np.linalg.norm(a - b)),
                 geo.screen(b + 12 * (b - a) / np.linalg.norm(a - b)))
            .set_stroke(WHITE, width=1.5, opacity=0.25)
            for a, b in geo.sides
        ])

        def P():
            return geo.screen(geo.point_on_circle(theta.get_value()))

        def feet_mob():
            p = P()
            group = VGroup()
            for f in geo.feet(theta.get_value()):
                fs = geo.screen(f)
                if np.linalg.norm(fs - p) > 1e-3:
                    group.add(DashedLine(p, fs, dash_length=0.08).set_stroke(CHALK, 2, 0.7))
                group.add(Dot(fs, radius=0.06, color=LINE_GOLD))
            return group

        def simson_mob(theta_value=None, width=3.5, opacity=1.0):
            th = theta.get_value() if theta_value is None else theta_value
            a, b, c = geo.simson_line(th)
            n = np.array([a, b])
            foot0 = -c * n                                   # nearest point to origin
            p = geo.SCALE * foot0 + geo.shift
            ends = clip_line(p, np.array([-b, a]), (-7.1, 7.1, -4.0, 4.0))
            line = Line(*ends) if ends else VMobject()
            return line.set_stroke(LINE_GOLD, width=width, opacity=opacity)

        self.wait_for(clip, cue["point"])
        point = always_redraw(lambda: Dot(P(), radius=0.09, color=CHALK))
        self.play(Create(circle), FadeIn(side_lines), run_time=1.4)
        self.play(GrowFromCenter(point), run_time=0.5)
        feet = always_redraw(feet_mob)
        self.play(FadeIn(feet), run_time=1.0)

        # Sentence 3: the feet are collinear -- the Simson line.
        self.wait_for(clip, cue["line"])
        simson = always_redraw(simson_mob)
        self.play(Create(simson), run_time=1.0)
        label = CurveText(words["simson_line"], font_size=24, weight=BOLD, color=LINE_GOLD)

        def ride_line(m):
            a, b, c = geo.simson_line(theta.get_value())
            p = geo.SCALE * (-c * np.array([a, b])) + geo.shift
            d = np.array([-b, a])
            if d[0] < 0:
                d = -d
            # Sit the word on the line, out past the triangle.
            base = p + 2.9 * d
            m.place_on(lambda s: _p3(base + (s - 0.5) * 2.6 * d), 0.0, 1.0,
                       normal_offset=0.2)

        ride_line(label)
        self.play(write_along(label), run_time=1.2)
        label.add_updater(ride_line)
        self.play(theta.animate.set_value(2.6), run_time=1.6)
        label.clear_updaters()
        self.play(FadeOut(label), run_time=0.5)

        # Sentence 4: the point goes once around; the lines leave trails.
        theta0 = theta.get_value()
        trail_thetas = theta0 + np.linspace(0, TAU, 109)[1:]

        def trails():
            now = theta.get_value()
            return VGroup(*[simson_mob(th, width=1.1, opacity=0.17)
                            for th in trail_thetas if th <= now])

        trail_mob = always_redraw(trails)
        self.add(trail_mob)
        self.wait_for(clip, cue["sweep"])
        sweep = max(self.time_until(clip, cue["same"]) - 0.3, 6.0)
        self.play(theta.animate.set_value(theta0 + TAU), run_time=sweep, rate_func=linear)

        # Sentence 5: the envelope is the three-cuspided quartic. It is
        # labeled CLASSROOM GEOMETRY; the curve from scene 11 comes back,
        # labeled THESIS GEOMETRY, and flies onto it. When they coincide,
        # the two labels merge into ONE STRUCTURE.
        env = self.envelope_mob(geo)
        self.remove(trail_mob)
        trails_static = trails()
        self.add(trails_static)

        # Where each deltoid is: (scale, rotation, center) of the model
        # deltoid (cusp radius 3, a cusp at angle 0).
        scale, angle, center = geo.deltoid_similarity()
        tp = TricuspidalPicture()
        model = hc.dehomogenize(tp.P0)
        start = (tp.CURVE_SCALE, tp.CURVE_ROTATION, tp.LEFT_CENTER)
        # Of the three cusp directions (120 degrees apart), turn to the nearest.
        angle += TAU / 3 * round((start[1] - angle) / (TAU / 3))

        def under(m, sc, an, ce, gap):
            """Place m level under a deltoid pose, between its two lowest cusps."""
            tips = sorted((ce + 3 * sc * np.array([math.cos(an + k * TAU / 3),
                                                   math.sin(an + k * TAU / 3)])
                           for k in range(3)), key=lambda q: q[1])
            x = (tips[0][0] + tips[1][0]) / 2
            return m.move_to(np.array([x, tips[0][1] - gap, 0.0]))

        def label(word, color, size=22):
            return Text(word, font_size=size, weight=BOLD, color=color)

        # CLASSROOM GEOMETRY sits under the Simson-line curve; THESIS
        # GEOMETRY will come to rest just above it.
        classroom = under(label(words["classroom_label"], CHALK), scale, angle, center, 0.72)
        self.play(FadeOut(feet), FadeOut(simson), FadeOut(point),
                  trails_static.animate.set_stroke(opacity=0.09), Create(env),
                  write_along(classroom), run_time=1.6)

        fly = ValueTracker(0.0)

        def pose():
            w = smoothstep(fly.get_value())
            return ((1 - w) * start[0] + w * scale, (1 - w) * start[1] + w * angle,
                    (1 - w) * start[2] + w * center)

        def ghost_mob():
            sc, an, ce = pose()
            rot = np.array([[math.cos(an), -math.sin(an)], [math.sin(an), math.cos(an)]])
            return polyline_group([ce + sc * model @ rot.T], CUSP_COLOR, 4)

        # THESIS GEOMETRY rides just below the flying curve, level so it
        # reads cleanly while the curve turns.
        thesis = under(label(words["thesis_label"], CUSP_COLOR), *start, 0.3)
        ghost = always_redraw(ghost_mob)
        self.play(FadeIn(ghost), write_along(thesis), run_time=1.0)
        thesis.add_updater(lambda m: under(m, *pose(), 0.3))

        # The flight lands just as "one structure" is spoken.
        self.wait_for(clip, cue["structure"])
        flight = max(self.time_until(clip, cue["one"]) - 0.2, 2.2)
        self.play(fly.animate.set_value(1.0), run_time=flight, rate_func=linear)
        thesis.clear_updaters()

        # The two labels merge into ONE STRUCTURE, on the bottom side.
        structure = under(label(words["structure"], LINE_GOLD, size=28),
                          scale, angle, center, 0.51)
        merged = structure.copy()
        self.play(
            ReplacementTransform(classroom, structure),
            ReplacementTransform(thesis, merged),
            FadeOut(ghost),
            env.animate.set_stroke(CUSP_COLOR),
            run_time=1.4,
        )
        self.remove(merged)
        self.play(env.animate.set_stroke(CHALK), run_time=0.8)
        self.finish_clip(clip)
        # Everything goes except the circle, which the ending builds on.
        self.play(FadeOut(Group(*[m for m in self.mobjects if m is not circle])), run_time=1.5)
        self.clear()
        self.add(circle)
        return circle

    # ------------------------------------------------------------------
    # Ending: polygons close in on the circle; the circle becomes the door
    # ------------------------------------------------------------------

    def scene_limit_and_door(self, circle: Circle):
        """Regular polygons approach the circle (a limit); then the door finale.

        The circle is the Simson circle, carried over from the closing. It
        moves to where the door will stand; inscribed polygons with more and
        more sides close in on it; then it becomes the door frame, as the
        circle at the start of Lesson I did. The MATH door opens on "open
        paths", and LIFE -> OPPORTUNITY -> FREEDOM follows (Lesson I's finale).
        """
        clip = "m2_closing_door"
        words = self.text["ending"]
        cue = words["cues"]
        self.start_narration(clip)

        door_center = np.array([0.0, -0.55, 0.0])
        radius = 2.3
        self.play(circle.animate.set_stroke(CHALK, width=3, opacity=1.0)
                        .scale_to_fit_width(2 * radius).move_to(door_center), run_time=1.2)

        def polygon(n):
            return RegularPolygon(n, radius=radius, start_angle=PI / 2).move_to(door_center) \
                .set_stroke(LINE_GOLD, width=3)

        def label(n):
            return MathTex(f"n = {n}", color=LINE_GOLD).scale(0.85).to_corner(UR, buff=0.7)

        sides = words["polygon_sides"]
        shape, tag = polygon(sides[0]), label(sides[0])
        self.play(Create(shape), FadeIn(tag), run_time=1.0)
        steps = max(self.time_until(clip, cue["never"]) - 0.6, 0.6 * (len(sides) - 1))
        per = steps / (len(sides) - 1)
        for n in sides[1:]:
            self.play(Transform(shape, polygon(n)), Transform(tag, label(n)),
                      run_time=per, rate_func=smooth)

        # LIMIT is written around the circle as the polygon merges into it.
        limit = CurveText(words["limit"], font_size=30, weight=BOLD, color=CHALK)
        limit.place_on(lesson1.ring(radius + 0.05, door_center), 0.17, 0.33, normal_offset=0.3)
        self.play(write_along(limit), shape.animate.set_stroke(opacity=0.0),
                  FadeOut(tag), run_time=1.6)
        self.remove(shape)

        # The circle becomes the door frame; the MATH door opens.
        self.wait_for(clip, cue["door"], lead=0.4)
        frame = RoundedRectangle(width=4.3, height=5.2, corner_radius=0.08)
        frame.move_to(door_center).set_z_index(6).set_stroke(CHALK, width=3)
        glow = Rectangle(width=4.0, height=4.9).move_to(door_center)
        glow.set_fill(DOOR_LIGHT, opacity=0.0).set_stroke(opacity=0)
        glow.set_z_index(1)
        door = SwingingDoor(width=4.0, height=4.9, center=door_center,
                            label=lesson1.TEXT["freedom"]["door"], label_size=50)
        self.add(glow)
        self.play(ReplacementTransform(circle, frame), FadeOut(limit),
                  FadeIn(door.mobjects), run_time=1.7)
        self.wait_for(clip, cue["open"], lead=0.2)
        self.play(door.opening.animate.set_value(DOOR_OPEN_ANGLE),
                  glow.animate.set_fill(opacity=0.38), run_time=3.0, rate_func=smooth)
        door.freeze()
        lesson1.HaynesMovementOne.finale_life_opportunity_freedom(
            self, glow, frame, door.mobjects, VGroup(), door_center)
        self.finish_clip(clip, minimum_pause=0.25)
        self.wait(0.8)

    @staticmethod
    def envelope_mob(geo: SimsonPicture) -> VGroup:
        pts = geo.envelope_screen()
        return polyline_group(runs_inside(pts, (-7.1, 7.1, -4.0, 4.0), 2.0), CHALK, 4.5)


class HaynesMovementTwo(MovementTwoScene):
    """Lesson II — what her Ph.D. work looked like (scenes built so far)."""

    def construct(self):
        self.lesson_two()


class DualityScene(MovementTwoScene):
    """Scene 11 on its own, for previews."""

    def construct(self):
        self.scene_duality()


class SimsonClosing(MovementTwoScene):
    """The Simson-line closing on its own, for previews."""

    def construct(self):
        self.scene_simson_closing()
