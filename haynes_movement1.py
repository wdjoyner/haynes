#!/usr/bin/env python3
"""Euphemia Haynes — Mathematics as an Open Door: Lesson I.

(The code still calls the lessons "movements": file and class names were
kept when the on-screen word became "Lesson". The whole film, both lessons
with one set of credits, is ``HaynesFilm`` in haynes_film.py.)

A standalone Manim prototype emphasizing kinetic mathematical imagery rather
than PAM character blocking. Narration is mixed directly into the rendered
MP4 with ``Scene.add_sound``.

Generate narration first::

    python generate_movement1_audio.py --voice af_heart --speed 0.94

Low-resolution preview::

    manim -pql haynes_movement1.py HaynesMovementOne

1080p render::

    manim -pqh haynes_movement1.py HaynesMovementOne

The script is written for Manim Community 0.22.x but intentionally uses only
stable core classes and methods.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from manim import *

HERE = Path(__file__).resolve().parent
AUDIO_DIR = HERE / "audio" / "movement1"
MANIFEST_PATH = AUDIO_DIR / "audio_manifest.json"
PORTRAIT_PATH = HERE / "assets" / "haynes_portrait_lineart.png"
CHALK = "#f2ead8"

TEXT_PATH = HERE / "haynes_movement1_text.json"


def _load_text(path: Path = TEXT_PATH) -> dict:
    """All on-screen words (and the narration) live in one JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; it holds every word in the film.")
    return json.loads(path.read_text(encoding="utf-8"))


TEXT = _load_text()


# ---------------------------------------------------------------------------
# Geometry / typography helpers
# ---------------------------------------------------------------------------

def _sample_curve(curve, start_alpha: float, end_alpha: float, samples: int):
    """Sample a VMobject (by proportion) or a function t -> point."""
    alphas = np.linspace(start_alpha, end_alpha, samples)
    if callable(curve) and not isinstance(curve, Mobject):
        points = np.array([np.asarray(curve(float(a)), dtype=float) for a in alphas])
    else:
        points = np.array([curve.point_from_proportion(float(a)) for a in alphas])
    return points


def _curve_placements(
    text: str,
    curve,
    start_alpha: float,
    end_alpha: float,
    normal_offset: float,
    keep_upright: bool,
    samples: int,
) -> list[tuple[np.ndarray, float]]:
    """Return (center, angle) for each non-space character of *text*.

    Characters are spaced by arc length, not by curve parameter, so letters
    don't bunch up where the curve bends sharply. Manim's
    ``point_from_proportion`` is not arc-length parameterized, hence the
    dense lookup table.
    """
    if end_alpha <= start_alpha:
        raise ValueError("end_alpha must be greater than start_alpha")
    points = _sample_curve(curve, start_alpha, end_alpha, samples)

    # If this stretch of curve runs right-to-left on screen (e.g. the top of
    # a counterclockwise circle), walking it in the curve's own direction
    # would lay the word down backwards ("SDNATSREDNU"). Walk it the other
    # way so the word always reads left to right.
    if keep_upright and points[-1][0] < points[0][0]:
        points = points[::-1]

    lengths = np.zeros(len(points))
    lengths[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    total_length = float(lengths[-1])

    weights = np.array([0.62 if ch.isspace() else 1.0 for ch in text], dtype=float)
    cumulative = np.cumsum(weights)
    centers = cumulative - 0.5 * weights
    targets = centers / cumulative[-1] * total_length

    placements = []
    for ch, target in zip(text, targets):
        if ch.isspace():
            continue
        idx = int(np.searchsorted(lengths, target, side="left"))
        idx = max(1, min(len(points) - 2, idx))
        tangent = points[idx + 1] - points[idx - 1]
        norm = float(np.linalg.norm(tangent))
        tangent = np.array([1.0, 0.0, 0.0]) if norm <= 1e-9 else tangent / norm

        angle = math.atan2(tangent[1], tangent[0])
        if keep_upright:
            if angle > PI / 2:
                angle -= PI
            elif angle < -PI / 2:
                angle += PI

        # Offset along the glyph's own "up" so letters sit above the curve
        # as read, even where a single glyph had to be flipped.
        up = np.array([-math.sin(angle), math.cos(angle), 0.0])
        placements.append((points[idx] + normal_offset * up, angle))
    return placements


class CurveText(VGroup):
    """A word whose letters follow a curve.

    The glyphs are created once; :meth:`place_on` moves and rotates them, so
    it is cheap enough to call from an updater (a word riding a circle whose
    radius is changing, say). *curve* may be a VMobject, or any function
    ``t -> point`` (fast, and allows ``t`` outside [0, 1]).

    Reveal it letter by letter, in step with the curve being drawn, with
    :func:`write_along`.
    """

    def __init__(self, text: str, *, font_size: float = 30, weight=NORMAL,
                 color=WHITE, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self._angles: list[float] = []
        for ch in text:
            if ch.isspace():
                continue
            self.add(Text(ch, font_size=font_size, weight=weight, color=color))
            self._angles.append(0.0)

    def place_on(
        self,
        curve,
        start_alpha: float = 0.05,
        end_alpha: float = 0.95,
        *,
        normal_offset: float = 0.16,
        keep_upright: bool = True,
        samples: int = 240,
    ) -> "CurveText":
        placements = _curve_placements(
            self.text, curve, start_alpha, end_alpha,
            normal_offset, keep_upright, samples,
        )
        for i, (glyph, (point, angle)) in enumerate(zip(self.submobjects, placements)):
            glyph.rotate(angle - self._angles[i])
            self._angles[i] = angle
            glyph.move_to(point)
        return self


def text_along_curve(
    text: str,
    curve,
    *,
    font_size: float = 34,
    start_alpha: float = 0.05,
    end_alpha: float = 0.95,
    normal_offset: float = 0.16,
    keep_upright: bool = True,
    weight=NORMAL,
) -> CurveText:
    """Convenience wrapper: build a :class:`CurveText` and place it once."""
    return CurveText(text, font_size=font_size, weight=weight).place_on(
        curve, start_alpha, end_alpha,
        normal_offset=normal_offset, keep_upright=keep_upright, samples=700,
    )


def write_along(word: VGroup, *, run_time: float = 2.0, fade_width: float = 1.5,
                **kwargs) -> Animation:
    """Reveal a word's letters in order, like a pen writing along its curve.

    Pair it with ``Create(curve)`` in the same ``self.play`` call and the
    letters appear just behind the moving tip of the curve.
    """
    glyphs = list(word.submobjects)
    n = max(len(glyphs), 1)
    for g in glyphs:
        g.set_opacity(0)

    def update(mob, alpha):
        front = alpha * (n + fade_width)
        for i, g in enumerate(glyphs):
            g.set_opacity(float(np.clip((front - i) / fade_width, 0.0, 1.0)))

    return UpdateFromAlphaFunc(word, update, run_time=run_time,
                               rate_func=linear, **kwargs)


def ring(radius: float, center=ORIGIN):
    """A circle as a fast parametric function, t in [0, 1) = one turn CCW."""
    c = np.asarray(center, dtype=float)
    return lambda t: c + radius * np.array([math.cos(TAU * t), math.sin(TAU * t), 0.0])


def square_on_segment(a: np.ndarray, b: np.ndarray, outward_sign: float = 1.0) -> Polygon:
    """Return the square erected on segment a--b in the xy-plane."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = b - a
    n = outward_sign * np.array([-v[1], v[0], 0.0])
    return Polygon(a, b, b + n, a + n)


def brick_arch() -> VGroup:
    """A stylized arch assembled from small rectangular 'calculation bricks'."""
    bricks = VGroup()
    for x in (-4.2, -3.55, -2.9, 2.9, 3.55, 4.2):
        for y in (-2.2, -1.55, -0.9, -0.25):
            bricks.add(
                RoundedRectangle(width=0.58, height=0.42, corner_radius=0.05)
                .move_to([x, y, 0])
            )
    for angle in np.linspace(PI * 0.15, PI * 0.85, 13):
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle) - 1.15
        brick = RoundedRectangle(width=0.58, height=0.42, corner_radius=0.05)
        brick.move_to([x, y, 0]).rotate(angle - PI / 2)
        bricks.add(brick)
    return bricks


def epicycloid(R: float = 2.2, r: float = 0.62, shift=ORIGIN) -> ParametricFunction:
    """Return an epicycloid traced by a circle of radius r outside radius R."""
    def f(t):
        return np.array([
            (R + r) * np.cos(t) - r * np.cos((R + r) / r * t),
            (R + r) * np.sin(t) - r * np.sin((R + r) / r * t),
            0.0,
        ]) + np.asarray(shift)

    return ParametricFunction(f, t_range=[0, TAU], stroke_width=4)


DOOR_FILL = "#2b3242"
DOOR_LIGHT = "#f3d58a"
DOOR_OPEN_ANGLE = 105 * DEGREES


class SwingingDoor:
    """A flat door hinged on its left edge that swings open toward the viewer.

    ``self.opening`` is a ValueTracker for the swing angle (0 = closed,
    105 degrees = swung out past the frame so the doorway is fully clear). The free edge moves toward the hinge as
    ``cos(angle)`` and grows slightly taller as it comes toward the camera,
    which reads as perspective without needing a 3-D camera. Any ``label``
    is painted on the door: it foreshortens with the door and dims as the
    door turns edge-on, so it never floats in the doorway.
    """

    PERSPECTIVE = 0.16

    def __init__(self, width: float, height: float, center=ORIGIN, label: str | None = None,
                 label_size: float = 58):
        self.width = width
        self.height = height
        self.center = np.asarray(center, dtype=float)
        self.hinge_x = self.center[0] - width / 2
        self.opening = ValueTracker(0.0)
        self._label_template = (
            Text(label, weight=BOLD, font_size=label_size) if label else None
        )

        self.panel = always_redraw(self._make_panel)
        self.panel.set_z_index(3)
        self.handle = always_redraw(self._make_handle)
        self.handle.set_z_index(5)
        self.label = always_redraw(self._make_label) if label else VMobject()
        self.label.set_z_index(4)

    def _geometry(self):
        theta = self.opening.get_value()
        c, s = math.cos(theta), math.sin(theta)
        free_x = self.hinge_x + self.width * c
        half_h = self.height / 2
        free_half_h = half_h * (1 + self.PERSPECTIVE * s)
        return c, s, free_x, half_h, free_half_h

    def _make_panel(self):
        _, _, free_x, half_h, free_half_h = self._geometry()
        cy = self.center[1]
        return Polygon(
            [self.hinge_x, cy + half_h, 0],
            [free_x, cy + free_half_h, 0],
            [free_x, cy - free_half_h, 0],
            [self.hinge_x, cy - half_h, 0],
        ).set_fill(DOOR_FILL, opacity=1.0).set_stroke(WHITE, width=3)

    def _make_handle(self):
        c, _, free_x, _, _ = self._geometry()
        # Past 90 degrees the door has swung beyond the frame; the handle
        # stays on the free edge's inner side either way.
        inset = 0.30 * c if abs(c) > 0.05 else 0.015 * np.sign(c or 1)
        return Dot(radius=0.09).move_to([free_x - inset, self.center[1], 0])

    def _make_label(self):
        c, s, free_x, _, _ = self._geometry()
        label = self._label_template.copy()
        label.stretch(max(c, 0.02), dim=0)
        label.stretch(1 + 0.5 * self.PERSPECTIVE * s, dim=1)
        label.move_to([(self.hinge_x + free_x) / 2, self.center[1], 0])
        label.set_opacity(max(0.0, min(1.0, 1.4 * c - 0.2)))
        return label

    @property
    def mobjects(self) -> VGroup:
        return VGroup(self.panel, self.label, self.handle)

    def freeze(self):
        """Stop the redraw updaters (e.g. once the door has finished moving)."""
        for m in self.mobjects:
            m.clear_updaters()


class FloorPerspective:
    """A flat plane seen as a floor in one-point perspective.

    The plane's y direction runs away from the viewer toward a horizon;
    vertical lines stay vertical (the camera is level), so a point raised
    off the floor moves straight up on screen. ``blend`` interpolates
    between the flat plane (0) and the floor (1).
    """

    def __init__(self, horizon: float = 0.95, eye_height: float = 1.6,
                 focal: float = 4.5, near: float = 2.0, y_min: float = -3.0):
        self.horizon, self.h, self.f = horizon, eye_height, focal
        self.near, self.y_min = near, y_min

    def _depth(self, q) -> float:
        return q[1] - self.y_min + self.near

    def screen(self, q) -> np.ndarray:
        z = self._depth(q)
        return np.array([self.f * q[0] / z, self.horizon - self.f * self.h / z, 0.0])

    def lift(self, q, height: float) -> np.ndarray:
        """Screen offset of a point *height* above the floor at q."""
        return np.array([0.0, self.f * height / self._depth(q), 0.0])

    def blend(self, q, a: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return (1 - a) * q + a * self.screen(q)


class ExtrudedWord:
    """A word drawn as solid block letters, by stacking layers in perspective.

    Copies of the word shrink toward a vanishing point below the frame,
    darker toward the back, with the front face on top. The result reads
    as 3-D lettering seen from slightly below, while the scene stays 2-D.
    :meth:`at` returns the word partway through growing out of a point.
    """

    LAYERS = 44
    DEPTH = 0.085                  # back face is 8.5% closer to the vanishing point
    VANISH = np.array([0.0, -7.5, 0.0])
    BACK, SIDE = "#2a251c", "#a08a58"

    def __init__(self, text: str, font_size: float = 66):
        self.face = Text(text, weight=BOLD, font_size=font_size, color=CHALK)
        self.face.move_to(ORIGIN)
        self.sides = [interpolate_color(ManimColor(self.BACK), ManimColor(self.SIDE),
                                        i / (self.LAYERS - 1)) for i in range(self.LAYERS)]

    def at(self, g: float, source, target, zoom: float = 1.0) -> VGroup:
        """The word at growth g in [0, 1]: from a speck at *source* to full size at *target*."""
        g = float(np.clip(g, 0.0, 1.0))
        center = np.asarray(source) + g * (np.asarray(target) - np.asarray(source))
        size = (0.03 + 0.97 * g) * zoom
        front = self.face.copy().scale(size).move_to(center)
        group = VGroup()
        depth = self.DEPTH * g
        for i, color in enumerate(self.sides):
            k = 1 - depth * (1 - i / (self.LAYERS - 1))
            layer = front.copy().scale(k, about_point=self.VANISH)
            group.add(layer.set_fill(color, opacity=1).set_stroke(width=0))
        front.set_stroke("#10131a", width=1.2)
        group.add(front)
        return group


# ---------------------------------------------------------------------------
# Narrated scene base
# ---------------------------------------------------------------------------

class NarratedScene(Scene):
    """A scene whose narration clips come from audio manifests.

    Lesson I uses the defaults below. Another lesson reuses this class by
    subclassing it and overriding these three class attributes; a scene that
    plays several lessons overrides :meth:`narration_sources` instead.
    """

    audio_dir: Path = AUDIO_DIR
    text: dict = TEXT
    audio_script: str = "generate_movement1_audio.py"

    def setup(self):
        super().setup()
        self._audio_manifest = self._load_audio_manifest()
        self._clip_starts: dict[str, float] = {}
        self._chapters: list[tuple[float, str]] = []

    def mark_chapter(self, key: str):
        """Note that a chapter (for the YouTube description) starts now."""
        self._chapters.append((self.renderer.time, key))

    def narration_sources(self) -> list[tuple[Path, dict, str]]:
        """(audio folder, text, script that generates the audio) per lesson."""
        return [(self.audio_dir, self.text, self.audio_script)]

    def _load_audio_manifest(self) -> dict:
        """All clips from every source's manifest, each remembering its folder."""
        merged: dict = {"clips": {}}
        for audio_dir, _, script in self.narration_sources():
            manifest_path = audio_dir / "audio_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Missing {manifest_path}. Generate narration first with:\n"
                    f"  python {script}"
                )
            for clip_id, clip in json.loads(manifest_path.read_text())["clips"].items():
                merged["clips"][clip_id] = dict(clip, _dir=str(audio_dir))
        return merged

    def narration_text(self, clip_id: str) -> str:
        for _, text, _ in self.narration_sources():
            if clip_id in text.get("narration", {}):
                narration = text["narration"][clip_id]
                return " ".join(narration) if isinstance(narration, list) else narration
        return ""

    def narration_duration(self, clip_id: str) -> float:
        return float(self._audio_manifest["clips"][clip_id]["duration"])

    def start_narration(self, clip_id: str) -> float:
        clip = self._audio_manifest["clips"][clip_id]
        path = Path(clip["_dir"]) / clip["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        self.add_sound(str(path))
        self._clip_starts[clip_id] = self.renderer.time
        return float(clip["duration"])

    def quote_time(self, clip_id: str, starts_with: str) -> float | None:
        """Seconds from the start of the film at which a quoted phrase is spoken.

        Uses the per-sentence start times that generate_movement1_audio.py
        records in the manifest; for older manifests without them, estimates
        from the phrase's position in the clip's text. Returns None if the
        clip hasn't played or the phrase isn't in its narration.
        """
        if clip_id not in self._clip_starts:
            return None
        clip = self._audio_manifest["clips"][clip_id]
        needle = " ".join(starts_with.lower().split())

        def find(haystack: str) -> int:
            return " ".join(haystack.lower().split()).find(needle)

        for i, seg in enumerate(clip.get("segments", [])):
            at = find(seg["text"])
            if at >= 0:
                seg_end = (clip["segments"][i + 1]["start"]
                           if i + 1 < len(clip["segments"]) else clip["duration"])
                frac = at / max(len(seg["text"]), 1)
                return self._clip_starts[clip_id] + seg["start"] + frac * (seg_end - seg["start"])

        text = self.narration_text(clip_id)
        at = find(text)
        if at < 0:
            return None
        return self._clip_starts[clip_id] + clip["duration"] * at / max(len(text), 1)

    def time_until(self, clip_id: str, starts_with: str) -> float:
        """Seconds from now until these words are spoken in a narration clip."""
        at = self.quote_time(clip_id, starts_with)
        if at is None:
            raise ValueError(f"{starts_with!r} is not in the narration of {clip_id}; "
                             "check the 'cues' in the text file")
        return at - self.renderer.time

    def wait_for(self, clip_id: str, starts_with: str, lead: float = 0.0):
        """Wait until *lead* seconds before these words are spoken (if not already past)."""
        remaining = self.time_until(clip_id, starts_with) - lead
        if remaining > 1 / config.frame_rate:
            self.wait(remaining)

    def finish_narration(
        self,
        duration: float,
        visual_elapsed: float,
        *,
        minimum_pause: float = 0.2,
    ):
        """Hold only as long as needed for the current narration clip."""
        remaining = duration - visual_elapsed
        if remaining > minimum_pause:
            self.wait(remaining)
        else:
            self.wait(minimum_pause)


# ---------------------------------------------------------------------------
# Main film
# ---------------------------------------------------------------------------

class HaynesMovementOne(NarratedScene):
    """Lesson I — Haynes's philosophy of teaching mathematics."""

    #: Credits line naming what this render contains (``{lessons}`` in the credits).
    lessons_credit: str = TEXT["lessons_credit"]

    def construct(self):
        self.title_card()
        self.lesson_one()
        self.end_credits()

    def lesson_one(self):
        """Lesson I without the title card and credits (the film reuses it)."""
        self.camera.background_color = "#10131a"
        self._opening_transition = None
        self.opening_computation_to_structure()
        self.mark_chapter("reason")
        self.scene_answer_to_reason()
        self.mark_chapter("inquiry")
        self.scene_inquiry_epicycloid()
        self.mark_chapter("constant_variable")
        self.scene_constant_variable()
        self.mark_chapter("relationships")
        self.scene_relationships()
        self.mark_chapter("freedom")
        self.scene_degrees_of_freedom()

    # ------------------------------------------------------------------
    # Opening: calculation -> mathematical structure -> title -> door
    # ------------------------------------------------------------------

    def opening_computation_to_structure(self):
        dur = self.start_narration("m1_00_opening")
        elapsed = 0.0

        words = TEXT["opening"]
        equations = [MathTex(tex) for tex in words["equations"]]

        field = VGroup()
        for row in range(5):
            for col in range(5):
                eq = equations[(row * 5 + col) % len(equations)].copy().scale(0.55)
                eq.move_to([-5.6 + 2.8 * col, 2.7 - 1.25 * row, 0])
                field.add(eq)

        self.play(
            LaggedStart(*[Write(eq) for eq in field], lag_ratio=0.025),
            run_time=3.4,
        )
        elapsed += 3.4

        word = Text(words["computation"], weight=BOLD).scale(1.1)
        self.play(
            FadeOut(field, scale=0.85),
            FadeIn(word, scale=1.15),
            run_time=1.4,
        )
        elapsed += 1.4

        bricks = VGroup(*[
            RoundedRectangle(width=1.25, height=0.45, corner_radius=0.05)
            for _ in range(20)
        ])
        bricks.arrange_in_grid(rows=4, cols=5, buff=0.16)
        self.play(ReplacementTransform(word, bricks), run_time=2.0)
        elapsed += 2.0

        arch = brick_arch()
        self.play(Transform(bricks, arch), run_time=3.8)
        elapsed += 3.8

        # STRUCTURE is written along a keystone arc riding over the arch.
        keystone_arc = Arc(
            radius=3.7, start_angle=58 * DEGREES, angle=64 * DEGREES,
            arc_center=[0, -1.15, 0],
        ).set_stroke(opacity=0.5)
        structure = CurveText(words["structure"], font_size=40, weight=BOLD).place_on(
            keystone_arc, 0.0, 1.0, normal_offset=0.28,
        )
        self.play(
            Create(keystone_arc),
            write_along(structure),
            run_time=1.6,
        )
        elapsed += 1.6

        circle = Circle(radius=0.7).move_to([0, 0.25, 0])
        self.play(Create(circle), run_time=1.0)
        self.play(
            circle.animate.scale(4.1),
            FadeOut(bricks),
            FadeOut(structure),
            FadeOut(keystone_arc),
            run_time=2.6,
        )
        elapsed += 3.6

        # The film title now has its own card; this is the movement title.
        title = VGroup(
            Text(words["movement_title"], weight=BOLD, font_size=46),
            Text(words["movement_subtitle"], font_size=32),
        ).arrange(DOWN, buff=0.28)
        self.play(FadeIn(title, shift=0.18 * UP), run_time=1.4)
        self.wait(1.6)
        elapsed += 3.0

        door_frame = RoundedRectangle(width=5.2, height=5.6, corner_radius=0.12)
        self.play(Transform(circle, door_frame), run_time=1.8)
        elapsed += 1.8

        door = SwingingDoor(width=4.8, height=5.2, center=door_frame.get_center())
        doorway_glow = Rectangle(width=4.8, height=5.2).move_to(door_frame)
        doorway_glow.set_fill(DOOR_LIGHT, opacity=0.16).set_stroke(opacity=0)
        doorway_glow.set_z_index(1)
        self.add(doorway_glow)

        self.play(
            FadeOut(title, shift=0.16 * UP),
            FadeIn(door.mobjects),
            run_time=1.0,
        )
        elapsed += 1.0

        self.play(
            door.opening.animate.set_value(DOOR_OPEN_ANGLE),
            doorway_glow.animate.set_fill(opacity=0.32),
            run_time=2.0,
            rate_func=smooth,
        )
        door.freeze()
        elapsed += 2.0

        self.finish_narration(dur, elapsed, minimum_pause=0.1)
        self._opening_transition = VGroup(circle, door.mobjects, doorway_glow)

    # ------------------------------------------------------------------
    # Scene 1: a correct answer is not the same as understanding
    # ------------------------------------------------------------------

    def scene_answer_to_reason(self):
        clip = "m1_01_reason"
        dur = self.start_narration(clip)
        words = TEXT["reason"]
        cue = words["cues"]

        # Everything before UNDERSTANDS (equation, triangle, squares, areas)
        # is paced to finish just as "He understands it" is spoken, so each
        # word appears with its own sentence; its natural length is 13.7 s.
        before_words = 13.7
        pace = float(np.clip((self.time_until(clip, cue["understands"]) - 0.2) / before_words,
                             0.45, 1.0))

        def rt(seconds):
            return seconds * pace

        equation = MathTex(words["equation"]).scale(1.55)
        opening_fade = []
        if self._opening_transition is not None:
            opening_fade.append(FadeOut(self._opening_transition))
            self._opening_transition = None
        self.play(Write(equation), *opening_fade, run_time=rt(1.8))

        check = Text("✓", font_size=64).next_to(equation, RIGHT)
        self.play(FadeIn(check, scale=1.4), run_time=rt(0.7))
        self.wait(rt(0.6))

        self.play(FadeOut(check), equation.animate.to_edge(UP), run_time=rt(1.2))

        # A 3-4-5 triangle with its three squares spans 7 units vertically
        # at unit scale; s = 0.48 keeps the whole figure below the equation.
        s = 0.55
        fig_shift = np.array([-1.2, -0.35, 0.0])
        A = s * np.array([-2.0, -1.5, 0.0]) + fig_shift
        B = s * np.array([2.0, -1.5, 0.0]) + fig_shift
        C = s * np.array([2.0, 1.5, 0.0]) + fig_shift
        tri = Polygon(A, B, C)
        labels = VGroup(
            # Inside the triangle, each near the side it names.
            MathTex("4").scale(0.7).move_to(B + np.array([-0.45, 0.17, 0])),
            MathTex("3").scale(0.7).move_to((B + C) / 2 + np.array([-0.2, 0.25, 0])),
            MathTex("5").scale(0.7).move_to((A + C) / 2 + np.array([0.14, -0.18, 0])),
        )
        self.play(
            Create(tri),
            LaggedStart(*[FadeIn(x) for x in labels], lag_ratio=0.15),
            run_time=rt(2.2),
        )

        sq4 = square_on_segment(A, B, outward_sign=-1)
        sq3 = square_on_segment(B, C, outward_sign=-1)
        sq5 = square_on_segment(C, A, outward_sign=-1)
        squares = VGroup(sq4, sq3, sq5)
        self.play(
            LaggedStart(*[Create(s) for s in squares], lag_ratio=0.2),
            run_time=rt(3.4),
        )

        areas = VGroup(
            MathTex("16").scale(0.85).move_to(sq4.get_center()),
            MathTex("9").scale(0.85).move_to(sq3.get_center()),
            MathTex("25").scale(0.85).move_to(sq5.get_center()),
        )
        self.play(
            LaggedStart(*[FadeIn(a, scale=0.7) for a in areas], lag_ratio=0.22),
            run_time=rt(1.6),
        )
        self.play(Indicate(areas[0]), Indicate(areas[1]), run_time=rt(1.2))
        self.play(Indicate(areas[2]), run_time=rt(1.0))

        # A gentle arc in the open space to the right of the figure.
        understands_arc = Arc(
            radius=7.0,
            start_angle=72 * DEGREES,
            angle=33 * DEGREES,
            arc_center=[3.4, -6.0, 0],
        )
        curved_word = text_along_curve(
            words["understands"],
            understands_arc,
            font_size=32,
            start_alpha=0.0,
            end_alpha=1.0,
            normal_offset=0.22,
        )
        understands_arc.set_stroke(opacity=0.35)
        # UNDERSTANDS is written as "He understands it" is spoken ...
        self.wait_for(clip, cue["understands"])
        self.play(
            Create(understands_arc),
            write_along(curved_word),
            run_time=2.6,
        )

        # ... and IMPOSED drops in on "imposed".
        imposed = Text(words["imposed"], weight=BOLD).scale(1.0).to_edge(DOWN)
        self.wait_for(clip, cue["imposed"], lead=0.5)
        self.play(FadeIn(imposed, shift=2 * DOWN), run_time=0.8)
        self.play(
            imposed.animate.shift(0.25 * DOWN),
            run_time=0.25,
            rate_func=there_and_back,
        )
        self.play(FadeOut(imposed), run_time=0.5)

        self.finish_narration(dur, self.renderer.time - self._clip_starts[clip])
        self.play(
            FadeOut(VGroup(equation, tri, labels, squares, areas, curved_word,
                           understands_arc)),
            run_time=1.2,
        )

    # ------------------------------------------------------------------
    # Scene 2: inquiry visualized by an epicycloid
    # ------------------------------------------------------------------

    def scene_inquiry_epicycloid(self):
        dur = self.start_narration("m1_02_inquiry")
        elapsed = 0.0

        # R/r an integer closes the curve (3 cusps, then 4 cusps), and
        # R + 2r <= 3.0 keeps both shapes inside the frame.
        R = 1.8
        curve1 = epicycloid(R=R, r=R / 3)
        base = Circle(radius=R).set_stroke(opacity=0.35)
        self.play(Create(base), run_time=1.0)
        self.play(Create(curve1), run_time=5.2, rate_func=linear)
        elapsed += 6.2

        # Haynes's cycle of inquiry, written around a ring: each word rides
        # its own arc, and arrowed arcs carry the eye on to the next word.
        # Top words sit just outside the ring, bottom words just below it,
        # so neither collides with the epicycloid inside.
        ring_r = 3.3
        cycle = ring(ring_r)
        cycle_words = TEXT["inquiry"]["cycle"]
        first, second, third, fourth = cycle_words
        span_list = [      # (start, end) as fractions of a full turn, CCW
            (0.035, 0.165),   # top right
            (0.300, 0.465),   # top left
            (0.540, 0.700),   # bottom left
            (0.800, 0.935),   # bottom right
        ]
        spans = dict(zip(cycle_words, span_list))
        words, arcs = {}, {}
        for name, (a0, a1) in spans.items():
            top = 0.0 < (a0 + a1) / 2 < 0.5
            words[name] = CurveText(name, font_size=28, weight=BOLD).place_on(
                cycle, a0, a1, normal_offset=0.24 if top else -0.24,
            )
            # Arrowed arc from this word toward the next one.
            arcs[name] = Arc(
                radius=ring_r, start_angle=TAU * a0, angle=TAU * (a1 - a0 + 0.08),
            ).set_stroke(opacity=0.4).add_tip(tip_length=0.16, tip_width=0.16)

        def reveal(name, *others, run_time=1.7):
            self.play(Create(arcs[name]), write_along(words[name]), *others,
                      run_time=run_time)

        reveal(first)
        elapsed += 1.7

        curve2 = epicycloid(R=R, r=R / 4)
        reveal(second, Transform(curve1, curve2), run_time=3.6)
        elapsed += 3.6

        rays = VGroup(*[
            Line(ORIGIN, curve1.point_from_proportion(a)).set_stroke(opacity=0.45)
            for a in np.linspace(0.05, 0.95, 9)
        ])
        reveal(
            third,
            LaggedStart(*[Create(r) for r in rays], lag_ratio=0.08),
            run_time=2.4,
        )
        elapsed += 2.4

        reveal(fourth)
        elapsed += 1.7

        # "...and then another look": the whole cycle turns once more.
        cycle_group = VGroup(*words.values(), *arcs.values())
        remaining = dur - elapsed
        if remaining > 0.4:
            self.play(
                Rotate(rays, angle=TAU / 5, about_point=ORIGIN),
                Rotate(cycle_group, angle=TAU / 10, about_point=ORIGIN),
                Indicate(words[first], scale_factor=1.12),
                run_time=remaining,
                rate_func=smooth,
            )
            elapsed += remaining
        self.wait(0.2)

        self.play(
            FadeOut(VGroup(base, curve1, rays, cycle_group)),
            run_time=1.0,
        )

    # ------------------------------------------------------------------
    # Scene 3: constant / variable
    # ------------------------------------------------------------------

    def scene_constant_variable(self):
        dur = self.start_narration("m1_03_constant_variable")
        elapsed = 0.0

        radius = ValueTracker(1.5)
        circle = always_redraw(lambda: Circle(radius=radius.get_value()))
        center = Dot()
        cv = TEXT["constant_variable"]
        const = Text(cv["constant"], font_size=30).next_to(center, DOWN, buff=0.35)
        # VARIABLE rides the top of the circle with a fixed length, so it
        # bends tightly when the radius is small and flattens as it grows.
        var = CurveText(cv["variable"], font_size=30)
        var_length = 2.5

        def ride_circle(word):
            r = radius.get_value()
            half = var_length / (2 * TAU * r)
            word.place_on(ring(r, circle.get_center()), 0.25 - half, 0.25 + half,
                          normal_offset=0.26)

        ride_circle(var)
        self.play(
            FadeIn(center),
            Create(circle),
            FadeIn(const),
            write_along(var),
            run_time=1.5,
        )
        var.add_updater(ride_circle)
        self.play(radius.animate.set_value(2.4), run_time=2.2)
        self.play(radius.animate.set_value(1.0), run_time=2.0)
        elapsed += 5.7

        moving_center = VGroup(center, const)
        self.play(moving_center.animate.shift(3.0 * RIGHT), run_time=2.0)
        self.play(
            moving_center.animate.shift(5.5 * LEFT),
            radius.animate.set_value(2.0),
            run_time=2.5,
        )
        elapsed += 4.5

        var.clear_updaters()
        self.play(FadeOut(VGroup(circle, center, const, var)), run_time=0.8)
        learner = Dot(radius=0.16)
        learner_word = Text(cv["learner"], weight=BOLD, font_size=32).next_to(learner, DOWN)
        qualities = VGroup(*[
            Text(t, font_size=25)
            for t in cv["qualities"]
        ])
        orbit = ValueTracker(0.0)
        for q, angle in zip(qualities, np.linspace(0, TAU, 4, endpoint=False)):
            # Orbit the learner while staying upright and readable.
            q.add_updater(
                lambda m, a=angle: m.move_to(
                    2.4 * np.array([
                        np.cos(a + orbit.get_value()),
                        np.sin(a + orbit.get_value()),
                        0,
                    ])
                )
            )
        self.play(
            FadeIn(learner),
            FadeIn(learner_word),
            LaggedStart(*[FadeIn(q) for q in qualities], lag_ratio=0.15),
            run_time=2.0,
        )
        self.play(orbit.animate.set_value(PI / 2), run_time=2.8)
        elapsed += 5.6

        self.finish_narration(dur, elapsed)
        for q in qualities:
            q.clear_updaters()
        self.play(FadeOut(VGroup(learner, learner_word, qualities)), run_time=0.8)

    # ------------------------------------------------------------------
    # Scene 4: function / interdependence
    # ------------------------------------------------------------------

    def scene_relationships(self):
        dur = self.start_narration("m1_04_relationships")
        elapsed = 0.0

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=7,
            y_length=4.5,
            tips=False,
        )
        graph = axes.plot(lambda x: 1.15 * np.sin(x), x_range=[-3, 3])
        tracker = ValueTracker(-2.8)
        dot = always_redraw(
            lambda: Dot(
                axes.c2p(tracker.get_value(), 1.15 * np.sin(tracker.get_value()))
            )
        )
        xdot = always_redraw(
            lambda: Dot(axes.c2p(tracker.get_value(), 0), radius=0.06)
        )
        guide = always_redraw(
            lambda: DashedLine(xdot.get_center(), dot.get_center(), dash_length=0.08)
        )

        self.play(
            Create(axes),
            Create(graph),
            FadeIn(dot),
            FadeIn(xdot),
            Create(guide),
            run_time=2.0,
        )
        # RELATIONSHIPS is written along the graph just behind the moving
        # point: the word literally follows the dependence of y on x.
        rel = TEXT["relationships"]
        relation = CurveText(rel["relationship"], font_size=34, weight=BOLD).place_on(
            graph, 0.04, 0.96, normal_offset=0.26,
        )
        self.play(
            tracker.animate.set_value(2.8),
            write_along(relation, fade_width=1.0),
            run_time=4.5,
            rate_func=linear,
        )
        elapsed += 6.5

        formula = MathTex(rel["formula"]).to_corner(UR)
        self.play(FadeIn(formula), run_time=1.0)
        elapsed += 1.0

        self.play(
            FadeOut(VGroup(axes, graph, dot, xdot, guide, formula)),
            relation.animate.move_to(ORIGIN),
            run_time=1.2,
        )
        elapsed += 1.2

        # Node labels come from the text file, in the order left, top,
        # right, bottom.
        node_places = (LEFT * 3, UP * 1.8, RIGHT * 3, DOWN * 1.8)
        nodes = VGroup(*[Dot(p, radius=0.14) for p in node_places])
        labels = VGroup(*[
            Text(name, font_size=24).next_to(dot_obj, direction)
            for name, dot_obj, direction in zip(
                rel["nodes"], nodes, (LEFT, UP, RIGHT, DOWN)
            )
        ])
        edges = VGroup(
            *[
                Line(nodes[i].get_center(), nodes[(i + 1) % 4].get_center())
                for i in range(4)
            ],
            Line(nodes[0].get_center(), nodes[2].get_center()),
            Line(nodes[1].get_center(), nodes[3].get_center()),
        )
        self.play(
            Transform(
                relation,
                Text(rel["interdependence"], weight=BOLD, font_size=34).to_edge(UP),
            ),
            Create(edges),
            FadeIn(nodes),
            FadeIn(labels),
            run_time=2.2,
        )
        self.play(
            Indicate(nodes[0]),
            Indicate(nodes[1]),
            Indicate(nodes[2]),
            Indicate(nodes[3]),
            run_time=2.0,
        )
        elapsed += 4.2

        self.finish_narration(dur, elapsed)
        self.play(FadeOut(VGroup(relation, edges, nodes, labels)), run_time=0.8)

    # ------------------------------------------------------------------
    # Scene 5: degrees of freedom -> MATH door -> LIFE
    # ------------------------------------------------------------------

    def scene_degrees_of_freedom(self):
        """0, 1, 2, then 3 degrees of freedom; MORE POSSIBILITIES rises out of space.

        Lesson I ends here, on an open note: the door finale (MATH -> LIFE ->
        OPPORTUNITY -> FREEDOM) now closes the whole film, after Lesson II.
        """
        clip = "m1_05_freedom"
        self.start_narration(clip)
        words = TEXT["freedom"]
        cue = words["cues"]

        p = Dot(radius=0.14)
        zero = Text(words["zero"], font_size=28).next_to(p, DOWN)
        self.play(FadeIn(p), FadeIn(zero), run_time=0.8)

        self.wait_for(clip, cue["line"], lead=0.3)
        line = Line(LEFT * 5, RIGHT * 5)
        one = Text(words["one_degree"], font_size=28).to_edge(UP)
        self.play(Create(line), Transform(zero, one), run_time=1.3)
        self.play(p.animate.shift(4 * RIGHT), run_time=1.6)
        self.play(p.animate.shift(7 * LEFT), run_time=1.8)

        self.wait_for(clip, cue["plane"], lead=0.3)
        plane = NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.25},
        )
        two = Text(words["two_degrees"], font_size=28).to_edge(UP)
        self.play(FadeOut(line), FadeIn(plane), Transform(zero, two), run_time=1.4)
        self.play(p.animate.move_to([3.5, 2.0, 0]), run_time=1.3)
        self.play(p.animate.move_to([-2.5, -1.5, 0]), run_time=1.3)

        # In space: the plane tilts back into a floor, a vertical axis rises
        # from it, and the point lifts off the floor.
        self.wait_for(clip, cue["space"], lead=0.3)
        floor = FloorPerspective()
        tilt = ValueTracker(0.0)
        flat = plane.copy()
        self.remove(plane)
        plane = always_redraw(lambda: flat.copy().apply_function(
            lambda q: floor.blend(q, tilt.get_value())))
        self.add(plane)
        world = np.array([-2.5, -1.5, 0.0])          # the point, on the floor
        height = ValueTracker(0.0)
        p.add_updater(lambda m: m.move_to(
            floor.blend(world, tilt.get_value()) + floor.lift(world, height.get_value())))
        three = Text(words["three_degrees"], font_size=28).to_edge(UP)
        self.play(tilt.animate.set_value(1.0), Transform(zero, three), run_time=1.6)

        z_axis = Line(floor.screen(ORIGIN), floor.screen(ORIGIN) + floor.lift(ORIGIN, 3.4))
        z_axis.set_stroke(BLUE_D, width=2)
        drop = always_redraw(lambda: DashedLine(
            floor.screen(world), p.get_center(), dash_length=0.08,
        ).set_stroke(WHITE, 1.5, 0.5) if height.get_value() > 0.05 else VMobject())
        self.add(drop)
        self.play(Create(z_axis), height.animate.set_value(2.4), run_time=1.6)
        tour = [np.array([2.0, 1.2, 0.0]), np.array([0.4, -0.6, 0.0])]
        for target, h in zip(tour, (1.7, 2.9)):
            start = world.copy()
            self.play(
                UpdateFromAlphaFunc(Mobject(), lambda _, a, s0=start, s1=target:
                                    world.__setitem__(slice(None), s0 + smooth(a) * (s1 - s0))),
                height.animate.set_value(h),
                run_time=1.3,
            )

        # MORE POSSIBILITIES grows out of the point, with solid lettering.
        self.wait_for(clip, cue["possibilities"], lead=0.4)
        origin_pt = p.get_center()
        p.clear_updaters()
        word = ExtrudedWord(words["possibilities"], font_size=66)
        grow = ValueTracker(0.0)
        title = always_redraw(lambda: word.at(grow.get_value(), origin_pt, np.array([0.0, 0.9, 0.0])))
        self.add(title)
        self.play(
            grow.animate.set_value(1.0),
            FadeOut(VGroup(p, zero, z_axis, drop), run_time=1.2),
            plane.animate.set_stroke(opacity=0.0),
            run_time=3.0,
            rate_func=rate_functions.ease_out_cubic,
        )
        self.remove(drop, plane)

        # Hold, drifting gently toward the viewer, until the lesson's last words.
        push = ValueTracker(1.0)
        held = always_redraw(lambda: word.at(1.0, origin_pt, np.array([0.0, 0.9, 0.0]),
                                             zoom=push.get_value()))
        self.remove(title)
        self.add(held)
        end = self._clip_starts[clip] + self.narration_duration(clip)
        hold = max(end - self.renderer.time + 1.2, 1.5)
        self.play(push.animate.set_value(1.07), run_time=hold, rate_func=linear)
        self.play(FadeOut(held), run_time=1.0)
        self.remove(held)

    # ------------------------------------------------------------------
    # Finale: LIFE -> OPPORTUNITY -> FREEDOM
    # ------------------------------------------------------------------

    def finale_life_opportunity_freedom(self, glow, frame, door, title, door_center) -> float:
        """Print LIFE in the open doorway, then let it grow.

        LIFE is printed letter by letter; the doorway light floods the
        screen; LIFE expands and morphs into OPPORTUNITY; OPPORTUNITY bends
        into a spinning ring that collapses and bursts out as FREEDOM; and
        the letters of FREEDOM fly off along spirals, leaving trails of
        light. Returns the time used.
        """
        t = 0.0

        # 1. LIFE is printed in the doorway.
        life = Text(TEXT["freedom"]["life"], weight=BOLD, font_size=76).move_to(door_center)
        life.set_z_index(7)
        self.add(life)
        self.play(write_along(life, fade_width=0.6), run_time=1.2)
        t += 1.2

        # 2. The light floods out through the doorway and fills the screen.
        self.play(
            glow.animate.stretch_to_fit_width(config.frame_width + 1)
                        .stretch_to_fit_height(config.frame_height + 1)
                        .move_to(ORIGIN)
                        .set_fill(opacity=0.20),
            FadeOut(VGroup(frame, door, title)),
            life.animate.scale(1.35).move_to(ORIGIN),
            run_time=1.5,
        )
        t += 1.5

        # 3. LIFE expands and becomes OPPORTUNITY.
        opportunity = Text(TEXT["freedom"]["opportunity"], weight=BOLD, font_size=64).set_z_index(7)
        self.play(
            ReplacementTransform(life, opportunity, path_arc=-PI / 3),
            run_time=1.6,
        )
        t += 1.6

        # 4. OPPORTUNITY bends into a ring (letters standing on the circle
        #    like the lettering on a coin) and spins.
        def clockwise(radius):
            return lambda u: radius * np.array([
                math.cos(TAU * (0.25 - u)), math.sin(TAU * (0.25 - u)), 0.0,
            ])

        opp_ring = CurveText(TEXT["freedom"]["opportunity"], font_size=46, weight=BOLD)
        opp_ring.place_on(clockwise(2.1), -0.43, 0.43, normal_offset=0.0,
                          keep_upright=False)
        opp_ring.set_z_index(7)
        self.play(Transform(opportunity, opp_ring), run_time=1.4)
        self.play(
            Rotate(opportunity, angle=-TAU, about_point=ORIGIN),
            run_time=1.8,
            rate_func=rate_functions.ease_in_quad,
        )
        t += 3.2

        # 5. The ring collapses to a point and FREEDOM bursts out of it.
        freedom = Text(TEXT["freedom"]["freedom"], weight=BOLD, font_size=84).set_z_index(7)
        self.play(
            opportunity.animate.scale(0.02).set_opacity(0),
            run_time=0.5,
            rate_func=rate_functions.ease_in_cubic,
        )
        self.remove(opportunity)
        self.play(
            GrowFromCenter(freedom),
            Flash(ORIGIN, color=DOOR_LIGHT, line_length=1.2, num_lines=16,
                  flash_radius=0.4),
            glow.animate.set_fill(opacity=0.30),
            run_time=0.8,
            rate_func=rate_functions.ease_out_back,
        )
        self.wait(0.7)
        t += 2.0

        # 6. The letters of FREEDOM take flight along spirals, each leaving
        #    a fading trail: every letter its own degree of freedom.
        paths, trails = [], []
        n = len(freedom)
        for i, glyph in enumerate(freedom):
            g0 = glyph.get_center()
            phi = PI / 2 + (i - (n - 1) / 2) * 0.42      # fan out upward
            turn = 1.6 if i % 2 else -1.6                # alternate spin
            reach = 9.0 + 0.6 * abs(i - (n - 1) / 2)

            def spiral(u, g0=g0, phi=phi, turn=turn, reach=reach):
                r = 0.15 + reach * u * u
                r0 = 0.15
                return (g0
                        + r * np.array([math.cos(phi + turn * u), math.sin(phi + turn * u), 0])
                        - r0 * np.array([math.cos(phi), math.sin(phi), 0]))

            paths.append(ParametricFunction(spiral, t_range=[0, 1]))
            trail = TracedPath(glyph.get_center, stroke_color=DOOR_LIGHT,
                               stroke_width=3, dissipating_time=0.9)
            trail.set_z_index(6)
            trails.append(trail)
        self.add(*trails)
        self.play(
            LaggedStart(
                *[MoveAlongPath(g, path, rate_func=rate_functions.ease_in_quad)
                  for g, path in zip(freedom, paths)],
                lag_ratio=0.06,
            ),
            glow.animate.set_fill(opacity=0.10),
            run_time=3.2,
        )
        self.wait(0.9)   # let the trails dissipate
        self.remove(*trails, freedom)
        self.play(FadeOut(glow), run_time=0.8)
        t += 4.9
        return t

    # ------------------------------------------------------------------
    # Title card: the portrait, with the title written on her board
    # ------------------------------------------------------------------

    def _portrait(self) -> ImageMobject:
        if not PORTRAIT_PATH.exists():
            raise FileNotFoundError(PORTRAIT_PATH)
        from PIL import Image
        pixels = np.array(Image.open(PORTRAIT_PATH).convert("RGBA"))
        # The drawing has a stray vertical stroke in its first few columns
        # (rows ~240-780), which shows as a line down the board where the
        # title card continues it. Cover it with the columns just to its
        # right, which hold only the board's horizontal lines there.
        pixels[200:820, 0:10] = pixels[200:820, 10:20]
        return ImageMobject(pixels)

    def title_card(self):
        """Haynes beside her easel; the board extends across the screen.

        The line drawing (960 x 1400 px) fills the frame height on the right.
        Its easel board is cut off by the drawing's left edge, so the board's
        frame lines are continued in Manim across the screen, and the film
        title is written on the board.
        """
        portrait = self._portrait()
        portrait.height = config.frame_height
        portrait.to_edge(RIGHT, buff=0)
        px = config.frame_height / 1400.0          # scene units per pixel
        left_x = portrait.get_left()[0]
        top_y = portrait.get_top()[1]

        def y_of(row):                             # image row -> scene y
            return top_y - row * px

        board_left = -config.frame_width / 2 + 0.45
        frame_w = 70 * px                          # thickness of board frame

        def x_of(col):                             # image column -> scene x
            return left_x + col * px

        # Rows (in the drawing) of the board's lines where the drawing is cut.
        top_outer, top_bevel, top_inner = 426.5, 487, 497
        bot_inner, bot_bevel, bot_outer = 1214, 1226, 1284   # outer: behind the tray
        tray_top, tray_edge, tray_bottom = 1276, 1282, 1302  # the chalk tray, in front
        border_rows = (5, 1393.5)                  # the drawing's own border
        tray_left = board_left + 0.9 * frame_w

        def hline(row, x_end):
            return Line([left_x, y_of(row), 0], [x_end, y_of(row), 0])

        lines = VGroup(
            hline(top_outer, board_left),
            hline(top_bevel, board_left + 0.85 * frame_w),
            hline(top_inner, board_left + frame_w),
            hline(bot_inner, board_left + frame_w),
            hline(bot_bevel, board_left + 0.85 * frame_w),
            # The drawing's top border starts ~20 px in; overlap it so they join.
            *[Line([x_of(22), y_of(r), 0], [-config.frame_width / 2, y_of(r), 0])
              for r in border_rows],
        )
        sides = VGroup(
            Line([board_left, y_of(top_outer), 0], [board_left, y_of(bot_outer), 0]),
            Line([board_left + frame_w, y_of(top_inner), 0],
                 [board_left + frame_w, y_of(bot_inner), 0]),
            Line([board_left + 0.85 * frame_w, y_of(top_bevel), 0],
                 [board_left + 0.85 * frame_w, y_of(bot_bevel), 0]),
            Line([board_left, y_of(top_outer), 0], [board_left + frame_w, y_of(top_inner), 0]),
            Line([board_left, y_of(bot_outer), 0], [board_left + frame_w, y_of(bot_inner), 0]),
            Line([board_left, y_of(bot_outer), 0], [tray_left, y_of(bot_outer), 0]),
        )
        VGroup(lines, sides).set_stroke(CHALK, width=2.2)

        # The chalk tray runs in front of the bottom rail, as in the drawing,
        # and ends in a rounded cap (like its other end in the drawing).
        cap_r = (y_of(tray_top) - y_of(tray_bottom)) / 2
        cap_center = np.array([tray_left + cap_r, (y_of(tray_top) + y_of(tray_bottom)) / 2, 0])
        tray = VMobject()
        tray.start_new_path(np.array([left_x, y_of(tray_top), 0.0]))
        tray.add_line_to(cap_center + cap_r * UP)
        tray.append_points(Arc(radius=cap_r, start_angle=PI / 2, angle=PI,
                               arc_center=cap_center).points)
        tray.add_line_to(np.array([left_x, y_of(tray_bottom), 0.0]))
        tray.set_fill(self.camera.background_color, opacity=1).set_stroke(CHALK, width=2.2)
        tray_edge_line = Line(cap_center + cap_r * 0.55 * UP + 0.12 * cap_r * LEFT,
                              [left_x, y_of(tray_edge), 0]).set_stroke(CHALK, width=1.8)
        tray_group = VGroup(tray, tray_edge_line).set_z_index(1)
        lines.add(tray_group)

        grain = self._wood_grain(
            x_seam=x_of(18), y_of=y_of, px=px,
            board_left=board_left, frame_w=frame_w,
            top=(top_outer, top_bevel), bottom=(bot_bevel, tray_top),
            stile_rows=(top_outer, bot_outer),
        )

        # The board's writing surface, between the extended frame lines.
        inner_left = board_left + frame_w
        inner_right = left_x + 281 * px
        board_center = np.array([(inner_left + inner_right) / 2,
                                 (y_of(496) + y_of(1214)) / 2, 0.0])

        card = TEXT["title_card"]
        name = Text(card["name"], font_size=50, weight=BOLD, color=CHALK)
        subtitle = Text(card["subtitle"], font_size=34, color=CHALK, slant=ITALIC)
        dates = Text(card["dates"], font_size=24, color=CHALK)
        title = VGroup(name, subtitle).arrange(DOWN, buff=0.3)
        title.move_to(board_center + 0.55 * UP)
        if title.width > inner_right - inner_left - 0.6:
            title.scale_to_fit_width(inner_right - inner_left - 0.6)

        # A chalk flourish: one period of a gentle wave under the subtitle,
        # with the dates written along it.
        flourish_w = subtitle.width * 0.9
        flourish_center = subtitle.get_bottom() + 0.75 * DOWN
        flourish = ParametricFunction(
            lambda t: flourish_center + np.array([
                flourish_w * (t - 0.5), 0.16 * math.sin(TAU * t), 0.0,
            ]),
            t_range=[0, 1],
        ).set_stroke(CHALK, width=2.5)
        dates.next_to(flourish, DOWN, buff=0.28)

        # Reveal the drawing from the bottom up, as if sketched in.
        cover = Rectangle(width=portrait.width + 0.1, height=portrait.height + 0.1)
        cover.move_to(portrait).set_fill(self.camera.background_color, opacity=1)
        cover.set_stroke(width=0)
        self.add(portrait, cover)
        self.play(
            cover.animate.stretch(0.001, dim=1, about_edge=UP),
            run_time=2.6,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.remove(cover)

        # The board grows out of the drawing toward the left.
        self.play(Create(lines), run_time=1.3, rate_func=rate_functions.ease_out_sine)
        self.play(Create(sides), Create(grain, lag_ratio=0.05), run_time=0.9)

        # Chalk title.
        self.play(Write(name), run_time=2.2)
        self.play(Write(subtitle), run_time=1.6)
        self.play(Create(flourish), FadeIn(dates, shift=0.1 * UP), run_time=1.2)
        self.wait(2.4)

        self.play(
            FadeOut(Group(portrait, lines, sides, grain, name, subtitle, flourish, dates)),
            run_time=1.4,
        )
        self.wait(0.3)

    def _wood_grain(self, *, x_seam, y_of, px, board_left, frame_w, top, bottom,
                    stile_rows, seed: int = 1943) -> VGroup:
        """Wood grain for the part of the board frame drawn in Manim.

        The drawing's rails have grain lines that taper off just before its
        left edge; lines here begin a few pixels inside the drawing at those
        same rows, so they join without a gap, then wander gently along the
        rail. A few more start partway along, and the left stile gets
        vertical grain, like the stile in the drawing. Rows are the
        drawing's pixel rows (``y_of`` converts them).
        """
        rng = np.random.default_rng(seed)
        bg = self.camera.background_color
        # Thinner and a little softer than the outlines, like the drawing's grain.
        width = 0.95
        GRAIN = interpolate_color(ManimColor(CHALK), ManimColor(bg), 0.18)
        group = VGroup()

        def wavy(x0, x1, row0, lane, lo, hi):
            """A grain line from x0 leftward to x1, starting at row0."""
            k1 = TAU / (px * rng.uniform(420, 620))
            k2 = TAU / (px * rng.uniform(170, 260))
            a1, a2 = rng.uniform(1.5, 3.0), rng.uniform(0.6, 1.4)
            p1, p2 = rng.uniform(0, TAU), rng.uniform(0, TAU)
            blend = 160 * px

            def row(x):
                w = min(abs(x0 - x) / blend, 1.0)
                w = w * w * (3 - 2 * w)
                wave = a1 * (math.sin(k1 * x + p1) - math.sin(k1 * x0 + p1)) \
                    + a2 * (math.sin(k2 * x + p2) - math.sin(k2 * x0 + p2))
                return float(np.clip((1 - w) * row0 + w * lane + wave, lo, hi))

            return ParametricFunction(
                lambda t: np.array([x0 + t * (x1 - x0), y_of(row(x0 + t * (x1 - x0))), 0.0]),
                t_range=[0, 1, 0.01],
            )

        def rail(rows_from_drawing, lo, hi, x_end):
            inner = (lo + 4, hi - 4)
            lanes = np.linspace(*inner, len(rows_from_drawing) + 2)[1:-1]
            for row0, lane in zip(rows_from_drawing, lanes):
                end = x_end + rng.uniform(0.0, 0.5)
                line = wavy(x_seam, end, row0, lane, *inner)
                # Solid where it meets the drawing, fading toward its far end.
                group.add(line.set_stroke([GRAIN, GRAIN, bg], width=width))
            # Shorter lines that start and end along the rail.
            for _ in range(3):
                a = rng.uniform(x_end + 0.6, x_seam - 1.6)
                b = a - rng.uniform(1.2, 2.8)
                lane = rng.uniform(*inner)
                line = wavy(a, max(b, x_end), lane, lane, *inner)
                group.add(line.set_stroke([bg, GRAIN, bg], width=width))

        stile_x = board_left + frame_w
        rail((440, 451, 466, 482), top[0], top[1], stile_x + 0.05)
        rail((1236.5, 1243, 1256.5, 1266.5), bottom[0], bottom[1], stile_x + 0.05)

        # Vertical grain in the left stile, inside its mitered corners.
        top_row, bottom_row = stile_rows
        miter = frame_w / px                       # miter depth in rows
        for f in (0.22, 0.42, 0.63, 0.8):
            x = board_left + f * frame_w
            r0, r1 = top_row + f * miter + 6, bottom_row - f * miter - 6
            k = TAU / (px * rng.uniform(260, 380))
            a, ph = rng.uniform(0.8, 1.6) * px, rng.uniform(0, TAU)
            line = ParametricFunction(
                lambda t, x=x, r0=r0, r1=r1, k=k, a=a, ph=ph: np.array([
                    x + a * math.sin(k * (r0 + t * (r1 - r0)) * px + ph),
                    y_of(r0 + t * (r1 - r0)), 0.0]),
                t_range=[0, 1, 0.01],
            )
            group.add(line.set_stroke([bg, GRAIN, GRAIN, bg], width=width))
        return group

    # ------------------------------------------------------------------
    # End credits
    # ------------------------------------------------------------------

    def end_credits(self):
        """Credits roll beside the portrait, then her 1945 words close the film."""
        portrait = self._portrait()
        portrait.height = config.frame_height
        portrait.to_edge(LEFT, buff=0)
        portrait.set_opacity(0.55)
        self.play(FadeIn(portrait), run_time=1.5)

        styles = {
            "b": dict(font_size=34, weight=BOLD),
            "h": dict(font_size=22, slant=ITALIC),
            "t": dict(font_size=28),
            "s": dict(font_size=21),
        }
        column_x = portrait.get_right()[0] + (config.frame_width / 2 - portrait.get_right()[0]) / 2
        column_w = config.frame_width / 2 - portrait.get_right()[0] - 0.6

        def wrapped(text, kind):
            """Break genuinely long text into lines that fit the credits column.

            A line only slightly too wide is kept whole and shrunk to fit.
            """
            if Text(text, **styles[kind]).width <= 1.3 * column_w:
                return [text]
            lines, current = [], ""
            for word in text.split():
                trial = f"{current} {word}".strip()
                if current and Text(trial, **styles[kind]).width > column_w:
                    lines.append(current)
                    current = word
                else:
                    current = trial
            return lines + ([current] if current else [])

        entries = []
        for kind, text in TEXT["credits"]:
            if kind == "quotations":
                entries.extend(self._quotation_credits())
            else:
                entries.append((kind, text.replace("{lessons}", self.lessons_credit)))

        roll = VGroup()
        y = 0.0
        for kind, text in entries:
            if not kind:
                y -= 0.42
                continue
            if kind == "h":
                y -= 0.18
            for piece in wrapped(text, kind):
                line = Text(piece, color=CHALK, **styles[kind])
                if line.width > column_w:
                    line.scale_to_fit_width(column_w)
                line.move_to([column_x, y - line.height / 2, 0])
                y -= line.height + 0.16
                roll.add(line)

        # Start just below the frame and roll until the last line is gone.
        roll.shift((-config.frame_height / 2 - 0.3 - roll.get_top()[1]) * UP)
        distance = roll.height + config.frame_height + 0.6
        speed = 0.9                                     # scene units per second
        self.add(roll)
        self.play(roll.animate.shift(distance * UP),
                  run_time=distance / speed, rate_func=linear)
        self.remove(roll)

        # Closing words, centered in the space beside the portrait.
        quote = VGroup(*[
            Text(line, font_size=40, slant=ITALIC, color=CHALK) for line in TEXT["closing_quote"]
        ]).arrange(DOWN, buff=0.22)
        attribution = Text(TEXT["closing_attribution"], font_size=24, color=CHALK)
        closing = VGroup(quote, attribution).arrange(DOWN, buff=0.5)
        closing.move_to([column_x, 0, 0])
        if closing.width > column_w:
            closing.scale_to_fit_width(column_w)

        self.play(LaggedStart(*[Write(l) for l in quote], lag_ratio=0.55), run_time=3.6)
        self.play(FadeIn(attribution, shift=0.1 * UP), run_time=1.0)
        self.wait(3.5)
        self.play(FadeOut(Group(portrait, closing)), run_time=2.0)
        self.wait(0.5)

    def _quotation_credits(self) -> list[tuple[str, str]]:
        """Credit lines naming the source and time stamp of each quotation."""
        q = TEXT.get("quotations")
        if not q:
            return []
        # Quotations from every lesson in this render, in film order.
        items = [item for _, text, _ in self.narration_sources()
                 for item in text.get("quotations", {}).get("items", [])]
        lines = [("h", q.get("heading", "Quotations"))]
        for item in items:
            seconds = self.quote_time(item["clip"], item["starts_with"])
            if seconds is None:
                stamp = "--:--:--"
            else:
                whole = int(round(seconds))
                stamp = f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"
            lines.append(("s", q["template"].format(
                time=stamp, excerpt=item.get("excerpt", ""), source=item.get("source", ""),
            )))
            lines.append(("", ""))
        return lines[:-1]

