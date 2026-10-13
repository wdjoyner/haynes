#!/usr/bin/env python3
"""Euphemia Haynes — Mathematics as an Open Door: the whole film.

Title card -> Lesson I -> Lesson II -> one set of end credits.

Each lesson keeps its own files (haynes_movement1.py / haynes_movement2.py,
their text JSON files and their audio folders); this scene just plays them
in order. Because it is one continuous render, the time stamps of the
quotations in the credits count from the start of the whole film.

Generate both lessons' narration first::

    python generate_movement1_audio.py --voice af_heart --speed 0.94
    python generate_movement2_audio.py --voice af_heart --speed 0.94

Low-resolution preview, then 1080p::

    manim -pql haynes_film.py HaynesFilm
    manim -pqh haynes_film.py HaynesFilm
"""

from __future__ import annotations

from pathlib import Path

from manim import *

import haynes_movement1 as lesson1
import haynes_movement2 as lesson2


class HaynesFilm(lesson2.MovementTwoScene, lesson1.HaynesMovementOne):
    """Both lessons, with the credits (and "We will teach ...") once, at the end."""

    lessons_credit = lesson2.TEXT["film"]["lessons_credit"]

    def narration_sources(self):
        return [
            (lesson1.AUDIO_DIR, lesson1.TEXT, "generate_movement1_audio.py"),
            (lesson2.AUDIO_DIR, lesson2.TEXT, "generate_movement2_audio.py"),
        ]

    def construct(self):
        self.mark_chapter("title")
        self.title_card()
        self.mark_chapter("lesson1")
        self.lesson_one()
        if self.mobjects:                  # anything Lesson I left on screen
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
            self.clear()
        self.lesson_two()
        self.mark_chapter("credits")
        self.end_credits()
        self.write_chapters()

    def write_chapters(self):
        """Write youtube_chapters.txt: chapter times to paste into the description.

        The times come from this render, so they match the narration it
        used. YouTube needs the first at 0:00, at least three chapters, and
        each at least ten seconds long.
        """
        names = lesson2.TEXT["film"]["chapters"]
        lines = []
        for seconds, key in self._chapters:
            whole = int(seconds)
            stamp = (f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"
                     if whole >= 3600 else f"{whole // 60}:{whole % 60:02d}")
            lines.append(f"{stamp} {names[key]}")
        out = Path(__file__).resolve().parent / "youtube_chapters.txt"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("Chapters for YouTube (also in youtube_chapters.txt):")
        print("\n".join(lines))
