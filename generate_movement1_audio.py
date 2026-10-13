#!/usr/bin/env python3
"""Generate Kokoro narration WAV files for the Manim Movement I prototype.

Run from the repository root::

    python generate_movement1_audio.py --voice af_heart --speed 0.94

Outputs are written to ``audio/movement1`` together with an
``audio_manifest.json`` file containing measured durations.  The Manim scene
reads that manifest at render time, so narration timing does not need to be
hard-coded into the animation source.

The narration text itself lives in ``haynes_movement1_text.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000

TEXT_PATH = Path(__file__).resolve().parent / "haynes_movement1_text.json"


def load_narration(path: Path = TEXT_PATH) -> dict[str, str]:
    """Read the narration from the shared text file.

    Each clip may be one string or a list of sentences; lists are joined
    with single spaces.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        clip_id: " ".join(text) if isinstance(text, list) else text
        for clip_id, text in data["narration"].items()
    }


NARRATION = load_narration()


def _audio_to_numpy(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


#: Silence between sentences, in seconds (Kokoro's own sentence pause is short
#: once each sentence is spoken separately).
SENTENCE_GAP = 0.32


def synthesize(
    pipeline: KPipeline, text, voice: str, speed: float,
) -> tuple[np.ndarray, list[dict]]:
    """Return the clip's audio and where each sentence starts.

    *text* is a list of sentences (or one string, treated as one sentence).
    Each sentence goes to Kokoro on its own, with a short silence between
    them: Kokoro rushes, slurs or drops words when a long passage is given
    to it in one piece. The start time of every sentence, in seconds from
    the start of the clip, is recorded so the animation can cue on exact
    sentences and the credits can give accurate quotation times.
    """
    sentences = [text] if isinstance(text, str) else list(text)
    gap = np.zeros(int(round(SENTENCE_GAP * SAMPLE_RATE)), dtype=np.float32)
    chunks: list[np.ndarray] = []
    segments: list[dict] = []
    position = 0
    for n, sentence in enumerate(sentences):
        pieces = []
        for result in pipeline(sentence, voice=voice, speed=speed):
            audio = result.audio if hasattr(result, "audio") else result[2]
            if audio is not None:
                pieces.append(_audio_to_numpy(audio))
        if not pieces:
            raise RuntimeError(f"Kokoro returned no audio for: {sentence!r}")
        if n:
            chunks.append(gap)
            position += len(gap)
        segments.append({"start": round(position / SAMPLE_RATE, 3), "text": sentence})
        for piece in pieces:
            chunks.append(piece)
            position += len(piece)
    return np.concatenate(chunks), segments


def load_sentences(path: Path = TEXT_PATH) -> dict[str, list[str]]:
    """Each clip's narration as a list of sentences."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {clip_id: text if isinstance(text, list) else [text]
            for clip_id, text in data["narration"].items()}


SENTENCES = load_sentences()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice ID")
    parser.add_argument("--speed", type=float, default=0.94, help="Speech speed")
    parser.add_argument("--lang", default="a", help="Kokoro language code; a = American English")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    audio_dir = here / "audio" / "movement1"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Imported here so other scripts can reuse this file's helpers
    # (e.g. generate_movement2_audio.py --placeholder) without Kokoro.
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=args.lang)
    manifest = {
        "engine": "kokoro",
        "language": args.lang,
        "voice": args.voice,
        "speed": args.speed,
        "sample_rate": SAMPLE_RATE,
        "clips": {},
    }

    for clip_id, text in NARRATION.items():
        print(f"Synthesizing {clip_id}...")
        audio, segments = synthesize(pipeline, SENTENCES[clip_id], args.voice,
                                     args.speed)
        wav_path = audio_dir / f"{clip_id}.wav"
        sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
        duration = len(audio) / SAMPLE_RATE
        manifest["clips"][clip_id] = {
            "file": wav_path.name,
            "duration": round(duration, 3),
            "text": text,
            "segments": segments,
        }
        print(f"  {wav_path.relative_to(here)} ({duration:.3f} s)")

    manifest_path = audio_dir / "audio_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {manifest_path.relative_to(here)}")


if __name__ == "__main__":
    main()
