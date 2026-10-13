#!/usr/bin/env python3
"""Generate the narration WAV files for Movement II.

Run from the repository root::

    python generate_movement2_audio.py --voice af_heart --speed 0.94

This works exactly like generate_movement1_audio.py (it reuses that file's
Kokoro code) but reads ``haynes_movement2_text.json`` and writes to
``audio/movement2``.

To preview the animation before recording, ``--placeholder`` writes silent
clips whose lengths are estimated from the word count, with an estimated
start time for every sentence. Kokoro is not needed for that. Run it again
without ``--placeholder`` to replace them with the real narration; the
animation picks up the new lengths automatically.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from generate_movement1_audio import SAMPLE_RATE, load_narration, load_sentences, synthesize

HERE = Path(__file__).resolve().parent
TEXT_PATH = HERE / "haynes_movement2_text.json"
AUDIO_DIR = HERE / "audio" / "movement2"

# Rough speaking rate of Kokoro's af_heart voice at speed 0.94, used only
# for --placeholder clips.
WORDS_PER_SECOND = 2.45
SENTENCE_PAUSE = 0.35


def placeholder_clip(sentences: list[str]) -> tuple[float, list[dict]]:
    """Estimated duration and sentence start times for silent preview audio."""
    position, segments = 0.0, []
    for sentence in sentences:
        segments.append({"start": round(position, 3), "text": sentence})
        position += len(sentence.split()) / WORDS_PER_SECOND + SENTENCE_PAUSE
    return position, segments


def write_silence(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * int(round(seconds * SAMPLE_RATE)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice ID")
    parser.add_argument("--speed", type=float, default=0.94, help="Speech speed")
    parser.add_argument("--lang", default="a", help="Kokoro language code; a = American English")
    parser.add_argument("--placeholder", action="store_true",
                        help="write silent clips of estimated length instead (no Kokoro needed)")
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(TEXT_PATH.read_text(encoding="utf-8"))
    joined = load_narration(TEXT_PATH)

    manifest = {
        "engine": "placeholder" if args.placeholder else "kokoro",
        "language": args.lang,
        "voice": args.voice,
        "speed": args.speed,
        "sample_rate": SAMPLE_RATE,
        "clips": {},
    }

    pipeline = None
    if not args.placeholder:
        from kokoro import KPipeline
        import soundfile as sf
        pipeline = KPipeline(lang_code=args.lang)

    for clip_id, text in joined.items():
        wav_path = AUDIO_DIR / f"{clip_id}.wav"
        if args.placeholder:
            raw = data["narration"][clip_id]
            sentences = raw if isinstance(raw, list) else [raw]
            duration, segments = placeholder_clip(sentences)
            write_silence(wav_path, duration)
        else:
            print(f"Synthesizing {clip_id}...")
            audio, segments = synthesize(pipeline, load_sentences(TEXT_PATH)[clip_id],
                                         args.voice, args.speed)
            sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
            duration = len(audio) / SAMPLE_RATE
        manifest["clips"][clip_id] = {
            "file": wav_path.name,
            "duration": round(duration, 3),
            "text": text,
            "segments": segments,
        }
        print(f"  {wav_path.relative_to(HERE)} ({duration:.3f} s)")

    manifest_path = AUDIO_DIR / "audio_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {manifest_path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
