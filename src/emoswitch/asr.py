"""Deterministic ASR for the speech gates (word error rate, spoken-instruction leak).

faster-whisper by default retries a hard segment by sampling from an unseeded generator, so the
same wav can score differently between two passes. Temperature 0, a seeded generator and a fixed
thread count make the transcript a function of the audio.
"""

from __future__ import annotations


def load_asr(cfg: dict, device: str = "cpu"):
    import ctranslate2
    from faster_whisper import WhisperModel

    ctranslate2.set_random_seed(cfg["seed"])
    return WhisperModel(
        cfg["model"],
        device=device,
        compute_type=cfg["compute_type"] if device == "cpu" else "float16",
        cpu_threads=cfg["cpu_threads"] if device == "cpu" else 0,
        num_workers=cfg["num_workers"],
        revision=cfg["revision"],
    )


def transcribe(asr, path: str, cfg: dict) -> str:
    """Raw transcript of one clip."""
    segments, _ = asr.transcribe(
        path, language=cfg["language"], beam_size=cfg["beam_size"], temperature=cfg["temperature"]
    )
    return " ".join(s.text for s in segments).strip()
