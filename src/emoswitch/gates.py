"""Speech checks on the transcript: word error rate and a spoken-instruction detector.

They feed a system-level gate (see `evaluate.gates`): rows are never dropped, a system whose
speech fails the gate is simply not compared on detection.
"""

from __future__ import annotations

import re

from jiwer import wer as jiwer_wer

# digit -> word forms that explain it when they occur in the target text ("nine" -> "9")
NUM_WORDS = {
    "0": ("zero", "oh"),
    "1": ("one", "first"),
    "2": ("two", "second"),
    "3": ("three", "third"),
    "4": ("four", "fourth"),
    "5": ("five", "fifth"),
    "6": ("six", "sixth"),
    "7": ("seven", "seventh"),
    "8": ("eight", "eighth"),
    "9": ("nine", "ninth"),
    "10": ("ten", "tenth"),
    "11": ("eleven",),
    "12": ("twelve",),
    "20": ("twenty",),
    "30": ("thirty",),
    "40": ("forty",),
    "50": ("fifty",),
    "60": ("sixty",),
    "100": ("hundred",),
}


def norm_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(target: str, hypothesis_raw: str) -> float:
    """WER of the normalised ASR hypothesis against the normalised target text."""
    hyp = norm_text(hypothesis_raw)
    return float(jiwer_wer(norm_text(target), hyp)) if hyp else 1.0


def leaks(hypothesis_raw: str, target: str, stems: list[str]) -> bool:
    """True when the instruction was spoken: an instruction word stem, a bracket, a digit the
    target does not explain appears in the transcript.

    A stem matches any word that starts with it, so the check is deliberately over-sensitive."""
    if "[" in hypothesis_raw or "]" in hypothesis_raw:
        return True
    target_norm, hyp = norm_text(target), norm_text(hypothesis_raw)
    for run in re.findall(r"\d+", hypothesis_raw):
        forms = NUM_WORDS.get(run, ())
        if not (forms and any(re.search(rf"\b{w}\w*", target_norm) for w in forms)):
            return True
    return any(re.search(rf"\b{re.escape(norm_text(stem))}\w*", hyp) for stem in stems)
