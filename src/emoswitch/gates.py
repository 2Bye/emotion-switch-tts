"""Speech checks on the transcript: word error rate and a spoken-instruction detector.

They feed a system-level gate (see `evaluate.gate_check`): rows are never dropped, a system whose
speech fails the gate is simply not compared on detection. They also decide whether a single clip is
valid (`evaluate.is_valid`).
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


# digits the ASR writes for a number word of the target, 0-20
DIGIT_WORDS = dict(enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split()
))  # fmt: skip
# British -> American spellings the ASR uses for words of the evaluation texts (results/baseline/eda.md)
SPELLING = {"neighbour": "neighbor", "neighbours": "neighbors"}


def norm_text(text: str) -> str:
    """Lower case, punctuation to spaces, digits 0-20 to words, one spelling per word of SPELLING."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    words = [DIGIT_WORDS.get(int(w), w) if w.isascii() and w.isdigit() else w for w in text.split()]
    return " ".join(SPELLING.get(w, w) for w in words)


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
