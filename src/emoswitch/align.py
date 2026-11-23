"""Forced alignment of the known text on the synthesised audio: where each word starts.

The text of every clip is known, so its words are aligned (torchaudio MMS_FA) rather than
recognised. Recognition would move a boundary whenever the recogniser mishears a word.
"""

from __future__ import annotations

import re
import warnings

import numpy as np

from emoswitch.detect import norm_word
from emoswitch.judge import MIN_SAMPLES


def align_token(word: str) -> str:
    """The aligner's dictionary is a-z plus the apostrophe."""
    return re.sub(r"[^a-z']", "", word.lower())


class ForcedAligner:
    """Callable: (16 kHz mono float32, text) -> words as {'w', 'start', 'end', 'score'}."""

    def __init__(self, sample_rate: int, device: str = "cpu"):
        import torch
        from torchaudio.pipelines import MMS_FA as bundle

        if sample_rate != bundle.sample_rate:
            raise ValueError(f"the aligner needs {bundle.sample_rate} Hz audio, got {sample_rate}")
        self.torch = torch
        self.device = device
        self.sample_rate = sample_rate
        self.model = bundle.get_model().to(device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

    def __call__(self, y: np.ndarray, text: str) -> list[dict]:
        raw = [w for w in re.split(r"\s+", text.strip()) if w]
        tokens = [align_token(w) for w in raw]
        keep = [i for i, t in enumerate(tokens) if t]
        if not keep or len(y) < MIN_SAMPLES:
            return []
        wav = self.torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # torchaudio announces the removal of forced_align after 2.8
            emission, _ = self.model(wav)
            spans = self.aligner(emission[0], self.tokenizer([tokens[i] for i in keep]))
        seconds_per_frame = wav.shape[1] / emission.shape[1] / self.sample_rate
        words = []
        for i, span in zip(keep, spans, strict=True):
            frames = sum(t.end - t.start for t in span) or 1
            words.append(
                {
                    "w": norm_word(raw[i]),
                    "start": round(float(span[0].start * seconds_per_frame), 3),
                    "end": round(float(span[-1].end * seconds_per_frame), 3),
                    "score": round(float(sum(t.score * (t.end - t.start) for t in span) / frames), 4),
                }
            )
        return words
