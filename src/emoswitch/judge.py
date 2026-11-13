"""The arousal judge: loudness-normalised audio -> windowed arousal profile.

The judge is a public speech-emotion regressor (wav2vec2, trained on MSP-Podcast). It is
frozen: nothing in this repository is trained against it.
"""

from __future__ import annotations

import numpy as np

MIN_SAMPLES = 400  # the judge returns nothing below 25 ms


def load_audio(path: str, sample_rate: int, lufs: float | None) -> np.ndarray:
    """Mono float32 at `sample_rate`, loudness-normalised to `lufs` (None = as is).

    The step is kept from the original instrument. It does not change the judge's output: the
    judge's own preprocessor rescales every window to zero mean and unit variance.
    """
    import librosa

    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if lufs is not None and len(y) >= int(0.4 * sample_rate):
        import pyloudnorm as pyln

        measured = float(pyln.Meter(sample_rate).integrated_loudness(y.astype(np.float64)))
        if np.isfinite(measured):
            y = pyln.normalize.loudness(y.astype(np.float64), measured, lufs)
            peak = float(np.max(np.abs(y))) if len(y) else 0.0
            if peak > 0.99:
                y = y / peak * 0.99
            y = y.astype(np.float32)
    return y


def _emotion_model_class():
    import torch
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = torch.nn.Dropout(config.final_dropout)
            self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    # The attribute names `wav2vec2` and `classifier` must match the checkpoint. A generic
    # AutoModel loader does not know the head and silently leaves it randomly initialised.
    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0]
            return self.classifier(hidden.mean(dim=1))

    return EmotionModel


class ArousalJudge:
    """Callable: 16 kHz mono float32 -> {'arousal', 'dominance', 'valence'}, each roughly in [0, 1]."""

    def __init__(self, model: str, revision: str, sample_rate: int, device: str = "cpu"):
        import torch
        from transformers import Wav2Vec2Processor

        self.torch = torch
        self.device = device
        self.sample_rate = sample_rate
        self.processor = Wav2Vec2Processor.from_pretrained(model, revision=revision)
        loaded, info = _emotion_model_class().from_pretrained(model, revision=revision, output_loading_info=True)
        if info["missing_keys"]:
            raise RuntimeError(f"judge weights not loaded for: {info['missing_keys'][:5]}")
        self.model = loaded.eval().to(device)

    def __call__(self, y: np.ndarray) -> dict:
        if len(y) < MIN_SAMPLES:
            return {"arousal": None, "dominance": None, "valence": None}
        inputs = self.processor(y, sampling_rate=self.sample_rate, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(inputs["input_values"].to(self.device))
        arousal, dominance, valence = (float(x) for x in out.reshape(-1)[:3].cpu().numpy())
        return {"arousal": round(arousal, 4), "dominance": round(dominance, 4), "valence": round(valence, 4)}


def window_profile(y: np.ndarray, judge, sample_rate: int, window_s: float, hop_s: float) -> tuple[list, list]:
    """Slide a fixed window over the clip. Returns (window centres in seconds, arousal per window)."""
    win, hop = int(window_s * sample_rate), int(hop_s * sample_rate)
    min_seg = max(MIN_SAMPLES, min(win, int(0.25 * sample_rate)))
    centres, arousal = [], []
    start = covered = 0
    while start + min(win, len(y)) <= len(y) and start < len(y):
        seg = y[start : start + win]
        if len(seg) < min_seg:
            break
        centres.append(round((start + len(seg) / 2) / sample_rate, 3))
        arousal.append(judge(seg)["arousal"])
        covered = start + len(seg)
        if start + win >= len(y):
            break
        start += hop
    if len(y) - covered >= min_seg:  # leftover longer than 0.25 s gets its own window
        seg = y[covered:]
        centres.append(round((covered + len(seg) / 2) / sample_rate, 3))
        arousal.append(judge(seg)["arousal"])
    return centres, arousal
