"""In-pass emotion switch on the fine-tuned AR model: one autoregressive pass, one codec decode.

The model runs in two phases. Phase 1 is an ordinary single pass under the instruction of the first
emotion; its codec frames up to the sentence boundary are kept. Phase 2 rebuilds the model input with
the instruction of the second emotion, appends the kept frames as context and lets the same decoder
continue to the end of the text. The whole phrase is decoded by the codec once, so there is no
waveform join. Two optional components act on phase 2:

  anchor  the first K frames of the context the model SEES are taken from a single-pass render of the
          same text in the second emotion; the emitted audio still starts with the real frames
  steer   the hidden state that emits each continued frame is shifted along the mean difference
          between the two emotions at one decoder layer (scripts/estimate_steer.py)

The vendor package builds the model input; nothing of it is re-implemented here: `capture_prefix`
lets the vendor's `generate` run up to its call of the decoder and keeps the arguments. Written
against qwen-tts 0.1.1 (custom_voice, non-streaming layout).
"""

from __future__ import annotations

import numpy as np

from emoswitch.detect import norm_word

CONTINUATION_SEED_OFFSET = 1000  # phase 2 is seeded apart from phase 1


def boundary_time(words: list[dict], switch_word: str, duration_s: float, search_from_frac: float) -> float | None:
    """Middle of the pause before the first sentence-2 word, from the aligned words of the phase-1 render."""
    target = norm_word(switch_word)
    hits = [i for i, w in enumerate(words) if w["start"] >= search_from_frac * duration_s and w["w"] == target]
    if not hits or hits[0] == 0:
        return None
    i = hits[0]
    return 0.5 * (words[i - 1]["end"] + words[i]["start"])


def anchor_frames(n_kept: int, n_source: int, share: float, min_tail: int) -> int:
    """How many leading context frames to take from the second-emotion render: `share` of the kept
    frames, leaving at least `min_tail` real frames before the boundary. 0 = the anchor does not fit."""
    return max(0, min(int(round(share * n_kept)), n_kept - min_tail, n_source - 1))


class _Captured(Exception):
    def __init__(self, kwargs: dict):
        super().__init__("captured")
        self.kwargs = kwargs


class PrefixRepetitionPenalty:
    """The repetition penalty of the single pass, applied over the context frames and the continuation."""

    def __init__(self, penalty: float, prefix_ids):
        self.penalty = float(penalty)
        self.prefix = prefix_ids  # [1, N]

    def __call__(self, input_ids, scores):
        import torch

        ids = torch.cat([self.prefix.to(input_ids.device).expand(input_ids.shape[0], -1), input_ids], dim=1)
        score = torch.gather(scores, 1, ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        return scores.scatter(1, ids, score)


class HiddenSteer:
    """Forward hook on one decoder layer: the last position of every forward gets
    h += alpha * ||h|| * direction. In phase 2 the decoder runs one prefill over [input | context] and then
    one position per generated frame, so the last position is always the state that emits a continued
    frame; the positions before the boundary are never edited. Use as a context manager."""

    def __init__(self, layer_module, alpha: float, direction):
        import torch

        self.module, self.alpha, self.calls, self.handle = layer_module, float(alpha), 0, None
        weight = next(layer_module.parameters())
        self.direction = direction.detach().to(weight.device, dtype=torch.float32).view(1, 1, -1)
        if abs(float(self.direction.norm()) - 1.0) > 1e-2:
            raise ValueError("the steering direction must have unit norm")

    def _hook(self, module, inputs, output):
        import torch

        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        last = h[:, -1:, :]
        shifted = last + (self.alpha * last.float().norm(dim=-1, keepdim=True) * self.direction).to(h.dtype)
        h = torch.cat([h[:, :-1, :], shifted], dim=1)
        self.calls += 1
        return (h, *output[1:]) if is_tuple else h

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        self.handle = None
        return False


class InPassEngine:
    """Input capture and continuation on a loaded qwen_tts custom-voice model, one utterance at a time."""

    def __init__(self, model, speaker: str, gen: dict):
        self.model, self.core, self.talker = model, model.model, model.model.talker
        self.speaker, self.language = speaker, gen["language"]
        self.max_new_tokens = int(gen["max_new_tokens"])
        keys = ("max_new_tokens", "do_sample", "repetition_penalty", "subtalker_dosample", "subtalker_temperature",
                "subtalker_top_k", "subtalker_top_p")  # fmt: skip
        self.merged = model._merge_generate_kwargs(**{k: gen[k] for k in keys})
        self.n_groups = self.talker.config.num_code_groups
        self.eos_id = self.core.config.talker_config.codec_eos_token_id
        codec = self.core.speech_tokenizer.model.config
        self.fps = float(codec.output_sample_rate) / float(codec.decode_upsample_rate)

    def capture_prefix(self, text: str, instruct: str | None) -> dict:
        """The decoder input the vendor builds for (instruction, text): embeddings and decoding arguments."""
        m, talker = self.model, self.talker
        input_ids = m._tokenize_texts([m._build_assistant_text(text)])
        instruct_ids = [m._tokenize_texts([m._build_instruct_text(instruct)])[0]] if instruct else [None]
        original = talker.generate

        def capture(*args, **kwargs):
            raise _Captured(kwargs)

        talker.generate = capture
        try:
            self.core.generate(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                languages=[self.language],
                speakers=[self.speaker],
                non_streaming_mode=True,
                **self.merged,
            )
            raise RuntimeError("the vendor generate did not call the decoder")
        except _Captured as captured:
            kw = captured.kwargs
        finally:
            talker.generate = original
        if int(kw["attention_mask"].sum()) != kw["inputs_embeds"].shape[1]:
            raise RuntimeError("unexpected padding in a one-row decoder input")
        named = ("inputs_embeds", "attention_mask", "trailing_text_hidden", "tts_pad_embed")
        return {
            "embeds": kw["inputs_embeds"],
            "trailing": kw["trailing_text_hidden"],
            "tts_pad": kw["tts_pad_embed"],
            "kwargs": {k: v for k, v in kw.items() if k not in named},
        }

    def codes_to_embeds(self, codes, tts_pad):
        """[T, groups] codec frames -> [1, T, d] decoder inputs: the sum of the codebook embeddings plus the
        pad embedding, as the vendor forms the input of every generated step."""
        import torch

        parts = [self.talker.get_input_embeddings()(codes[:, 0])]
        for i in range(1, self.n_groups):
            parts.append(self.talker.code_predictor.get_input_embeddings()[i - 1](codes[:, i]))
        return (torch.stack(parts, dim=1).sum(1).unsqueeze(0) + tts_pad).to(tts_pad.dtype)

    def _run(self, embeds, prefix: dict, max_new: int, seed: int, context_ids=None):
        """Decode from `embeds` to the end of speech -> ([T, groups] complete frames, end reached)."""
        import torch
        from transformers import LogitsProcessorList, set_seed

        kw = {**prefix["kwargs"], "max_new_tokens": int(max_new)}
        processors = LogitsProcessorList()
        if context_ids is not None and kw.get("repetition_penalty", 1.0) != 1.0:
            processors.append(PrefixRepetitionPenalty(kw["repetition_penalty"], context_ids))
            kw["repetition_penalty"] = 1.0
        set_seed(seed)
        out = self.talker.generate(
            inputs_embeds=embeds,
            attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device),
            trailing_text_hidden=prefix["trailing"],
            tts_pad_embed=prefix["tts_pad"],
            logits_processor=processors,
            **kw,
        )
        frames = [h[-1] for h in out.hidden_states if h[-1] is not None]  # the vendor returns each step's codes here
        if not frames:
            return torch.zeros((0, self.n_groups), dtype=torch.long, device=embeds.device), True
        codes = torch.stack(frames, dim=1)[0]
        stop = (codes[:, 0] == self.eos_id).nonzero()
        if stop.numel():
            return codes[: int(stop[0, 0])], True
        return codes, bool(out.sequences.numel() and int(out.sequences[0, -1]) == self.eos_id)

    def single_pass(self, text: str, instruct: str | None, seed: int):
        """One ordinary pass -> ([T, groups] codes, end reached)."""
        prefix = self.capture_prefix(text, instruct)
        return self._run(prefix["embeds"], prefix, self.max_new_tokens, seed)

    def continue_from(self, text: str, kept, instruct: str | None, seed: int, context=None, steer=None):
        """Phase 2: the input rebuilt with `instruct`, `context` (default: the kept frames) as what the model
        sees, continued to the end. `steer` = (layer index, alpha, unit direction [d]).
        Returns (kept frames + continuation, end reached)."""
        import torch

        prefix = self.capture_prefix(text, instruct)
        seen = kept if context is None else context
        embeds = torch.cat([prefix["embeds"], self.codes_to_embeds(seen, prefix["tts_pad"])], dim=1)
        remaining = max(2, self.max_new_tokens - int(kept.shape[0]))
        args = (embeds, prefix, remaining, seed + CONTINUATION_SEED_OFFSET, seen[:, 0:1].T)
        if steer is None:
            continuation, ended = self._run(*args)
        else:
            layer, alpha, direction = steer
            with HiddenSteer(self.talker.model.layers[int(layer)], alpha, direction):
                continuation, ended = self._run(*args)
        return torch.cat([kept, continuation], dim=0), ended

    def layer_means(self, text: str, instruct: str | None, codes):
        """Teacher-forced forward over [input | codes]: mean output of every decoder layer over the codec
        positions -> [layers, d] float32 on the CPU."""
        import torch

        prefix = self.capture_prefix(text, instruct)
        n_prefix = int(prefix["embeds"].shape[1])
        embeds = torch.cat([prefix["embeds"], self.codes_to_embeds(codes, prefix["tts_pad"])], dim=1)
        layers = self.talker.model.layers
        got = [None] * len(layers)

        def keep(i):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                got[i] = h[0, n_prefix:, :].float().mean(0).cpu()

            return hook

        handles = [layer.register_forward_hook(keep(i)) for i, layer in enumerate(layers)]
        try:
            with torch.inference_mode():
                self.talker(
                    inputs_embeds=embeds,
                    attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device),
                    use_cache=False,
                    output_hidden_states=False,
                )
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack(got, 0)

    def decode(self, codes) -> tuple[np.ndarray, int]:
        import torch

        wavs, sample_rate = self.core.speech_tokenizer.decode([{"audio_codes": codes}])
        wav = wavs[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().to(torch.float32).numpy()
        return np.asarray(wav).reshape(-1), int(sample_rate)


def load_model(checkpoint: str, device: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen3TTSModel.from_pretrained(checkpoint, device_map=device, dtype=dtype)
    # The vendor wrapper refuses `instruct` for 0.6B custom-voice models (its check is a substring
    # test against "0b6"); the fine-tuned checkpoint was trained with the instruction channel.
    model.model.tts_model_size = "0b6+instruct-ft"
    return model
