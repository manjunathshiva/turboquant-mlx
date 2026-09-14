# Copyright 2026 Manjunath Janardhan
"""Vision path for Qwen3.8-Flash-Next (``model_type: "qwen4_exp"``).

The vendored text model (:mod:`turboquant_mlx.models.qwen4_exp`) is text-only:
upstream mlx-lm PR #1788 drops ``model.visual.*``. This module adds the missing
half without touching the text path.

**The tower is a Qwen3-VL tower.** All 333 vision tensors match
``mlx_vlm.models.qwen3_vl.vision`` name for name and load ``strict=True``, so the
implementation is reused rather than reimplemented. mlx-vlm is imported lazily
and only here — the text model must not depend on it.

Flow: image -> tower -> features (N, 2560) -> spliced into the embedding stream at
``image_token_id`` -> text model, with 3-D MRoPE positions for the image tokens
supplied through the text model's optional ``rope_cs`` argument.

**Known approximation:** Qwen Sparse Attention's block indexer keeps 1-D sequence
positions for its pooled key blocks; only token positions become 3-D. Whether the
reference implementation gives blocks 3-D coordinates is unverified.
"""

from typing import Sequence

import mlx.core as mx


class _MXShim:
    """mlx-vlm 0.6.14 calls ``mx.repeat(x, grid_thw[i, 0])`` with an ``mx.array``
    repeat count; mlx >= 0.32 requires a Python int. Patching this one call is
    cheaper than upgrading mlx-vlm, which carries the shipped Muse /
    Qwen3.8-27B serving path. Remove once mlx-vlm ships the int conversion.
    """

    def __getattr__(self, name):
        return getattr(mx, name)

    @staticmethod
    def repeat(a, repeats, axis=None):
        if isinstance(repeats, mx.array):
            repeats = int(repeats.item())
        return mx.repeat(a, repeats, axis=axis)


def load_vision_tower(vision_config: dict, weights_path: str):
    """Build the Qwen3-VL tower and load ``vision.safetensors`` into it.

    ``strict=True`` on purpose: a partial load would silently produce plausible
    but wrong features.
    """
    from mlx_vlm.models.qwen3_vl.config import VisionConfig
    from mlx_vlm.models.qwen3_vl.vision import VisionModel
    import mlx_vlm.models.qwen3_vl.vision as _v

    _v.mx = _MXShim()
    fields = set(VisionConfig.__dataclass_fields__)
    kw = {k: v for k, v in vision_config.items() if k in fields}
    # The tower rejects unknown model types; the layout is identical.
    kw["model_type"] = "qwen3_vl"
    tower = VisionModel(VisionConfig(**kw))
    raw = mx.load(weights_path)
    stripped = {k.replace("model.visual.", "", 1): v for k, v in raw.items()}
    tower.load_weights(list(tower.sanitize(stripped).items()), strict=True)
    mx.eval(tower.parameters())
    return tower


def encode_image(tower, processor, image) -> tuple:
    """Return (features (N, out_hidden), grid_thw) for one PIL image."""
    enc = processor(images=image, return_tensors="np")
    pixel_values = mx.array(enc["pixel_values"]).astype(mx.bfloat16)
    grid_thw = mx.array(enc["image_grid_thw"])
    feats = tower(pixel_values, grid_thw)
    feats = feats[0] if isinstance(feats, tuple) else feats
    mx.eval(feats)
    return feats, grid_thw


def build_position_ids(ids: Sequence[int], grid_thw, image_token_id: int,
                       merge_size: int) -> mx.array:
    """3-D (t, h, w) positions, Qwen2-VL style -> shape (3, 1, len(ids)).

    Text tokens advance all three components together. An image block takes grid
    coordinates from its own origin, and the sequence resumes at
    ``origin + max(t, grid_h, grid_w)`` so later text never collides with the
    image's span.
    """
    t, h, w = (int(x) for x in grid_thw)
    gh, gw = h // merge_size, w // merge_size
    rows: list = [[], [], []]
    i = cur = 0
    n = len(ids)
    while i < n:
        if ids[i] == image_token_id:
            for f in range(t):
                for r in range(gh):
                    for c in range(gw):
                        rows[0].append(cur + f)
                        rows[1].append(cur + r)
                        rows[2].append(cur + c)
            cur += max(t, gh, gw)
            i += t * gh * gw
        else:
            for d in range(3):
                rows[d].append(cur)
            cur += 1
            i += 1
    return mx.array(rows)[:, None, :]


def mrope_cos_sin(text_config: dict, hidden: mx.array, position_ids: mx.array):
    """Interleaved MRoPE (cos, sin) for the given 3-D positions.

    Built with mlx-vlm's ``MRoPERotaryEmbedding`` rather than re-deriving the
    section layout: ``mrope_section`` partitions the rotary frequencies between
    the t/h/w components, and getting that wrong fails silently.
    """
    from mlx_vlm.models.rope_utils import MRoPERotaryEmbedding

    rp = text_config["rope_parameters"]
    rotary_dim = int(text_config["head_dim"] * rp["partial_rotary_factor"])
    emb = MRoPERotaryEmbedding(
        rotary_dim,
        max_position_embeddings=text_config.get("max_position_embeddings", 262144),
        base=float(rp["rope_theta"]),
        rope_parameters=rp,
        style="interleaved",
    )
    return emb(hidden, position_ids)


def splice_image_features(embeddings: mx.array, ids: mx.array, features: mx.array,
                          image_token_id: int) -> mx.array:
    """Replace the embedding rows at image placeholders with tower features."""
    mask = ids == image_token_id
    n_slots = int(mask.sum().item())
    if n_slots != features.shape[0]:
        raise ValueError(
            f"image placeholders ({n_slots}) != vision tokens ({features.shape[0]}); "
            "expand '<|image_pad|>' to one token per merged patch before encoding")
    out = mx.where(mask[..., None], mx.zeros_like(embeddings), embeddings)
    positions = mx.array([i for i, v in enumerate(mask[0].tolist()) if v])
    scatter = mx.zeros_like(out)
    scatter[0, positions] = features.astype(out.dtype)
    return out + scatter


def prepare_inputs(model, ids: mx.array, features: mx.array, grid_thw,
                   config: dict, use_mrope: bool = True):
    """Return (embeddings, rope_cs) ready for ``model(ids, cache, emb, rope_cs)``."""
    image_token_id = config["image_token_id"]
    emb = model.model.embed_tokens(ids)
    emb = splice_image_features(emb, ids, features, image_token_id)
    rope_cs = None
    if use_mrope:
        merge = config["vision_config"]["spatial_merge_size"]
        pos = build_position_ids(ids[0].tolist(), list(grid_thw[0]), image_token_id, merge)
        rope_cs = mrope_cos_sin(config["text_config"], emb, pos)
    return emb, rope_cs
