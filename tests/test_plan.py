"""Tests for the preflight planner (turboquant-plan / turboquant-doctor).

The load-bearing tests are the FIELD ones: the projection is calibrated against
what the 16 GB mini actually did, and those numbers are the whole reason to trust
the tool. If a refactor drifts the calibration, these fail.
"""

import json
import struct

import pytest

from turboquant_mlx.plan import (
    _RESERVE_BYTES,
    build_plan,
    estimate_wss,
    machine,
    footprint,
    _attention_layers,
    _moe_prefill_bytes,
    _resolve,
    is_complete_checkpoint,
    kv_bytes,
    prefill_workspace_bytes,
    read_safetensors_index,
    render,
    run_doctor,
)

GB = 1e9

# Qwen3.6-35B-A3B geometry (the real one, from the shipped config).
# NOTE: this fixture carries neither `vocab_size` nor `moe_intermediate_size`,
# so the lm_head and MoE-routing terms of prefill_workspace_bytes are both
# ZERO here. That is why the field calibration below is unaffected by changes
# to either — it is not coverage of them. Use MUSE / LAGUNA for those.
Q35 = {
    "model_type": "qwen3_5_moe",
    "quantization": {"mode": "turboquant", "bits": 3, "group_size": 64},
    "text_config": {
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        # 10 of 40 layers are full attention (hybrid GatedDeltaNet)
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
    },
}


# Muse-Glimmer-30B geometry (the real one). Unlike Q35 this carries a
# vocab_size, and a big one — it is the fixture for anything lm_head-shaped.
MUSE = {
    "model_type": "muse_glimmer",
    "quantization": {"mode": "turboquant", "bits": 3, "group_size": 64},
    "text_config": {
        "num_hidden_layers": 32,
        "hidden_size": 6656,
        "intermediate_size": 19968,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 202048,
        "sliding_window": 2048,
        "layer_types": (["sliding_attention"] * 3 + ["full_attention"]) * 8,
    },
}


# Laguna-S-2.1 geometry, read off the shipped config.json. Unlike Q35 this
# carries moe_intermediate_size + num_experts_per_tok, so it is the fixture for
# anything that depends on expert routing.
LAGUNA = {
    "model_type": "laguna",
    "quantization": {"mode": "turboquant", "bits": 3, "group_size": 64},
    "text_config": {
        "num_hidden_layers": 48,
        "hidden_size": 3072,
        "moe_intermediate_size": 1024,
        "intermediate_size": 12288,      # the DENSE mlp — must not be used
        "num_experts": 256,
        "num_experts_per_tok": 10,
        "num_attention_heads": 48,
        "vocab_size": 100352,
        "layer_types": (["sliding_attention"] * 3 + ["full_attention"]) * 12,
    },
}


def _write_model(tmp_path, cfg, tensors):
    """Minimal on-disk model: config.json + one shard whose HEADER is real but
    whose payload is zero-filled (the planner must never read the payload)."""
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    header, off = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        size = n * {"U32": 4, "F16": 2, "F32": 4}[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + size]}
        off += size
    blob = json.dumps(header).encode()
    with open(tmp_path / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        # NOTE: no payload written. A planner that reads tensors breaks here.
    (tmp_path / "tokenizer.json").write_text("{}")
    return str(tmp_path)


class TestFootprint:
    def test_splits_experts_from_resident(self):
        idx = {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": ("U32", (256, 768, 96)),
            "model.layers.0.mlp.switch_mlp.gate_proj.scales": ("F16", (256, 768, 32)),
            "model.layers.0.self_attn.q_proj.weight": ("U32", (2048, 96)),
            "model.embed_tokens.weight": ("F16", (150000, 2048)),
        }
        fp = footprint(idx)
        assert fp["expert_bytes"] > 0
        assert fp["resident_bytes"] > 0
        assert fp["total_bytes"] == fp["expert_bytes"] + fp["resident_bytes"]

    def test_router_gate_is_not_an_expert(self):
        # the router must stay resident — it is not a switch_mlp tensor
        idx = {"model.layers.0.mlp.gate.weight": ("F16", (256, 2048))}
        assert footprint(idx)["expert_bytes"] == 0

    def test_header_only_never_reads_payload(self, tmp_path):
        p = _write_model(tmp_path, Q35,
                         {"model.embed_tokens.weight": ("F16", (1000, 64))})
        idx = read_safetensors_index(p)          # payload absent from the file
        assert idx["model.embed_tokens.weight"] == ("F16", (1000, 64))


class TestKV:
    def test_hybrid_counts_only_full_attention_layers(self):
        n_full, n_slide, n_layers, note = _attention_layers(Q35)
        assert (n_full, n_slide, n_layers) == (10, 0, 40)
        _, per_tok, _ = kv_bytes(Q35, 1024)
        # 2 (K+V) * 2 kv-heads * 256 head_dim * 10 layers * 2 bytes
        assert per_tok == 2 * 2 * 256 * 10 * 2
        assert "hybrid" in note

    def test_field_measurement_32k(self):
        """Ternary 35B measured 0.62 GB of KV at 32K. Predict within 10%, and
        on the conservative side — a planner that under-predicts is a crash."""
        total, *_ = kv_bytes(Q35, 32768)
        pred = total / GB
        assert pred >= 0.62, "planner must not under-predict KV"
        assert abs(pred - 0.62) / 0.62 < 0.10

    def test_kv_bits_halves_at_8(self):
        fp16, *_ = kv_bytes(Q35, 4096)
        kv8, *_ = kv_bytes(Q35, 4096, kv_bits=8)
        assert kv8 == pytest.approx(fp16 / 2)

    def test_missing_geometry_returns_none(self):
        total, _, note = kv_bytes({"text_config": {}}, 1024)
        assert total is None and "lacks" in note

    def test_mamba_hybrid_reads_the_override_pattern(self):
        """Nemotron-H: only the '*' layers are attention. Counting
        num_hidden_layers instead over-predicts KV by 11x."""
        cfg = {"text_config": {
            "num_hidden_layers": 88, "num_key_value_heads": 2, "head_dim": 128,
            "hybrid_override_pattern":
                "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
                "EMEMEMEMEM*EMEMEMEM*EMEMEMEME",
        }}
        n_full, _, n_layers, note = _attention_layers(cfg)
        assert (n_full, n_layers) == (8, 88)
        assert "Mamba" in note
        _, per_tok, _ = kv_bytes(cfg, 4096)
        assert per_tok == 2 * 2 * 128 * 8 * 2      # 8 attention layers, not 88

    def test_sliding_window_layers_are_counted_but_bounded(self):
        """GPT-OSS/Gemma sliding layers hold real KV capped at the window.
        Ignoring them under-predicts, which is the direction that OOMs."""
        cfg = {"text_config": {
            "num_hidden_layers": 4, "num_key_value_heads": 8, "head_dim": 64,
            "layer_types": ["sliding_attention", "full_attention"] * 2,
            "sliding_window": 128,
        }}
        n_full, n_slide, _, note = _attention_layers(cfg)
        assert (n_full, n_slide) == (2, 2)
        assert "sliding" in note
        per_layer_tok = 2 * 8 * 64 * 2
        # at 4096 ctx: 2 full layers grow; 2 sliding stop at 128 tokens
        total, _, _ = kv_bytes(cfg, 4096)
        assert total == 2 * 4096 * per_layer_tok + 2 * 128 * per_layer_tok
        # sliding cost must stop growing once past the window
        t8k, *_ = kv_bytes(cfg, 8192)
        assert t8k - total == 2 * 4096 * per_layer_tok   # only full layers grew


class TestMalformedConfigs:
    """config.json comes from arbitrary HF repo ids — untrusted input. Review
    findings (PR #52): a garbage value must degrade, never under-predict."""

    BASE = {"num_hidden_layers": 4, "num_key_value_heads": 8, "head_dim": 64,
            "layer_types": ["sliding_attention", "full_attention"] * 2}
    PER_LAYER_TOK = 2 * 8 * 64 * 2

    def _kv(self, ctx=4096, **over):
        return kv_bytes({"text_config": {**self.BASE, **over}}, ctx)[0]

    def test_full_window_is_the_floor_not_a_subtraction(self):
        """`sliding_window: -1` made the sliding layers subtract KV — an
        under-prediction. Any non-positive/garbage window means 'not windowed'."""
        full_ctx = 4 * 4096 * self.PER_LAYER_TOK      # all 4 layers, no window
        for bad in (-1, 0, None, "", "none", 3.4e38):
            got = self._kv(sliding_window=bad)
            assert got == full_ctx, f"sliding_window={bad!r} -> {got}"

    def test_a_real_window_still_bounds_the_cost(self):
        got = self._kv(sliding_window=128)
        assert got == (2 * 4096 + 2 * 128) * self.PER_LAYER_TOK

    def test_fractional_interval_does_not_divide_by_zero(self):
        """int(0.5) == 0 -> ZeroDivisionError in the old code."""
        for bad in (0, 0.5, -3, "x", None):
            n_full, _, n_layers, _ = _attention_layers(
                {"text_config": {"num_hidden_layers": 8,
                                 "full_attention_interval": bad}})
            assert (n_full, n_layers) == (8, 8)      # falls back to dense

    def test_valid_interval_still_works(self):
        n_full, _, _, note = _attention_layers(
            {"text_config": {"num_hidden_layers": 8,
                             "full_attention_interval": 4}})
        assert n_full == 2 and "every 4th" in note

    def test_layer_types_supplies_the_layer_count(self):
        """num_hidden_layers missing: layer_types is itself authoritative, so
        the projection should still resolve instead of going unknown."""
        cfg = {"text_config": {k: v for k, v in self.BASE.items()
                               if k != "num_hidden_layers"}}
        _, _, n_layers, _ = _attention_layers(cfg)
        assert n_layers == 4
        total, per_tok, note = kv_bytes(cfg, 4096)
        assert total and per_tok and "lacks" not in note


class TestWorkspace:
    def test_scales_with_chunk_and_context(self):
        a = prefill_workspace_bytes(Q35, 21000, 2048)
        b = prefill_workspace_bytes(Q35, 21000, 128)
        assert a == pytest.approx(b * 16)          # linear in chunk
        c = prefill_workspace_bytes(Q35, 42000, 128)
        assert c == pytest.approx(b * 2)           # linear in context

    def test_chunked_prefill_does_not_pay_for_the_lm_head(self):
        """The chunk loop discards the forward's output and evaluates only the
        cache state; MLX being lazy, the lm_head matmul never runs. Measured on
        an M4 Max at 2048x202048: dropping the output costs 0 MB, while slicing
        [:, -1:] out of it costs the full 763 MiB.

        So a big-vocab model prefilled in chunks must be billed attention
        scratch, NOT chunk x vocab x 2 (0.83 GB at 2048 x 202048 on Muse).
        """
        w = prefill_workspace_bytes(MUSE, 5068, 2048)
        attn = 2048 * 5068 * MUSE["text_config"]["num_attention_heads"] * 2
        assert w == pytest.approx(attn)

        naive = attn + 2048 * MUSE["text_config"]["vocab_size"] * 2
        assert naive - w > 0.8 * GB, "the term this test exists to keep out"

    def test_unchunked_prefill_does_pay_for_the_lm_head(self):
        """A prompt that fits in ONE forward gets no such relief: the engine
        slices logits[:, -1, :] after the matmul has already run."""
        w = prefill_workspace_bytes(MUSE, 1500, 2048)
        heads = MUSE["text_config"]["num_attention_heads"]
        assert w == pytest.approx(1500 * 1500 * heads * 2
                                  + 1500 * MUSE["text_config"]["vocab_size"] * 2)

    def test_chunk_never_exceeds_the_prompt(self):
        """context <= step means the forward is prompt-wide, not step-wide."""
        assert (prefill_workspace_bytes(MUSE, 900, 2048)
                == prefill_workspace_bytes(MUSE, 900, 900))

    def test_moe_term_bounds_the_measured_per_routing_cost(self):
        """SwitchGLU fans each token out to top_k rows, so an expert block's
        transient scales with chunk x top_k.

        MEASURED on an M4 Max, Laguna-S-2.1's exact geometry (256 experts,
        hidden 3072, moe_inter 1024, top-10), one gate/up/down block, sorted
        prefill, peak over resident packed weights: 30,833 B/routing at step
        256 and 30,819 at step 1024. The model must BOUND that, not match it —
        this tool exists to prevent OOMs.
        """
        step, top_k = 1024, LAGUNA["text_config"]["num_experts_per_tok"]
        per_routing = (_moe_prefill_bytes(LAGUNA, step, LAGUNA["quantization"])
                       / (step * top_k))
        assert per_routing >= 30833, "must not under-predict the measurement"
        assert per_routing < 30833 * 1.25, "bound has drifted loosely"

    def test_moe_term_scales_with_top_k_not_just_chunk(self):
        few = dict(LAGUNA, text_config=dict(LAGUNA["text_config"],
                                            num_experts_per_tok=5))
        assert (_moe_prefill_bytes(LAGUNA, 512, None)
                == pytest.approx(_moe_prefill_bytes(few, 512, None) * 2))
        assert (_moe_prefill_bytes(LAGUNA, 1024, None)
                == pytest.approx(_moe_prefill_bytes(LAGUNA, 512, None) * 2))

    def test_moe_uses_the_expert_width_not_the_dense_mlp(self):
        """moe_intermediate_size (1024), never intermediate_size (12288) — a
        12x error on any model that carries both."""
        step = 512
        c = LAGUNA["text_config"]
        routings = step * c["num_experts_per_tok"]
        assert _moe_prefill_bytes(LAGUNA, step, None) == pytest.approx(
            routings * 8 * (c["hidden_size"] + c["moe_intermediate_size"]))

    def test_sorted_prefill_pays_no_expert_dequantization(self):
        """polar_gather_qmm tiles over PACKED weights when output dims are a
        multiple of 64, materializing nothing. Measured: a 256-expert block at
        chunk 2048 peaks at 357.8 MB, against 933.8 MB for a control that does
        dequantize all experts — the 768 MB tensor never appears.

        So the term is activations only. Break the tiling constraint and the
        fallback cost shows up.
        """
        step = 512
        c = LAGUNA["text_config"]
        q = LAGUNA["quantization"]
        routings = step * c["num_experts_per_tok"]

        # Real geometry: activations only, even with TurboQuant weights.
        assert _moe_prefill_bytes(LAGUNA, step, q) == pytest.approx(
            routings * 8 * (c["hidden_size"] + c["moe_intermediate_size"])), \
            "no dequant term should appear for 64-divisible expert dims"

        # 1000 is not a multiple of 64, so gate/up cannot be tiled and fall
        # back to dequantizing every expert: 256 x 1000 x 3072 x 2 B = 1.57 GB,
        # under the switch layer's 2 GiB cap.
        odd = dict(LAGUNA, text_config=dict(c, moe_intermediate_size=1000))
        assert (_moe_prefill_bytes(odd, step, q)
                - routings * 8 * (3072 + 1000)) == pytest.approx(
            256 * 1000 * 3072 * 2)

        # Past the cap the switch layer uses the gather kernels instead, which
        # materialize nothing — so a huge expert tensor is NOT charged.
        huge = dict(LAGUNA, text_config=dict(c, moe_intermediate_size=1000,
                                             num_experts=4096))
        assert _moe_prefill_bytes(huge, step, q) == pytest.approx(
            routings * 8 * (3072 + 1000))

    def test_no_moe_term_for_a_dense_model(self):
        assert (prefill_workspace_bytes(MUSE, 5068, 512, is_moe=False)
                == prefill_workspace_bytes(MUSE, 5068, 512))

    def test_moe_term_skipped_when_routing_is_unknown(self):
        """Q35 has no moe_intermediate_size; guessing it would move a verdict
        on no evidence, so the term stays out."""
        assert _moe_prefill_bytes(Q35, 512, Q35["quantization"]) == 0.0

    def test_chunking_crossover_is_a_drop_not_a_jump(self):
        """Crossing the chunk size must not make the estimate worse: past the
        bound the lm_head term is elided, which is a saving, not a cost."""
        just_under = prefill_workspace_bytes(MUSE, 2048, 2048)
        just_over = prefill_workspace_bytes(MUSE, 2100, 2048)
        assert just_over < just_under


class TestFieldCalibration:
    """The mini datapoints. These are the reason the tool is trustworthy."""

    def _plan(self, tmp_path, weight_gb, **kw):
        # one fake expert tensor sized to hit the target total
        n = int(weight_gb * GB / 4)
        p = _write_model(tmp_path, Q35, {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": ("U32", (n,)),
        })
        return build_plan(p, wired_gb=10.5, ram_gb=16, **kw)

    def test_ternary_9_4gb_short_context_fits_default_cap(self, tmp_path):
        """Field: 10.42 GB peak at 512 tokens, runs under the DEFAULT wired cap
        with no sudo. The published model card promises exactly this."""
        pl = self._plan(tmp_path, 9.45, context=512)
        assert pl["projection"]["peak_bytes"] / GB == pytest.approx(10.42, abs=0.15)
        assert pl["verdict"]["mode"] == "resident"
        assert not pl["verdict"]["needs_wired_bump"], \
            "must not tell users to sudo for the no-sudo build"

    def test_down4_12_6gb_needs_the_wired_bump(self, tmp_path):
        """Field: 12.6 GB build needs `sysctl iogpu.wired_limit_mb=13824`, and
        then runs RESIDENT — it must NOT be told to stream."""
        pl = self._plan(tmp_path, 12.59, context=8000, kv_bits=8)
        assert pl["verdict"]["mode"] == "resident"
        assert pl["verdict"]["needs_wired_bump"]
        bump = [f for f in pl["flags"] if "iogpu.wired_limit_mb" in f]
        assert bump, "should recommend the sysctl raise"
        mb = int(bump[0].split("iogpu.wired_limit_mb=")[1].split()[0])
        assert 13000 <= mb <= 14500, f"bump {mb} MiB is far from the field's 13824"

    def test_long_agent_context_forces_a_small_prefill_step(self, tmp_path):
        """Field: 21K context on the mini needs --prefill-step-size 128; the
        2048 default OOMs.

        WEAKENED from `<= 256` to `< 2048`, deliberately and with the reason
        recorded, when `--ram-gb` was corrected from decimal GB to GiB. That
        fix raised the simulated mini's ceiling 14.40 -> 15.46 GB (matching
        what the real machine reports), after which the chooser picks 512 here
        instead of 128.

        The measured fact is only that **2048 OOMed and 128 worked** — 512 and
        256 were never tried, so the field data does not actually contradict
        512. What it does suggest is that two errors were cancelling: the
        undersized-RAM bug was compensating for an unmodelled MoE term. This
        assertion therefore pins the part that was measured and no more.

        That MoE term is now modelled (`_moe_prefill_bytes`), and it turned out
        NOT to be expert dequantization — sorted prefill tiles over packed
        weights and materializes nothing. It is routing-expanded activations,
        chunk x top_k wide. This fixture still does not exercise it, because
        Q35 carries no `moe_intermediate_size`; see the note on the fixture.

        TODO: re-measure the real step ceiling for a MoE at 21K on the mini,
        then tighten this back up.
        """
        pl = self._plan(tmp_path, 12.59, context=21000, kv_bits=8)
        step = pl["projection"]["prefill_step_size"]
        assert step < 2048, "the 2048 default OOMed in the field"
        assert any("prefill-step-size" in f for f in pl["flags"])

    def test_roomy_machine_keeps_the_fast_default(self, tmp_path):
        n = int(12.59 * GB / 4)
        p = _write_model(tmp_path, Q35, {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": ("U32", (n,)),
        })
        pl = build_plan(p, wired_gb=55.7, ram_gb=68.7, context=16384)
        assert pl["verdict"]["mode"] == "resident"
        assert not pl["verdict"]["needs_wired_bump"]
        assert pl["projection"]["prefill_step_size"] == 2048

    def test_model_far_past_ram_streams(self, tmp_path):
        """A 122B-class build on a 16 GB mini: experts page from disk."""
        pl = self._plan(tmp_path, 30.9, context=4096)
        assert pl["verdict"]["mode"] == "streaming"
        assert any("--streaming" in f for f in pl["flags"])
        assert any("cache-budget-gb" in f for f in pl["flags"])

    def test_reserve_does_not_double_count_kv_and_prefill(self):
        """The projection counts KV and workspace explicitly, so the reserve
        must be the *residue* (~1 GB measured), not serve.py's 2 GB budget."""
        assert _RESERVE_BYTES == pytest.approx(1.0 * GB)


class TestMachineIsOneMachine:
    """Review findings (PR #50): wss and ram must describe the SAME machine, and
    an unknown cap must not turn every verdict into 'needs a sudo bump'."""

    def _model(self, tmp_path, gb):
        n = int(gb * GB / 4)
        return _write_model(tmp_path, Q35, {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": ("U32", (n,)),
        })

    def test_assumed_ram_does_not_borrow_this_machines_wss(self, tmp_path):
        """Planning a 30 GB model for a 16 GB mini must not read the 55 GB
        working set of the machine running the command."""
        pl = build_plan(self._model(tmp_path, 30), ram_gb=16)
        assert pl["machine"]["wss_estimated"]
        assert pl["machine"]["wss_bytes"] < 12 * GB, \
            "used the host's working set for an assumed 16 GB machine"
        assert pl["verdict"]["mode"] == "streaming"

    def test_small_model_is_never_told_to_sudo(self, tmp_path):
        """A 2 GB model on 16 GB fits the default cap; recommending a wired
        bump for it is noise."""
        pl = build_plan(self._model(tmp_path, 2), ram_gb=16)
        assert pl["verdict"]["mode"] == "resident"
        assert not pl["verdict"]["needs_wired_bump"]
        assert not any("sysctl" in f for f in pl["flags"])

    def test_wss_estimate_tracks_the_measured_caps(self):
        """16 GB mini reports 10.5 GB; 68.7 GB M4 Max reports 55.7 GB. The
        estimate must be close on small machines and never optimistic."""
        assert estimate_wss(16 * GB) == pytest.approx(10.5 * GB, abs=0.3 * GB)
        assert estimate_wss(68.72 * GB) <= 55.66 * GB
        assert estimate_wss(None) is None

    def test_explicit_wired_gb_wins(self, tmp_path):
        pl = build_plan(self._model(tmp_path, 2), ram_gb=16, wired_gb=13.5)
        assert pl["machine"]["wss_bytes"] == pytest.approx(13.5 * GB)
        assert not pl["machine"]["wss_estimated"]

    def test_no_metal_device_falls_back_to_the_estimate(self, tmp_path, monkeypatch):
        """On a box with no Metal (CI/Linux) the cap is unknown — estimate it
        from RAM instead of flagging a bump for everything."""
        import builtins
        real = builtins.__import__

        def no_mlx(name, *a, **k):
            if name == "mlx.core":
                raise ImportError("no metal here")
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_mlx)
        m = machine()          # no overrides: must not raise, must not be None
        assert m["wss_bytes"] is None or m["wss_estimated"]


class TestRemotePlanning:
    """A repo id must be planned from headers over the network — downloading it
    would defeat the entire purpose of a preflight tool."""

    def test_repo_id_uses_remote_headers_not_a_download(self, monkeypatch):
        import turboquant_mlx.plan as P
        calls = {"cfg": 0, "idx": 0}

        def fake_cfg(repo):
            calls["cfg"] += 1
            return Q35

        def fake_idx(repo):
            calls["idx"] += 1
            n = int(9.45 * GB / 4)
            return {"model.layers.0.mlp.switch_mlp.gate_proj.weight":
                    ("U32", (n,))}

        monkeypatch.setattr(P, "read_remote_config", fake_cfg)
        monkeypatch.setattr(P, "read_remote_index", fake_idx)
        # a downloader in this path would be a bug: make it explode
        monkeypatch.setattr(P, "read_safetensors_index", lambda p: 1 / 0)

        pl = P.build_plan("org/some-repo", remote=True, wired_gb=10.5, ram_gb=16,
                          context=512)
        assert calls == {"cfg": 1, "idx": 1}
        assert pl["model"]["source"] == "huggingface"
        assert pl["model"]["name"] == "org/some-repo"
        assert pl["verdict"]["mode"] == "resident"

    def test_doctor_skips_on_disk_checks_for_a_remote_model(self, monkeypatch):
        import turboquant_mlx.plan as P
        monkeypatch.setattr(P, "read_remote_config", lambda r: Q35)
        monkeypatch.setattr(P, "read_remote_index", lambda r: {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight":
                ("U32", (int(9.45 * GB / 4),))})
        pl = P.build_plan("org/repo", remote=True, wired_gb=10.5, ram_gb=16)
        checks = P.run_doctor("org/repo", pl)
        ids = {c[0] for c in checks}
        assert "model.remote" in ids
        assert "model.dir" not in ids and "model.tokenizer" not in ids


class TestOutput:
    def test_render_and_doctor(self, tmp_path):
        n = int(9.45 * GB / 4)
        p = _write_model(tmp_path, Q35, {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": ("U32", (n,)),
        })
        pl = build_plan(p, wired_gb=10.5, ram_gb=16, context=512)
        text = render(pl)
        assert "Verdict:" in text and "RESIDENT" in text
        checks = run_doctor(p, pl)
        ids = {c[0] for c in checks}
        assert {"model.config", "model.weights", "model.tokenizer",
                "fit.projection"} <= ids
        assert all(s in ("pass", "warn", "fail") for _, s, _ in checks)

    def test_json_is_serialisable(self, tmp_path):
        n = int(9.45 * GB / 4)
        p = _write_model(tmp_path, Q35, {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": ("U32", (n,)),
        })
        pl = build_plan(p, wired_gb=10.5, ram_gb=16)
        json.dumps(pl)          # must not raise
        assert pl["schema"] == 1

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_plan(str(tmp_path))


class TestPartialCache:
    """Planning a repo id caches config.json, which creates a snapshot dir with
    no shards in it. Trusting that dir made `turboquant-plan --model <repo>`
    work once and then fail forever — it took its own cache droppings for a
    downloaded model. A cache hit now has to prove it holds the weights."""

    def _index(self, tmp_path, shards):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"t{i}": s for i, s in enumerate(shards)}}))

    def test_config_only_snapshot_is_not_a_checkpoint(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        self._index(tmp_path, ["model-00001-of-00002.safetensors"])
        assert is_complete_checkpoint(str(tmp_path)) is False

    def test_interrupted_download_is_not_a_checkpoint(self, tmp_path):
        """The dangerous case: some shards present. Sizes would be summed from
        only those headers, so a half-downloaded model reports half its weight
        and a falsely confident RESIDENT verdict."""
        self._index(tmp_path, ["a.safetensors", "b.safetensors"])
        (tmp_path / "a.safetensors").write_bytes(b"")
        assert is_complete_checkpoint(str(tmp_path)) is False

    def test_all_shards_present_is_a_checkpoint(self, tmp_path):
        self._index(tmp_path, ["a.safetensors", "b.safetensors"])
        (tmp_path / "a.safetensors").write_bytes(b"")
        (tmp_path / "b.safetensors").write_bytes(b"")
        assert is_complete_checkpoint(str(tmp_path)) is True

    def test_single_shard_without_index_is_a_checkpoint(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert is_complete_checkpoint(str(tmp_path)) is True

    def test_corrupt_index_is_not_a_checkpoint(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text("{not json")
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert is_complete_checkpoint(str(tmp_path)) is False

    @pytest.mark.parametrize("payload", [
        [],                                   # top-level list: no .get
        ["a"],
        "hello",                              # top-level str
        42,
        None,
        {"weight_map": ["a.safetensors"]},    # weight_map not a dict: no .values
        {"weight_map": "a.safetensors"},
        {"weight_map": {"t0": 5}},            # non-str value: os.path.join
        {"weight_map": {"t0": []}},           # unhashable value: set()
        {"weight_map": {"t0": None}},
    ])
    def test_malformed_index_degrades_instead_of_raising(self, tmp_path, payload):
        """The index is fetched from an arbitrary repo id — untrusted input.
        Every shape here raised an uncaught AttributeError/TypeError."""
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(payload))
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert is_complete_checkpoint(str(tmp_path)) is False

    def test_empty_weight_map_falls_back_to_globbing(self, tmp_path):
        """An index with nothing to say must not veto a shard that is present."""
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {}}))
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert is_complete_checkpoint(str(tmp_path)) is True

    def test_missing_dir_is_not_a_checkpoint(self, tmp_path):
        assert is_complete_checkpoint(str(tmp_path / "nope")) is False

    def test_partial_cache_hit_falls_back_to_remote(self, tmp_path, monkeypatch):
        """The regression itself: a snapshot with no shards must send the repo
        id back to the network, not be planned as a local model."""
        (tmp_path / "config.json").write_text("{}")
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                            lambda *a, **k: str(tmp_path))
        target, remote = _resolve("some/repo")
        assert (target, remote) == ("some/repo", True)

    def test_complete_cache_hit_is_used_locally(self, tmp_path, monkeypatch):
        (tmp_path / "model.safetensors").write_bytes(b"")
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                            lambda *a, **k: str(tmp_path))
        assert _resolve("some/repo") == (str(tmp_path), False)

    def test_explicit_local_path_warns_when_shards_are_missing(self, tmp_path):
        """A local path is the user's word, so it is planned — but the numbers
        are undercounted and the plan has to say so."""
        n = int(9.45 * GB / 4)
        p = _write_model(tmp_path, Q35, {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": ("U32", (n,)),
        })
        self._index(tmp_path, ["model.safetensors", "missing.safetensors"])
        pl = build_plan(p, wired_gb=10.5, ram_gb=16)
        assert any("UNDER-counted" in w for w in pl["warnings"])
        assert "UNDER-counted" in render(pl)


K3_TEXT = {
    # Kimi K3 geometry (moonshotai/Kimi-K3 text_config), the KDA/MLA dialect:
    # layer lists are 1-based; only the 24 MLA layers hold growing KV, and MLA
    # caches the shared latent + rope key once per layer, not per head.
    "num_hidden_layers": 93,
    "num_attention_heads": 96,
    "num_key_value_heads": 96,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "head_dim": None,
    "hidden_size": 7168,
    "linear_attn_config": {
        "num_heads": 96,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "kda_layers": [i for i in range(1, 94) if i % 4 != 0 and i != 93],
        "full_attn_layers": [i for i in range(4, 93, 4)] + [93],
    },
}


class TestKDAHybrid:
    def test_kda_hybrid_reads_full_attn_layers(self):
        cfg = {"text_config": K3_TEXT}
        n_full, n_slide, n_layers, note = _attention_layers(cfg)
        assert (n_full, n_slide, n_layers) == (24, 0, 93)
        assert "KDA" in note

    def test_mla_latent_kv_geometry(self):
        """The dense formula (2*96 heads*128 dim) over-predicts K3's KV ~160x:
        MLA caches (kv_lora_rank + qk_rope_head_dim) once per layer."""
        cfg = {"text_config": K3_TEXT}
        context = 65536
        total, per_tok, note = kv_bytes(cfg, context)
        lac = K3_TEXT["linear_attn_config"]
        fixed_kda = len(lac["kda_layers"]) * (
            96 * 128 * 128 * 4.0 + 3 * 3 * 96 * 128 * 2.0
        )
        grow = 24 * context * (512 + 64) * 2.0
        assert total == pytest.approx(grow + fixed_kda)
        assert "MLA latent" in note and "KDA state" in note
        # sanity: a 1M-token context stays under 30 GB of KV
        total_1m, *_ = kv_bytes(cfg, 1_048_576)
        assert total_1m < 30 * GB

    def test_kda_fixed_state_is_charged_at_zero_context(self):
        """The recurrent state exists before the first token; never report 0."""
        cfg = {"text_config": K3_TEXT}
        total, *_ = kv_bytes(cfg, 1)
        assert total > 0.4 * GB


class TestAssumedMachineUnits:
    """`--ram-gb 16` must describe a machine SOLD as 16 GB, i.e. 16 GiB.

    FIELD-CALIBRATED. Getting this wrong is not cosmetic: reading `--ram-gb 16`
    as 16e9 bytes understates a real 16 GB mini by 7.4% (14.40 vs 15.46 GB
    usable) and produced a WILL-NOT-RUN verdict for Muse-Glimmer-30B tq3 —
    a model that then ran resident on that exact machine (Mac16,10, macOS 26.5.2,
    iogpu.wired_limit_mb=14336), peaking at 14.13 GiB.
    """

    def test_ram_gb_is_gibibytes_not_decimal(self):
        from turboquant_mlx.plan import machine
        m = machine(ram_gb=16)
        # a 16 GB Mac reports hw.memsize = 17179869184
        assert m["ram_bytes"] == 17179869184
        assert m["assumed"] is True

    @pytest.mark.parametrize("gb,expected", [
        (8, 8589934592),
        (16, 17179869184),
        (36, 38654705664),
        (64, 68719476736),
        (128, 137438953472),
    ])
    def test_common_mac_sizes_match_hw_memsize(self, gb, expected):
        from turboquant_mlx.plan import machine
        assert machine(ram_gb=gb)["ram_bytes"] == expected

    def test_assumed_16gb_reproduces_the_real_mini_ceiling(self):
        """The measured mini reported 15.46 GB usable with the wired cap
        raised. The simulation must agree, or planning for a machine you do
        not own is worthless."""
        from turboquant_mlx.plan import machine, estimate_wss
        m = machine(ram_gb=16)
        raisable = min(m["ram_bytes"] * 0.90, m["ram_bytes"] - 1.5 * 1e9)
        assert 15.4e9 < raisable < 15.5e9
        # and the un-raised default cap is well below the model, which is why
        # the wired bump is required rather than optional
        assert estimate_wss(m["ram_bytes"]) < 12e9
