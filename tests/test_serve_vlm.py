"""Serving TurboQuant VLMs through mlx-vlm's server.

Three things go silently wrong without this module, and none of them raise:

1. mlx-vlm's server cannot load a polar checkpoint at all.
2. Muse Glimmer's reasoning channel lands in `message.content`, because
   mlx-vlm's ATEM parser only strips the envelope when a tool call was parsed.
   An agent harness then reads the model's private deliberation as its answer.
3. `reasoning_effort` never reaches the template, which reads
   `reasoning_strength` — so the model always deliberates at `high`.
"""

import json

import pytest

from turboquant_mlx.serve_vlm import (
    CHANNEL_DELIMITERS,
    build_server_argv,
    channel_delimiters_for,
    chat_template_text,
    is_turboquant_checkpoint,
    map_reasoning_effort,
    model_architecture,
)


def _write_model(tmp_path, config=None, template=None, template_in_config=False):
    (tmp_path / "config.json").write_text(json.dumps(config or {}))
    if template is not None:
        if template_in_config:
            (tmp_path / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": template})
            )
        else:
            (tmp_path / "chat_template.jinja").write_text(template)
    return tmp_path


class TestCheckpointDetection:
    def test_turboquant_checkpoint_is_detected(self, tmp_path):
        _write_model(tmp_path, {"quantization": {"mode": "turboquant"}})
        assert is_turboquant_checkpoint(tmp_path) is True

    def test_affine_mlx_checkpoint_is_not_hijacked(self, tmp_path):
        """An ordinary mlx-vlm model must keep mlx-vlm's own loader."""
        _write_model(tmp_path, {"quantization": {"group_size": 64, "bits": 4}})
        assert is_turboquant_checkpoint(tmp_path) is False

    def test_unquantized_model_is_not_a_turboquant_checkpoint(self, tmp_path):
        _write_model(tmp_path, {"model_type": "muse_glimmer"})
        assert is_turboquant_checkpoint(tmp_path) is False

    def test_missing_or_unreadable_config_is_false_not_an_exception(self, tmp_path):
        assert is_turboquant_checkpoint(tmp_path) is False
        (tmp_path / "config.json").write_text("{not json")
        assert is_turboquant_checkpoint(tmp_path) is False


class TestReasoningEffortMapping:
    @pytest.mark.parametrize(
        "effort,expected",
        [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("HIGH", "high"),
            # Muse has no "off" level; the floor is "low".
            ("minimal", "low"),
            ("none", "low"),
        ],
    )
    def test_openai_vocabulary_maps_onto_muse_levels(self, effort, expected):
        assert map_reasoning_effort(effort) == expected

    def test_absent_effort_stays_absent(self):
        assert map_reasoning_effort(None) is None
        assert map_reasoning_effort("") is None

    def test_unknown_value_passes_through_rather_than_being_dropped(self):
        """Better to hand the template something it may know than nothing."""
        assert map_reasoning_effort("ludicrous") == "ludicrous"


class TestChannelDelimiters:
    def test_muse_glimmer_gets_delimiters(self, tmp_path):
        _write_model(tmp_path, {"model_type": "muse_glimmer"})
        assert channel_delimiters_for(tmp_path) == CHANNEL_DELIMITERS["muse_glimmer"]

    def test_the_span_ends_at_the_routing_header_not_at_eom(self):
        """Ending at `<|eom|>` would leave the routing header on every reply.

        Muse emits `…<|eom|><|start|>assistant to=user<|message|>answer`, so the
        end delimiter has to swallow the header for content to start at the
        first real answer token.
        """
        start, end = CHANNEL_DELIMITERS["muse_glimmer"]
        assert start == "to=self<|message|>"
        assert end == "<|start|>assistant to=user<|message|>"
        assert end != "<|eom|>"

    def test_delimiters_split_a_real_generation(self):
        """Replays the exact shape measured from the served model."""
        start, end = CHANNEL_DELIMITERS["muse_glimmer"]
        raw = (
            " to=self<|message|>We have tool output: temp_c 17.\n\nJust respond."
            "<|eom|><|start|>assistant to=user<|message|>"
            "Paris is currently 17°C with light rain."
        )
        assert start in raw and end in raw
        reasoning, _, content = raw.partition(end)
        assert content == "Paris is currently 17°C with light rain."
        assert "to=user" not in content
        assert "<|message|>" not in content
        assert "We have tool output" in reasoning

    def test_unknown_architecture_gets_nothing(self, tmp_path):
        _write_model(tmp_path, {"model_type": "qwen3_vl"})
        assert channel_delimiters_for(tmp_path) is None

    def test_architecture_is_read_from_config(self, tmp_path):
        _write_model(tmp_path, {"model_type": "muse_glimmer"})
        assert model_architecture(tmp_path) == "muse_glimmer"


class TestServerArgv:
    def test_delimiters_are_injected_for_muse_glimmer(self, tmp_path):
        _write_model(tmp_path, {"model_type": "muse_glimmer"})
        argv = build_server_argv(["--model", str(tmp_path)], tmp_path)
        start, end = CHANNEL_DELIMITERS["muse_glimmer"]
        assert argv[-4:] == [
            "--thinking-start-token",
            start,
            "--thinking-end-token",
            end,
        ]

    def test_an_explicit_delimiter_is_never_overridden(self, tmp_path):
        """The user asked for something specific; do not argue."""
        _write_model(tmp_path, {"model_type": "muse_glimmer"})
        argv = ["--model", str(tmp_path), "--thinking-start-token", "<think>"]
        assert build_server_argv(argv, tmp_path) == argv

    def test_an_explicit_end_delimiter_alone_also_wins(self, tmp_path):
        _write_model(tmp_path, {"model_type": "muse_glimmer"})
        argv = ["--model", str(tmp_path), "--thinking-end-token=</think>"]
        assert build_server_argv(argv, tmp_path) == argv

    def test_other_architectures_are_left_alone(self, tmp_path):
        _write_model(tmp_path, {"model_type": "qwen3_vl"})
        argv = ["--model", str(tmp_path)]
        assert build_server_argv(argv, tmp_path) == argv


class TestChatTemplateDiscovery:
    def test_template_read_from_jinja_file(self, tmp_path):
        _write_model(tmp_path, {}, template="{{ reasoning_strength }}")
        assert "reasoning_strength" in chat_template_text(tmp_path)

    def test_template_read_from_tokenizer_config(self, tmp_path):
        _write_model(
            tmp_path, {}, template="{{ reasoning_strength }}", template_in_config=True
        )
        assert "reasoning_strength" in chat_template_text(tmp_path)

    def test_missing_template_is_empty_not_an_error(self, tmp_path):
        _write_model(tmp_path, {})
        assert chat_template_text(tmp_path) == ""
