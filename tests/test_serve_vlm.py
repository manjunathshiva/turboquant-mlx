"""Serving TurboQuant VLMs through mlx-vlm's server.

Four things go silently wrong without this module, and none of them raise:

1. mlx-vlm's server cannot load a polar checkpoint at all.
2. Muse Glimmer's reasoning channel lands in `message.content`, because
   mlx-vlm's ATEM parser only strips the envelope when a tool call was parsed.
   An agent harness then reads the model's private deliberation as its answer.
3. `reasoning_effort` never reaches the template, which reads
   `reasoning_strength` — so the model always deliberates at `high`.
4. While streaming, ATEM's `tool_call_start` is the same string Muse Glimmer
   opens every turn with, so mlx-vlm suppresses every content delta and the
   client receives an empty answer.
"""

import json
import sys

import pytest

from turboquant_mlx import serve_vlm
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


class TestHelp:
    def test_help_documents_our_own_flag(self, capsys, monkeypatch):
        """`--help` is handled by mlx-vlm's parser, which never sees our flag.

        Without this, `--reasoning-strength` is documented on the model cards
        but absent from `--help`, which reads as "the flag does not exist".
        """
        import turboquant_mlx.serve_vlm as serve_vlm

        monkeypatch.setattr(serve_vlm, "install_turboquant_loader", lambda: None)

        def fake_server_main():
            raise SystemExit(0)

        import types

        fake_cli = types.ModuleType("mlx_vlm.server.cli")
        fake_cli.main = fake_server_main
        monkeypatch.setitem(sys.modules, "mlx_vlm.server.cli", fake_cli)

        with pytest.raises(SystemExit):
            serve_vlm.main(["--help"])

        out = capsys.readouterr().out
        assert "--reasoning-strength" in out
        # and it must not swallow mlx-vlm's own help
        assert "mlx-vlm" in out


class TestDefaultHost:
    """mlx-vlm binds 0.0.0.0 by default; ours binds loopback unless asked."""

    def test_loopback_is_added_when_no_host_given(self):
        assert serve_vlm.with_default_host(["--port", "8080"]) == [
            "--port", "8080", "--host", "127.0.0.1"]

    @pytest.mark.parametrize("argv", [
        ["--host", "0.0.0.0"], ["--host=0.0.0.0"], ["--host", "127.0.0.1"]])
    def test_an_explicit_host_is_kept(self, argv):
        assert serve_vlm.with_default_host(list(argv)) == argv

    def test_main_hands_mlx_vlm_the_loopback_default(self, monkeypatch):
        import types

        seen = {}

        def fake_server_main():
            seen["argv"] = list(sys.argv[1:])

        fake_cli = types.ModuleType("mlx_vlm.server.cli")
        fake_cli.main = fake_server_main
        monkeypatch.setitem(sys.modules, "mlx_vlm.server.cli", fake_cli)
        monkeypatch.setattr(serve_vlm, "install_turboquant_loader", lambda: None)
        monkeypatch.setattr(serve_vlm, "install_tool_call_suppression_fix",
                            lambda: None)
        monkeypatch.setattr(sys, "argv", ["turboquant-serve-vlm"])

        serve_vlm.main(["--port", "8080"])
        argv = seen["argv"]
        assert argv[argv.index("--host") + 1] == "127.0.0.1"

        serve_vlm.main(["--host", "0.0.0.0"])
        assert seen["argv"].count("--host") == 1
        assert "0.0.0.0" in seen["argv"]


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


class TestToolCallSuppressionFix:
    """The reasoning-channel opener must not suppress the streamed answer.

    ATEM declares `tool_call_start = "to=self<|message|>"`, which is exactly
    what Muse Glimmer emits at the start of every turn. mlx-vlm's streaming
    suppressor treats that string as a call opening, so without this fix a
    tools-enabled request streams no content: 0.6 dropped it outright, 0.7's
    `ToolCallStreamState` holds the whole turn back until the stream ends.
    """

    ROUTER = "to=self<|message|>"
    CALL_TAG = "<atem:function_calls>"
    CALL_END = "</atem:function_calls>"
    # A turn with no tool call: router, reasoning, then the answer.
    TURN = [ROUTER, "thinking", "<|start|>assistant to=user<|message|>",
            "The answer ", "is 42."]

    def test_the_router_is_remapped_to_the_real_call_tag(self):
        assert serve_vlm.TOOL_CALL_SUPPRESSION_TRIGGER[self.ROUTER] == self.CALL_TAG

    def test_unknown_parsers_pass_through_untouched(self):
        assert "<tool_call>" not in serve_vlm.TOOL_CALL_SUPPRESSION_TRIGGER

    @pytest.fixture
    def state_cls(self, monkeypatch):
        """mlx-vlm's *real* stream state, with its constructor restored after.

        Deliberately the real upstream class rather than a fake: these tests
        assert what mlx-vlm actually does, so a hand-written stand-in would
        only prove the stand-in matches itself. That makes them require the
        optional `[vlm]` extra, which CI does not install.
        """
        pytest.importorskip(
            "mlx_vlm", reason="needs the [vlm] extra for the real stream state"
        )
        from mlx_vlm.server.responses_state import ToolCallStreamState

        # Start from upstream's own constructor even if another test already
        # installed the fix on this shared class; restored afterwards.
        init = ToolCallStreamState.__init__
        while getattr(init, "_turboquant_wrapper", False):
            init = init.__wrapped__
        monkeypatch.setattr(ToolCallStreamState, "__init__", init)
        return ToolCallStreamState

    def _stream(self, cls, chunks):
        state = cls(self.ROUTER, self.CALL_END)
        return [state.feed(c, last=(i == len(chunks) - 1))
                for i, c in enumerate(chunks)]

    def test_reasoning_no_longer_swallows_the_answer(self, state_cls):
        before = self._stream(state_cls, self.TURN)
        assert before[:-1] == [None] * (len(self.TURN) - 1), (
            "precondition: upstream holds the turn back until the end")

        serve_vlm.install_tool_call_suppression_fix()

        after = self._stream(state_cls, self.TURN)
        assert after[-2:] == ["The answer", " is 42."], "the answer streams"

    def test_a_real_tool_call_is_still_suppressed(self, state_cls):
        serve_vlm.install_tool_call_suppression_fix()

        out = self._stream(state_cls, [
            self.ROUTER, "plan", self.CALL_TAG, '{"name": "f"}', self.CALL_END,
            " done"])
        shown = "".join(x for x in out if x)
        assert self.CALL_TAG not in shown and '"name"' not in shown, (
            "tool markup must not reach the user")
        assert shown.endswith("done")

    def test_installing_twice_does_not_double_wrap(self, state_cls):
        serve_vlm.install_tool_call_suppression_fix()
        once = state_cls.__init__
        serve_vlm.install_tool_call_suppression_fix()
        assert state_cls.__init__ is once
