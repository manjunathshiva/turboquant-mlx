"""Regression tests for the K3 tokenizer trust gate (turboquant_mlx.compat).

Contract: ``trust_remote_code`` is auto-injected ONLY for a local directory
whose config.json declares model_type == "kimi_k3" AND that the user named
on a command line (``compat.trust_local_kimi_k3``, called before anything is
resolved). A Hub repo id never qualifies, and neither does the cache snapshot
it resolves to — every loader downloads before the tokenizer loads, so the
snapshot is "a local kimi_k3 directory" by the time the tokenizer sees it.
A model named in a server request is never auto-trusted unless it is the
operator's own ``--model`` directory. An explicit ``trust_remote_code`` from
the caller always wins.

The end-to-end tests drive the real resolution path (mlx-lm's ``_download``,
``load_tokenizer``, ``load``, ``turboquant_mlx.generate.load_turboquant`` and
``mlx_lm.server.ModelProvider`` behind ``turboquant-serve``'s loader) with only
``snapshot_download`` and model construction stubbed. The checkpoint ships a
``tokenization_marker.py`` that writes a file when executed: the marker is the
proof of whether checkpoint code ran.
"""

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import turboquant_mlx.compat as compat


def _model_dir(tmp_path, model_type):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return d


def test_local_kimi_k3_dir_is_trusted(tmp_path):
    assert compat.is_local_kimi_k3(_model_dir(tmp_path, "kimi_k3")) is True


def test_other_local_model_is_not_trusted(tmp_path):
    # the review's repro: a local qwen3_moe dir must NOT execute its own code
    assert compat.is_local_kimi_k3(_model_dir(tmp_path, "qwen3_moe")) is False


def test_hub_repo_id_is_not_trusted():
    assert compat.is_local_kimi_k3("moonshotai/Kimi-K3") is False


def test_dir_without_config_is_not_trusted(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert compat.is_local_kimi_k3(d) is False


def test_corrupt_config_fails_safe(tmp_path):
    d = tmp_path / "corrupt"
    d.mkdir()
    (d / "config.json").write_text("{not json")
    assert compat.is_local_kimi_k3(d) is False


# --------------------------------------------------------------------------
# End-to-end: the trust decision through the real loaders
# --------------------------------------------------------------------------

_TOKENIZER_SRC = '''
import pathlib
pathlib.Path({marker!r}).write_text("ran")

from transformers import PreTrainedTokenizer


class MarkerTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        self._vocab = {{"a": 0, "b": 1}}
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        return [c for c in text if c in self._vocab]

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, 0)

    def _convert_id_to_token(self, index):
        return {{0: "a", 1: "b"}}[index]

    def save_vocabulary(self, *args, **kwargs):
        return ()
'''


class _TinyArgs:
    @classmethod
    def from_dict(cls, config):
        return cls()


class _Tiny(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def make_cache(self):
        return []


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Fresh trust registry; dynamic modules land in tmp; never prompt."""
    import transformers.dynamic_module_utils as dmu

    monkeypatch.setattr(compat, "_TRUSTED_K3_DIRS", set())
    monkeypatch.setattr(dmu, "HF_MODULES_CACHE", str(tmp_path / "hf_modules"))
    # 0 = raise instead of the interactive "run the custom code? [y/N]"
    monkeypatch.setattr(dmu, "TIME_OUT_REMOTE_CODE", 0)


def _checkpoint(d, marker, model_type="kimi_k3", tq=False):
    """A checkpoint whose tokenizer code writes ``marker`` when executed."""
    d.mkdir(parents=True)
    cfg = {"model_type": model_type}
    if tq:
        cfg["quantization"] = {"mode": "turboquant", "bits": 3, "group_size": 64}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "MarkerTokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_marker.MarkerTokenizer", None]},
    }))
    (d / "tokenization_marker.py").write_text(_TOKENIZER_SRC.format(marker=str(marker)))
    mx.save_safetensors(str(d / "model.safetensors"), {
        "proj.weight": mx.zeros((2, 2)), "proj.bias": mx.zeros((2,)),
    })
    return d


@pytest.fixture
def stub_model(monkeypatch):
    """Build a stand-in model instead of a real K3; the tokenizer path is real."""
    import mlx_lm.utils as mlx_utils

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (_Tiny, _TinyArgs))
    # TurboQuant's loader resolves kimi_k3 through its own port, not _get_classes.
    monkeypatch.setattr(compat, "get_model_classes",
                        lambda config: (_Tiny, _TinyArgs))
    monkeypatch.setattr(
        mlx_utils, "load_model",
        lambda model_path, lazy=False, model_config=None, **kw:
            (_Tiny(), mlx_utils.load_config(model_path)))


@pytest.fixture
def hub(monkeypatch, tmp_path, stub_model):
    """A fake Hub: ``attacker/k3`` resolves, via mlx-lm's real ``_download``,
    to a K3 snapshot in a HF-style cache directory."""
    import mlx_lm.utils as mlx_utils

    marker = tmp_path / "hub_code_ran"
    snapshot = _checkpoint(
        tmp_path / "hub" / "models--attacker--k3" / "snapshots" / "0123abcd",
        marker, tq=True)
    calls = []

    def snapshot_download(repo_id, revision=None, allow_patterns=None, **kw):
        calls.append(repo_id)
        assert repo_id == "attacker/k3", repo_id
        return str(snapshot)

    monkeypatch.setattr(mlx_utils, "snapshot_download", snapshot_download)
    return SimpleNamespace(repo="attacker/k3", snapshot=snapshot,
                           marker=marker, calls=calls)


@pytest.fixture
def local_k3(tmp_path, stub_model):
    marker = tmp_path / "local_code_ran"
    d = _checkpoint(tmp_path / "Kimi-K3-tq", marker, tq=True)
    return SimpleNamespace(path=d, marker=marker)


def _refused(fn, *args, **kwargs):
    """The load must reach the tokenizer and have transformers refuse it."""
    with pytest.raises(ValueError, match="trust_remote_code"):
        fn(*args, **kwargs)


# -- the registry ----------------------------------------------------------

def test_registry_only_takes_local_k3_dirs(tmp_path, local_k3):
    other = _model_dir(tmp_path, "qwen3_moe")
    assert compat.trust_local_kimi_k3("attacker/k3") is False
    assert compat.trust_local_kimi_k3(other) is False
    assert compat.trust_local_kimi_k3(local_k3.path) is True
    assert compat.is_trusted_kimi_k3(local_k3.path)
    assert not compat.is_trusted_kimi_k3(other)


def test_registry_matches_by_realpath(tmp_path, local_k3):
    link = tmp_path / "k3-link"
    link.symlink_to(local_k3.path)
    compat.trust_local_kimi_k3(link)
    assert compat.is_trusted_kimi_k3(local_k3.path)
    assert compat.is_trusted_kimi_k3(str(local_k3.path) + "/")


def test_unregistered_local_k3_dir_is_not_auto_trusted(local_k3):
    """Being a local kimi_k3 dir is not enough — a resolved snapshot is one."""
    from mlx_lm.utils import load_tokenizer

    _refused(load_tokenizer, local_k3.path)
    assert not local_k3.marker.exists()


# -- generate / library path ----------------------------------------------

def test_hub_repo_id_does_not_run_tokenizer_code(hub):
    from mlx_lm.utils import load_tokenizer

    _refused(load_tokenizer, hub.repo)
    assert hub.calls == ["attacker/k3"]           # the real _download ran
    assert not hub.marker.exists()


def test_hub_repo_id_through_load_turboquant(hub):
    from turboquant_mlx.generate import load_turboquant

    _refused(load_turboquant, hub.repo)
    assert hub.calls, "load_turboquant must have resolved the repo id"
    assert not hub.marker.exists()


def test_hub_repo_id_through_mlx_lm_load(hub):
    import mlx_lm.utils as mlx_utils

    _refused(mlx_utils.load, hub.repo)
    assert not hub.marker.exists()


def test_cli_registering_a_hub_repo_id_does_not_trust_its_snapshot(hub):
    """What turboquant-generate --model attacker/k3 does now."""
    from turboquant_mlx.generate import load_turboquant

    assert compat.trust_local_kimi_k3(hub.repo) is False
    _refused(load_turboquant, hub.repo)
    assert not hub.marker.exists()


def test_explicit_trust_flag_still_loads_hub_repo(hub):
    """--trust-remote-code is the opt-in for a K3 repo id."""
    from turboquant_mlx.generate import load_turboquant

    _model, tok = load_turboquant(
        hub.repo, tokenizer_config={"trust_remote_code": True})
    assert hub.marker.exists()
    assert tok.encode("ab", add_special_tokens=False) == [0, 1]


def test_registered_local_k3_dir_loads_without_flag(local_k3):
    """The reason the patch exists: a converted local K3 loads, no prompt."""
    from turboquant_mlx.generate import load_turboquant

    compat.trust_local_kimi_k3(str(local_k3.path))
    _model, tok = load_turboquant(local_k3.path)
    assert local_k3.marker.exists()
    assert tok.encode("ab", add_special_tokens=False) == [0, 1]


def test_explicit_false_beats_registration(local_k3):
    from mlx_lm.utils import load_tokenizer

    compat.trust_local_kimi_k3(local_k3.path)
    _refused(load_tokenizer, local_k3.path, {"trust_remote_code": False})
    assert not local_k3.marker.exists()


# -- turboquant-serve: models named in requests ----------------------------

@pytest.fixture
def serve_loader(monkeypatch):
    """Install turboquant-serve's loader; restore mlx-lm's afterwards."""
    import mlx_lm.server as server_mod
    import mlx_lm.utils as mlx_utils

    from turboquant_mlx import serve

    monkeypatch.setattr(server_mod, "load", server_mod.load)
    monkeypatch.setattr(mlx_utils, "load", mlx_utils.load)
    serve._patch_loader()
    return serve


def _provider(model=None, trust_remote_code=False):
    """mlx-lm's ModelProvider, as turboquant-serve's mlx_lm.server builds it."""
    from mlx_lm.server import ModelProvider

    return ModelProvider(SimpleNamespace(
        model=model, adapter_path=None, draft_model=None,
        trust_remote_code=trust_remote_code, chat_template="",
        use_default_chat_template=False, pipeline=False,
    ))


def test_serve_request_for_hub_repo_turboquant(serve_loader, hub):
    _refused(_provider().load, hub.repo)
    assert hub.calls == ["attacker/k3"]
    assert not hub.marker.exists()


def test_serve_request_for_hub_repo_with_operator_k3(serve_loader, hub, local_k3):
    """The operator's own trusted K3 dir does not extend to other models."""
    serve_loader._trust_operator_model(["--model", str(local_k3.path)])
    provider = _provider(model=str(local_k3.path))
    _refused(provider.load, hub.repo)
    assert hub.calls == ["attacker/k3"]
    assert not hub.marker.exists()


def test_serve_request_for_hub_repo_non_turboquant(serve_loader, hub):
    """A non-TurboQuant checkpoint takes mlx-lm's own loader; same answer."""
    cfg = json.loads((hub.snapshot / "config.json").read_text())
    del cfg["quantization"]
    (hub.snapshot / "config.json").write_text(json.dumps(cfg))
    _refused(_provider().load, hub.repo)
    assert not hub.marker.exists()


def test_serve_request_naming_a_snapshot_path_does_not_run_code(serve_loader, hub):
    """A cached snapshot is a local kimi_k3 directory; naming it directly
    in a request must not count as the operator's choice."""
    _refused(_provider().load, str(hub.snapshot))
    assert hub.calls == []                         # local path: no download
    assert not hub.marker.exists()


def test_serve_request_for_draft_model_does_not_run_code(serve_loader, hub, local_k3):
    serve_loader._trust_operator_model(["--model", str(local_k3.path)])
    provider = _provider(model=str(local_k3.path))
    _refused(provider.load, "default_model", None, hub.repo)
    assert not hub.marker.exists()


def test_serve_operator_local_k3_model_loads(serve_loader, local_k3):
    serve_loader._trust_operator_model([f"--model={local_k3.path}", "--port", "8080"])
    _model, tok = _provider(model=str(local_k3.path)).load("default_model")
    assert local_k3.marker.exists()
    assert tok.encode("ab", add_special_tokens=False) == [0, 1]


def test_serve_trust_flag_reaches_turboquant_loader(serve_loader, hub):
    """mlx-lm's --trust-remote-code used to be dropped on the TurboQuant path."""
    _model, tok = _provider(trust_remote_code=True).load(hub.repo)
    assert hub.marker.exists()


def test_serve_operator_hub_repo_is_not_registered(serve_loader, hub):
    serve_loader._trust_operator_model(["--model", hub.repo])
    assert compat._TRUSTED_K3_DIRS == set()
    _refused(_provider(model=hub.repo).load, "default_model")
    assert not hub.marker.exists()


def test_serve_streaming_path_passes_trust_through(monkeypatch, hub):
    """--cache-budget-gb routes TurboQuant models through load_streaming."""
    import mlx_lm.server as server_mod
    import mlx_lm.utils as mlx_utils

    import turboquant_mlx.stream.loader as loader
    from turboquant_mlx import serve

    seen = []

    def fake_load_streaming(path, tokenizer_config=None, **kw):
        seen.append(tokenizer_config)
        raise _Stop

    monkeypatch.setattr(server_mod, "load", server_mod.load)
    monkeypatch.setattr(mlx_utils, "load", mlx_utils.load)
    monkeypatch.setattr(loader, "load_streaming", fake_load_streaming)
    serve._patch_loader(stream_config={"cache_budget_gb": 4.0})
    for flag in (False, True):
        with pytest.raises(_Stop):
            _provider(trust_remote_code=flag).load(hub.repo)
    # An unregistered repo keeps the server's explicit False (a refusal, never
    # transformers' interactive prompt); only the operator's registered K3
    # directory is mapped back to "unset" (test_serve_operator_local_k3_model_loads).
    assert seen == [{"trust_remote_code": False}, {"trust_remote_code": True}]


def test_load_streaming_passes_tokenizer_config_on(monkeypatch, local_k3):
    import turboquant_mlx.stream.loader as loader

    seen = []

    def fake_load_turboquant(path, tokenizer_config=None, **kw):
        seen.append(tokenizer_config)
        raise _Stop

    monkeypatch.setattr(loader, "load_turboquant", fake_load_turboquant)
    with pytest.raises(_Stop):
        loader.load_streaming(local_k3.path, use_page_cache=False,
                              tokenizer_config={"trust_remote_code": True})
    assert seen == [{"trust_remote_code": True}]


# -- CLI entry points: what they register and pass ---------------------------

class _Stop(Exception):
    pass


def _run_cli(monkeypatch, module, loader_name, argv):
    """Run a CLI's main() up to its loader call; return (kwargs, registry)."""
    import sys

    seen = {}

    def fake_loader(model, **kwargs):
        seen["model"] = model
        seen["kwargs"] = kwargs
        seen["trusted"] = set(compat._TRUSTED_K3_DIRS)
        raise _Stop

    monkeypatch.setattr(module, loader_name, fake_loader)
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(_Stop):
        module.main()
    return seen


@pytest.mark.parametrize("cli", ["generate", "stream_generate"])
def test_cli_registers_only_a_local_k3_model(monkeypatch, cli, hub, local_k3):
    import os

    import turboquant_mlx.generate as gen
    import turboquant_mlx.stream.stream_generate as sgen

    module, loader_name = ((gen, "load_turboquant") if cli == "generate"
                           else (sgen, "load_streaming"))

    seen = _run_cli(monkeypatch, module, loader_name, ["--model", hub.repo])
    assert seen["trusted"] == set()
    assert seen["kwargs"]["tokenizer_config"] is None
    assert hub.calls == []                   # decided before any download

    compat._TRUSTED_K3_DIRS.clear()
    seen = _run_cli(monkeypatch, module, loader_name, ["--model", str(local_k3.path)])
    assert seen["trusted"] == {os.path.realpath(local_k3.path)}

    compat._TRUSTED_K3_DIRS.clear()
    seen = _run_cli(monkeypatch, module, loader_name,
                    ["--model", hub.repo, "--trust-remote-code"])
    assert seen["trusted"] == set()
    assert seen["kwargs"]["tokenizer_config"] == {"trust_remote_code": True}


def test_serve_main_registers_the_operator_model(monkeypatch, local_k3):
    import os
    import sys

    import mlx_lm.server as server_mod
    import mlx_lm.utils as mlx_utils

    from turboquant_mlx import serve

    seen = {}

    def fake_server_main():
        seen["argv"] = list(sys.argv[1:])
        seen["trusted"] = set(compat._TRUSTED_K3_DIRS)

    monkeypatch.setattr(server_mod, "load", server_mod.load)
    monkeypatch.setattr(mlx_utils, "load", mlx_utils.load)
    monkeypatch.setattr(server_mod, "main", fake_server_main)
    monkeypatch.setattr(sys, "argv", [
        "turboquant-serve", "--model", str(local_k3.path), "--port", "8080"])
    serve.main()
    assert seen["trusted"] == {os.path.realpath(local_k3.path)}
    assert seen["argv"] == ["--model", str(local_k3.path), "--port", "8080"]


def test_convert_cli_flag_reaches_convert(monkeypatch):
    import sys

    import turboquant_mlx.convert as conv

    seen = {}
    monkeypatch.setattr(conv, "convert", lambda **kw: seen.update(kw))
    monkeypatch.setattr(sys, "argv", ["prog", "--hf-path", "org/repo"])
    conv.main()
    assert seen["trust_remote_code"] is False
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--hf-path", "org/repo", "--trust-remote-code"])
    conv.main()
    assert seen["trust_remote_code"] is True


def test_convert_trust_flag_loads_hub_repo(hub, tmp_path, monkeypatch):
    import mlx_lm.utils as mlx_utils

    from turboquant_mlx.convert import convert

    def save(*a, **k):
        raise _Stop

    monkeypatch.setattr(mlx_utils, "save", save)
    with pytest.raises(_Stop):
        convert(hub.repo, str(tmp_path / "out"), trust_remote_code=True)
    assert hub.marker.exists()


@pytest.mark.parametrize("flag", [True, False])
def test_convert_model_code_gate_follows_only_the_users_flag(
        local_k3, tmp_path, monkeypatch, flag):
    """mlx-lm 0.32 refuses a source's own model code (`model_file`) unless
    `load_model` gets trust_remote_code=True. `--trust-remote-code` must reach
    it; the automatic local-K3 trust is for the tokenizer only and must not."""
    import mlx_lm.utils as mlx_utils

    from turboquant_mlx.convert import convert

    seen = []
    stub = mlx_utils.load_model

    def load_model(*args, trust_remote_code=False, **kw):
        seen.append(trust_remote_code)
        return stub(*args, **kw)

    def save(*a, **k):
        raise _Stop

    monkeypatch.setattr(mlx_utils, "load_model", load_model)
    monkeypatch.setattr(mlx_utils, "save", save)
    compat.trust_local_kimi_k3(local_k3.path)
    with pytest.raises(_Stop):
        convert(str(local_k3.path), str(tmp_path / "out"),
                trust_remote_code=flag)
    assert seen == [flag]


# -- converters -------------------------------------------------------------

def test_convert_hub_repo_id_does_not_run_code(hub, tmp_path, monkeypatch):
    import mlx_lm.utils as mlx_utils

    from turboquant_mlx.convert import convert

    def save(*a, **k):  # also keeps a regression off the network (model card)
        raise AssertionError("convert got past the tokenizer")

    monkeypatch.setattr(mlx_utils, "save", save)

    _refused(convert, hub.repo, str(tmp_path / "out"))
    assert not hub.marker.exists()
