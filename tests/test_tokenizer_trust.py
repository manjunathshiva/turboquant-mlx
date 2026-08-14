"""Regression tests for the K3 tokenizer trust gate (compat.is_local_kimi_k3).

Contract (PR review blocker): trust_remote_code is auto-injected ONLY for a
local directory whose config.json declares model_type == "kimi_k3". Any other
local model, hub repo id, or missing/corrupt config must be left alone —
downloading tokenizer code and executing it are separate trust decisions,
and a blanket local-directory default collapses them.
"""

import json

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
