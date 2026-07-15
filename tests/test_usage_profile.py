"""Tests for the learning cache (colibrì ③): a per-user hot-expert profile.

The properties that matter: counting must be per-ROUTING (so hot experts stay
hot), an immature profile must not pin, a corrupt/read-only cache must never
take the run down, and the emitted spec must be the same schema the shipped
hotlist already uses.
"""

import json
import os

import numpy as np

from turboquant_mlx.stream.usage_profile import (
    _MIN_ROUTINGS,
    UsageProfile,
    profile_path,
)


class TestRecording:
    def test_counts_every_routing_not_the_unique_set(self):
        """A prefill forward touches nearly every expert once; the hot ones are
        hot because MANY tokens pick them. Counting the set would erase that."""
        p = UsageProfile()
        # expert 3 picked 10x, expert 5 once — one forward
        p.record(0, np.array([3] * 10 + [5], dtype=np.int64))
        spec = p.pin_spec()
        assert spec["pin"][0] == [0, 3], "hottest expert must rank first"
        assert p.routings == 11

    def test_accumulates_across_layers_and_forwards(self):
        p = UsageProfile()
        p.record(0, np.array([1, 1], dtype=np.int64))
        p.record(1, np.array([2], dtype=np.int64))
        p.record(0, np.array([1], dtype=np.int64))
        assert p.routings == 4
        assert p.pin_spec()["pin"][0] == [0, 1]

    def test_empty_record_is_a_noop(self):
        p = UsageProfile()
        p.record(0, np.array([], dtype=np.int64))
        assert p.routings == 0 and p.pin_spec()["pin"] == []

    def test_grows_when_a_later_forward_sees_more_experts(self):
        p = UsageProfile()
        p.record(0, np.array([1], dtype=np.int64))
        p.record(0, np.array([99], dtype=np.int64))     # wider than before
        assert p.num_experts >= 100
        assert [0, 1] in p.pin_spec()["pin"] and [0, 99] in p.pin_spec()["pin"]


class TestMaturity:
    def test_immature_profile_does_not_pin(self):
        p = UsageProfile()
        p.record(0, np.arange(100) % 8)
        assert not p.is_mature()

    def test_mature_after_enough_routings(self):
        p = UsageProfile()
        p.record(0, np.zeros(_MIN_ROUTINGS, dtype=np.int64))
        assert p.is_mature()


class TestPersistence:
    def test_round_trip(self, tmp_path):
        p = UsageProfile()
        p.record(0, np.array([1, 1, 2], dtype=np.int64))
        f = str(tmp_path / "u.json")
        assert p.save(f)
        q = UsageProfile.load(f)
        assert q.routings == 3
        assert q.pin_spec()["pin"][0] == [0, 1]

    def test_merge_decays_history(self, tmp_path):
        f = str(tmp_path / "u.json")
        old = UsageProfile()
        old.record(0, np.array([7] * 100, dtype=np.int64))
        old.save(f)
        new = UsageProfile()
        new.record(0, np.array([9] * 80, dtype=np.int64))
        new.update_on_disk(f, decay=0.5)
        got = UsageProfile.load(f)
        # 7: 100*0.5 = 50 ; 9: 80  -> the newer workload takes over
        assert got.pin_spec()["pin"][0] == [0, 9]
        assert got.routings == 50 + 80

    def test_update_on_disk_is_idempotent(self, tmp_path):
        """An explicit flush + the atexit hook both fire in one process; the
        second must not double-weight this session against the history."""
        f = str(tmp_path / "u.json")
        p = UsageProfile()
        p.record(0, np.array([1] * 10, dtype=np.int64))
        assert p.update_on_disk(f) is True
        assert p.update_on_disk(f) is False          # no second merge
        assert UsageProfile.load(f).routings == 10

    def test_corrupt_profile_degrades_to_empty(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not json at all")
        assert UsageProfile.load(str(f)).routings == 0

    def test_valid_json_that_is_not_an_object_does_not_crash(self, tmp_path):
        """A truncated/edited cache file can be valid JSON but not a dict.
        d.get() then raises AttributeError, which the OSError/ValueError guard
        does not catch — crashing the loader at startup over an *optimisation*
        file. (Review finding, PR #55.)"""
        for payload in ("[]", "null", "42", '"a string"', "[1, 2, 3]"):
            f = tmp_path / f"x{abs(hash(payload))}.json"
            f.write_text(payload)
            assert UsageProfile.load(str(f)).routings == 0

    def test_bare_relative_filename_actually_persists(self, tmp_path, monkeypatch):
        """dirname("profile.json") is "" -> makedirs("") raised -> the OSError
        guard turned it into a SILENT never-saved. A bare --usage-file is a
        reasonable thing to pass. (Review finding, PR #55.)"""
        monkeypatch.chdir(tmp_path)
        p = UsageProfile()
        p.record(0, np.array([1, 1], dtype=np.int64))
        assert p.save("profile.json") is True
        assert (tmp_path / "profile.json").exists()
        assert UsageProfile.load("profile.json").routings == 2

    def test_failed_write_leaves_no_orphan_tmp(self, tmp_path, monkeypatch):
        """A mid-write failure must not litter the cache dir with .tmp files."""
        import json as _json
        p = UsageProfile()
        p.record(0, np.array([1], dtype=np.int64))
        monkeypatch.setattr(_json, "dump",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
        assert p.save(str(tmp_path / "u.json")) is False
        assert not list(tmp_path.glob("*.tmp")), "orphaned temp file left behind"

    def test_wrong_version_is_ignored(self, tmp_path):
        f = tmp_path / "v.json"
        f.write_text(json.dumps({"version": 999, "num_experts": 4,
                                 "counts": {"0": [1, 2, 3, 4]}}))
        assert UsageProfile.load(str(f)).routings == 0

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert UsageProfile.load(str(tmp_path / "nope.json")).routings == 0

    def test_unwritable_dir_never_raises(self):
        """A profile is an optimisation — a read-only cache dir must not take
        the generation down with it."""
        p = UsageProfile()
        p.record(0, np.array([1], dtype=np.int64))
        assert p.save("/proc/nonexistent/u.json") is False   # returns, no raise


class TestPaths:
    def test_profile_path_is_stable_and_outside_the_model_dir(self, tmp_path):
        m = str(tmp_path / "some-model")
        a, b = profile_path(m), profile_path(m)
        assert a == b                       # deterministic
        assert not a.startswith(m), "must not write into the model directory"
        assert "some-model" in os.path.basename(a)   # readable for humans

    def test_different_models_do_not_collide(self, tmp_path):
        assert profile_path(str(tmp_path / "a")) != profile_path(str(tmp_path / "b"))

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TURBOQUANT_USAGE_DIR", str(tmp_path / "custom"))
        assert str(tmp_path / "custom") in profile_path("/models/x")


class TestSchema:
    def test_pin_spec_matches_the_shipped_hotlist_schema(self, tmp_path):
        """It must feed the existing loader path unchanged — and be uploadable
        as a hot_experts.json if it turns out to be a good profile."""
        from turboquant_mlx.stream.loader import _load_pin_spec
        p = UsageProfile()
        p.record(0, np.array([1, 1, 2], dtype=np.int64))
        p.record(3, np.array([5], dtype=np.int64))
        f = tmp_path / "hot_experts.json"
        f.write_text(json.dumps(p.pin_spec()))
        pin_layers, pin_order = _load_pin_spec(str(f))   # the real parser
        assert pin_order[0] == (0, 1)
        assert 1 in pin_layers[0] and 5 in pin_layers[3]

    def test_limit_keeps_the_hottest(self):
        p = UsageProfile()
        p.record(0, np.array([1] * 5 + [2] * 3 + [3], dtype=np.int64))
        assert p.pin_spec(limit=2)["pin"] == [[0, 1], [0, 2]]
