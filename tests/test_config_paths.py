"""Config hashing, inheritance, and path resolution."""

from __future__ import annotations

import pytest
import yaml

from mutstab import paths
from mutstab.config import Config, load_config


def write_config(directory, name, values):
    path = directory / f"{name}.yaml"
    with path.open("w") as fh:
        yaml.safe_dump(values, fh)
    return path


def test_hash_ignores_formatting_but_not_values(tmp_path):
    a = write_config(tmp_path, "a", {"name": "x", "model": {"kind": "ridge", "alpha": 1.0}})
    b = write_config(tmp_path, "b", {"model": {"alpha": 1.0, "kind": "ridge"}, "name": "x"})
    c = write_config(tmp_path, "c", {"name": "x", "model": {"kind": "ridge", "alpha": 2.0}})

    assert load_config(a).hash == load_config(b).hash
    assert load_config(a).hash != load_config(c).hash


def test_run_id_carries_the_hash(tmp_path):
    cfg = load_config(write_config(tmp_path, "baseline", {"name": "baseline", "x": 1}))
    assert cfg.run_id.startswith("baseline-")
    assert cfg.run_id.endswith(cfg.hash[:8])


def test_extends_merges_over_the_parent(tmp_path):
    write_config(tmp_path, "parent", {"name": "parent", "model": {"kind": "ridge", "alpha": 1.0}})
    child_path = write_config(
        tmp_path,
        "child",
        {"extends": str(tmp_path / "parent.yaml"), "name": "child", "model": {"alpha": 99.0}},
    )
    child = load_config(child_path)
    assert child["model"]["alpha"] == 99.0
    assert child["model"]["kind"] == "ridge"  # inherited
    assert child.name == "child"


def test_dotted_access_and_defaults():
    cfg = Config({"data": {"filters": {"abs_dtm_max": 30.0}}})
    assert cfg.get_path("data.filters.abs_dtm_max") == 30.0
    assert cfg.get_path("data.filters.missing", "fallback") == "fallback"
    assert cfg.get_path("absent.deeply.nested") is None
    assert cfg.data.filters.abs_dtm_max == 30.0


def test_helpful_error_on_unknown_key():
    cfg = Config({"present": 1})
    with pytest.raises(AttributeError, match="present"):
        _ = cfg.nope


def test_missing_config_names_what_it_tried():
    with pytest.raises(FileNotFoundError, match="no config at"):
        load_config("definitely-not-a-real-config")


def test_data_root_follows_the_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MUTSTAB_DATA_ROOT", str(tmp_path))
    assert paths.data_root() == tmp_path.resolve()
    assert paths.raw_dir() == tmp_path.resolve() / "raw"
    assert paths.cache_dir() == tmp_path.resolve() / "cache"


def test_stage_copies_once(tmp_path, monkeypatch):
    data_root = tmp_path / "drive"
    scratch = tmp_path / "local"
    monkeypatch.setenv("MUTSTAB_DATA_ROOT", str(data_root))
    monkeypatch.setenv("MUTSTAB_SCRATCH", str(scratch))

    source = paths.processed_dir(create=True) / "data.parquet"
    source.write_bytes(b"first")

    staged = paths.stage(source)
    assert staged.read_bytes() == b"first"
    assert scratch in staged.parents

    # A changed source is not re-copied unless forced — staging is a cache.
    source.write_bytes(b"second")
    assert paths.stage(source).read_bytes() == b"first"
    assert paths.stage(source, force=True).read_bytes() == b"second"


def test_stage_reports_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MUTSTAB_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        paths.stage(tmp_path / "nope.parquet")
