"""Experiment configs: one YAML per experiment, hashed so runs stay traceable.

The hash is over the canonical JSON form of the config, so key order and
formatting in the YAML do not change it, but any value change does. Output
filenames carry the hash, which is what lets you look at a metrics file six
weeks later and know exactly which settings produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from mutstab import paths

__all__ = ["Config", "load_config"]


class Config(Mapping[str, Any]):
    """Read-only nested config with attribute access and a stable hash."""

    def __init__(self, values: Mapping[str, Any], source: Path | None = None):
        self._values = dict(values)
        self._source = Path(source) if source else None

    # -- mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._values[key]
        return Config(value) if isinstance(value, Mapping) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(
                f"config has no key {key!r}; present keys: {sorted(self._values)}"
            ) from exc

    def __repr__(self) -> str:
        return f"Config({self.hash[:8]}, keys={sorted(self._values)})"

    # -- extras -----------------------------------------------------------
    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted path: cfg.get_path('model.alpha')."""
        node: Any = self._values
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._values, sort_keys=True, default=str))

    @property
    def source(self) -> Path | None:
        return self._source

    @property
    def hash(self) -> str:
        """sha256 of the canonical form. Independent of YAML formatting."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def name(self) -> str:
        explicit = self._values.get("name")
        if explicit:
            return str(explicit)
        return self._source.stem if self._source else "unnamed"

    @property
    def run_id(self) -> str:
        """Human-readable name plus enough hash to be unique."""
        return f"{self.name}-{self.hash[:8]}"


def load_config(path: str | Path) -> Config:
    """Load a YAML config.

    A bare name resolves against `configs/`, so both of these work::

        load_config("baseline")
        load_config("configs/baseline.yaml")
    """
    candidate = Path(path)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")
    if not candidate.exists():
        in_configs = paths.repo_root() / "configs" / candidate.name
        if in_configs.exists():
            candidate = in_configs
        else:
            raise FileNotFoundError(
                f"no config at {path!r}; looked at {candidate} and {in_configs}"
            )

    with candidate.open() as fh:
        values = yaml.safe_load(fh) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{candidate} must contain a YAML mapping at the top level")

    # `extends` keeps variants honest: a sweep config says only what differs
    # from the baseline, instead of being a near-duplicate that drifts.
    parent_name = values.pop("extends", None)
    if parent_name:
        parent = load_config(parent_name)
        values = _deep_merge(parent.to_dict(), values)

    return Config(values, source=candidate)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out
