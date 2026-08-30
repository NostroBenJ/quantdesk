"""Config loading. No strategy parameter is ever a literal in the code.

JSON is handled with the stdlib. YAML is imported lazily inside the loader so the
core package stays importable with zero third-party dependencies - the same
pattern `black_scholes.py::plot_greeks` already uses for matplotlib.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

_MISSING = object()


class ConfigError(KeyError):
    """Raised when a required config key is absent or the file is malformed."""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`, returning a new dict."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable nested config with dotted-path access.

    `get` returns a default; `require` raises. Prefer `require` for anything that
    changes trading behaviour - a silently-defaulted risk limit is worse than a
    crash on startup.
    """

    data: Mapping[str, Any] = field(default_factory=dict)
    source: str = "<literal>"

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # lazy: keeps the core dependency-free
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ConfigError(
                    "PyYAML is required to read .yaml configs; "
                    "install it or use .json"
                ) from exc
            loaded = yaml.safe_load(text) or {}
        elif p.suffix.lower() == ".json":
            loaded = json.loads(text)
        else:
            raise ConfigError(f"unsupported config extension: {p.suffix}")
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"config root must be a mapping, got {type(loaded).__name__}")
        return cls(data=loaded, source=str(p))

    @classmethod
    def from_files(cls, *paths: str | Path) -> "Config":
        """Load and deep-merge several files, later files winning.

        The intended use is `Config.from_files(defaults, account_overrides)` so a
        prop-firm account file only has to state what differs.
        """
        merged: dict[str, Any] = {}
        names: list[str] = []
        for path in paths:
            cfg = cls.from_file(path)
            merged = _deep_merge(merged, cfg.data)
            names.append(cfg.source)
        return cls(data=merged, source=" + ".join(names))

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"required config key {dotted!r} missing from {self.source}")
        return value

    def section(self, dotted: str) -> "Config":
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"config section {dotted!r} missing from {self.source}")
        if not isinstance(value, Mapping):
            raise ConfigError(f"config key {dotted!r} is not a section")
        return Config(data=value, source=f"{self.source}:{dotted}")

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new Config with `overrides` deep-merged on top.

        Used by the walk-forward optimiser to sweep parameters without ever
        mutating the shared config object.
        """
        return Config(data=_deep_merge(self.data, overrides), source=f"{self.source}+overrides")

    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, _MISSING) is not _MISSING

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)
