"""Experiment configuration: loading, validation and hashing.

A configuration is a JSON file with the keys described in ``SCHEMA``. ``load`` merges it over
``configs/base.json`` (if the file names a ``base``), checks every key against the schema and returns a
plain dict. The SHA-256 of the canonical JSON is the configuration's identity in the run directory name.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConfigError(ValueError):
    """Raised when a configuration is missing a key or has a value of the wrong type or range."""


# key -> (type or tuple of types, required, validator or None)
SCHEMA: Dict[str, tuple] = {
    "experiment": (str, True, None),
    "seed": (int, True, lambda v: v >= 0),
    "engine_config": (str, True, None),
    "data": (dict, True, None),
    "out_root": (str, True, None),
    "n_jobs": (int, False, lambda v: v >= 1),
    "params": (dict, False, None),
    "base": (str, False, None),
    "description": (str, False, None),
}
DATA_KEYS = ("discovery_expr", "discovery_meta")


def _merge(a: Dict, b: Dict) -> Dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(path: str) -> str:
    """Paths in configurations are relative to the project root unless absolute."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    unknown = set(cfg) - set(SCHEMA)
    if unknown:
        raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")
    for key, (typ, required, check) in SCHEMA.items():
        if key not in cfg:
            if required:
                raise ConfigError(f"missing required key '{key}'")
            continue
        if not isinstance(cfg[key], typ) or (typ is int and isinstance(cfg[key], bool)):
            raise ConfigError(f"'{key}' must be {typ.__name__}, got {type(cfg[key]).__name__}")
        if check is not None and not check(cfg[key]):
            raise ConfigError(f"'{key}' has an invalid value: {cfg[key]!r}")
    for k in DATA_KEYS:
        if k not in cfg["data"]:
            raise ConfigError(f"data.{k} is required")
    return cfg


def load(path: str, check_files: bool = True) -> Dict[str, Any]:
    with open(path) as fh:
        cfg = json.load(fh)
    if "base" in cfg:
        base_path = resolve(cfg["base"])
        with open(base_path) as fh:
            base = json.load(fh)
        base.pop("base", None)
        cfg = _merge(base, {k: v for k, v in cfg.items() if k != "base"})
    cfg = validate(cfg)
    if check_files:
        for k in DATA_KEYS + ("engine_config",):
            p = resolve(cfg["data"][k] if k in DATA_KEYS else cfg[k])
            if not os.path.exists(p):
                raise ConfigError(f"file not found for '{k}': {p}")
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def engine_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The configuration dict expected by the verified engine (``tep_analysis.load_config``)."""
    with open(resolve(cfg["engine_config"])) as fh:
        ec = json.load(fh)
    ec["discovery_expr"] = resolve(cfg["data"]["discovery_expr"])
    ec["discovery_meta"] = resolve(cfg["data"]["discovery_meta"])
    ec["seed"] = cfg["seed"]
    ec["n_jobs"] = cfg.get("n_jobs", ec.get("n_jobs", 2))
    ec["sel_n_jobs"] = cfg.get("n_jobs", ec.get("sel_n_jobs", 2))
    ec["out_dir"] = resolve(os.path.join(cfg["out_root"], "_engine"))   # engine log only
    return ec
