"""Versioned run directories, structured logging and run metadata.

Every experiment run writes to ``<out_root>/<experiment>/<UTC timestamp>_<config hash[:8]>/``:

    config.json     the resolved configuration
    metadata.json   package versions, platform, seed, SHA-256 of the input files and of the code,
                    start/end time and status
    log.jsonl       one JSON object per log event
    cache/          per-unit intermediate results (lets an interrupted run resume)
    results/        tables (CSV) and figures
    summary.json    the headline numbers of the run

``<out_root>/<experiment>/LATEST`` holds the name of the most recent completed run.
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import platform
import sys
import time
from importlib import metadata as _md
from typing import Any, Dict, Optional

from . import __version__
from .config import ROOT, config_hash, resolve

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "statsmodels", "joblib", "pyarrow")


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def code_hash() -> str:
    """SHA-256 over every source file of the package and the engine, in sorted order."""
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(ROOT, "pmtrd", "**", "*.py"), recursive=True) +
                    glob.glob(os.path.join(ROOT, "engine", "*.py"))):
        h.update(os.path.relpath(p, ROOT).encode()); h.update(open(p, "rb").read())
    return h.hexdigest()


def _versions() -> Dict[str, str]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = _md.version(p)
        except _md.PackageNotFoundError:
            out[p] = "not installed"
    return out


class Run:
    """A tracked experiment run. Use as a context manager so that the status is always recorded."""

    def __init__(self, cfg: Dict[str, Any], resume: Optional[str] = None):
        self.cfg = cfg
        self.hash = config_hash(cfg)
        base = os.path.join(resolve(cfg["out_root"]), cfg["experiment"])
        if resume:
            self.dir = resume if os.path.isabs(resume) else os.path.join(base, resume)
            if not os.path.isdir(self.dir):
                raise FileNotFoundError(self.dir)
            old = json.load(open(os.path.join(self.dir, "config.json")))
            if config_hash(old) != self.hash:
                raise ValueError("cannot resume: the configuration differs from the one of that run")
        else:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.dir = os.path.join(base, f"{stamp}_{self.hash[:8]}")
        for sub in ("cache", "results"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self._log = open(os.path.join(self.dir, "log.jsonl"), "a")
        self.t0 = time.time()
        self.code_sha = code_hash()          # the code as it was when the run started

    # ------------------------------------------------------------------ paths
    def path(self, *parts: str) -> str:
        return os.path.join(self.dir, *parts)

    # ------------------------------------------------------------------ logging
    def log(self, event: str, **fields: Any) -> None:
        rec = {"t": round(time.time() - self.t0, 2), "event": event, **fields}
        self._log.write(json.dumps(rec, default=str) + "\n"); self._log.flush()
        msg = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[{rec['t']:>8.1f}s] {event} {msg}", flush=True)

    # ------------------------------------------------------------------ metadata
    def _write_meta(self, status: str, extra: Optional[Dict] = None) -> None:
        inputs = {k: resolve(v) for k, v in self.cfg["data"].items()}
        meta = {
            "pmtrd_version": __version__, "status": status, "config_sha256": self.hash,
            "code_sha256": self.code_sha, "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": _versions(), "seed": self.cfg["seed"],
            "inputs": {k: {"path": p, "sha256": _sha256(p)} for k, p in inputs.items() if os.path.isfile(p)},
            "started_utc": _dt.datetime.fromtimestamp(self.t0, _dt.timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - self.t0, 1),
        }
        if extra:
            meta.update(extra)
        json.dump(meta, open(self.path("metadata.json"), "w"), indent=1)

    def __enter__(self) -> "Run":
        json.dump(self.cfg, open(self.path("config.json"), "w"), indent=1, sort_keys=True)
        self._write_meta("running")
        self.log("run_started", dir=self.dir)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "completed" if exc_type is None else f"failed: {exc_type.__name__}: {exc}"
        self._write_meta(status)
        self.log("run_finished", status=status)
        self._log.close()
        if exc_type is None:
            with open(os.path.join(os.path.dirname(self.dir), "LATEST"), "w") as fh:
                fh.write(os.path.basename(self.dir) + "\n")
        return False

    def summary(self, values: Dict[str, Any]) -> None:
        json.dump(values, open(self.path("summary.json"), "w"), indent=1, default=float)
