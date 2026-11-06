import json
import os

import pytest

from pmtrd import config as C
from pmtrd import tracking


def _cfg(tmp_path, **over):
    expr = tmp_path / "x.parquet"; meta = tmp_path / "m.csv"; eng = tmp_path / "e.json"
    for f in (expr, meta, eng):
        f.write_text("{}")
    cfg = {"experiment": "demo", "seed": 1, "engine_config": str(eng),
           "data": {"discovery_expr": str(expr), "discovery_meta": str(meta)}, "out_root": str(tmp_path / "runs")}
    cfg.update(over)
    p = tmp_path / "c.json"; p.write_text(json.dumps(cfg))
    return p


def test_valid_config_loads_and_hash_is_stable(tmp_path):
    c = C.load(str(_cfg(tmp_path)))
    assert C.config_hash(c) == C.config_hash(json.loads(json.dumps(c)))


@pytest.mark.parametrize("over,msg", [({"seed": -1}, "invalid"), ({"seed": "1"}, "must be int"),
                                      ({"bogus": 1}, "unknown"), ({"seed": True}, "must be int")])
def test_invalid_configs(tmp_path, over, msg):
    with pytest.raises(C.ConfigError, match=msg):
        C.load(str(_cfg(tmp_path, **over)))


def test_missing_file(tmp_path):
    p = _cfg(tmp_path); c = json.loads(p.read_text()); c["data"]["discovery_expr"] = str(tmp_path / "nope")
    p.write_text(json.dumps(c))
    with pytest.raises(C.ConfigError, match="file not found"):
        C.load(str(p))


def test_run_directory(tmp_path):
    c = C.load(str(_cfg(tmp_path)))
    with tracking.Run(c) as run:
        run.log("hello", value=1); run.summary({"x": 1})
    meta = json.load(open(run.path("metadata.json")))
    assert meta["status"] == "completed" and meta["seed"] == 1 and meta["config_sha256"] == C.config_hash(c)
    assert open(os.path.join(os.path.dirname(run.dir), "LATEST")).read().strip() == os.path.basename(run.dir)
    with pytest.raises(ValueError):          # resuming with a different configuration is refused
        c2 = dict(c); c2["seed"] = 2
        tracking.Run(c2, resume=os.path.basename(run.dir))


def test_failed_run_is_recorded(tmp_path):
    c = C.load(str(_cfg(tmp_path)))
    with pytest.raises(RuntimeError):
        with tracking.Run(c) as run:
            raise RuntimeError("boom")
    assert json.load(open(run.path("metadata.json")))["status"].startswith("failed")
