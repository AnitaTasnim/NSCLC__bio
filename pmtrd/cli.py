"""Command line: ``python -m pmtrd <experiment> --config configs/<experiment>.json [--mode run|summarise|all]
[--resume <run dir name>]``. ``python -m pmtrd list`` prints the available experiments."""
from __future__ import annotations

import argparse
import importlib
import sys

from . import config as C
from . import tracking

EXPERIMENTS = {
    "replicates": "pmtrd.experiments.replicates",
    "benchmark": "pmtrd.experiments.benchmark",
    "internal_calibration": "pmtrd.experiments.internal_calibration",
    "tuning": "pmtrd.experiments.tuning",
    "ppi_sensitivity": "pmtrd.experiments.ppi_sensitivity",
    "composite_ablation": "pmtrd.experiments.composite_ablation",
    "column_order": "pmtrd.experiments.column_order",
    "pathway_model": "pmtrd.experiments.pathway_model",
    "selector_cards": "pmtrd.experiments.selector_cards",
    "replicates_generic": "pmtrd.experiments.replicates_generic",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pmtrd", description=__doc__)
    ap.add_argument("experiment", choices=sorted(EXPERIMENTS) + ["list"])
    ap.add_argument("--config", help="experiment configuration (JSON)")
    ap.add_argument("--mode", default="all", choices=["all", "run", "summarise"])
    ap.add_argument("--resume", help="name of an existing run directory to continue")
    a = ap.parse_args(argv)
    if a.experiment == "list":
        print("\n".join(sorted(EXPERIMENTS))); return 0
    if not a.config:
        ap.error("--config is required")
    try:
        cfg = C.load(a.config)
    except C.ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr); return 2
    if cfg["experiment"] != a.experiment:
        print(f"configuration error: file is for '{cfg['experiment']}', not '{a.experiment}'", file=sys.stderr)
        return 2
    mod = importlib.import_module(EXPERIMENTS[a.experiment])
    from .engine import load
    with tracking.Run(cfg, resume=a.resume) as run:
        data = load(cfg)
        run.log("data_loaded", samples=data.X.shape[0], genes=data.X.shape[1], nsclc=int(data.y.sum()))
        mod.main(run, data, cfg.get("params", {}), mode=a.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
