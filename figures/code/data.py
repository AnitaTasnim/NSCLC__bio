"""Paths and loaders for the figure inputs. Every plotted number is read from a saved result of the analysis
code (results/tables, the PMTRD run directories, or the macro file the manuscript reads)."""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

# Two layouts are supported.
#  * Repository (TEP-PMTRD on GitHub): set PMTRD_ROOT to the repository root (figures/make_figures.sh does).
#  * Submission package: the defaults below (code/figures_v7 inside the package; PMTRD_V2 = experiment project).
REPO = os.environ.get("PMTRD_ROOT")
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO:
    REPO = os.path.abspath(REPO)
    TAB = os.path.join(REPO, "original", "results", "tables")
    INPUTS = os.path.join(REPO, "figures", "inputs")
    MACROS = os.path.join(INPUTS, "results_macros.tex")
    PANEL_CSV = os.path.join(INPUTS, "panel_development_run.csv")
    DE_IMAGE = os.path.join(INPUTS, "fig05_de_landscape.png")
    V2 = REPO
    OUT = os.environ.get("TEP_FIG_OUT", os.path.join(REPO, "figures", "build"))
    MAN = None
else:
    TAB = os.path.join(PKG, "results", "tables")
    MAN = os.path.join(PKG, "manuscript")
    MACROS = os.path.join(MAN, "results", "results_macros.tex")
    PANEL_CSV = None
    DE_IMAGE = os.path.join(MAN, "figures", "fig05_de_landscape.png")
    V2 = os.environ.get("PMTRD_V2", os.path.abspath(os.path.join(PKG, "..", "v2", "TEP-PMTRD-v2")))
    OUT = os.environ.get("TEP_FIG_OUT", os.path.join(MAN, "figures_v7"))


def tab(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(TAB, name if name.endswith(".csv") else name + ".csv"))


def macros() -> dict:
    """\\renewcommand{\\Name}{value} pairs from results/results_macros.tex (the values the paper prints)."""
    s = open(MACROS).read()
    out = {}
    for m in re.finditer(r"\\(?:re)?newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", s):
        out[m.group(1)] = m.group(2)
    return out


def num(v) -> float:
    v = str(v).replace("$-$", "-").replace("{,}", "").replace("\\%", "").replace("$", "").strip()
    m = re.match(r"^[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?", v)
    return float(m.group(0)) if m else np.nan


def latest(exp: str, sub: str | None = None) -> str:
    base = os.path.join(V2, "runs", sub, exp) if sub else os.path.join(V2, "runs", exp)
    return os.path.join(base, open(os.path.join(base, "LATEST")).read().strip())


def run_results(exp: str, sub: str | None = None) -> str:
    return os.path.join(latest(exp, sub), "results")


def export_json() -> dict:
    return json.load(open(os.path.join(V2, "runs", "export", "macros_v2.json")))
