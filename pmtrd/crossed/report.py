"""Summaries, pre-registered criterion check and figures for the E1-E4 simulation, read only from the saved
unit files of a run directory.

    python -m pmtrd.crossed.report --run runs/crossed_sim/v1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

STATS = ["S", "Q", "SQ", "E", "gamma", "Q+gamma", "eq3"]
MODES = ["pigeonhole", "draws", "patients"]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(run: str) -> tuple:
    rows_t, rows_r = [], []
    for f in glob.glob(os.path.join(run, "units", "*.json")):
        u = json.load(open(f))
        key = {k: u[k] for k in ("config", "selector", "clf", "T")}
        key.update({"n": u["data"]["n"], "p": u["data"]["p"], "rho": u["data"]["rho"], "auc": u["data"]["auc"],
                    "f": u["data"]["f"]})
        if u["kind"] == "truth":
            for s in STATS:
                val = u["eq3"] if s == "eq3" else u["share"][s]
                rows_t.append({**key, "stat": s, "truth": val, "truth_mc_se": u["mc_se"][s],
                               "seconds": u["seconds"], "cpu_seconds": u["cpu_seconds"], "n_fits": u["n_fits"]})
        else:
            for s in STATS:
                val = u["eq3"] if s == "eq3" else u["share"][s]
                r = {**key, "R": u["R"], "rep": u["rep"], "stat": s, "est": val, "seconds": u["seconds"],
                     "cpu_seconds": u["cpu_seconds"], "n_fits": u["n_fits"]}
                for m in MODES:
                    r[f"lo_{m}"], r[f"hi_{m}"] = u[f"ci_{m}"][s]
                rows_r.append(r)
    return pd.DataFrame(rows_t), pd.DataFrame(rows_r)


def summarise(T: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    keys = ["config", "selector", "clf", "T", "stat"]
    df = E.merge(T[keys + ["truth", "truth_mc_se"]], on=keys, how="inner")
    # Eq. 3 is also compared with the truth of Q + gamma (the quantity it is claimed to approximate)
    qg = T[T.stat == "Q+gamma"][keys[:-1] + ["truth"]].rename(columns={"truth": "truth_Qgamma"})
    df = df.merge(qg, on=keys[:-1], how="left")
    df["err"] = df.est - df.truth
    for m in MODES:
        df[f"cov_{m}"] = (df[f"lo_{m}"] <= df.truth) & (df.truth <= df[f"hi_{m}"])
        df[f"width_{m}"] = df[f"hi_{m}"] - df[f"lo_{m}"]
    out = []
    g = ["config", "selector", "clf", "R", "stat"]
    for k, d in df.groupby(g):
        n = len(d)
        nT = d["T"].nunique()
        mc = d.groupby("T").truth_mc_se.first()
        r = dict(zip(g, k))
        r.update(n_reps=n, n_T=nT, n=d.n.iloc[0], p=d.p.iloc[0], rho=d.rho.iloc[0], auc=d.auc.iloc[0],
                 f=d.f.iloc[0], truth_mean=d.groupby("T").truth.first().mean(),
                 bias=d.err.mean(), rmse=float(np.sqrt((d.err ** 2).mean())), est_sd=d.est.std(ddof=1),
                 bias_se=float(np.sqrt(d.err.var(ddof=1) / n + (mc ** 2).mean() / nT)),
                 frac_negative=float((d.est < 0).mean()), truth_mc_se=float(mc.mean()))
        for m in MODES:
            kk = int(d[f"cov_{m}"].sum())
            lo, hi = wilson(kk, n)
            r.update({f"coverage_{m}": kk / n, f"coverage_{m}_lo": lo, f"coverage_{m}_hi": hi,
                      f"width_{m}": d[f"width_{m}"].mean()})
        if k[-1] == "eq3":
            r["eq3_minus_Qgamma_truth"] = float((d.truth - d.truth_Qgamma).mean())
            r["eq3_err_vs_Qgamma"] = float((d.est - d.truth_Qgamma).mean())
        out.append(r)
    return pd.DataFrame(out)


def criterion(S: pd.DataFrame, crit: Dict) -> tuple:
    """Pre-registered check (docs/crossed_simulation.md, sections 5-6). Returns the cell table and the verdict."""
    c = S[(S.R == crit["R"]) & (S.n >= crit["n_min"]) & (S.stat.isin(crit["statistics"]))].copy()
    c = c[~((c.clf == "logistic") & (c.stat == "E"))]          # identically 0: reported, not counted
    m = crit["ci_method"]
    c["bias_ok"] = c.bias.abs() <= crit["abs_bias_max"]
    c["cov_ok"] = (c[f"coverage_{m}"] >= crit["coverage_low"]) & (c[f"coverage_{m}"] <= crit["coverage_high"])
    c["pass"] = c.bias_ok & c.cov_ok
    c["mc_uncertain"] = (((c.bias.abs() - crit["abs_bias_max"]).abs() < 1.96 * c.bias_se)
                         | ((c[f"coverage_{m}_lo"] < crit["coverage_low"]) & (c[f"coverage_{m}_hi"] > crit["coverage_low"]))
                         | ((c[f"coverage_{m}_lo"] < crit["coverage_high"]) & (c[f"coverage_{m}_hi"] > crit["coverage_high"])))
    key_ok = c[c.stat.isin(["Q+gamma", "S"])]["pass"].all()
    verdict = "GO" if c["pass"].all() else ("CONDITIONAL GO" if key_ok else "NO-GO")
    return c, verdict


def runtime(T: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    t = T.drop_duplicates(["config", "selector", "clf", "T"]).assign(kind="truth", R=200)
    e = E.drop_duplicates(["config", "selector", "clf", "T", "R", "rep"]).assign(kind="replicate")
    a = pd.concat([t, e])
    return (a.groupby(["clf", "kind", "R"]).agg(units=("seconds", "size"), mean_seconds=("seconds", "mean"),
                                                 mean_cpu_seconds=("cpu_seconds", "mean"), mean_fits=("n_fits", "mean"),
                                                 total_cpu_hours=("cpu_seconds", lambda x: x.sum() / 3600))
            .reset_index())


# ----------------------------------------------------------------------------------------------- figures
# Colours: reference categorical slots (fixed order) and a blue <-> red diverging pair with a grey midpoint.
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
COL = {"pigeonhole": SLOTS[0], "draws": SLOTS[2], "patients": SLOTS[1]}
MODE_LABEL = {"pigeonhole": "pigeonhole (pre-registered)", "draws": "draws only", "patients": "patients only (as in the paper)"}
SEL = {"mw": "MW", "l1": "L1", "random": "Random"}
STAT_LABEL = {"S": "Π_S", "Q": "Π_Q", "SQ": "Π_SQ", "E": "Π_E", "gamma": "Π_γ", "Q+gamma": "Π_Q+Π_γ", "eq3": "Eq. 3"}
CFG_ORDER = ["base", "n=100", "n=600", "p=10000", "rho=0.0", "rho=0.9", "auc=0.75", "auc=0.97",
             "f=0.5", "f=0.632", "f=0.9"]
NEUTRAL, BLUE_LO, BLUE_HI, RED_LO, RED_HI = "#f0efec", "#cde2fb", "#1c5cab", "#f6cfce", "#a8302f"


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "savefig.facecolor": SURF, "figure.dpi": 200,
                         "savefig.bbox": "tight", "axes.titlesize": 8, "axes.titleweight": "bold"})
    return plt


def _mix(c1: str, c2: str, t: float):
    import matplotlib.colors as mc
    a, b = np.array(mc.to_rgb(c1)), np.array(mc.to_rgb(c2))
    return tuple(a + (b - a) * float(np.clip(t, 0, 1)))


def _diverge(v: float, lo: float, hi: float, lo_end: float, hi_end: float):
    """Grey inside [lo, hi]; blue arm above hi (to hi_end), red arm below lo (to lo_end)."""
    if not np.isfinite(v):
        return (1, 1, 1)
    if lo <= v <= hi:
        return _mix(NEUTRAL, NEUTRAL, 0)
    if v > hi:
        return _mix(BLUE_LO, BLUE_HI, (v - hi) / (hi_end - hi))
    return _mix(RED_LO, RED_HI, (lo - v) / (lo - lo_end))


def _rows(S: pd.DataFrame) -> List[tuple]:
    keys = S[["clf", "selector", "config"]].drop_duplicates()
    order = {c: i for i, c in enumerate(CFG_ORDER)}
    keys = keys.assign(c0=keys.clf.map({"logistic": 0, "xgboost": 1}), s0=keys.selector.map({"mw": 0, "l1": 1, "random": 2}),
                       k0=keys.config.map(lambda c: order.get(c, 99)))
    return [tuple(r) for r in keys.sort_values(["c0", "s0", "k0"])[["clf", "selector", "config"]].values]


def _heat(plt, S, value, fmt, colour, title, note, crit_ok, fname, out, stats):
    rows = _rows(S)
    fig, ax = plt.subplots(figsize=(5.6, 0.16 * len(rows) + 1.1))
    for i, (c, s, cf) in enumerate(rows):
        for j, st in enumerate(stats):
            d = S[(S.clf == c) & (S.selector == s) & (S.config == cf) & (S.stat == st)]
            if not len(d) or (c == "logistic" and st == "E"):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=SURF, ec=GRID, lw=0.3))
                ax.text(j + .5, i + .5, "—" if (c == "logistic" and st == "E") else "", ha="center", va="center",
                        fontsize=5.5, color=MUTED)
                continue
            r = d.iloc[0]
            v = r[value]
            fc = colour(v)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=fc, ec=SURF, lw=1.0))
            dark = np.mean(fc) < 0.62
            counted = r.n >= 300 and st != "eq3"
            ax.text(j + .5, i + .5, fmt(v), ha="center", va="center", fontsize=5.5,
                    color="#ffffff" if dark else (INK if counted else INK2),
                    fontweight="bold" if counted and not crit_ok(r) else "normal",
                    fontstyle="normal" if counted else "italic")
    ax.set_xlim(0, len(stats))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(stats)) + .5)
    ax.set_xticklabels([STAT_LABEL[s] for s in stats])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)) + .5)
    ax.set_yticklabels([f"{c}  {SEL[s]}  {cf}" for c, s, cf in rows], fontsize=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    prev = None
    for i, (c, s, _) in enumerate(rows):
        if prev is not None and (c, s) != prev:
            ax.axhline(i, color=INK2, lw=0.6)
        prev = (c, s)
    ax.set_title(title, loc="left", pad=22)
    import textwrap
    ax.text(-0.0, len(rows) + 0.5, textwrap.fill(note, 105), fontsize=6, color=INK2, ha="left", va="top",
            clip_on=False)
    f = os.path.join(out, fname)
    fig.savefig(f)
    plt.close(fig)
    return f


def figures(S: pd.DataFrame, rt: pd.DataFrame, out: str, crit: Dict) -> List[str]:
    plt = _setup()
    files = []
    stats = ["S", "Q", "SQ", "E", "gamma", "Q+gamma"]
    r10 = S[S.R == 10]
    lo, hi, bmax = crit["coverage_low"], crit["coverage_high"], crit["abs_bias_max"]

    # 1. bias at R = 10: annotated heat map, grey within the +/-0.02 criterion
    files.append(_heat(
        plt, r10, "bias", lambda v: f"{v:+.3f}",
        lambda v: _diverge(v, -bmax, bmax, -2 * bmax, 2 * bmax),
        "Bias of the share estimators, R = 10",
        f"Cell = mean(estimate − truth) over replicates. Grey: |bias| ≤ {bmax} (pre-registered); blue: above +{bmax}; "
        f"red: below −{bmax}. Bold: fails the criterion. Italic: not counted (n = 100, or Eq. 3). —: Π_E is identically 0 "
        "for logistic regression.", lambda r: abs(r.bias) <= bmax, "fig_bias_R10.png", out, stats + ["eq3"]))

    # 2. pigeonhole coverage at R = 10: annotated heat map, grey inside the pre-registered band
    files.append(_heat(
        plt, r10, "coverage_pigeonhole", lambda v: f"{v:.3f}",
        lambda v: _diverge(v, lo, hi, 0.5, 1.0),
        "Coverage of the pre-registered pigeonhole 95% interval, R = 10",
        f"Grey: inside {lo}–{hi}; blue: over-coverage (interval too wide); red: under-coverage. Bold: fails the "
        "criterion. Italic: not counted (n = 100, or Eq. 3, which is reported but not part of the criterion).",
        lambda r: lo <= r.coverage_pigeonhole <= hi, "fig_coverage_R10.png", out, stats + ["eq3"]))

    # 3. the three interval methods at the base point: dot plot, one panel per statistic
    b10 = r10[(r10.config == "base")]
    combos = [(c, s) for c in ("logistic", "xgboost") for s in ("mw", "l1", "random")
              if len(b10[(b10.clf == c) & (b10.selector == s)])]
    pan = stats + ["eq3"]
    fig, axes = plt.subplots(1, len(pan), figsize=(7.2, 0.28 * len(combos) + 1.0), sharey=True)
    for ax, st in zip(axes, pan):
        ax.axvspan(lo, hi, color="#e9e8e3", zorder=0, lw=0)
        ax.axvline(0.95, color="#c3c2b7", lw=0.6, zorder=0)
        for i, (c, s) in enumerate(combos):
            d = b10[(b10.clf == c) & (b10.selector == s) & (b10.stat == st)]
            if not len(d) or (c == "logistic" and st == "E"):
                continue
            r = d.iloc[0]
            for k, m in enumerate(MODES):
                ax.plot([r[f"coverage_{m}_lo"], r[f"coverage_{m}_hi"]], [i + (k - 1) * .22] * 2, color=COL[m], lw=0.8)
                ax.plot(r[f"coverage_{m}"], i + (k - 1) * .22, "o", ms=3.2, color=COL[m], mec=SURF, mew=0.6)
        ax.set_title(STAT_LABEL[st], fontsize=7.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, .5, 1])
        ax.set_xticklabels(["0", ".5", "1"])
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(combos)))
    axes[0].set_yticklabels([f"{c} {SEL[s]}" for c, s in combos])
    axes[0].set_ylim(len(combos) - .5, -.5)
    handles = [plt.Line2D([], [], marker="o", ls="-", lw=.8, ms=3.2, color=COL[m], label=MODE_LABEL[m]) for m in MODES]
    fig.legend(handles=handles, ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.06), fontsize=6.5)
    fig.suptitle("Coverage of three 95% interval methods, base configuration, R = 10", x=0.0, ha="left", y=1.13,
                 fontsize=8, fontweight="bold")
    fig.text(0.0, -0.04, f"Grey band: pre-registered {lo}–{hi}. Lines: Wilson 95% intervals for the coverage.",
             fontsize=6, color=INK2)
    f = os.path.join(out, "fig_coverage_methods_base.png")
    fig.savefig(f)
    plt.close(fig)
    files.append(f)

    # 4. RMSE and pigeonhole coverage against R at the base point
    b = S[(S.config == "base") & S.stat.isin(stats)]
    fig, axes = plt.subplots(2, len(combos), figsize=(7.2, 3.6), squeeze=False, sharey="row")
    for j, (c, s) in enumerate(combos):
        d = b[(b.clf == c) & (b.selector == s)]
        for k, st in enumerate(stats):
            dd = d[d.stat == st].sort_values("R")
            if c == "logistic" and st == "E":
                continue
            axes[0, j].plot(dd.R, dd.rmse, "o-", ms=2.8, lw=1.2, color=SLOTS[k], label=STAT_LABEL[st])
            axes[1, j].plot(dd.R, dd.coverage_pigeonhole, "o-", ms=2.8, lw=1.2, color=SLOTS[k])
        axes[0, j].set_yscale("log")
        axes[1, j].axhspan(lo, hi, color="#e9e8e3", zorder=0, lw=0)
        axes[1, j].set_ylim(0.6, 1.02)
        axes[0, j].set_title(f"{c} {SEL[s]}", fontsize=7)
        axes[1, j].set_xlabel("R (draws)")
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.set_xticks([5, 10, 20, 50])
            ax.set_xticklabels(["5", "10", "20", "50"])
            ax.minorticks_off()
            ax.grid(axis="y", color=GRID, lw=0.4)
    axes[0, 0].set_ylabel("RMSE (share)")
    axes[1, 0].set_ylabel("pigeonhole coverage")
    h, l = axes[0, 0].get_legend_handles_labels()
    if "Π_E" not in l:
        for ax in axes[0]:
            hh, ll = ax.get_legend_handles_labels()
            if "Π_E" in ll:
                h, l = hh, ll
                break
    fig.legend(h, l, ncol=6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.04), fontsize=6.5)
    fig.suptitle("Base configuration: RMSE and coverage against the number of draws", x=0.0, ha="left", y=1.09,
                 fontsize=8, fontweight="bold")
    f = os.path.join(out, "fig_R_base.png")
    fig.savefig(f)
    plt.close(fig)
    files.append(f)

    # 5. Eq. 3 against its own truth and against Pi_Q + Pi_gamma: dot plot
    e = r10[r10.stat == "eq3"].copy()
    if len(e):
        rows = _rows(e)
        fig, ax = plt.subplots(figsize=(4.8, 0.15 * len(rows) + 1.0))
        ax.axvspan(-bmax, bmax, color="#e9e8e3", zorder=0, lw=0)
        ax.axvline(0, color="#c3c2b7", lw=0.6)
        for i, (c, s, cf) in enumerate(rows):
            r = e[(e.clf == c) & (e.selector == s) & (e.config == cf)].iloc[0]
            ax.plot(r.eq3_minus_Qgamma_truth, i, "o", ms=3.4, color=SLOTS[6], mec=SURF, mew=.6)
            ax.plot(r.bias, i, "D", ms=2.8, color=SLOTS[1], mec=SURF, mew=.6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{c}  {SEL[s]}  {cf}" for c, s, cf in rows], fontsize=6)
        ax.set_ylim(len(rows) - .5, -.5)
        ax.set_xlabel("difference (share scale)")
        ax.grid(axis="x", color=GRID, lw=0.4)
        hs = [plt.Line2D([], [], marker="o", ls="", ms=3.4, color=SLOTS[6], label="truth(Eq. 3) − truth(Π_Q+Π_γ)"),
              plt.Line2D([], [], marker="D", ls="", ms=2.8, color=SLOTS[1], label="bias of Eq. 3 at R = 10")]
        ax.legend(handles=hs, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0 + 0.9 / len(rows)), ncol=2,
                  fontsize=6.5)
        ax.set_title("Eq. 3 versus the crossed decomposition", loc="left", pad=16)
        fig.text(0.0, -0.01, f"Grey band: ±{bmax}.", fontsize=6, color=INK2, va="top")
        f = os.path.join(out, "fig_eq3.png")
        fig.savefig(f)
        plt.close(fig)
        files.append(f)

    # 6. runtime
    rr = rt.copy().sort_values(["clf", "kind", "R"])
    rr["label"] = rr.clf + " · " + rr.kind + " · R=" + rr.R.astype(str)
    fig, ax = plt.subplots(figsize=(4.6, 0.22 * len(rr) + 0.8))
    y = np.arange(len(rr))
    ax.barh(y, rr.mean_seconds, height=0.62, color=SLOTS[0])
    for yi, v, n in zip(y, rr.mean_seconds, rr.units):
        ax.text(v * 1.08, yi, f"{v:.1f} s  (n = {n})", va="center", fontsize=6, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(rr.label, fontsize=6.5)
    ax.set_ylim(len(rr) - .5, -.5)
    ax.set_xscale("log")
    ax.set_xlim(rr.mean_seconds.min() / 2, rr.mean_seconds.max() * 8)
    ax.set_xlabel("mean wall seconds per unit, one core (log scale)")
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    f = os.path.join(out, "fig_runtime.png")
    fig.savefig(f)
    plt.close(fig)
    files.append(f)
    return files


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/crossed_sim/v1")
    a = ap.parse_args(argv)
    cfg = json.load(open(os.path.join(a.run, "metadata.json")))["config"]
    T, E = load(a.run)
    out = os.path.join(a.run, "summary")
    os.makedirs(out, exist_ok=True)
    T.to_csv(os.path.join(out, "truth.csv"), index=False)
    E.to_csv(os.path.join(out, "replicates_raw.csv"), index=False)
    S = summarise(T, E)
    S.to_csv(os.path.join(out, "summary.csv"), index=False)
    C, verdict = criterion(S, cfg["criteria"])
    C.to_csv(os.path.join(out, "criterion_cells.csv"), index=False)
    rt = runtime(T, E)
    rt.to_csv(os.path.join(out, "runtime.csv"), index=False)
    figures(S, rt, out, cfg["criteria"])
    json.dump({"verdict": verdict, "cells": int(len(C)), "passed": int(C["pass"].sum()),
               "failed": C.loc[~C["pass"], ["config", "selector", "clf", "stat", "bias", "coverage_pigeonhole"]]
               .to_dict("records")}, open(os.path.join(out, "verdict.json"), "w"), indent=1, default=float)
    print("verdict:", verdict, f"({int(C['pass'].sum())}/{len(C)} counted cells pass)")


if __name__ == "__main__":
    main()
