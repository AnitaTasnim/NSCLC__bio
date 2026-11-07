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
COL = {"pigeonhole": "#2a78d6", "draws": "#14967a", "patients": "#eb6834"}
SEL = {"mw": "Mann–Whitney", "l1": "L1-logistic", "random": "Random"}
STAT_LABEL = {"S": "Π_S", "Q": "Π_Q", "SQ": "Π_SQ", "E": "Π_E", "gamma": "Π_γ", "Q+gamma": "Π_Q+Π_γ", "eq3": "Eq. 3"}


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    return plt


def figures(S: pd.DataFrame, rt: pd.DataFrame, out: str, crit: Dict) -> List[str]:
    plt = _setup()
    files = []
    stats = ["S", "Q", "SQ", "E", "gamma", "Q+gamma"]
    r10 = S[S.R == 10]
    combos = [(c, s) for c in ("logistic", "xgboost") for s in ("mw", "l1", "random")
              if len(r10[(r10.clf == c) & (r10.selector == s)])]

    # 1. bias at R = 10 for every configuration
    fig, axes = plt.subplots(len(combos), 1, figsize=(7.2, 1.6 * len(combos) + 0.4), sharex=True, squeeze=False)
    for ax, (c, s) in zip(axes[:, 0], combos):
        d = r10[(r10.clf == c) & (r10.selector == s) & r10.stat.isin(stats)]
        cfgs = list(dict.fromkeys(d.sort_values(["n", "p", "rho", "auc", "f"]).config))
        for j, st in enumerate(stats):
            dd = d[d.stat == st].set_index("config").reindex(cfgs)
            x = np.arange(len(cfgs)) + (j - 2.5) * 0.12
            ax.errorbar(x, dd.bias, yerr=1.96 * dd.bias_se, fmt="o", ms=2.5, lw=0.8, label=STAT_LABEL[st])
        ax.axhspan(-crit["abs_bias_max"], crit["abs_bias_max"], color="0.9", zorder=0)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_ylabel(f"{c}\n{SEL[s]}\nbias")
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, rotation=45, ha="right")
    axes[0, 0].legend(ncol=6, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.35))
    fig.suptitle("Bias of the share estimators at R = 10 (grey band: ±0.02 criterion; bars: 95% MC interval)",
                 fontsize=8, y=1.02)
    f = os.path.join(out, "fig_bias_R10.png")
    fig.savefig(f)
    plt.close(fig)
    files.append(f)

    # 2. coverage at R = 10, three interval methods
    fig, axes = plt.subplots(len(combos), 1, figsize=(7.2, 1.6 * len(combos) + 0.4), sharex=True, squeeze=False)
    for ax, (c, s) in zip(axes[:, 0], combos):
        d = r10[(r10.clf == c) & (r10.selector == s) & r10.stat.isin(stats)]
        d = d[~((d.clf == "logistic") & (d.stat == "E"))]
        cfgs = list(dict.fromkeys(d.sort_values(["n", "p", "rho", "auc", "f"]).config))
        labels = []
        for i, cf in enumerate(cfgs):
            for j, st in enumerate([x for x in stats if x in set(d[d.config == cf].stat)]):
                labels.append((i, j, cf, st))
        for m_i, m in enumerate(MODES):
            xs, ys, lo, hi = [], [], [], []
            for (i, j, cf, st) in labels:
                row = d[(d.config == cf) & (d.stat == st)].iloc[0]
                xs.append(i + (j - 2.5) * 0.13 + (m_i - 1) * 0.035)
                ys.append(row[f"coverage_{m}"])
                lo.append(row[f"coverage_{m}"] - row[f"coverage_{m}_lo"])
                hi.append(row[f"coverage_{m}_hi"] - row[f"coverage_{m}"])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=2, lw=0.5, color=COL[m], label=m)
        ax.axhspan(crit["coverage_low"], crit["coverage_high"], color="0.9", zorder=0)
        ax.axhline(0.95, color="0.4", lw=0.6)
        ax.set_ylim(0.3, 1.02)
        ax.set_ylabel(f"{c}\n{SEL[s]}\ncoverage")
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, rotation=45, ha="right")
    axes[0, 0].legend(ncol=3, fontsize=6.5, frameon=False, loc="lower left")
    fig.suptitle("Coverage of 95% intervals at R = 10 (grey band: 0.92–0.97; per configuration the six points\n"
                 "are Π_S, Π_Q, Π_SQ, Π_E (XGBoost only), Π_γ, Π_Q+Π_γ; bars: Wilson 95% intervals)", fontsize=8, y=1.03)
    f = os.path.join(out, "fig_coverage_R10.png")
    fig.savefig(f)
    plt.close(fig)
    files.append(f)

    # 3. RMSE and coverage against R at the base point
    b = S[(S.config == "base") & S.stat.isin(stats)]
    fig, axes = plt.subplots(2, len(combos), figsize=(1.6 * len(combos) + 1, 3.6), squeeze=False, sharey="row")
    for j, (c, s) in enumerate(combos):
        d = b[(b.clf == c) & (b.selector == s)]
        for st in stats:
            dd = d[d.stat == st].sort_values("R")
            if c == "logistic" and st == "E":
                continue
            axes[0, j].plot(dd.R, dd.rmse, "o-", ms=2.5, lw=0.8, label=STAT_LABEL[st])
            axes[1, j].plot(dd.R, dd.coverage_pigeonhole, "o-", ms=2.5, lw=0.8)
        axes[0, j].set_xscale("log")
        axes[0, j].set_yscale("log")
        axes[1, j].set_xscale("log")
        axes[1, j].axhspan(crit["coverage_low"], crit["coverage_high"], color="0.9", zorder=0)
        axes[0, j].set_title(f"{c}\n{SEL[s]}", fontsize=7)
        axes[1, j].set_xlabel("R")
        for ax in axes[:, j]:
            ax.set_xticks([5, 10, 20, 50])
            ax.set_xticklabels(["5", "10", "20", "50"])
    axes[0, 0].set_ylabel("RMSE")
    axes[1, 0].set_ylabel("pigeonhole coverage")
    axes[0, -1].legend(fontsize=6, frameon=False, loc="upper right")
    fig.suptitle("Base configuration: RMSE and coverage against the number of draws R", fontsize=8)
    f = os.path.join(out, "fig_R_base.png")
    fig.savefig(f)
    plt.close(fig)
    files.append(f)

    # 4. Eq. 3 against its own truth and against Π_Q + Π_γ
    e = S[(S.stat == "eq3") & (S.R == 10)].copy()
    if len(e):
        e = e.sort_values(["clf", "selector", "n", "p", "rho", "auc", "f"])
        fig, ax = plt.subplots(figsize=(7.2, 2.4))
        x = np.arange(len(e))
        ax.bar(x - 0.2, e.eq3_minus_Qgamma_truth, 0.4, color="#9b59b6", label="truth of Eq. 3 − truth of Π_Q+Π_γ")
        ax.bar(x + 0.2, e.bias, 0.4, color="#8c96a3", label="bias of Eq. 3 (R = 10) against its own truth")
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.clf[:3]} {r.selector} {r.config}" for r in e.itertuples()], rotation=60,
                           ha="right", fontsize=6)
        ax.set_ylabel("difference (share)")
        ax.legend(fontsize=6.5, frameon=False)
        f = os.path.join(out, "fig_eq3.png")
        fig.savefig(f)
        plt.close(fig)
        files.append(f)

    # 5. runtime
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    rr = rt.copy()
    rr["label"] = rr.clf + " " + rr.kind + " R=" + rr.R.astype(str)
    ax.barh(rr.label, rr.mean_seconds, color="#2a78d6")
    ax.set_xscale("log")
    ax.set_xlabel("mean wall seconds per unit (one core)")
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
