#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harm_sim.py - harmonisation leakage experiment inside a single site (R3-sim).

Question: can the alignment steps used in the original external validation change the AUC by
themselves - without any real batch effect - depending on the case mix of the "external" cohort?

Design (Vumc only, so there is NO site/batch difference between the two parts):
  repeat R times:  stratified split of Vumc into a discovery part (60%) and an external pool (40%)
  for each NSCLC prevalence p in {0.2, 0.4, 0.6, 0.8}: draw an "external cohort" of n samples
      with that prevalence from the pool and score it with a model trained on the discovery part:
        S0  no alignment (discovery scaler applied to the external samples)
        Sz  each cohort standardised on itself  (the procedure behind the original AUC 0.7801)
        S1  joint ComBat-seq of both parts (cohort = batch, no labels), model re-trained on the
            adjusted discovery data (the procedure behind the original AUC 0.9100)
        S2  reference-batch ComBat-seq (discovery fixed, only the external part adjusted),
            frozen model
  Because both parts come from the same site, any systematic difference between S0 and the
  other settings, and any dependence on prevalence, is created by the alignment itself.
Features: the 22-gene subset of the development model, log2(x+1).
Classifier: XGBoost as in the main analysis (300 trees, depth 4, subsample/colsample 0.8, seed 42).

    python harm_sim.py config.json [repeats] [n_external]
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tep_analysis as T  # noqa: E402

warnings.filterwarnings("ignore")


def combat_seq(C1, C2, ref_batch):
    from inmoose.pycombat import pycombat_seq
    allc = pd.concat([C1, C2], axis=0)
    batch = [0] * len(C1) + [1] * len(C2)
    adj = pycombat_seq(allc.T.values, batch, ref_batch=ref_batch)
    adj = pd.DataFrame(np.asarray(adj), index=allc.columns, columns=allc.index).T
    return adj.iloc[: len(C1)], adj.iloc[len(C1):]


def main(cfg_path, R=30, n_ext=50):
    cfg = T.load_config(cfg_path)
    cfg["out_dir"] = cfg["out_dir"]
    d = T.load_data(cfg)
    run = T.Run(cfg, d)
    P = [g for g in cfg["external_panel"] if g in d.X.columns]
    v = run.vumc
    X = d.X.loc[v]
    y = d.y.values[v]
    C = X.clip(lower=0).round().astype(np.int64)
    top = C.mean(0).sort_values(ascending=False).index[:500]
    keep = sorted(set(top) | set(P))
    Cred = C[keep].copy()
    Cred["__REST__"] = C.drop(columns=keep).sum(axis=1)
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    t0 = time.time()
    for r in range(R):
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        dp, dn = int(0.6 * len(pos)), int(0.6 * len(neg))
        disc = np.r_[pos[:dp], neg[:dn]]
        pool_p, pool_n = pos[dp:], neg[dn:]
        Ld = np.log2(X.iloc[disc][P].values + 1)
        sc = StandardScaler().fit(Ld)
        m0 = run.xgb(n_jobs=1).fit(sc.transform(Ld), y[disc])
        for prev in (0.2, 0.4, 0.6, 0.8):
            k1 = int(round(prev * n_ext)); k0 = n_ext - k1
            if k1 > len(pool_p) or k0 > len(pool_n):
                continue
            ext = np.r_[rng.choice(pool_p, k1, replace=False), rng.choice(pool_n, k0, replace=False)]
            ye = y[ext]
            Le = np.log2(X.iloc[ext][P].values + 1)
            row = dict(rep=r, prevalence=prev,
                       auc_none=roc_auc_score(ye, m0.predict_proba(sc.transform(Le))[:, 1]),
                       auc_percohort_z=roc_auc_score(ye, m0.predict_proba(StandardScaler().fit_transform(Le))[:, 1]),
                       mean_prob_none=float(m0.predict_proba(sc.transform(Le))[:, 1].mean()),
                       mean_prob_percohort_z=float(m0.predict_proba(StandardScaler().fit_transform(Le))[:, 1].mean()))
            try:
                Cd, Ce = Cred.iloc[disc], Cred.iloc[ext]
                jd, je = combat_seq(Cd, Ce, None)
                Ljd, Lje = np.log2(jd[P].values + 1), np.log2(je[P].values + 1)
                scj = StandardScaler().fit(Ljd)
                mj = run.xgb(n_jobs=1).fit(scj.transform(Ljd), y[disc])
                row["auc_joint_combat_retrained"] = roc_auc_score(ye, mj.predict_proba(scj.transform(Lje))[:, 1])
                _, re_ = combat_seq(Cd, Ce, 0)
                row["auc_ref_combat_frozen"] = roc_auc_score(ye, m0.predict_proba(sc.transform(np.log2(re_[P].values + 1)))[:, 1])
            except Exception as e:  # pragma: no cover
                row["combat_error"] = repr(e)[:80]
            rows.append(row)
        print(f"rep {r + 1}/{R} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = cfg["out_dir"]
    df.to_csv(os.path.join(out, "tables", "R3sim_alignment_within_vumc.csv"), index=False)
    cols = ["auc_none", "auc_percohort_z", "auc_joint_combat_retrained", "auc_ref_combat_frozen"]
    g = df.groupby("prevalence")[[c for c in cols if c in df]].mean()
    print(g.round(4))
    m = {}
    for c, nm in [("auc_percohort_z", "Z"), ("auc_joint_combat_retrained", "Joint"), ("auc_ref_combat_frozen", "Ref")]:
        if c not in df:
            continue
        dlt = (df[c] - df.auc_none)
        m[f"Sim{nm}Delta"] = f"{100 * dlt.mean():+.1f}".replace("-", "$-$")
        m[f"Sim{nm}DeltaRange"] = (f"{100 * dlt.quantile(.025):+.1f} to {100 * dlt.quantile(.975):+.1f}").replace("-", "$-$")
        gp = df.groupby("prevalence").apply(lambda s: (s[c] - s.auc_none).mean())
        m[f"Sim{nm}PrevSpread"] = f"{100 * (gp.max() - gp.min()):.1f}"
        m[f"Sim{nm}DeltaLowPrev"] = f"{100 * gp.iloc[0]:+.1f}".replace("-", "$-$")
        m[f"Sim{nm}DeltaHighPrev"] = f"{100 * gp.iloc[-1]:+.1f}".replace("-", "$-$")
    m["SimNoneAUC"] = f"{df.auc_none.mean():.3f}"
    m["SimR"] = str(R); m["SimN"] = str(n_ext)
    gp = df.groupby("prevalence").auc_none.mean()
    m["SimNonePrevSpread"] = f"{100 * (gp.max() - gp.min()):.1f}"
    json.dump(m, open(os.path.join(out, "macros", "r3sim.json"), "w"), indent=1)
    print(m)
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3))
    lab = {"auc_none": "no alignment", "auc_percohort_z": "each cohort standardised on itself",
           "auc_joint_combat_retrained": "joint ComBat-seq + re-trained model",
           "auc_ref_combat_frozen": "reference-batch ComBat-seq, frozen model"}
    sty = {"auc_none": ("#555", "s--"), "auc_percohort_z": ("#C0392B", "o-"),
           "auc_joint_combat_retrained": ("#7D3C98", "^-"), "auc_ref_combat_frozen": ("#1F618D", "D-")}
    for c in [c for c in cols if c in df]:
        mu = df.groupby("prevalence")[c].mean()
        axs[0].plot(mu.index, mu.values, sty[c][1], c=sty[c][0], label=lab[c], ms=4)
    axs[0].set_xlabel("NSCLC prevalence of the 'external' part"); axs[0].set_ylabel("Mean AUC")
    axs[0].set_title("A  AUC by case mix (no true batch effect)", fontsize=9, loc="left")
    axs[0].legend(frameon=False, fontsize=6.5)
    for c in [c for c in cols if c in df and c != "auc_none"]:
        dd = (df[c] - df.auc_none)
        mu = dd.groupby(df.prevalence).mean()
        axs[1].plot(mu.index, 100 * mu.values, sty[c][1], c=sty[c][0], label=lab[c], ms=4)
    axs[1].axhline(0, c="k", lw=.7, ls=":")
    axs[1].set_xlabel("NSCLC prevalence of the 'external' part"); axs[1].set_ylabel("AUC change vs no alignment (pp)")
    axs[1].set_title("B  Change caused by the alignment step", fontsize=9, loc="left")
    for a in axs:
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"R3sim_alignment_within_vumc.{e}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30, int(sys.argv[3]) if len(sys.argv) > 3 else 50)
