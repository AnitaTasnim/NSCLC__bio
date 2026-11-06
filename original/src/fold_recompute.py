#!/usr/bin/env python3
"""Fully fold-recomputable selection: every data-dependent term of the six-stage pipeline is
re-derived from the training data of each outer fold.

    python fold_recompute.py config.json fi_full          # feature-importance consensus on all 633 samples
    python fold_recompute.py config.json nested           # nested CV, FI recomputed in every training fold
    python fold_recompute.py config.json mrmr_union       # genes that need STRING scores (for the PPI term)
    python fold_recompute.py config.json ppi <edges.tsv>  # add the PPI term from a STRING edge table, re-score
    python fold_recompute.py config.json summarise

Stage 0 (feature-importance consensus, s_FI).  The development notebook that produced G106 is not
available; the procedure is re-implemented from its description and fixed before any result was seen:
eight classifiers (XGBoost, CatBoost, AdaBoost, linear SVC, logistic regression, random forest,
k-nearest neighbours, SGD) are compared by three-fold stratified inner CV AUC on the training data;
the five best models that expose feature importances (kNN does not) each contribute their 3,000
genes with the largest non-zero importance (|coefficient| for linear models, impurity/gain
importance for tree models); the consensus is the set of genes present in all five lists, or in at
least four of the five if fewer than 20 genes are present in all five. Fixed settings are used (XGBoost 100
trees, CatBoost 200 iterations of depth 4 with 10% feature sampling, AdaBoost 50 stumps, random forest 200 trees). Tree models use the
processed matrix; linear models and kNN use standardised log2(x+1) values. All seeds are fixed and
ties are broken by gene name.

PPI term (s_PPI).  STRING scores are external, label-free knowledge, so the only data-dependent part
is which genes enter the MRMR top 30 (Stage 4), and that is computed inside each fold. The weighted
degree among the fold's MRMR genes is computed from a STRING edge table (combined score >= 0.4)
supplied with `ppi`; STRING could not be queried from the analysis environment.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tep_analysis as T  # noqa: E402
import selection_pipeline as S  # noqa: E402

TOP_K = 3000
N_MODELS = 5
SEED = 42


def _models(n_jobs):
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    return {
        "XGBoost": ("tree", lambda: XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8,
                                                  colsample_bytree=0.8, random_state=SEED, tree_method="hist",
                                                  eval_metric="logloss", n_jobs=n_jobs, verbosity=0)),
        "CatBoost": ("tree", lambda: CatBoostClassifier(iterations=200, depth=4, rsm=0.1, border_count=64, random_seed=SEED, verbose=0,
                                                        thread_count=n_jobs)),
        "AdaBoost": ("tree", lambda: AdaBoostClassifier(n_estimators=50, random_state=SEED)),
        "RandomForest": ("tree", lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=n_jobs)),
        "LinearSVC": ("linear", lambda: LinearSVC(C=1.0, max_iter=5000, random_state=SEED)),
        "LogisticRegression": ("linear", lambda: LogisticRegression(C=1.0, solver="liblinear", max_iter=2000,
                                                                    random_state=SEED)),
        "SGD": ("linear", lambda: SGDClassifier(loss="hinge", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=SEED)),
        "kNN": ("knn", lambda: KNeighborsClassifier(n_neighbors=5)),
    }


def _score(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.decision_function(X)


def _importance(m, kind):
    if kind == "tree":
        if hasattr(m, "get_feature_importance"):
            return np.asarray(m.get_feature_importance(), float)
        return np.asarray(m.feature_importances_, float)
    if kind == "linear":
        return np.abs(np.ravel(m.coef_))
    return None


def fi_consensus(X: pd.DataFrame, y, n_jobs=2, log=print):
    """Feature-importance consensus from training data only. Returns (genes, info)."""
    t0 = time.time()
    y = np.asarray(y).astype(int)
    genes = np.array(X.columns)
    Xt = X.values.astype(np.float32)
    Xl = np.log2(np.clip(X.values, 0, None) + 1.0)
    inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xt, y))
    res = {}
    for name, (kind, make) in _models(n_jobs).items():
        aucs = []
        for tr, te in inner:
            if kind == "tree":
                m = make().fit(Xt[tr], y[tr]); aucs.append(roc_auc_score(y[te], _score(m, Xt[te])))
            else:
                sc = StandardScaler().fit(Xl[tr])
                m = make().fit(sc.transform(Xl[tr]), y[tr])
                aucs.append(roc_auc_score(y[te], _score(m, sc.transform(Xl[te]))))
        res[name] = dict(kind=kind, inner_auc=float(np.mean(aucs)))
        log(f"      FI model {name:<18} inner AUC {np.mean(aucs):.4f}  ({time.time() - t0:.0f}s)")
    ranked = sorted([k for k in res if res[k]["kind"] != "knn"], key=lambda k: (-res[k]["inner_auc"], k))
    chosen = ranked[:N_MODELS]
    tops = {}
    for name in chosen:
        kind, make = _models(n_jobs)[name]
        if kind == "tree":
            m = make().fit(Xt, y)
        else:
            m = make().fit(StandardScaler().fit_transform(Xl), y)
        imp = _importance(m, kind)
        order = sorted([i for i in range(len(genes)) if imp[i] > 0], key=lambda i: (-imp[i], genes[i]))
        tops[name] = [genes[i] for i in order[:TOP_K]]
        res[name]["n_nonzero"] = int((imp > 0).sum())
    counts = pd.Series(0, index=genes)
    for name in chosen:
        counts[tops[name]] += 1
    cons = sorted(counts.index[counts == len(chosen)])
    rule = "all five"
    if len(cons) < 20:
        cons = sorted(counts.index[counts >= len(chosen) - 1]); rule = "at least four of five"
    info = dict(models=res, chosen=chosen, rule=rule, n=len(cons), seconds=round(time.time() - t0, 1),
                list_sizes={k: len(v) for k, v in tops.items()})
    log(f"      FI consensus: models {chosen}; rule {rule}; {len(cons)} genes ({time.time() - t0:.0f}s)")
    return cons, info


def ppi_scores_from_edges(mrmr, edges):
    """STRING weighted degree among the MRMR genes, divided by its maximum (as in the notebook)."""
    deg = {g: 0.0 for g in mrmr}
    if edges is not None and len(mrmr):
        s = set(mrmr)
        e = edges[edges.a.isin(s) & edges.b.isin(s) & (edges.score >= 0.4)]
        for a, b, sc in e[["a", "b", "score"]].itertuples(index=False):
            deg[a] += sc; deg[b] += sc
    mx = max(deg.values()) if deg and max(deg.values()) > 0 else 1.0
    return {g: v / mx for g, v in deg.items()}


def rescore(details, prior, ppi=None, weights=None, top_n=30):
    """Stage 6 again from saved Stage 1-5 results (cheap): lets the PPI term be added later."""
    sc = details["scores"].copy()
    W = dict(S.DEFAULT_WEIGHTS if weights is None else weights)
    sc["ml_prior"] = [1.0 if g in set(prior) else 0.2 for g in sc.index]
    if ppi is None:
        W["ppi"] = 0.0; sc["ppi_score"] = 0.0
    else:
        sc["ppi_score"] = [float(ppi.get(g, 0.0)) for g in sc.index]
    sc["composite"] = (W["stat"] * sc.stat_score + W["effect"] * sc.effect_score + W["lasso"] * sc.lasso_score +
                       W["boot"] * sc.boot_score + W["ppi"] * sc.ppi_score + W["prior"] * sc.ml_prior)
    sc = sc.assign(_g=sc.index).sort_values(["composite", "_g"], ascending=[False, True]).drop(columns="_g")
    top = sc.head(top_n) if len(sc) >= 20 else sc[sc.composite >= 0.35]
    return [g for g in top.index if S.is_clean_gene(g)]


def _setup(cfg):
    d = T.load_data(cfg); run = T.Run(cfg, d)
    return d, run


def main():
    cfg = T.load_config(sys.argv[1]); what = sys.argv[2]
    out = cfg["out_dir"]; cdir = os.path.join(out, "cache"); tdir = os.path.join(out, "tables")
    for p in (cdir, tdir, os.path.join(out, "macros")):
        os.makedirs(p, exist_ok=True)
    nj = cfg.get("n_jobs", 2)
    if what == "fi_full":
        d, run = _setup(cfg)
        cons, info = fi_consensus(d.X, d.y.values, n_jobs=nj, log=run.log)
        g106 = set(S.G106) & set(d.X.columns)
        ov = len(set(cons) & g106)
        info["overlap_with_G106"] = ov; info["G106_in_matrix"] = len(g106)
        json.dump(dict(genes=cons, info=info), open(os.path.join(cdir, "FI_full.json"), "w"), indent=1)
        run.log(f"  full-cohort FI consensus: {len(cons)} genes, {ov} of {len(g106)} G106 genes recovered")
        # full-cohort selection with the re-derived FI term (no PPI yet)
        panel, det = S.phase1_select(d.X, d.y.values, run=run, B=cfg["selection"]["boot_B"], use_prior=True,
                                     prior_genes=cons, run_mrmr=True, n_jobs=cfg.get("sel_n_jobs", 2),
                                     return_details=True)
        pickle.dump(dict(panel=panel, details=det, fi=cons), open(os.path.join(cdir, "FR_full.pkl"), "wb"))
        P = set(cfg["panel"])
        run.log(f"  full-cohort panel with re-derived FI: {len(panel)} genes, {len(P & set(panel))} of 27 shared")
    elif what == "nested":
        d, run = _setup(cfg)
        y = d.y.values; folds = run.folds()
        part = os.path.join(cdir, "FR_nested.pkl")
        R = pickle.load(open(part, "rb")) if os.path.exists(part) else {}
        for f, (tr, te) in enumerate(folds):
            if f in R:
                continue
            t0 = time.time()
            Xtr = d.X.iloc[tr]
            fi, info = fi_consensus(Xtr, y[tr], n_jobs=nj, log=run.log)
            panel, det = S.phase1_select(Xtr, y[tr], run=run, B=cfg["selection"]["boot_B"], use_prior=True,
                                         prior_genes=fi, run_mrmr=True, n_jobs=cfg.get("sel_n_jobs", 2),
                                         return_details=True)
            R[f] = dict(train=tr, test=te, fi=fi, fi_info=info, panel=panel,
                        details={k: det[k] for k in ("log", "scores", "stability", "coding")},
                        mrmr=det["log"]["mrmr"])
            pickle.dump(R, open(part, "wb"))
            run.log(f"  [fold {f + 1}] FI {len(fi)} genes; panel {len(panel)} genes ({time.time() - t0:.0f}s)")
    elif what == "mrmr_union":
        R = pickle.load(open(os.path.join(cdir, "FR_nested.pkl"), "rb"))
        full = pickle.load(open(os.path.join(cdir, "FR_full.pkl"), "rb"))
        genes = set(full["details"]["log"]["mrmr"])
        for f in R:
            genes |= set(R[f]["mrmr"])
        genes = sorted(genes)
        open(os.path.join(tdir, "FR_mrmr_union_genes.txt"), "w").write("\n".join(genes) + "\n")
        print(len(genes), "genes need STRING scores ->", os.path.join(tdir, "FR_mrmr_union_genes.txt"))
    elif what in ("ppi", "summarise"):
        evaluate(cfg, sys.argv[3] if what == "ppi" else None)


def evaluate(cfg, edges_path=None):
    """Train XGBoost on each fold panel, pool out-of-fold predictions, compare with the published nested CV."""
    out = cfg["out_dir"]; cdir = os.path.join(out, "cache"); tdir = os.path.join(out, "tables")
    d, run = _setup(cfg)
    y = d.y.values
    R = pickle.load(open(os.path.join(cdir, "FR_nested.pkl"), "rb"))
    full = pickle.load(open(os.path.join(cdir, "FR_full.pkl"), "rb"))
    fi_full = json.load(open(os.path.join(cdir, "FI_full.json")))
    edges = None
    if edges_path:
        e = pd.read_csv(edges_path, sep="\t")
        cols = {c.lower(): c for c in e.columns}
        a = cols.get("preferredname_a", cols.get("node1")); b = cols.get("preferredname_b", cols.get("node2"))
        s = cols.get("score", cols.get("combined_score"))
        edges = pd.DataFrame({"a": e[a], "b": e[b], "score": e[s].astype(float)})
        if edges.score.max() > 1.5:
            edges["score"] /= 1000.0
    variants = {"fi": None}
    if edges is not None:
        variants["fi_ppi"] = edges
    P27 = set(cfg["panel"])
    rows, M, summary = [], {}, {}
    for vname, ed in variants.items():
        oof = np.full(len(y), np.nan); panels = []
        for f in sorted(R):
            r = R[f]
            ppi = ppi_scores_from_edges(r["mrmr"], ed) if ed is not None else None
            panel = rescore(r["details"], r["fi"], ppi)
            m = run.xgb().fit(d.X.iloc[r["train"]][panel].values, y[r["train"]])
            oof[r["test"]] = m.predict_proba(d.X.iloc[r["test"]][panel].values)[:, 1]
            panels.append(panel)
            rows.append(dict(variant=vname, fold=f + 1, n_genes=len(panel), fi_genes=len(r["fi"]),
                             fold_auc=roc_auc_score(y[r["test"]], oof[r["test"]]),
                             jaccard_vs_27=len(set(panel) & P27) / len(set(panel) | P27),
                             fi_models=";".join(r["fi_info"]["chosen"]), panel=";".join(panel)))
        a, lo, hi, _ = T.delong_ci(y, oof)
        pj = [len(set(p) & set(q)) / len(set(p) | set(q)) for i, p in enumerate(panels) for q in panels[i + 1:]]
        jv = [len(set(p) & P27) / len(set(p) | P27) for p in panels]
        fullp = rescore(full["details"], full["fi"],
                        ppi_scores_from_edges(full["details"]["log"]["mrmr"], ed) if ed is not None else None)
        summary[vname] = dict(auc=a, lo=lo, hi=hi, jacc_vs27=float(np.mean(jv)), jacc_pair=float(np.mean(pj)),
                              full_panel=fullp, full_overlap=len(set(fullp) & P27), oof=oof)
        pd.DataFrame({"y": y, "oof": oof}).to_csv(os.path.join(tdir, f"FR_nested_oof_{vname}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "FR_nested_folds.csv"), index=False)
    # paired DeLong against the published nested CV (R4_nested_oof.csv)
    pub = pd.read_csv(os.path.join(os.path.dirname(HERE), "results", "tables", "R4_nested_oof.csv"))
    best = "fi_ppi" if "fi_ppi" in summary else "fi"
    s = summary[best]
    if "oof_prob" in pub.columns and len(pub) == len(y):
        dlt = T.delong_paired(y, s["oof"], pub["oof_prob"].values)
        M["FrVsPubDelta"] = T.pp(dlt["diff"]); M["FrVsPubP"] = T.pmac(dlt["p"])
        M["FrVsPubDeltaCI"] = f"{T.pp(dlt['lo95'])} to {T.pp(dlt['hi95'])}"
    for vname, s in summary.items():
        tag = "FrPpi" if vname == "fi_ppi" else "Fr"
        M[f"{tag}AUC"] = T.f4(s["auc"]); M[f"{tag}CI"] = f"{T.f4(s['lo'])}--{T.f4(s['hi'])}"
        M[f"{tag}JVsPanel"] = T.f3(s["jacc_vs27"]); M[f"{tag}JPair"] = T.f3(s["jacc_pair"])
        M[f"{tag}FullN"] = str(len(s["full_panel"])); M[f"{tag}FullOverlap"] = str(s["full_overlap"])
    # selection optimism: fixed 27-gene panel on the same folds minus the fully fold-recomputable estimate
    oof_fix, _ = T.cv_oof(d.X[run.panel_in(d.X)], y, run.folds(), run.xgb)   # same column order as the fixed-panel estimate
    g = T.delong_paired(y, oof_fix, summary[best]["oof"])
    M["FrFixedCheck"] = T.f4(roc_auc_score(y, oof_fix))
    M["FrGap"] = f"{100 * g['diff']:.1f}"; M["FrGapCI"] = f"{T.pp(g['lo95'])} to {T.pp(g['hi95'])}"; M["FrGapP"] = T.pmac(g["p"])
    F = pd.DataFrame([r for r in rows if r["variant"] == best])
    M["FrFoldMin"] = T.f4(F.fold_auc.min()); M["FrFoldMax"] = T.f4(F.fold_auc.max())
    M["FrSizes"] = f"{F.n_genes.min()}--{F.n_genes.max()}"
    # old (four-term) versus revised (fully fold-recomputable) nested CV, fold by fold
    old_f = pd.read_csv(os.path.join(os.path.dirname(HERE), "results", "tables", "R4_nested_folds.csv"))
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Four-term selection} & \multicolumn{3}{c}{Five-term selection} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Outer fold & AUC & Genes & Jaccard & AUC & Genes ($|\mathcal{G}^{(k)}_{\mathrm{FI}}|$) & Jaccard \\", r"\midrule"]
    for _, r in F.iterrows():
        o = old_f[old_f.fold == r.fold].iloc[0]
        lines.append(f"{int(r.fold)} & {T.f4(o.auc)} & {int(o.n_genes)} & {o.jaccard_vs_original:.3f} & "
                     f"{T.f4(r.fold_auc)} & {int(r.n_genes)} ({int(r.fi_genes)}) & {r.jaccard_vs_27:.3f} \\\\")
    pub_auc, pub_lo, pub_hi, _ = T.delong_ci(y, pub["oof_prob"].values)
    sb = summary[best]
    lines += [r"\midrule",
              f"Pooled out-of-fold & \\multicolumn{{3}}{{c}}{{{T.f4(pub_auc)} ({T.f4(pub_lo)}--{T.f4(pub_hi)})}} & "
              f"\\multicolumn{{3}}{{c}}{{{T.f4(sb['auc'])} ({T.f4(sb['lo'])}--{T.f4(sb['hi'])})}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(tdir, "FR_nested_compare.tex"), "w").write("\n".join(lines) + "\n")
    # how much the re-derived feature-importance term changed each fold panel
    jj, nd = [], []
    for _, r in F.iterrows():
        a = set(old_f[old_f.fold == r.fold].iloc[0].genes.split(";")); b = set(r.panel.split(";"))
        jj.append(len(a & b) / len(a | b)); nd.append(len(a ^ b) // 2 if len(a) == len(b) else max(len(a - b), len(b - a)))
    M["FrVsFourJMean"] = T.f3(np.mean(jj)); M["FrVsFourJRange"] = f"{min(jj):.3f}--{max(jj):.3f}"
    M["FrVsFourChanged"] = f"{min(nd)}--{max(nd)}"
    M["FrFiFullN"] = str(len(fi_full["genes"])); M["FrFiFullOverlap"] = str(fi_full["info"]["overlap_with_G106"])
    M["FrFiGInMatrix"] = str(fi_full["info"]["G106_in_matrix"])
    M["FrFiPanelInFi"] = str(len(set(fi_full["genes"]) & P27)); M["FrGPanelInG"] = str(len(set(S.G106) & P27))
    M["FrFiFullModels"] = ", ".join(fi_full["info"]["chosen"])
    M["FrFiFoldNRange"] = f"{min(len(R[f]['fi']) for f in R)}--{max(len(R[f]['fi']) for f in R)}"
    json.dump(M, open(os.path.join(out, "macros", "foldrec.json"), "w"), indent=1)
    print(json.dumps(M, indent=1))
    print(pd.DataFrame(rows).drop(columns=["panel"]).to_string(index=False))


if __name__ == "__main__":
    main()
