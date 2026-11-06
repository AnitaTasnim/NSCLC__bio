#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tep_analysis.py
===============
New analyses requested in the review of

    Tasnim A, Alam TT. "Decomposing Biomarker Robustness in Tumour-Educated Platelet
    Diagnostics: Predictive, Molecular, and Technical Stability of a 27-Gene NSCLC Signature"

Review point  ->  function(s)
-----------------------------------------------------------------------------------------
R1  site confounding ...................  r1_site_table, r1_gene_effects,
                                          r1_vumc_rederivation, r1_vumc_nested
R2  external validation (22 vs 27) .....  r2_panel_equivalence, r2_external_operating
R3  ComBat-seq leakage audit ...........  r3_harmonisation_audit
R4  gene-identity stability ............  r4_stability
R5  contamination genes ................  r5_contamination
R6  clinical covariates ................  r6_covariates
R7  biology / enrichment ...............  r7_enrichment
R8  head-to-head vs Sheng et al. .......  r8_head_to_head
R9  10,000 random panels ...............  r9_random_panels
R10 tissue-transfer interpretation .....  r10_tissue
minor: 95% CI for every AUC ............  r0_core_estimates + auc_summary
minor: SHAP after selection ............  shap_out_of_fold
minor: simpler figures .................  fig_auc_forest, fig_nested_summary, fig_rfecv

Every analysis writes CSV tables to <out>/tables, figures to <out>/figures and stores each
number it produces as a LaTeX macro.  run.write_macros() saves them to
<out>/results_macros.tex; the manuscript \\input's that file, so the paper updates
itself when you re-run the code on your data.

IMPORTANT
---------
* Every analysis uses the SAME classifier settings as the paper (XGBoost, 300 trees,
  learning rate 0.03, depth 4, subsample 0.8, seed 42).
* `six_stage_select()` below is a re-implementation of the six-stage pipeline written from
  the manuscript text.  For numbers that must match the paper exactly, pass YOUR OWN
  selection function (same signature: f(X_train, y_train, run) -> list of genes).
"""
from __future__ import annotations

import os
import re
import io
import json
import math
import time
import warnings
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
except ImportError as e:  # pragma: no cover
    raise ImportError("Please `pip install xgboost`") from e

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------------------
# The 27-gene panel (manuscript Table 8) and other fixed gene lists
# ----------------------------------------------------------------------------------------
PANEL27 = [
    "P2RY10", "MGAT5", "FCER1A", "HSD17B3", "JOSD1", "USP27X", "DUOXA1", "ITGB3BP",
    "LCE3B", "SPHAR", "NTNG2", "TMEM39B", "OR52B6", "PRSS42", "KRTAP22-1", "C1QA",
    "PCSK5", "HOXC12", "TNFRSF17", "LY6G6D", "DDAH1", "PTMS", "TM4SF1", "MFN2",
    "GJB2", "CXCL14", "TYMS",
]
# Table 8: bootstrap frequency and stability-selection probability (used for R5 'updated stability')
TABLE8 = {
    "P2RY10": (0.992, 0.995), "MGAT5": (0.966, 0.925), "FCER1A": (0.716, 0.975),
    "HSD17B3": (0.866, 0.930), "JOSD1": (0.980, 0.950), "USP27X": (0.986, 0.980),
    "DUOXA1": (0.902, 0.945), "ITGB3BP": (0.764, 0.835), "LCE3B": (0.990, 0.995),
    "SPHAR": (0.874, 0.945), "NTNG2": (0.734, 0.970), "TMEM39B": (0.912, 0.975),
    "OR52B6": (0.930, 0.945), "PRSS42": (0.934, 0.995), "KRTAP22-1": (0.916, 0.915),
    "C1QA": (0.998, 0.995), "PCSK5": (0.966, 0.865), "HOXC12": (0.970, 0.960),
    "TNFRSF17": (0.942, 0.860), "LY6G6D": (0.940, 0.960), "DDAH1": (0.888, 0.980),
    "PTMS": (0.928, 0.985), "TM4SF1": (0.972, 0.760), "MFN2": (0.964, 0.930),
    "GJB2": (0.898, 0.875), "CXCL14": (0.994, 0.590), "TYMS": (0.964, 0.790),
}
SUSPECT_CONTAMINATION = ["KRTAP22-1", "OR52B6", "LCE3B"]

# Marker sets for the contamination / cell-of-origin screen (R5).  Missing genes are skipped.
MARKER_SETS = {
    "skin": ["KRT1", "KRT10", "KRT2", "FLG", "LOR", "DSC1", "DSG1", "KRT14", "KRT5", "KRTDAP", "SBSN"],
    "leukocyte": ["PTPRC", "CD3E", "CD2", "CD247", "MS4A1", "CD19", "CD79A", "CD14", "LYZ",
                  "CSF3R", "FCGR3B"],
    "erythroid": ["HBB", "HBA1", "HBA2", "HBD", "ALAS2", "CA1"],
    "platelet": ["ITGA2B", "GP9", "PF4", "PPBP", "GP1BA", "TUBB1"],
}
SEX_Y_GENES = ["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "USP9Y", "ZFY", "NLGN4Y", "TXLNGY"]
SEX_X_GENES = ["XIST", "TSIX"]
# Genes most consistently associated with smoking in whole-blood transcriptomes
# (exploratory only: platelets are not whole blood).  Verify against Huan et al. 2016.
SMOKING_GENES = ["LRRN3", "AHRR", "GPR15", "SASH1", "PID1", "FUCA1", "CLDND1", "SEMA6B",
                 "P2RY6", "CYP1B1"]

NUM_WORDS = {5: "Five", 10: "Ten", 15: "Fifteen", 20: "Twenty", 22: "TwentyTwo",
             24: "TwentyFour", 25: "TwentyFive", 26: "TwentySix", 27: "TwentySeven",
             30: "Thirty", 40: "Forty", 47: "FortySeven", 50: "Fifty"}

# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------
DEFAULT_CFG: Dict = {
    # ---- inputs (see README.md for formats) ----
    "discovery_expr": None,        # GSE89843 ML matrix (633 samples x 28,089 genes; either orientation)
    "discovery_meta": None,        # CSV: sample id, label, GEO title (or site)
    "discovery_counts": None,      # optional raw counts (for ComBat-seq); default = discovery_expr
    "external_expr": None,         # GSE68086 matrix (94 independent samples)
    "external_meta": None,         # CSV: sample id, label
    "external_counts": None,       # optional raw counts for GSE68086
    "id_col": None,                # sample-id column in meta files (default: first column)
    "label_col": None,             # label column (auto-detected if None)
    "title_col": None,             # GEO title column (auto-detected if None)
    "site_col": None,              # site column if you already have one
    "fold_panels": None,           # optional JSON: list of the 5 nested-fold panels from your run
    "nested_oof": None,            # optional CSV: sample, fold, oof_prob from your nested run
    "sheng48": None,               # optional text file: Sheng et al. 48 genes, one per line
    "tissue": {},                  # optional {"GSE40791": {"expr": path, "meta": path}, ...}
    "protein_coding_list": None,   # optional text file of HGNC protein-coding symbols
    "ppi_edges": None,             # optional STRING edge list (gene1,gene2[,score]) for Stage 6
    "gene_sets": ["MSigDB_Hallmark_2020", "Reactome_2022", "GO_Biological_Process_2023"],
    # ---- general ----
    "out_dir": "outputs",
    "seed": 42,
    "n_jobs": -1,
    "values_are_log": False,       # True if your ML matrix is already log2-transformed
    "normalize_for_stats": "auto", # raw counts -> CPM before statistics/correlations (classifier input unchanged)
    "vumc_site": "Vumc",
    "site_regex": r"Platelets?_([A-Za-z0-9]+)[-_ ]",
    "panel": PANEL27,
    "suspect_contamination": SUSPECT_CONTAMINATION,
    # ---- classifier (manuscript Section 2.6) ----
    "xgb_params": dict(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8,
                       random_state=42, tree_method="hist", eval_metric="logloss"),
    "cv_folds": 5,
    "cv_repeats": 20,
    "equivalence_margin": 0.02,    # TOST margin on the AUC scale (pre-specified)
    "bootstrap_B": 2000,
    # ---- R9 random panels ----
    "n_random_panels": 10000,
    "n_matched_panels": 10000,
    "matched_bins": 10,
    "random_checkpoint_every": 250,
    # ---- R3 harmonisation audit ----
    "harm": dict(run_combat_seq=True, calib_k=[5, 10, 20, 30, 47], calib_repeats=200,
                 prev_levels=[0.2, 0.4, 0.6, 0.8], prev_n=40, prev_repeats=200,
                 prev_repeats_combat=30, random_panels=200, combat_genes=2000,
                 rank_min_cpm=1.0, rank_min_frac=0.2),
    # ---- six-stage selection re-implementation ----
    "selection": dict(tier2_padj=0.05, tier2_lfc=0.5, max_stage1=4000, n_C=12, inner_folds=5,
                      boot_B=100, boot_B_full=500, boot_threshold=0.60, mrmr_top=30, panel_size=27,
                      weights=[0.20, 0.15, 0.15, 0.30, 0.10, 0.10], redundancy_rho=0.90,
                      artefact_regex=[r"\.\d+$", r"^MT-", r"^RP[LS]\d", r"^MRP[LS]\d"]),
    # ---- R8 Sheng et al. MRMR-IFS-SVM ----
    "sheng": dict(prefilter_n=4722, mrmr_K=150, ifs_max=150, ifs_step=1, inner_folds=5,
                  svm_kernel="linear", svm_C=1.0, ifs_metric="mcc", mrmr_scheme="MID",
                  discretize_alpha=0.5, loocv=True),
}


def load_config(path_or_dict=None) -> Dict:
    """Merge user settings (dict, .json or .yaml file) into DEFAULT_CFG."""
    import copy
    cfg = copy.deepcopy(DEFAULT_CFG)
    if path_or_dict is None:
        return cfg
    if isinstance(path_or_dict, dict):
        user = path_or_dict
    else:
        with open(path_or_dict) as fh:
            if str(path_or_dict).endswith((".yml", ".yaml")):
                import yaml
                user = yaml.safe_load(fh)
            else:
                user = json.load(fh)
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# ----------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------
def read_matrix(path: str) -> pd.DataFrame:
    """Read an expression matrix (csv/tsv/txt[.gz], parquet, feather, pickle). First column = row ids."""
    p = str(path).lower()
    if p.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif p.endswith(".feather"):
        df = pd.read_feather(path)
        df = df.set_index(df.columns[0])
    elif p.endswith((".pkl", ".pickle")):
        df = pd.read_pickle(path)
    else:
        sep = "\t" if re.search(r"\.(tsv|txt)(\.gz)?$", p) else ","
        df = pd.read_csv(path, sep=sep, index_col=0)
    df.index = df.index.map(str)
    df.columns = df.columns.map(str)
    return df


def orient_samples_by_genes(df: pd.DataFrame, panel: Sequence[str],
                            sample_ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Return the matrix as samples x genes, detecting orientation from the panel genes."""
    in_cols = sum(g in df.columns for g in panel)
    in_rows = sum(g in df.index for g in panel)
    if sample_ids is not None:
        sid = set(map(str, sample_ids))
        if len(sid & set(df.index)) > len(sid & set(df.columns)):
            return df
        if len(sid & set(df.columns)) > 0:
            return df.T
    if in_rows > in_cols:
        df = df.T
    return df


def parse_labels(s: pd.Series) -> pd.Series:
    """Map labels to 1 = NSCLC, 0 = healthy.  Accepts 0/1 or text."""
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype(int)
        if not set(out.unique()) <= {0, 1}:
            raise ValueError("numeric labels must be 0 (healthy) / 1 (NSCLC)")
        return out

    def f(v):
        t = str(v).lower()
        if any(k in t for k in ["nsclc", "lung", "cancer", "carcinoma", "tumor", "tumour"]):
            return 1
        if any(k in t for k in ["healthy", "control", "hc", "hd", "normal", "donor"]):
            return 0
        return np.nan
    out = s.map(f)
    if out.isna().any():
        bad = s[out.isna()].unique()[:5]
        raise ValueError(f"Could not parse labels such as {list(bad)} - give 0/1 labels instead")
    return out.astype(int)


def site_from_title(title: str, regex: str) -> str:
    m = re.search(regex, str(title))
    return m.group(1) if m else "UNKNOWN"


@dataclass
class TEPData:
    X: pd.DataFrame                       # discovery ML matrix, samples x genes
    y: pd.Series                          # 1 NSCLC / 0 healthy
    site: pd.Series                       # collection-site code
    meta: pd.DataFrame
    X_ext: Optional[pd.DataFrame] = None  # external ML matrix, samples x genes
    y_ext: Optional[pd.Series] = None
    meta_ext: Optional[pd.DataFrame] = None
    counts: Optional[pd.DataFrame] = None
    counts_ext: Optional[pd.DataFrame] = None
    tissue: Dict[str, Tuple[pd.DataFrame, pd.Series]] = field(default_factory=dict)


def _pick_col(df: pd.DataFrame, given: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if given:
        return given
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in low:
            return low[c]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            return c
    return None


def _load_meta(path, cfg):
    meta = pd.read_csv(path, sep=None, engine="python")
    idc = cfg.get("id_col") or meta.columns[0]
    meta[idc] = meta[idc].astype(str)
    meta = meta.set_index(idc)
    return meta


def load_data(cfg: Dict) -> TEPData:
    """Load discovery (and optionally external, counts, tissue) data from the paths in cfg."""
    panel = cfg["panel"]
    meta = _load_meta(cfg["discovery_meta"], cfg)
    X = orient_samples_by_genes(read_matrix(cfg["discovery_expr"]), panel, meta.index)
    common = [s for s in meta.index if s in X.index]
    if len(common) == 0:
        raise ValueError("No sample ids shared between discovery_expr and discovery_meta")
    X = X.loc[common].astype(float)
    meta = meta.loc[common]
    lab = _pick_col(meta, cfg.get("label_col"), ["label", "y", "condition", "group", "class",
                                                  "disease", "patient group", "status"])
    y = parse_labels(meta[lab]).rename("y")
    sc = _pick_col(meta, cfg.get("site_col"), ["site", "hospital", "centre", "center"])
    if sc is not None and cfg.get("site_col") is not None:
        site = meta[sc].astype(str)
    else:
        tc = _pick_col(meta, cfg.get("title_col"), ["title", "sample_title", "geo_title"])
        if tc is None:
            raise ValueError("Need a 'title' (GEO title) or 'site' column in discovery_meta")
        site = meta[tc].map(lambda t: site_from_title(t, cfg["site_regex"]))
    site = site.rename("site")
    X = X.loc[:, (X != 0).any(axis=0)]                     # zero-count filter
    d = TEPData(X=X, y=y, site=site, meta=meta)
    if cfg.get("external_expr"):
        me = _load_meta(cfg["external_meta"], cfg)
        Xe = orient_samples_by_genes(read_matrix(cfg["external_expr"]), panel, me.index)
        ce = [s for s in me.index if s in Xe.index]
        Xe = Xe.loc[ce].astype(float)
        me = me.loc[ce]
        labe = _pick_col(me, cfg.get("label_col"), ["label", "y", "condition", "group", "class",
                                                     "disease", "patient group", "status"])
        d.X_ext, d.y_ext, d.meta_ext = Xe, parse_labels(me[labe]).rename("y"), me
    if cfg.get("discovery_counts"):
        C = orient_samples_by_genes(read_matrix(cfg["discovery_counts"]), panel, X.index)
        d.counts = C.loc[[s for s in X.index if s in C.index]]
    if cfg.get("external_counts") and d.X_ext is not None:
        Ce = orient_samples_by_genes(read_matrix(cfg["external_counts"]), panel, d.X_ext.index)
        d.counts_ext = Ce.loc[[s for s in d.X_ext.index if s in Ce.index]]
    for name, spec in (cfg.get("tissue") or {}).items():
        mt = _load_meta(spec["meta"], cfg)
        Xt = orient_samples_by_genes(read_matrix(spec["expr"]), panel, mt.index)
        ct = [s for s in mt.index if s in Xt.index]
        labt = _pick_col(mt, cfg.get("label_col"), ["label", "y", "condition", "group", "class",
                                                     "tissue", "status"])
        d.tissue[name] = (Xt.loc[ct].astype(float), parse_labels(mt.loc[ct, labt]))
    return d


def fetch_geo_metadata(gse: str = "GSE89843", destdir: str = "geo_cache") -> pd.DataFrame:
    """(Colab helper, needs internet + `pip install GEOparse`) - sample titles and every
    'characteristics' field (age, gender, smoking ... if deposited) for a GEO series."""
    import GEOparse
    g = GEOparse.get_GEO(geo=gse, destdir=destdir, silent=True)
    rows = []
    for gsm, s in g.gsms.items():
        md = s.metadata
        row = {"gsm": gsm, "title": md.get("title", [""])[0],
               "source": md.get("source_name_ch1", [""])[0]}
        for ch in md.get("characteristics_ch1", []):
            if ":" in ch:
                k, v = ch.split(":", 1)
                row[k.strip().lower()] = v.strip()
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_panel(genes: Sequence[str] = PANEL27) -> pd.DataFrame:
    """(Colab helper, needs internet + `pip install mygene`) current symbol, name and biotype of
    each panel gene, to verify the 'protein-coding' claim (e.g. PRSS42 -> PRSS42P, pseudogene)."""
    import mygene
    mg = mygene.MyGeneInfo()
    res = mg.querymany(list(genes), scopes="symbol,alias", fields="symbol,name,type_of_gene",
                       species="human", as_dataframe=True, returnall=False)
    return res.reset_index()[["query", "symbol", "name", "type_of_gene"]]


# ----------------------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------------------
def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(N)
    out[J] = T + 1
    return out


def _fast_delong(preds: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014) fast DeLong.  preds: (k, n) with the m positives first."""
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)); sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_ci(y, p, alpha: float = 0.05) -> Tuple[float, float, float, float]:
    """AUC with DeLong 100(1-alpha)% CI.  Returns (auc, lo, hi, se)."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    aucs, cov = _fast_delong(p[order][None, :], m)
    auc = float(aucs[0]); se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z = stats.norm.ppf(1 - alpha / 2)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se), se


def delong_paired(y, p1, p2, margin: Optional[float] = None) -> Dict:
    """Paired DeLong comparison of two correlated ROC curves (AUC1 - AUC2).
    If `margin` is given, also a TOST equivalence test (equivalent if the 90% CI of the
    difference lies inside (-margin, +margin))."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    P = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    aucs, cov = _fast_delong(P, m)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = math.sqrt(max(var, 1e-12))
    z = diff / se
    res = dict(auc1=float(aucs[0]), auc2=float(aucs[1]), diff=diff, se=se,
               lo95=diff - 1.959964 * se, hi95=diff + 1.959964 * se,
               lo90=diff - 1.644854 * se, hi90=diff + 1.644854 * se,
               p=float(2 * stats.norm.sf(abs(z))))
    if margin is not None:
        p_lower = float(stats.norm.sf((diff + margin) / se))   # H0: diff <= -margin
        p_upper = float(stats.norm.cdf((diff - margin) / se))  # H0: diff >= +margin
        res.update(margin=margin, p_tost=max(p_lower, p_upper),
                   equivalent=bool(max(p_lower, p_upper) < 0.05))
    return res


def bootstrap_auc_ci(y, p, B: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Stratified bootstrap percentile CI for the AUC."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals[b] = roc_auc_score(y[idx], p[idx])
    return (float(roc_auc_score(y, p)), float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


def binary_metrics(y, p, thr: float) -> Dict:
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(threshold=thr, sensitivity=sens, specificity=spec,
                ppv=tp / (tp + fp) if tp + fp else np.nan,
                npv=tn / (tn + fn) if tn + fn else np.nan,
                accuracy=(tp + tn) / len(y), balanced_accuracy=np.nanmean([sens, spec]),
                mcc=matthews_corrcoef(y, yhat), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    t = thr[j]
    return float(min(max(t, 0.0), 1.0)) if np.isfinite(t) else 0.5


def epsilon_squared(H, n):
    """Rank-based effect size for Kruskal-Wallis (Tomczak & Tomczak 2014): H / (n - 1)."""
    return np.asarray(H, float) / (n - 1)


def rank_corr_matrix(A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
    """Spearman correlation between columns of A and columns of B (vectorised)."""
    def rz(M):
        R = np.apply_along_axis(stats.rankdata, 0, M)
        R = (R - R.mean(0)) / (R.std(0) + 1e-12)
        return R
    RA = rz(A)
    RB = RA if B is None else rz(B)
    return RA.T @ RB / A.shape[0]


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


# ----------------------------------------------------------------------------------------
# Formatting helpers for LaTeX macros
# ----------------------------------------------------------------------------------------
def _num(x, nd: int) -> str:
    if x is None:
        return "n/a"
    try:
        if np.isnan(x):
            return "n/a"
    except TypeError:
        pass
    return f"{x:.{nd}f}".replace("-", "$-$")


def f4(x) -> str:
    return _num(x, 4)


def f3(x) -> str:
    return _num(x, 3)


def f2(x) -> str:
    return _num(x, 2)


def ci(lo, hi, nd=4) -> str:
    return f"{lo:.{nd}f}--{hi:.{nd}f}"


def pp(x) -> str:
    """Difference on the AUC scale -> signed percentage points (LaTeX-safe minus sign)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{100 * x:+.2f}".replace("-", "$-$")


def pfmt(p, floor: Optional[float] = 0.001) -> str:
    """p-value for the text.  Values below `floor` print as '<0.001' (journal style);
    use floor=None for an exact scientific value (e.g. the site x label chi-square)."""
    if p is None or np.isnan(p):
        return "n/a"
    if p <= 0:
        return "$<10^{-300}$"
    if floor is not None and p < floor:
        return f"$<{floor:g}$"
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def pmac(p, floor: Optional[float] = 0.001) -> str:
    """p-value as a complete LaTeX expression for the text: '$p = 0.012$', '$p < 0.001$'."""
    v = pfmt(p, floor)
    if v == "n/a":
        return "$p$ = n/a"
    v = v.strip("$")
    if v.startswith("<"):
        return f"$p {v[0]} {v[1:]}$"
    return f"$p = {v}$"


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def tex_stack(*lines, align: str = "c") -> str:
    """A table cell (or header) on several lines: \\begin{tabular}[c]{@{}c@{}}a\\\\b\\end{tabular}."""
    return "\\begin{tabular}[c]{@{}" + align + "@{}}" + "\\\\".join(lines) + "\\end{tabular}"


def tex_auc_ci(a, lo, hi, nd: int = 3) -> str:
    """AUC on the first line and its CI, smaller, on the second line."""
    return tex_stack(f"{a:.{nd}f}", f"\\scriptsize ({lo:.{nd}f}--{hi:.{nd}f})")


def write_tex_table(path: str, rows: List[List[str]], headers: List[str], colspec: Optional[str] = None,
                    size: str = "\\small"):
    """Write a booktabs tabular (cells already LaTeX-formatted) that the manuscript \\input's."""
    colspec = colspec or ("l" + "r" * (len(headers) - 1))
    out = [size, "\\begin{tabular}{" + colspec + "}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(" & ".join(str(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


# ----------------------------------------------------------------------------------------
# The Run object: data + config + outputs
# ----------------------------------------------------------------------------------------
class Run:
    def __init__(self, cfg: Dict, data: TEPData, select_fn: Optional[Callable] = None):
        self.cfg = cfg
        self.d = data
        self.select_fn = select_fn or six_stage_select
        self.out = cfg["out_dir"]
        for sub in ["tables", "figures", "cache"]:
            os.makedirs(os.path.join(self.out, sub), exist_ok=True)
        self.macros: Dict[str, str] = {}
        self.aucs: List[Dict] = []
        self.cache: Dict = {}
        self._log = open(os.path.join(self.out, "log.txt"), "a")
        if cfg.get("external_panel"):
            P = [g for g in cfg["panel"] if g in data.X.columns]
            self.cache["ext_shared"] = [g for g in cfg["external_panel"] if g in P]
            self.cache["ext_missing"] = [g for g in P if g not in self.cache["ext_shared"]]
        self.log(f"=== run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        self.log(f"discovery: {data.X.shape[0]} samples x {data.X.shape[1]} genes; "
                 f"NSCLC={int(data.y.sum())}, healthy={int((1 - data.y).sum())}")
        if data.X_ext is not None:
            self.log(f"external: {data.X_ext.shape[0]} samples; NSCLC={int(data.y_ext.sum())}")

    # ---- logging / saving ----
    def log(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self._log.write(msg + "\n"); self._log.flush()

    def table(self, df: pd.DataFrame, name: str, index: bool = False):
        path = os.path.join(self.out, "tables", name + ".csv")
        df.to_csv(path, index=index)
        return path

    def fig(self, fig, name: str):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(self.out, "figures", f"{name}.{ext}"), dpi=300,
                        bbox_inches="tight")
        plt.close(fig)

    def macro(self, name: str, value):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"LaTeX macro names may contain letters only: {name}")
        self.macros[name] = str(value)

    def register_auc(self, level: str, analysis: str, auc: float, lo: float, hi: float,
                     method: str, n: int, note: str = "", sd: Optional[float] = None):
        self.aucs.append(dict(level=level, analysis=analysis, auc=auc, ci_lo=lo, ci_hi=hi,
                              ci_method=method, n=n, fold_sd=sd, note=note))

    def write_macros(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.out, "results_macros.tex")
        with open(path, "w") as fh:
            fh.write("% Auto-generated by tep_analysis.py - do not edit by hand.\n")
            fh.write(f"% {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for k in sorted(self.macros):
                fh.write(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{self.macros[k]}}}\n")
        self.log(f"wrote {len(self.macros)} macros -> {path}")
        return path

    # ---- data helpers ----
    @property
    def vumc(self) -> np.ndarray:
        return (self.d.site == self.cfg["vumc_site"]).values

    def panel_in(self, X: pd.DataFrame, panel: Optional[Sequence[str]] = None) -> List[str]:
        panel = panel or self.cfg["panel"]
        miss = [g for g in panel if g not in X.columns]
        if miss:
            self.log(f"  note: {len(miss)} panel genes not in matrix: {miss}")
        return [g for g in panel if g in X.columns]

    @property
    def is_counts(self) -> bool:
        """True if the ML matrix looks like raw counts (integers with varying library size).
        Then statistics and correlations use CPM so that sequencing depth does not make every
        gene correlate with every other gene.  The classifier input is never changed."""
        if "_is_counts" not in self.cache:
            mode = self.cfg.get("normalize_for_stats", "auto")
            if self.cfg["values_are_log"] or mode is False:
                self.cache["_is_counts"] = False
            elif mode is True:
                self.cache["_is_counts"] = True
            else:
                v = self.d.X.values[:, : min(3000, self.d.X.shape[1])]
                libs = self.d.X.sum(axis=1)
                self.cache["_is_counts"] = bool(np.allclose(v, np.round(v)) and libs.std() / libs.mean() > 0.05)
            self.log(f"  statistics use {'CPM-normalised counts' if self.cache['_is_counts'] else 'the matrix as supplied'}")
        return self.cache["_is_counts"]

    def rawX(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Non-log expression on which group means / log2FC are computed (CPM for raw counts)."""
        X = self.d.X if X is None else X
        if self.cfg["values_are_log"]:
            return 2.0 ** X - 1.0
        if self.is_counts:
            lib = X.sum(axis=1).replace(0, 1)
            return X.div(lib, axis=0) * 1e6
        return X

    def logX(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        X = self.d.X if X is None else X
        if self.cfg["values_are_log"]:
            return X
        return np.log2(self.rawX(X).clip(lower=0) + 1.0)

    def xgb(self, n_jobs: Optional[int] = None):
        params = dict(self.cfg["xgb_params"])
        nj = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
        return XGBClassifier(**params, n_jobs=nj, verbosity=0)

    def folds(self, y: Optional[np.ndarray] = None, seed: Optional[int] = None, k=None):
        y = self.d.y.values if y is None else np.asarray(y)
        k = k or self.cfg["cv_folds"]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.cfg["seed"] if seed is None else seed)
        return list(skf.split(np.zeros(len(y)), y))

    def repeated_folds(self, y=None, R=None):
        R = R or self.cfg["cv_repeats"]
        return [self.folds(y, seed=self.cfg["seed"] + r) for r in range(R)]


def tier2_genes(run: "Run") -> List[str]:
    """Tier-2 pool exactly as defined by the paper's Stage 1: Mann-Whitney U on the processed
    matrix, BH, padj < 0.05 and |log2((mean_NSCLC+1e-6)/(mean_healthy+1e-6))| > 0.5 (3,557 genes
    in GSE89843).  Cached."""
    if "_tier2" not in run.cache:
        y = run.d.y.values
        V = run.d.X.values
        A, B = V[y == 1], V[y == 0]
        keep = ~((A.std(0) == 0) & (B.std(0) == 0))
        p = stats.mannwhitneyu(A[:, keep], B[:, keep], axis=0, alternative="two-sided").pvalue
        lfc = np.log2((A[:, keep].mean(0) + 1e-6) / (B[:, keep].mean(0) + 1e-6))
        q = bh(p)
        g = run.d.X.columns[keep]
        run.cache["_tier2"] = list(g[(q < run.cfg["selection"]["tier2_padj"]) &
                                     (np.abs(lfc) > run.cfg["selection"]["tier2_lfc"])])
    return run.cache["_tier2"]


def cv_oof(X, y, folds, factory: Callable) -> Tuple[np.ndarray, List[float]]:
    """Out-of-fold predicted probabilities and per-fold AUCs."""
    X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y = np.asarray(y).astype(int)
    oof = np.full(len(y), np.nan)
    aucs = []
    for tr, te in folds:
        m = factory()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X[te])
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))
    return oof, aucs


def fit_predict(Xtr, ytr, Xte, factory: Callable) -> np.ndarray:
    m = factory()
    m.fit(np.asarray(Xtr), np.asarray(ytr).astype(int))
    if hasattr(m, "predict_proba"):
        return m.predict_proba(np.asarray(Xte))[:, 1]
    return m.decision_function(np.asarray(Xte))


def _sk_version() -> Tuple[int, int]:
    import sklearn
    v = re.findall(r"\d+", sklearn.__version__)
    return int(v[0]), int(v[1])


def make_logreg(l1_ratio: float, C: float, max_iter: int = 3000, seed: int = 42,
                warm_start: bool = False) -> LogisticRegression:
    """L1 / elastic-net logistic regression that works on old and new scikit-learn."""
    if _sk_version() >= (1, 8):
        return LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=max_iter,
                                  random_state=seed, warm_start=warm_start, tol=1e-3)
    pen = "l1" if l1_ratio >= 1.0 else ("l2" if l1_ratio <= 0 else "elasticnet")
    kw = dict(penalty=pen, C=C, solver="saga", max_iter=max_iter, random_state=seed,
              warm_start=warm_start, tol=1e-3)
    if pen == "elasticnet":
        kw["l1_ratio"] = l1_ratio
    return LogisticRegression(**kw)


# ========================================================================================
# Feature-selection methods
# ========================================================================================
NONCODING_PATTERNS = [r"^LINC\d", r"^LOC\d", r"^MIR\d", r"^MIR\w+HG$", r"^SNOR[AD]\d", r"^SCARNA",
                      r"^RNU\d", r"^RNA5", r"^RN7S", r"^Y_RNA", r"^ENSG\d", r"-AS\d", r"-IT\d",
                      r"-DT$", r"-OT\d*$", r"^RP\d+-", r"^CT[ABCD]-", r"^A[CLP]\d{6}"]


def protein_coding_filter(genes: Sequence[str], run: Optional["Run"] = None) -> List[str]:
    """Stage 3.  Uses an HGNC protein-coding list if cfg['protein_coding_list'] is given,
    otherwise name patterns of non-coding transcripts (as described in the manuscript)."""
    path = run.cfg.get("protein_coding_list") if run is not None else None
    if path:
        if "_pc_set" not in run.cache:
            with open(path) as fh:
                run.cache["_pc_set"] = {l.strip() for l in fh if l.strip()}
        pc = run.cache["_pc_set"]
        return [g for g in genes if g in pc]
    pat = re.compile("|".join(NONCODING_PATTERNS))
    return [g for g in genes if not pat.search(g)]


def discretize(X: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Peng's 3-state discretisation: below mean-alpha*sd -> 0, above mean+alpha*sd -> 2."""
    mu = X.mean(0); sd = X.std(0) + 1e-12
    D = np.ones(X.shape, dtype=np.int8)
    D[X < mu - alpha * sd] = 0
    D[X > mu + alpha * sd] = 2
    return D


def _mi_cols_vec(Donehot: List[np.ndarray], v: np.ndarray, n: int) -> np.ndarray:
    """Mutual information between every column of a discretised matrix (given as one-hot
    boolean slabs) and one discrete vector v."""
    G = Donehot[0].shape[1]
    mi = np.zeros(G)
    pD = [d.mean(0) for d in Donehot]
    for val in np.unique(v):
        rows = (v == val)
        pv = rows.mean()
        for i, d in enumerate(Donehot):
            pij = d[rows].sum(0) / n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = pij * np.log(pij / (pD[i] * pv))
            mi += np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return mi


def mrmr_rank(X: np.ndarray, y: np.ndarray, K: int, scheme: str = "MID",
              alpha: float = 0.5) -> List[int]:
    """Max-relevance min-redundancy ranking (Peng et al. 2005) on discretised data.
    Returns the column indices of the first K selected features."""
    D = discretize(np.asarray(X, float), alpha)
    n, G = D.shape
    onehot = [(D == i) for i in range(3)]
    rel = _mi_cols_vec(onehot, np.asarray(y).astype(int), n)
    K = min(K, G)
    selected = [int(np.argmax(rel))]
    red_sum = np.zeros(G)
    for _ in range(1, K):
        red_sum += _mi_cols_vec(onehot, D[:, selected[-1]], n)
        red = red_sum / len(selected)
        score = rel - red if scheme.upper() == "MID" else rel / (red + 1e-6)
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return selected


def _minmax(v) -> np.ndarray:
    v = np.asarray(v, float)
    if np.all(~np.isfinite(v)) or np.nanmax(v) == np.nanmin(v):
        return np.zeros_like(v)
    return (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))


def _cv_choose_C(Z: np.ndarray, y: np.ndarray, l1_ratio: float, n_C: int, k: int, seed: int):
    """Choose C for an L1/EN logistic model by stratified k-fold AUC (warm-started path)."""
    from sklearn.svm import l1_min_c
    cmin = l1_min_c(Z, y, loss="log") * 1.2
    Cs = np.logspace(np.log10(cmin), np.log10(cmin) + 3, n_C)
    folds = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(Z, y))
    sc = np.zeros((len(folds), len(Cs)))
    for i, (tr, te) in enumerate(folds):
        clf = make_logreg(l1_ratio, Cs[0], seed=seed, warm_start=True)
        for j, C in enumerate(Cs):
            clf.set_params(C=C)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(Z[tr], y[tr])
            sc[i, j] = roc_auc_score(y[te], clf.decision_function(Z[te])) if clf.coef_.any() else 0.5
    mean = sc.mean(0)
    # one-standard-error rule towards sparser models
    se = sc.std(0, ddof=1) / np.sqrt(len(folds))
    best = int(np.argmax(mean))
    ok = np.where(mean >= mean[best] - se[best])[0]
    return float(Cs[ok.min()]), Cs, mean


def six_stage_select(X: pd.DataFrame, y, run: "Run", B: Optional[int] = None,
                     panel_size: Optional[int] = None, exclude: Sequence[str] = (),
                     return_details: bool = False):
    """Re-implementation of the manuscript's six-stage pipeline (Section 2.5).

    Stage 1  Mann-Whitney U + BH; keep padj < 0.05 and |log2FC| > 0.5   (Tier 2)
    Stage 2  LASSO  intersect  ElasticNet (logistic, C by 5-fold CV, 1-SE rule)
    Stage 3  protein-coding filter
    Stage 4  MRMR ranking of the Stage-3 pool (top 30)            } parallel evidence
    Stage 5  bootstrap LASSO selection frequency f_g (stable: >= 0.60) }
    Stage 6  composite score S = .20 stat + .15 effect + .15 LASSO + .30 boot + .10 PPI + .10 FI
             (PPI weight re-distributed if no STRING edge list is given); artefact QC;
             redundancy control; top `panel_size` stable genes.

    NOTE: written from the paper's description.  Replace with your own function for exact
    reproduction (same signature)."""
    cfg = run.cfg["selection"]
    seed = run.cfg["seed"]
    B = B or cfg["boot_B"]
    panel_size = panel_size or cfg["panel_size"]
    y = np.asarray(y).astype(int)
    X = X.loc[:, [c for c in X.columns if c not in set(exclude)]]
    X = X.loc[:, (X > 0).any(axis=0)]
    L = run.logX(X); R = run.rawX(X)
    # ---- Stage 1
    A, Bm = L.values[y == 1], L.values[y == 0]
    pv = stats.mannwhitneyu(A, Bm, axis=0, alternative="two-sided").pvalue
    q = bh(pv)
    lfc = np.log2(R.values[y == 1].mean(0) + 1) - np.log2(R.values[y == 0].mean(0) + 1)
    s1 = np.where((q < cfg["tier2_padj"]) & (np.abs(lfc) > cfg["tier2_lfc"]))[0]
    if len(s1) > cfg["max_stage1"]:
        s1 = s1[np.argsort(pv[s1])[: cfg["max_stage1"]]]
    genes1 = L.columns[s1]
    if len(genes1) < 5:
        return ([], None) if return_details else []
    Z = StandardScaler().fit_transform(L[genes1].values)
    # ---- Stage 2
    C1, _, _ = _cv_choose_C(Z, y, 1.0, cfg["n_C"], cfg["inner_folds"], seed)
    C2, _, _ = _cv_choose_C(Z, y, 0.5, cfg["n_C"], cfg["inner_folds"], seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        las = make_logreg(1.0, C1, seed=seed).fit(Z, y)
        en = make_logreg(0.5, C2, seed=seed).fit(Z, y)
    nz1 = set(genes1[np.abs(las.coef_[0]) > 1e-8]); nz2 = set(genes1[np.abs(en.coef_[0]) > 1e-8])
    s2 = [g for g in genes1 if g in nz1 and g in nz2] or [g for g in genes1 if g in nz1 | nz2]
    coef = dict(zip(genes1, np.abs(las.coef_[0])))
    # ---- Stage 3
    s3 = protein_coding_filter(s2, run)
    if len(s3) == 0:
        return ([], None) if return_details else []
    L3 = L[s3].values
    Z3 = StandardScaler().fit_transform(L3)
    # ---- Stage 4 (MRMR ranking, top 30)
    order = mrmr_rank(L3, y, K=min(len(s3), cfg["mrmr_top"]))
    mrmr_pos = {s3[i]: r for r, i in enumerate(order)}
    # ---- Stage 5 (bootstrap stability)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    freq = np.zeros(len(s3))
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_logreg(1.0, C1, seed=seed, max_iter=1000).fit(Z3[idx], y[idx])
        freq += (np.abs(m.coef_[0]) > 1e-8)
    freq /= B
    # ---- Stage 6 (composite score)
    idx3 = [list(genes1).index(g) for g in s3]
    s_stat = _minmax(-np.log10(np.clip(q[s1][idx3], 1e-300, 1)))
    s_eff = _minmax(np.abs(lfc[s1][idx3]))
    s_las = _minmax([coef.get(g, 0.0) for g in s3])
    s_boot = freq
    w = np.array(cfg["weights"], float)
    deg = np.zeros(len(s3))
    if run.cfg.get("ppi_edges"):
        if "_ppi" not in run.cache:
            E = pd.read_csv(run.cfg["ppi_edges"], sep=None, engine="python")
            run.cache["_ppi"] = pd.concat([E.iloc[:, 0], E.iloc[:, 1]]).value_counts()
        deg = run.cache["_ppi"].reindex(s3).fillna(0).values
    else:
        w[4] = 0.0
    s_ppi = _minmax(deg)
    fi_model = run.xgb(n_jobs=1).fit(L3, y)
    s_fi = _minmax(fi_model.feature_importances_)
    w = w / w.sum()
    S = w[0] * s_stat + w[1] * s_eff + w[2] * s_las + w[3] * s_boot + w[4] * s_ppi + w[5] * s_fi
    det = pd.DataFrame(dict(gene=s3, s_stat=s_stat, s_effect=s_eff, s_lasso=s_las, boot_freq=freq,
                            s_ppi=s_ppi, s_fi=s_fi, score=S,
                            mrmr_rank=[mrmr_pos.get(g, np.nan) for g in s3]))
    det["stable"] = det.boot_freq >= cfg["boot_threshold"]
    art = re.compile("|".join(cfg["artefact_regex"])) if cfg["artefact_regex"] else None
    det["artefact"] = det.gene.map(lambda g: bool(art.search(g)) if art else False)
    det = det.sort_values(["score", "mrmr_rank"], ascending=[False, True]).reset_index(drop=True)
    cand = det[det.stable & ~det.artefact]
    chosen: List[str] = []
    Lc = L[cand.gene.tolist()] if len(cand) else None
    for g in cand.gene:
        if len(chosen) >= panel_size:
            break
        if chosen:
            rho = rank_corr_matrix(Lc[[g]].values, Lc[chosen].values)
            if np.nanmax(np.abs(rho)) > cfg["redundancy_rho"]:
                continue
        chosen.append(g)
    det["selected"] = det.gene.isin(chosen)
    return (chosen, det) if return_details else chosen


def sheng_mrmr_ifs_svm(X: pd.DataFrame, y, run: "Run", return_curve: bool = False):
    """Re-implementation of Sheng, Dong & Xie (2018): unsupervised expression pre-filter
    (to the size of their 4,722-gene matrix) -> mRMR ranking -> incremental feature selection
    with an SVM (inner CV, MCC) -> final SVM.  Returns (genes, fitted_pipeline[, ifs_curve])."""
    sc = run.cfg["sheng"]
    seed = run.cfg["seed"]
    y = np.asarray(y).astype(int)
    L = run.logX(X)
    L = L.loc[:, L.var(axis=0) > 0]
    if L.shape[1] > sc["prefilter_n"]:
        keep = L.mean(0).sort_values(ascending=False).index[: sc["prefilter_n"]]
        L = L[keep]
    order = mrmr_rank(L.values, y, K=sc["mrmr_K"], scheme=sc["mrmr_scheme"],
                      alpha=sc["discretize_alpha"])
    ranked = L.columns[order]
    folds = list(StratifiedKFold(sc["inner_folds"], shuffle=True, random_state=seed).split(L, y))
    ks = list(range(1, min(sc["ifs_max"], len(ranked)) + 1, sc["ifs_step"]))
    curve = []
    for k in ks:
        Xk = L[ranked[:k]].values
        yhat = np.zeros(len(y), int)
        for tr, te in folds:
            mdl = make_pipeline(StandardScaler(), SVC(kernel=sc["svm_kernel"], C=sc["svm_C"]))
            mdl.fit(Xk[tr], y[tr])
            yhat[te] = mdl.predict(Xk[te])
        curve.append((k, matthews_corrcoef(y, yhat), (yhat == y).mean()))
    curve = pd.DataFrame(curve, columns=["k", "mcc", "accuracy"])
    kbest = int(curve.sort_values(["mcc", "k"], ascending=[False, True]).iloc[0].k)
    genes = list(ranked[:kbest])
    final = make_pipeline(StandardScaler(), SVC(kernel=sc["svm_kernel"], C=sc["svm_C"]))
    final.fit(L[genes].values, y)
    return (genes, final, curve) if return_curve else (genes, final)


def call_select(run: "Run", X: pd.DataFrame, y, B: Optional[int] = None) -> List[str]:
    """Call the selection function, passing B (bootstrap resamples) only if it accepts it."""
    if B is None:
        return list(run.select_fn(X, y, run))
    try:
        return list(run.select_fn(X, y, run, B=B))
    except TypeError:
        return list(run.select_fn(X, y, run))


def nested_cv(run: "Run", X: pd.DataFrame, y, folds, method: str = "six_stage",
              B: Optional[int] = None) -> Dict:
    """Nested CV: the whole selection is re-run inside each outer training fold.
    method: 'six_stage' (run.select_fn + XGBoost) or 'sheng' (mRMR-IFS-SVM)."""
    y = np.asarray(y).astype(int)
    oof = np.full(len(y), np.nan); yhat = np.zeros(len(y), int)
    panels, aucs = [], []
    for f, (tr, te) in enumerate(folds):
        t0 = time.time()
        Xtr = X.iloc[tr]
        if method == "six_stage":
            genes = call_select(run, Xtr, y[tr], B=B)
            m = run.xgb().fit(Xtr[genes].values, y[tr])
            p = m.predict_proba(X.iloc[te][genes].values)[:, 1]
            oof[te] = p; yhat[te] = (p >= 0.5).astype(int)
        else:
            genes, mdl = sheng_mrmr_ifs_svm(Xtr, y[tr], run)
            Lte = run.logX(X.iloc[te])[genes].values
            oof[te] = mdl.decision_function(Lte); yhat[te] = mdl.predict(Lte)
        panels.append(list(genes))
        aucs.append(roc_auc_score(y[te], oof[te]))
        run.log(f"    [{method}] outer fold {f + 1}: {len(genes)} genes, AUC={aucs[-1]:.4f} "
                f"({time.time() - t0:.0f}s)")
    return dict(oof=oof, yhat=yhat, panels=panels, fold_aucs=aucs)


# ========================================================================================
# R0  Core estimates with 95% CIs (also checks that the data reproduce the paper)
# ========================================================================================
def r0_core_estimates(run: Run) -> pd.DataFrame:
    """Re-computes the manuscript's core AUCs with 95% CIs (minor point: CIs everywhere).
    Expected (paper): fixed 0.9638, site-stratified 0.9582, Vumc-only 0.9653,
    held-out Vumc 0.7573, GSE68086 0.7801.  Small differences = different random split."""
    d, cfg = run.d, run.cfg
    run.log("[R0] core estimates with 95% CIs")
    P = run.panel_in(d.X)
    X, y = d.X[P].values, d.y.values
    rows = []
    # fixed-panel 5-fold CV
    oof, fa = cv_oof(X, y, run.folds(), run.xgb)
    run.cache["oof_P27"] = oof
    a, lo, hi, _ = delong_ci(y, oof)
    run.register_auc("Within cohort", "Fixed-panel 5-fold CV (pooled OOF)", a, lo, hi, "DeLong",
                     len(y), sd=np.std(fa, ddof=1))
    run.macro("CoreFixedAUC", f4(a)); run.macro("CoreFixedCI", ci(lo, hi))
    run.macro("CoreFixedFoldMean", f4(np.mean(fa))); run.macro("CoreFixedFoldSD", f4(np.std(fa, ddof=1)))
    rows.append(("Fixed-panel 5-fold CV", a, lo, hi, np.mean(fa), np.std(fa, ddof=1), len(y)))
    # sensitivity 1: library-size-normalised input (log2 CPM) instead of the processed values
    if run.is_counts:
        oof_c, fa_c = cv_oof(run.logX()[P].values, y, run.folds(), run.xgb)
        a_c, lo_c, hi_c, _ = delong_ci(y, oof_c)
        run.macro("CoreCpmAUC", f4(a_c)); run.macro("CoreCpmCI", ci(lo_c, hi_c))
        run.macro("CoreCpmFoldMean", f4(np.mean(fa_c)))
        rows.append(("Fixed-panel 5-fold CV, log2-CPM input", a_c, lo_c, hi_c, np.mean(fa_c), np.std(fa_c, ddof=1), len(y)))
    # sensitivity 2: how much does the fold assignment alone move the fold-mean AUC?
    n_perm = int(cfg.get("n_order_perms", 0) or 0)
    if n_perm:
        rng_o = np.random.default_rng(cfg["seed"])
        fm = []
        for _ in range(n_perm):
            perm = rng_o.permutation(len(y))
            _, fa_o = cv_oof(X[perm], y[perm], run.folds(y[perm]), run.xgb)
            fm.append(np.mean(fa_o))
        fm = np.array(fm)
        pd.DataFrame({"fold_mean_auc": fm}).to_csv(os.path.join(run.out, "tables", "R0_fold_assignment_sensitivity.csv"), index=False)
        run.macro("CoreOrderN", str(n_perm)); run.macro("CoreOrderMin", f4(fm.min()))
        run.macro("CoreOrderMax", f4(fm.max())); run.macro("CoreOrderMean", f4(fm.mean()))
        run.log(f"  fold-mean AUC over {n_perm} random fold assignments: {fm.mean():.4f} ({fm.min():.4f}-{fm.max():.4f})")
    # site-stratified 5-fold CV (strata = site x label with >= k members)
    k = cfg["cv_folds"]
    strata = d.site.astype(str) + "|" + d.y.astype(str)
    ok = strata.map(strata.value_counts()).values >= k
    skf = StratifiedKFold(k, shuffle=True, random_state=cfg["seed"])
    fs = list(skf.split(np.zeros(ok.sum()), strata.values[ok]))
    oof_s, fa_s = cv_oof(X[ok], y[ok], fs, run.xgb)
    a, lo, hi, _ = delong_ci(y[ok], oof_s)
    run.register_auc("Site-aware audit", "Site-stratified 5-fold CV", a, lo, hi, "DeLong",
                     int(ok.sum()), sd=np.std(fa_s, ddof=1))
    run.macro("CoreSiteStratAUC", f4(a)); run.macro("CoreSiteStratCI", ci(lo, hi))
    run.macro("CoreSiteStratN", str(int(ok.sum())))
    rows.append(("Site-stratified 5-fold CV", a, lo, hi, np.mean(fa_s), np.std(fa_s, ddof=1), int(ok.sum())))
    # Vumc-only CV
    v = run.vumc
    oof_v, fa_v = cv_oof(X[v], y[v], run.folds(y[v]), run.xgb)
    run.cache["oof_vumc_P27"] = oof_v
    a, lo, hi, _ = delong_ci(y[v], oof_v)
    run.register_auc("Site-aware audit", "Vumc-only 5-fold CV (site constant)", a, lo, hi, "DeLong",
                     int(v.sum()), sd=np.std(fa_v, ddof=1))
    run.macro("CoreVumcAUC", f4(a)); run.macro("CoreVumcCI", ci(lo, hi))
    rows.append(("Vumc-only 5-fold CV", a, lo, hi, np.mean(fa_v), np.std(fa_v, ddof=1), int(v.sum())))
    # held-out site: train on all other sites, test on Vumc
    p = fit_predict(X[~v], y[~v], X[v], run.xgb)
    a, lo, hi, _ = delong_ci(y[v], p)
    run.register_auc("Site-aware audit", "Held-out site (train other sites, test Vumc)", a, lo, hi,
                     "DeLong", int(v.sum()))
    run.macro("CoreHeldOutAUC", f4(a)); run.macro("CoreHeldOutCI", ci(lo, hi))
    rows.append(("Held-out site (test = Vumc)", a, lo, hi, np.nan, np.nan, int(v.sum())))
    # external frozen classifier on the shared genes
    if d.X_ext is not None:
        base = cfg.get("external_panel") or P
        shared = [g for g in P if g in base and g in d.X_ext.columns]
        missing = [g for g in P if g not in shared]
        run.cache["ext_shared"], run.cache["ext_missing"] = shared, missing
        pe = fit_predict(d.X[shared].values, y, d.X_ext[shared].values, run.xgb)
        run.cache["p_ext_frozen"] = pe
        ye = d.y_ext.values
        a, lo, hi, _ = delong_ci(ye, pe)
        _, blo, bhi = bootstrap_auc_ci(ye, pe, cfg["bootstrap_B"], cfg["seed"])
        run.register_auc("Independent platelet cohort", f"GSE68086 frozen ({len(shared)} genes)", a,
                         blo, bhi, "bootstrap", len(ye))
        run.macro("CoreExtAUC", f4(a)); run.macro("CoreExtCIboot", ci(blo, bhi))
        run.macro("CoreExtCIdelong", ci(lo, hi)); run.macro("CoreExtNgenes", str(len(shared)))
        run.macro("CoreExtMissing", ", ".join(missing) if missing else "none")
        rows.append((f"GSE68086 frozen ({len(shared)} genes)", a, blo, bhi, np.nan, np.nan, len(ye)))
    df = pd.DataFrame(rows, columns=["analysis", "auc", "ci_lo", "ci_hi", "fold_mean", "fold_sd", "n"])
    run.table(df, "R0_core_estimates")
    run.log(df.round(4).to_string(index=False))
    return df


# ========================================================================================
# R1  Site confounding
# ========================================================================================
def r1_site_table(run: Run) -> pd.DataFrame:
    """Site x label cross-tabulation and chi-square test (reproduces Table 6)."""
    d = run.d
    ct = pd.crosstab(d.site, d.y).rename(columns={0: "healthy", 1: "NSCLC"})
    for c in ["healthy", "NSCLC"]:
        if c not in ct.columns:
            ct[c] = 0
    chi2, p, dof, _ = stats.chi2_contingency(ct[["healthy", "NSCLC"]].values)
    ct["total"] = ct.sum(1)
    run.table(ct.reset_index(), "R1_site_by_label")
    run.macro("SiteChi", f2(chi2)); run.macro("SiteChiDf", str(dof)); run.macro("SiteChiP", pmac(p, None))
    run.log(f"[R1] site x label chi2={chi2:.2f}, df={dof}, p={p:.2e}")
    return ct


def _site_groups(run: Run, cls: int, min_n: int = 3) -> Dict[str, np.ndarray]:
    """Sample masks per site within one class; single-sample codes are pooled
    ('singleton batch', manuscript Section 3.1)."""
    d = run.d
    m_cls = (d.y.values == cls)
    counts = d.site[m_cls].value_counts()
    groups = {}
    single = counts[counts == 1].index
    for s, n in counts.items():
        if n >= min_n:
            groups[s] = m_cls & (d.site.values == s)
    if len(single) >= min_n:
        groups["singleton-batch"] = m_cls & d.site.isin(single).values
    return groups


def r1_gene_effects(run: Run, genome_wide: bool = True) -> pd.DataFrame:
    """Gene-level disease effect (within Vumc, site held constant) versus site effect
    (across sites within each class).  Epsilon-squared (rank-based eta^2) for both."""
    d, cfg = run.d, run.cfg
    run.log("[R1] gene-level site vs disease effects")
    L = run.logX(); R = run.rawX()
    y = d.y.values; v = run.vumc
    genes = list(L.columns) if genome_wide else run.panel_in(L)
    Lg = L[genes].values
    # disease effect within Vumc
    A, Bv = Lg[v & (y == 1)], Lg[v & (y == 0)]
    mw = stats.mannwhitneyu(A, Bv, axis=0, alternative="two-sided")
    auc_v = mw.statistic / (A.shape[0] * Bv.shape[0])
    H_d = stats.kruskal(A, Bv, axis=0).statistic
    eps_d = epsilon_squared(H_d, A.shape[0] + Bv.shape[0])
    Rg = R[genes].values
    lfc_v = np.log2(Rg[v & (y == 1)].mean(0) + 1) - np.log2(Rg[v & (y == 0)].mean(0) + 1)
    # full-cohort effect (as in the paper)
    mw_f = stats.mannwhitneyu(Lg[y == 1], Lg[y == 0], axis=0, alternative="two-sided")
    lfc_f = np.log2(Rg[y == 1].mean(0) + 1) - np.log2(Rg[y == 0].mean(0) + 1)
    q_f = bh(mw_f.pvalue)
    # site effects within class
    out = {}
    for cls, nm in [(0, "healthy"), (1, "nsclc")]:
        grp = _site_groups(run, cls)
        arrs = [Lg[m] for m in grp.values()]
        n = sum(len(a) for a in arrs)
        if len(arrs) >= 2:
            kw = stats.kruskal(*arrs, axis=0)
            out[nm] = (epsilon_squared(kw.statistic, n), kw.pvalue, list(grp))
        else:
            out[nm] = (np.full(len(genes), np.nan), np.full(len(genes), np.nan), list(grp))
    df = pd.DataFrame(dict(gene=genes, log2fc_full=lfc_f, q_full=q_f, log2fc_vumc=lfc_v,
                           p_vumc=mw.pvalue, q_vumc_genomewide=bh(mw.pvalue), auc_vumc=auc_v,
                           eps2_disease_vumc=eps_d, eps2_site_healthy=out["healthy"][0],
                           p_site_healthy=out["healthy"][1], eps2_site_nsclc=out["nsclc"][0],
                           p_site_nsclc=out["nsclc"][1]))
    df["eps2_site_max"] = df[["eps2_site_healthy", "eps2_site_nsclc"]].max(axis=1)
    df["dominant"] = np.where(df.eps2_disease_vumc >= df.eps2_site_max, "disease", "site")
    df["direction_concordant"] = np.sign(df.log2fc_full) == np.sign(df.log2fc_vumc)
    df["tier2"] = df.gene.isin(set(tier2_genes(run)))
    P = run.panel_in(L)
    pan = df[df.gene.isin(P)].set_index("gene").loc[P].reset_index()
    pan["q_vumc_panel"] = bh(pan.p_vumc.values)
    # site-adjusted linear model (site fixed effects; label effect identified within Vumc)
    try:
        import statsmodels.api as sm
        site_c = d.site.copy()
        vc = site_c.value_counts()
        site_c[site_c.map(vc) < 3] = "small"
        Dm = pd.get_dummies(site_c, drop_first=True, dtype=float)
        Xd = sm.add_constant(pd.concat([pd.Series(y, index=d.X.index, name="NSCLC"), Dm], axis=1))
        betas, ps = [], []
        for g in P:
            fit = sm.OLS(L[g].values, Xd.values).fit(cov_type="HC3")
            betas.append(fit.params[1]); ps.append(fit.pvalues[1])
        pan["beta_site_adjusted"] = betas; pan["p_site_adjusted"] = ps
        pan["q_site_adjusted"] = bh(np.array(ps))
    except Exception as e:  # statsmodels missing
        run.log("  site-adjusted OLS skipped:", e)
        pan["q_site_adjusted"] = np.nan
    run.table(pan, "R1_panel_site_vs_disease")
    run.cache["r1_panel_table"] = pan
    rows = [[r.gene, "UP" if r.log2fc_full > 0 else "DOWN", f2(r.log2fc_vumc), pfmt(r.q_vumc_panel),
             f3(r.auc_vumc), f3(r.eps2_disease_vumc), f3(r.eps2_site_max), r.dominant] for r in pan.itertuples()]
    write_tex_table(os.path.join(run.out, "tables", "R1_panel_site_vs_disease.tex"), rows,
                    ["Gene", "Dir.", "log$_2$FC (Vumc)", "$q$ (Vumc)", "AUC (Vumc)",
                     "$\\varepsilon^2$ disease", "$\\varepsilon^2$ site", "Dominant"], "llrrrrrl", "\\footnotesize")
    if genome_wide:
        run.table(df, "R1_genomewide_site_vs_disease")
    # macros
    n = len(P)
    run.macro("GeneNpanel", str(n))
    run.macro("GeneVumcSig", str(int((pan.q_vumc_panel < 0.05).sum())))
    run.macro("GeneVumcSigGW", str(int((pan.q_vumc_genomewide < 0.05).sum())))
    run.macro("GeneDirConc", str(int(pan.direction_concordant.sum())))
    run.macro("GeneDisDom", str(int((pan.dominant == "disease").sum())))
    run.macro("GeneSiteDom", str(int((pan.dominant == "site").sum())))
    run.macro("GeneAdjSig", str(int((pan.q_site_adjusted < 0.05).sum())) if "q_site_adjusted" in pan else "n/a")
    run.macro("GeneMedEpsDis", f3(pan.eps2_disease_vumc.median()))
    run.macro("GeneMedEpsSite", f3(pan.eps2_site_max.median()))
    run.macro("GeneSiteDomList", ", ".join(pan.loc[pan.dominant == "site", "gene"]) or "none")
    run.macro("GeneNotSigList", ", ".join(pan.loc[pan.q_vumc_panel >= 0.05, "gene"]) or "none")
    if genome_wide:
        t2 = df[df.tier2 & ~df.gene.isin(P)]
        a_ = int((pan.dominant == "disease").sum()); b_ = n - a_
        c_ = int((t2.dominant == "disease").sum()); d_ = len(t2) - c_
        orat, pf = stats.fisher_exact([[a_, b_], [c_, d_]])
        run.macro("TierTwoN", f"{len(t2) + n:,}".replace(",", "{,}"))
        run.macro("TierTwoDisDomPct", f"{100 * c_ / max(len(t2), 1):.1f}")
        run.macro("PanelDisDomPct", f"{100 * a_ / n:.1f}")
        run.macro("DisDomFisherOR", f2(orat)); run.macro("DisDomFisherP", pmac(pf))
        if pf < 0.05 and a_ / n > c_ / max(len(t2), 1):
            v = "The multi-criteria selection therefore preferentially retained disease-dominant genes"
        elif pf < 0.05:
            v = "The panel was therefore enriched for site-dominant genes relative to other Tier~2 genes, which is a reason for caution"
        else:
            v = "The panel was therefore neither enriched nor depleted for disease-dominant genes relative to other Tier~2 genes"
        run.macro("GeneSiteVerdict", v)
        allg = df
        run.macro("GWSiteDomPct", f"{100 * (allg.dominant == 'site').mean():.1f}")
        # figure
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.scatter(t2.eps2_site_max, t2.eps2_disease_vumc, s=5, c="#B8B8B8", alpha=.5,
                   label="other Tier-2 genes", rasterized=True)
        ax.scatter(pan.eps2_site_max, pan.eps2_disease_vumc, s=28, c="#C0392B", label="27-gene panel",
                   edgecolor="k", lw=.4, zorder=3)
        for _, r in pan.iterrows():
            ax.annotate(r.gene, (r.eps2_site_max, r.eps2_disease_vumc), fontsize=6, xytext=(2, 2),
                        textcoords="offset points")
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], ls="--", c="k", lw=.8)
        ax.set_xlabel("Site effect, max within-class $\\varepsilon^2$")
        ax.set_ylabel("Disease effect within Vumc, $\\varepsilon^2$")
        ax.text(.97, .03, "site-dominant", transform=ax.transAxes, ha="right", fontsize=8, color="#555")
        ax.text(.03, .97, "disease-dominant", transform=ax.transAxes, va="top", fontsize=8, color="#555")
        ax.legend(frameon=False, fontsize=8, loc="center right")
        run.fig(fig, "R1_site_vs_disease_effects")
    run.log(pan[["gene", "q_vumc_panel", "auc_vumc", "eps2_disease_vumc", "eps2_site_max",
                 "dominant"]].round(4).to_string(index=False))
    return pan


def r1_vumc_rederivation(run: Run, vumc_panel: Optional[List[str]] = None) -> Dict:
    """Re-run the six-stage selection on Vumc ONLY (site constant), compare the genes with
    the original panel, and test transfer of a Vumc-only model to the 15 other site codes and
    to GSE68086."""
    d, cfg = run.d, run.cfg
    run.log("[R1] six-stage selection inside Vumc only")
    v = run.vumc; y = d.y.values
    Xv, yv = d.X.loc[v], y[v]
    if vumc_panel is None:
        t0 = time.time()
        vumc_panel = call_select(run, Xv, yv, B=cfg["selection"]["boot_B_full"])
        run.log(f"  Vumc-only panel: {len(vumc_panel)} genes ({time.time() - t0:.0f}s)")
    run.cache["vumc_panel"] = vumc_panel
    P = run.panel_in(d.X)
    ov = sorted(set(P) & set(vumc_panel))
    G = d.X.shape[1]
    p_hyper = stats.hypergeom.sf(len(ov) - 1, G, len(P), len(vumc_panel))
    exp_ov = len(P) * len(vumc_panel) / G
    # correlated substitutes (within-class co-expression inside Vumc)
    L = run.logX()
    Lv = L.loc[v]
    resid = Lv - Lv.groupby(yv).transform("mean")
    new = [g for g in vumc_panel if g not in P]
    sub_rho = {}
    if new:
        Rm = rank_corr_matrix(resid[new].values, resid[P].values)
        for i, g in enumerate(new):
            j = int(np.argmax(np.abs(Rm[i])))
            sub_rho[g] = (P[j], float(Rm[i, j]))
    rng = np.random.default_rng(cfg["seed"])
    pool = [g for g in L.columns if g not in set(P) | set(vumc_panel)]
    null = rng.choice(pool, size=min(1000, len(pool)), replace=False)
    Rn = rank_corr_matrix(resid[list(null)].values, resid[P].values)
    null_max = np.abs(Rn).max(1)
    frac_new = np.mean([abs(r) >= 0.5 for _, r in sub_rho.values()]) if sub_rho else np.nan
    # transfer: Vumc-trained -> 15 other site codes (no non-Vumc sample used for selection or training)
    pt = fit_predict(Xv[vumc_panel].values, yv, d.X.loc[~v, vumc_panel].values, run.xgb)
    a, lo, hi, _ = delong_ci(y[~v], pt)
    run.register_auc("Site-aware audit", "Vumc-derived panel, Vumc-trained -> other 15 site codes",
                     a, lo, hi, "DeLong", int((~v).sum()))
    # same with the original panel (selection saw all sites -> optimistic, reported for contrast)
    po = fit_predict(Xv[P].values, yv, d.X.loc[~v, P].values, run.xgb)
    a2, lo2, hi2, _ = delong_ci(y[~v], po)
    # within-Vumc CV with the Vumc panel (selection used all of Vumc -> fixed-panel estimate)
    oof_vp, _ = cv_oof(Xv[vumc_panel].values, yv, run.folds(yv), run.xgb)
    a3, lo3, hi3, _ = delong_ci(yv, oof_vp)
    res = dict(n_vumc_panel=len(vumc_panel), overlap=len(ov), overlap_genes=ov,
               jaccard=jaccard(P, vumc_panel), p_hyper=p_hyper, expected_overlap=exp_ov,
               frac_new_correlated=frac_new, null_frac_correlated=float(np.mean(null_max >= 0.5)),
               transfer_auc=a, transfer_ci=(lo, hi), transfer_orig_auc=a2, transfer_orig_ci=(lo2, hi2),
               vumc_fixed_auc=a3, vumc_fixed_ci=(lo3, hi3))
    # external
    if d.X_ext is not None:
        sh = [g for g in vumc_panel if g in d.X_ext.columns]
        pe = fit_predict(Xv[sh].values, yv, d.X_ext[sh].values, run.xgb)
        ae, loe, hie = bootstrap_auc_ci(d.y_ext.values, pe, cfg["bootstrap_B"], cfg["seed"])
        res.update(ext_auc=ae, ext_ci=(loe, hie), ext_ngenes=len(sh))
        run.register_auc("Independent platelet cohort", "Vumc-derived panel, Vumc-trained -> GSE68086",
                         ae, loe, hie, "bootstrap", len(d.y_ext))
        run.macro("VumcPanelExtAUC", f4(ae)); run.macro("VumcPanelExtCI", ci(loe, hie))
        run.macro("VumcPanelExtNgenes", str(len(sh)))
    # tables / macros
    pd.DataFrame({"gene": vumc_panel, "in_original_panel": [g in P for g in vumc_panel],
                  "closest_original_gene": [sub_rho.get(g, (g, 1.0))[0] for g in vumc_panel],
                  "within_class_spearman": [sub_rho.get(g, (g, 1.0))[1] for g in vumc_panel]}
                 ).to_csv(os.path.join(run.out, "tables", "R1_vumc_only_panel.csv"), index=False)
    run.macro("VumcPanelN", str(len(vumc_panel))); run.macro("VumcPanelOverlap", str(len(ov)))
    run.macro("VumcPanelOverlapGenes", ", ".join(ov) if ov else "none")
    run.macro("VumcPanelJaccard", f3(res["jaccard"])); run.macro("VumcPanelHyperP", pmac(p_hyper, None))
    run.macro("VumcPanelExpOverlap", f"{exp_ov:.2f}")
    run.macro("VumcPanelCorrPct", f"{100 * frac_new:.0f}" if not np.isnan(frac_new) else "n/a")
    nullf = res["null_frac_correlated"]
    if not np.isnan(frac_new) and frac_new >= 0.3 and frac_new >= 2 * nullf:
        run.macro("VumcSubVerdict", "indicating that much of the turnover consists of correlated substitutes rather than unrelated signal")
    else:
        run.macro("VumcSubVerdict", "so the turnover is not explained by correlated substitution alone")
    pan = run.cache.get("r1_panel_table")
    nsig = int((pan.q_vumc_panel < 0.05).sum()) if pan is not None else None
    strong = (a >= 0.75) and (nsig is None or nsig >= len(P) / 2)
    run.macro("SiteVerdict", "These results substantially weaken the strongest form of the confounding concern"
              if strong else "These results only partly address the confounding concern")
    run.macro("SiteConclusion", "supporting genuine biological signal rather than a site-detection artefact"
              if strong else "which only partly separates biological signal from site effects")
    run.macro("VumcPanelCorrNullPct", f"{100 * res['null_frac_correlated']:.0f}")
    run.macro("VumcTransferAUC", f4(a)); run.macro("VumcTransferCI", ci(lo, hi))
    run.macro("VumcTransferOrigAUC", f4(a2)); run.macro("VumcTransferOrigCI", ci(lo2, hi2))
    run.macro("VumcTransferN", str(int((~v).sum())))
    run.macro("VumcTransferNsclc", str(int(y[~v].sum()))); run.macro("VumcTransferHealthy", str(int((1 - y[~v]).sum())))
    run.macro("VumcPanelFixedAUC", f4(a3)); run.macro("VumcPanelFixedCI", ci(lo3, hi3))
    # figure: predicted probability by site and class
    oof_v27 = run.cache.get("oof_vumc_P27")
    sites = d.site.values
    rows = []
    for i, s in enumerate(np.where(v)[0]):
        rows.append(("Vumc (within-site CV)", y[s], oof_vp[i]))
    for i, s in enumerate(np.where(~v)[0]):
        sname = sites[s] if (d.site == sites[s]).sum() > 1 else "singleton batch"
        rows.append((sname, y[s], pt[i]))
    sc = pd.DataFrame(rows, columns=["site", "y", "p"])
    run.table(sc, "R1_scores_by_site")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order_s = ["Vumc (within-site CV)"] + [s for s in sc.site.unique() if s != "Vumc (within-site CV)"]
    pos = 0; ticks = []; labels = []
    for s in order_s:
        for cls, col in [(0, "#2E86C1"), (1, "#C0392B")]:
            vals = sc[(sc.site == s) & (sc.y == cls)].p.values
            if len(vals) == 0:
                continue
            ax.boxplot(vals, positions=[pos], widths=.6, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=col, alpha=.35), medianprops=dict(color=col))
            ax.scatter(np.full(len(vals), pos) + np.random.default_rng(0).uniform(-.2, .2, len(vals)),
                       vals, s=4, c=col, alpha=.6)
            ticks.append(pos); labels.append(f"{s}\n{'NSCLC' if cls else 'healthy'} (n={len(vals)})")
            pos += 1
        pos += .6
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Predicted P(NSCLC)"); ax.axhline(.5, ls=":", c="k", lw=.7)
    ax.set_title("Model trained on Vumc only, applied to the other site codes", fontsize=9)
    run.fig(fig, "R1_scores_by_site")
    run.log(f"  overlap {len(ov)}/{len(P)} (expected {exp_ov:.2f}, p={p_hyper:.2e}); "
            f"transfer AUC {a:.4f} [{lo:.4f}-{hi:.4f}]")
    return res


def r1_vumc_nested(run: Run, B: Optional[int] = None) -> Dict:
    """Fully clean within-site estimate: nested 5-fold CV inside Vumc (selection re-run in
    every outer training fold)."""
    run.log("[R1] nested CV inside Vumc")
    v = run.vumc; y = run.d.y.values
    Xv, yv = run.d.X.loc[v], y[v]
    res = nested_cv(run, Xv, yv, run.folds(yv), "six_stage", B=B)
    a, lo, hi, _ = delong_ci(yv, res["oof"])
    run.register_auc("Site-aware audit", "Nested CV inside Vumc (selection re-run per fold)", a, lo, hi,
                     "DeLong", int(v.sum()), sd=np.std(res["fold_aucs"], ddof=1))
    run.macro("VumcNestedAUC", f4(a)); run.macro("VumcNestedCI", ci(lo, hi))
    run.macro("VumcNestedFoldMean", f4(np.mean(res["fold_aucs"])))
    run.macro("VumcNestedFoldSD", f4(np.std(res["fold_aucs"], ddof=1)))
    json.dump(res["panels"], open(os.path.join(run.out, "tables", "R1_vumc_nested_panels.json"), "w"))
    return res


# ========================================================================================
# R2  External validation: 22-gene versus 27-gene panel
# ========================================================================================
def compare_panels(run: Run, panels: Dict[str, List[str]], ref: str, tag: str,
                   within_vumc: bool = True) -> pd.DataFrame:
    """Paired comparison of gene panels on identical repeated CV splits.
    Point estimate: pooled OOF AUC (mean +- SD over repeats).  Inference: paired DeLong on the
    OOF probabilities averaged over repeats, plus TOST equivalence (margin cfg)."""
    d, cfg = run.d, run.cfg
    margin = cfg["equivalence_margin"]
    rows = []
    for scope in (["all", "vumc"] if within_vumc else ["all"]):
        m = np.ones(len(d.y), bool) if scope == "all" else run.vumc
        X, y = d.X.loc[m], d.y.values[m]
        reps = run.repeated_folds(y)
        avg, per_rep = {}, {}
        for name, genes in list(panels.items()):
            g = [x for x in genes if x in X.columns]
            if len(g) == 0:
                run.log(f"  panel {name} has no genes in the matrix - skipped")
                panels.pop(name); continue
            Xs = X[g].values
            oofs = []
            for fl in reps:
                o, _ = cv_oof(Xs, y, fl, run.xgb)
                oofs.append(o)
            avg[name] = np.mean(oofs, axis=0)
            per_rep[name] = np.array([roc_auc_score(y, o) for o in oofs])
        for name, genes in panels.items():
            a, lo, hi, _ = delong_ci(y, avg[name])
            row = dict(comparison=tag, scope=scope, panel=name, n_genes=len([x for x in genes if x in X.columns]),
                       auc_rep_mean=per_rep[name].mean(), auc_rep_sd=per_rep[name].std(ddof=1),
                       auc_avgoof=a, ci_lo=lo, ci_hi=hi)
            if name != ref:
                dp = delong_paired(y, avg[ref], avg[name], margin)
                dd = per_rep[ref] - per_rep[name]
                row.update(delta_vs_ref=dp["diff"], delta_lo95=dp["lo95"], delta_hi95=dp["hi95"],
                           delta_lo90=dp["lo90"], delta_hi90=dp["hi90"], p_delong=dp["p"],
                           p_tost=dp["p_tost"], equivalent=dp["equivalent"],
                           delta_rep_mean=dd.mean(), delta_rep_q025=np.quantile(dd, .025),
                           delta_rep_q975=np.quantile(dd, .975))
            rows.append(row)
    df = pd.DataFrame(rows)
    run.table(df, f"compare_{tag}")
    run.log(df[["scope", "panel", "n_genes", "auc_rep_mean", "auc_rep_sd"] +
               [c for c in ["delta_vs_ref", "delta_lo95", "delta_hi95", "p_tost", "equivalent"]
                if c in df]].round(4).to_string(index=False))
    return df


def _panel_macros(run: Run, df: pd.DataFrame, name: str, prefix: str):
    for scope, sfx in [("all", ""), ("vumc", "Vumc")]:
        r = df[(df.panel == name) & (df.scope == scope)]
        if r.empty:
            continue
        r = r.iloc[0]
        run.macro(f"{prefix}AUC{sfx}", f4(r.auc_rep_mean)); run.macro(f"{prefix}SD{sfx}", f4(r.auc_rep_sd))
        run.macro(f"{prefix}CI{sfx}", ci(r.ci_lo, r.ci_hi))
        if "delta_vs_ref" in r and pd.notna(r.get("delta_vs_ref", np.nan)):
            run.macro(f"{prefix}Delta{sfx}", pp(r.delta_vs_ref))
            run.macro(f"{prefix}DeltaCI{sfx}", f"{pp(r.delta_lo95)} to {pp(r.delta_hi95)}")
            run.macro(f"{prefix}DeltaNinety{sfx}", f"{pp(r.delta_lo90)} to {pp(r.delta_hi90)}")
            run.macro(f"{prefix}TostP{sfx}", pmac(r.p_tost))
            run.macro(f"{prefix}Equiv{sfx}", "met" if r.equivalent else "did not meet")


def r2_panel_equivalence(run: Run) -> pd.DataFrame:
    """Rebuild the 22-gene (externally testable) version in GSE89843 and test equivalence
    with the 27-gene panel; also the 5 missing genes on their own."""
    d = run.d
    run.log("[R2] 22-gene vs 27-gene equivalence")
    P = run.panel_in(d.X)
    shared = run.cache.get("ext_shared") or ([g for g in P if g in d.X_ext.columns] if d.X_ext is not None else P[:22])
    missing = [g for g in P if g not in shared]
    panels = {"P27": P, f"P{len(shared)}": shared}
    if missing:
        panels[f"missing{len(missing)}"] = missing
    df = compare_panels(run, panels, "P27", "R2_P27_vs_P22")
    run.macro("MissingGenes", ", ".join(missing) if missing else "none")
    run.macro("MarginPP", f"{100 * run.cfg['equivalence_margin']:.0f}")
    _panel_macros(run, df, "P27", "PanelRef")
    _panel_macros(run, df, f"P{len(shared)}", "PanelShared")
    if missing:
        _panel_macros(run, df, f"missing{len(missing)}", "PanelMissing")
    return df


def r2_external_operating(run: Run) -> pd.DataFrame:
    """Operating characteristics and calibration of the frozen classifier in GSE68086
    (threshold fixed in the discovery cohort), plus a decomposition of the AUC drop."""
    d, cfg = run.d, run.cfg
    if d.X_ext is None:
        run.log("[R2] no external cohort - skipped"); return pd.DataFrame()
    run.log("[R2] external operating characteristics")
    shared = run.cache.get("ext_shared") or [g for g in run.panel_in(d.X) if g in d.X_ext.columns]
    y, ye = d.y.values, d.y_ext.values
    oof22, _ = cv_oof(d.X[shared].values, y, run.folds(), run.xgb)
    thr = youden_threshold(y, oof22)
    pe = run.cache.get("p_ext_frozen")
    if pe is None:
        pe = fit_predict(d.X[shared].values, y, d.X_ext[shared].values, run.xgb)
    rows = []
    for nm, t in [("discovery Youden threshold", thr), ("0.5", 0.5)]:
        m = binary_metrics(ye, pe, t); m["rule"] = nm
        # bootstrap CIs for sensitivity / specificity
        rng = np.random.default_rng(cfg["seed"]); bs = []
        for _ in range(1000):
            i = rng.choice(len(ye), len(ye))
            if ye[i].min() == ye[i].max():
                continue
            mm = binary_metrics(ye[i], pe[i], t); bs.append((mm["sensitivity"], mm["specificity"]))
        bs = np.array(bs)
        m["sens_lo"], m["sens_hi"] = np.quantile(bs[:, 0], [.025, .975])
        m["spec_lo"], m["spec_hi"] = np.quantile(bs[:, 1], [.025, .975])
        rows.append(m)
    df = pd.DataFrame(rows)
    # calibration (logistic recalibration)
    try:
        import statsmodels.api as sm
        lp = np.log(np.clip(pe, 1e-6, 1 - 1e-6) / (1 - np.clip(pe, 1e-6, 1 - 1e-6)))
        slope_fit = sm.GLM(ye, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        citl_fit = sm.GLM(ye, np.ones((len(ye), 1)), family=sm.families.Binomial(), offset=lp).fit()
        df["calib_intercept"] = float(citl_fit.params[0]); df["calib_slope"] = float(slope_fit.params[1])
    except Exception as e:
        run.log("  calibration skipped:", e)
    df["brier"] = float(np.mean((pe - ye) ** 2))
    run.table(df, "R2_external_operating_characteristics")
    r0 = df.iloc[0]
    run.macro("ExtThr", f3(thr))
    run.macro("ExtSens", f3(r0.sensitivity)); run.macro("ExtSpec", f3(r0.specificity))
    run.macro("ExtSensCI", ci(r0.sens_lo, r0.sens_hi, 3)); run.macro("ExtSpecCI", ci(r0.spec_lo, r0.spec_hi, 3))
    run.macro("ExtPPV", f3(r0.ppv)); run.macro("ExtNPV", f3(r0.npv)); run.macro("ExtBalAcc", f3(r0.balanced_accuracy))
    run.macro("ExtAccHalf", f3(df.iloc[1].accuracy))
    if "calib_intercept" in df:
        run.macro("ExtCalInt", f2(r0.calib_intercept)); run.macro("ExtCalSlope", f2(r0.calib_slope))
    run.macro("ExtBrier", f3(r0.brier))
    # decomposition of the external drop
    cmp_ = pd.read_csv(os.path.join(run.out, "tables", "compare_R2_P27_vs_P22.csv")) \
        if os.path.exists(os.path.join(run.out, "tables", "compare_R2_P27_vs_P22.csv")) else None
    if cmp_ is not None:
        a27 = cmp_[(cmp_.panel == "P27") & (cmp_.scope == "all")].auc_rep_mean.iloc[0]
        a22 = cmp_[(cmp_.panel == f"P{len(shared)}") & (cmp_.scope == "all")].auc_rep_mean.iloc[0]
        aext = roc_auc_score(ye, pe)
        run.macro("DropGeneLoss", pp(a27 - a22)); run.macro("DropCohort", pp(a22 - aext))
        tot = a27 - aext
        run.macro("DropCohortPct", f"{100 * (a22 - aext) / tot:.0f}" if tot > 0 else "n/a")
        gene_loss, cohort = a27 - a22, a22 - aext
        if cohort > 0 and gene_loss <= 0.25 * cohort:
            v = "is therefore attributable mainly to the change of cohort rather than to the missing genes"
        elif cohort > 0:
            v = "reflects both the missing genes and the change of cohort"
        else:
            v = "is not explained by the change of cohort in this analysis"
        run.macro("DropVerdict", v)
    run.log(df.round(3).to_string(index=False))
    return df


# ========================================================================================
# R3  ComBat-seq harmonisation: leakage audit and deployment feasibility
# ========================================================================================
def _counts_pair(run: Run) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = run.d
    if d.counts is None or d.counts_ext is None:
        run.log("  note: raw counts not supplied - using rounded ML-matrix values as counts")
    C = d.counts if d.counts is not None else run.rawX(d.X)
    Ce = d.counts_ext if d.counts_ext is not None else run.rawX(d.X_ext)
    shared = sorted(set(C.columns) & set(Ce.columns))
    C = C[shared].clip(lower=0).round().astype(np.int64)
    Ce = Ce[shared].clip(lower=0).round().astype(np.int64)
    keep = (C.sum(0) > 0) & (Ce.sum(0) > 0)
    return C.loc[:, keep], Ce.loc[:, keep]


def logcpm(C: pd.DataFrame) -> pd.DataFrame:
    lib = C.sum(axis=1).replace(0, 1)
    return np.log2(C.div(lib, axis=0) * 1e6 + 1.0)


def _reduce_with_rest(C: pd.DataFrame, keep: Sequence[str]) -> pd.DataFrame:
    R = C[list(keep)].copy()
    R["__REST__"] = C.drop(columns=list(keep)).sum(axis=1)
    return R


def _combat_seq(C_ref: pd.DataFrame, C_new: pd.DataFrame, ref_batch: Optional[int]):
    """ComBat-seq (inmoose) on samples x genes count tables; labels are never passed."""
    from inmoose.pycombat import pycombat_seq
    allc = pd.concat([C_ref, C_new], axis=0)
    batch = [0] * len(C_ref) + [1] * len(C_new)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adj = pycombat_seq(allc.T, batch, ref_batch=ref_batch)
    adj = pd.DataFrame(np.asarray(adj), index=allc.columns, columns=allc.index).T
    return adj.iloc[: len(C_ref)], adj.iloc[len(C_ref):]


def _loc_scale(ref: pd.DataFrame, calib: pd.DataFrame, apply: pd.DataFrame) -> pd.DataFrame:
    """Reference-based location/scale ('add-on') alignment on log-CPM: new-site parameters
    are estimated on unlabelled calibration samples only, the reference is never changed."""
    mu_r, sd_r = ref.mean(0), ref.std(0, ddof=1)
    mu_t, sd_t = calib.mean(0), calib.std(0, ddof=1)
    sd_t = sd_t.where(sd_t > 1e-8, sd_r)
    return (apply - mu_t) / sd_t * sd_r + mu_r


def within_sample_ranks(L: pd.DataFrame, genes_universe: Sequence[str]) -> pd.DataFrame:
    """Single-sample representation: percentile rank of each gene within its own sample."""
    U = L[list(genes_universe)]
    return U.rank(axis=1, pct=True)


def r3_harmonisation_audit(run: Run) -> pd.DataFrame:
    d, cfg = run.d, run.cfg
    if d.X_ext is None:
        run.log("[R3] no external cohort - skipped"); return pd.DataFrame()
    h = cfg["harm"]
    run.log("[R3] harmonisation leakage audit")
    rng = np.random.default_rng(cfg["seed"])
    C, Ce = _counts_pair(run)
    P = [g for g in (run.cache.get("ext_shared") or run.panel_in(d.X)) if g in C.columns]
    y, ye = d.y.values, d.y_ext.values
    Ld, Le = logcpm(C), logcpm(Ce)
    frozen = run.xgb().fit(Ld[P].values, y)
    pred = lambda M: frozen.predict_proba(M[P].values)[:, 1]
    rows = []

    def add(key, desc, labels, unlab, retrain, single, p, extra=""):
        a, lo, hi = bootstrap_auc_ci(ye, p, cfg["bootstrap_B"], cfg["seed"])
        rows.append(dict(setting=key, description=desc, uses_external_labels=labels,
                         uses_unlabelled_external_data=unlab, model_retrained=retrain,
                         works_for_single_new_sample=single, auc=a, ci_lo=lo, ci_hi=hi, note=extra))
        return a, lo, hi

    a0 = add("S0", "No harmonisation (frozen model)", "No", "No", "No", "Yes", pred(Le))
    run.macro("HarmBaseAUC", f4(a0[0])); run.macro("HarmBaseCI", ci(a0[1], a0[2]))
    Lref_adj = None
    if h["run_combat_seq"]:
        try:
            t0 = time.time()
            Cj_d, Cj_e = _combat_seq(C, Ce, ref_batch=None)
            Ljd, Lje = logcpm(Cj_d), logcpm(Cj_e)
            pj = fit_predict(Ljd[P].values, y, Lje[P].values, run.xgb)
            a1 = add("S1a", "Joint ComBat-seq (both cohorts adjusted), model retrained on adjusted "
                     "discovery data", "No", "Yes (joint with discovery)", "Yes", "No", pj)
            a1b = add("S1b", "Joint ComBat-seq, frozen model (not re-trained)", "No", "Yes (joint)", "No", "No", pred(Lje))
            Cr_d, Cr_e = _combat_seq(C, Ce, ref_batch=0)
            unchanged = bool(np.allclose(Cr_d.values, C.values))
            Lref_adj = logcpm(Cr_e)
            a2 = add("S2", "Reference-batch ComBat-seq (discovery = fixed reference; only the new "
                     "cohort is adjusted), frozen model", "No", "Yes (new cohort only)", "No",
                     "No (needs a batch)", pred(Lref_adj), f"reference unchanged: {unchanged}")
            run.macro("HarmJointAUC", f4(a1[0])); run.macro("HarmJointCI", ci(a1[1], a1[2]))
            run.macro("HarmJointFrozenAUC", f4(a1b[0])); run.macro("HarmJointFrozenCI", ci(a1b[1], a1b[2]))
            run.macro("HarmRefAUC", f4(a2[0])); run.macro("HarmRefCI", ci(a2[1], a2[2]))
            run.log(f"  ComBat-seq runs: {time.time() - t0:.0f}s (reference unchanged: {unchanged})")
        except Exception as e:
            run.log("  ComBat-seq skipped:", repr(e))
    # transductive location/scale on the whole external cohort
    pls = pred(_loc_scale(Ld[P], Le[P], Le[P]))
    a3 = add("S3", "Reference location/scale alignment estimated on all external samples "
             "(unlabelled), frozen model", "No", "Yes (new cohort only)", "No", "No (needs a batch)", pls)
    run.macro("HarmLSAUC", f4(a3[0])); run.macro("HarmLSCI", ci(a3[1], a3[2]))
    # ---- add-on (inductive) alignment: calibrate on k unlabelled samples, test on the others
    cal_rows = []
    for k in h["calib_k"]:
        if k >= len(ye) - 10:
            continue
        for r in range(h["calib_repeats"]):
            idx = rng.permutation(len(ye)); cal, ev = idx[:k], idx[k:]
            if ye[ev].min() == ye[ev].max():
                continue
            Xev = _loc_scale(Ld[P], Le.iloc[cal][P], Le.iloc[ev][P])
            cal_rows.append(dict(k=k, rep=r, auc_addon=roc_auc_score(ye[ev], pred(Xev)),
                                 auc_none=roc_auc_score(ye[ev], pred(Le.iloc[ev]))))
    cal = pd.DataFrame(cal_rows)
    run.table(cal, "R3_addon_calibration_size")
    for k, g in cal.groupby("k"):
        w = NUM_WORDS.get(int(k), None)
        if w:
            run.macro(f"HarmAddon{w}AUC", f4(g.auc_addon.median()))
            run.macro(f"HarmAddon{w}Range", ci(g.auc_addon.quantile(.025), g.auc_addon.quantile(.975)))
            run.macro(f"HarmAddon{w}Gain", pp((g.auc_addon - g.auc_none).median()))
    rows.append(dict(setting="S4", description="Add-on alignment: parameters from k unlabelled new-site "
                     "samples, evaluated on the remaining samples (see R3_addon_calibration_size.csv)",
                     uses_external_labels="No", uses_unlabelled_external_data="Yes (calibration subset only)",
                     model_retrained="No", works_for_single_new_sample="Yes, after a k-sample calibration",
                     auc=float(cal[cal.k == cal.k.max()].auc_addon.median()) if len(cal) else np.nan,
                     ci_lo=np.nan, ci_hi=np.nan, note="median over repeats at the largest k"))
    # ---- prevalence sensitivity
    pos, neg = np.where(ye == 1)[0], np.where(ye == 0)[0]
    prev_rows = []
    keep_genes = None
    if h["run_combat_seq"] and Lref_adj is not None:
        top = Ld.mean(0).sort_values(ascending=False).index[: h["combat_genes"]]
        keep_genes = sorted(set(top) | set(P))
        Cd_red = _reduce_with_rest(C, keep_genes)
    for pi in h["prev_levels"]:
        npos = int(round(pi * h["prev_n"])); nneg = h["prev_n"] - npos
        if npos > len(pos) or nneg > len(neg) or npos < 2 or nneg < 2:
            continue
        for r in range(h["prev_repeats"]):
            idx = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)])
            sub = Le.iloc[idx][P]
            row = dict(prevalence=pi, rep=r, auc_none=roc_auc_score(ye[idx], pred(sub)),
                       auc_locscale=roc_auc_score(ye[idx], pred(_loc_scale(Ld[P], sub, sub))))
            if keep_genes is not None and r < h["prev_repeats_combat"]:
                try:
                    _, adj = _combat_seq(Cd_red, _reduce_with_rest(Ce.iloc[idx], keep_genes), ref_batch=0)
                    row["auc_combat_ref"] = roc_auc_score(ye[idx], pred(logcpm(adj)))
                except Exception:
                    pass
            prev_rows.append(row)
    prev = pd.DataFrame(prev_rows)
    run.table(prev, "R3_prevalence_sensitivity")
    if len(prev):
        g = prev.groupby("prevalence").mean(numeric_only=True)
        for col, nm in [("auc_none", "None"), ("auc_locscale", "LS"), ("auc_combat_ref", "Ref")]:
            if col in g:
                run.macro(f"HarmPrevRange{nm}", f"{g[col].min():.3f}--{g[col].max():.3f}")
                run.macro(f"HarmPrevSpread{nm}", pp(g[col].max() - g[col].min()))
    # ---- random-panel control: does harmonisation make ANY gene set look good?
    mexp = Ld.mean(0)
    pool = mexp[(mexp > 1) & ~mexp.index.isin(run.cfg["panel"])].index.tolist()
    rnd_rows = []
    for r in range(h["random_panels"]):
        gs = list(rng.choice(pool, len(P), replace=False))
        m = run.xgb().fit(Ld[gs].values, y)
        row = dict(rep=r, auc_none=roc_auc_score(ye, m.predict_proba(Le[gs].values)[:, 1]),
                   auc_locscale=roc_auc_score(ye, m.predict_proba(_loc_scale(Ld[gs], Le[gs], Le[gs]).values)[:, 1]))
        if Lref_adj is not None:
            row["auc_combat_ref"] = roc_auc_score(ye, m.predict_proba(Lref_adj[gs].values)[:, 1])
        rnd_rows.append(row)
    rnd = pd.DataFrame(rnd_rows)
    run.table(rnd, "R3_random_panel_control")
    hc = "auc_combat_ref" if "auc_combat_ref" in rnd else "auc_locscale"
    cur_h = [r for r in rows if r["setting"] == ("S2" if hc == "auc_combat_ref" else "S3")][0]["auc"]
    run.macro("HarmRandBaseMed", f4(rnd.auc_none.median()))
    run.macro("HarmRandHarmMed", f4(rnd[hc].median()))
    run.macro("HarmRandHarmMax", f4(rnd[hc].max()))
    run.macro("HarmRandDeltaMed", pp((rnd[hc] - rnd.auc_none).median()))
    run.macro("HarmCuratedDelta", pp(cur_h - a0[0]))
    run.macro("HarmRandExceed", str(int((rnd[hc] >= cur_h).sum())))
    run.macro("HarmRandN", str(len(rnd)))
    frac_ex = (rnd[hc] >= cur_h).mean()
    run.macro("HarmRandVerdict", "alignment improves transfer generally but does not make arbitrary genes perform like the panel"
              if frac_ex < 0.05 else "alignment can make a notable fraction of arbitrary gene sets perform as well as the panel, so harmonised AUCs must be interpreted with caution")
    # ---- single-sample rank representation (no batch needed)
    fd = ((2 ** Ld - 1) > h["rank_min_cpm"]).mean(0)
    fe = ((2 ** Le - 1) > h["rank_min_cpm"]).mean(0)
    expressed = fd.index[(fd >= h["rank_min_frac"]) & (fe >= h["rank_min_frac"])].tolist()
    Rd, Re = within_sample_ranks(Ld, expressed), within_sample_ranks(Le, expressed)
    Pr = [g for g in P if g in expressed]
    prk = fit_predict(Rd[Pr].values, y, Re[Pr].values, run.xgb)
    a6 = add("S5", f"Single-sample within-sample rank features ({len(Pr)} panel genes ranked among "
             f"{len(expressed)} expressed genes), no harmonisation", "No", "No", "Yes (rank model)", "Yes", prk)
    run.macro("HarmRankAUC", f4(a6[0])); run.macro("HarmRankCI", ci(a6[1], a6[2]))
    run.macro("HarmRankNgenes", str(len(Pr))); run.macro("HarmRankUniverse", f"{len(expressed):,}".replace(",", "{,}"))
    oof_r, _ = cv_oof(Rd[Pr].values, y, run.folds(), run.xgb)
    run.macro("HarmRankDiscAUC", f4(roc_auc_score(y, oof_r)))
    audit = pd.DataFrame(rows)
    run.table(audit, "R3_harmonisation_audit")
    short = {"S0": "None (frozen model)", "S1a": "Joint ComBat-seq + retrained model",
             "S1b": "Joint ComBat-seq, frozen model", "S2": "Reference-batch ComBat-seq, frozen model",
             "S3": "Reference location/scale, frozen model", "S4": "Add-on location/scale (k calibration samples)",
             "S5": "Single-sample rank model, no alignment"}
    trows = []
    for r in audit.itertuples():
        auc_s = f"{f4(r.auc)} ({ci(r.ci_lo, r.ci_hi)})" if not np.isnan(r.ci_lo) else f"{f4(r.auc)}$^{{a}}$"
        trows.append([r.setting, short.get(r.setting, r.setting), r.uses_unlabelled_external_data.split(" (")[0],
                      r.model_retrained.split(" (")[0], r.works_for_single_new_sample.split(" (")[0].split(",")[0], auc_s])
    write_tex_table(os.path.join(run.out, "tables", "R3_harmonisation_audit.tex"), trows,
                    ["", "Alignment", "Unlabelled new-cohort data", "Model retrained", "Single new sample", "AUC (95\\% CI)"],
                    "llllll", "\\footnotesize")
    for _, r in audit.iterrows():
        if not np.isnan(r.ci_lo):
            run.register_auc("Independent platelet cohort", f"GSE68086 {r.setting}: {r.description[:60]}",
                             r.auc, r.ci_lo, r.ci_hi, "bootstrap", len(ye))
    # ---- figure
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    if len(cal):
        g = cal.groupby("k")
        ks = np.array(list(g.groups.keys()))
        axs[0].plot(ks, g.auc_addon.median(), "o-", c="#C0392B", label="add-on alignment")
        axs[0].fill_between(ks, g.auc_addon.quantile(.1), g.auc_addon.quantile(.9), color="#C0392B", alpha=.15)
        axs[0].plot(ks, g.auc_none.median(), "s--", c="#7F8C8D", label="no alignment")
        axs[0].set_xlabel("Unlabelled calibration samples from the new cohort (k)")
        axs[0].set_ylabel("AUC on the remaining samples"); axs[0].legend(frameon=False, fontsize=7)
        axs[0].set_title("A  How many samples does a new site need?", fontsize=9, loc="left")
    if len(prev):
        g = prev.groupby("prevalence").mean(numeric_only=True)
        for col, lab, st in [("auc_none", "no alignment", "s--"), ("auc_locscale", "location/scale", "o-"),
                             ("auc_combat_ref", "reference ComBat-seq", "^-")]:
            if col in g:
                axs[1].plot(g.index, g[col], st, label=lab)
        axs[1].set_xlabel("NSCLC prevalence in the external subsample"); axs[1].set_ylabel("Mean AUC")
        axs[1].legend(frameon=False, fontsize=7)
        axs[1].set_title("B  Sensitivity to case mix", fontsize=9, loc="left")
    axs[2].hist(rnd[hc], bins=30, color="#B8B8B8", label="random gene panels, harmonised")
    axs[2].axvline(cur_h, c="#C0392B", lw=2, label="curated panel, harmonised")
    axs[2].axvline(rnd.auc_none.median(), c="k", ls=":", lw=1, label="random panels, median before")
    axs[2].set_xlabel("External AUC"); axs[2].legend(frameon=False, fontsize=7)
    axs[2].set_title("C  Can harmonisation make random genes look good?", fontsize=9, loc="left")
    fig.tight_layout()
    run.fig(fig, "R3_harmonisation_audit")
    run.log(audit[["setting", "auc", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    return audit


# ========================================================================================
# R4  Gene-identity stability: chance-corrected, recurrence, correlated substitutes
# ========================================================================================
def nogueira_stability(panels: List[List[str]], d_total: int) -> float:
    """Nogueira, Sechidis & Brown (2018) stability estimator; 0 = chance level, 1 = identical."""
    M = len(panels)
    genes = sorted(set().union(*panels))
    Z = np.array([[g in set(p) for g in genes] for p in panels], float)
    pf = Z.mean(0)
    s2 = M / (M - 1) * pf * (1 - pf)
    kbar = Z.sum(1).mean()
    denom = (kbar / d_total) * (1 - kbar / d_total)
    return float(1 - (s2.sum() / d_total) / denom)


def expected_random_jaccard(k1: int, k2: int, n: int, sims: int = 20000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    ov = rng.hypergeometric(k1, n - k1, k2, size=sims)
    j = ov / (k1 + k2 - ov)
    return float(j.mean()), float(np.quantile(j, 0.999))


def r4_stability(run: Run, fold_panels: Optional[List[List[str]]] = None,
                 run_nested_if_missing: bool = True) -> Dict:
    d, cfg = run.d, run.cfg
    run.log("[R4] gene-identity stability")
    P = run.panel_in(d.X)
    if fold_panels is None and cfg.get("fold_panels"):
        fold_panels = json.load(open(cfg["fold_panels"]))
    if fold_panels is None and "nested_six_stage" in run.cache:
        fold_panels = run.cache["nested_six_stage"]["panels"]
    if fold_panels is None and run_nested_if_missing:
        res = nested_cv(run, d.X, d.y.values, run.folds(), "six_stage")
        run.cache["nested_six_stage"] = res
        fold_panels = res["panels"]
    if not fold_panels:
        run.log("  no fold panels available - skipped"); return {}
    M = len(fold_panels)
    L = run.logX()
    y = d.y.values
    tier2 = tier2_genes(run)
    G_all, G_t2 = L.shape[1], max(len(tier2), len(P))
    j_orig = [jaccard(P, f) for f in fold_panels]
    j_pair = [jaccard(a, b) for a, b in itertools.combinations(fold_panels, 2)]
    ej_t2, ej_t2_hi = expected_random_jaccard(len(P), int(np.mean([len(f) for f in fold_panels])), G_t2)
    ej_all, _ = expected_random_jaccard(len(P), int(np.mean([len(f) for f in fold_panels])), G_all)
    phi_t2 = nogueira_stability(fold_panels, G_t2)
    phi_all = nogueira_stability(fold_panels, G_all)
    # recurrence of each gene across fold panels
    allg = sorted(set().union(*fold_panels) | set(P))
    rec = pd.DataFrame(dict(gene=allg, n_folds=[sum(g in set(f) for f in fold_panels) for g in allg],
                            in_original=[g in P for g in allg]))
    rec = rec.sort_values(["n_folds", "in_original"], ascending=False)
    core = rec[(rec.in_original) & (rec.n_folds >= math.ceil(M / 2))].gene.tolist()
    # correlated substitutes: fold-panel genes absent from the original panel
    resid = L - L.groupby(y).transform("mean")
    subs = [g for g in set().union(*fold_panels) if g not in P and g in L.columns]
    sub_tab = []
    if subs:
        Rm = rank_corr_matrix(resid[subs].values, resid[P].values)
        for i, g in enumerate(subs):
            j = int(np.argmax(np.abs(Rm[i])))
            sub_tab.append((g, P[j], float(Rm[i, j])))
    sub_df = pd.DataFrame(sub_tab, columns=["substitute_gene", "closest_original_gene", "within_class_spearman"])
    rng = np.random.default_rng(cfg["seed"])
    pool = [g for g in tier2 if g not in set(P) | set(subs)] or [g for g in L.columns if g not in P]
    null = rng.choice(pool, size=min(1000, len(pool)), replace=False)
    null_max = np.abs(rank_corr_matrix(resid[list(null)].values, resid[P].values)).max(1)
    frac_sub = float((sub_df.within_class_spearman.abs() >= 0.5).mean()) if len(sub_df) else np.nan
    run.table(rec, "R4_gene_recurrence")
    run.table(sub_df, "R4_correlated_substitutes")
    out = dict(j_orig_mean=np.mean(j_orig), j_pair_mean=np.mean(j_pair), exp_j_tier2=ej_t2,
               exp_j_tier2_999=ej_t2_hi, exp_j_all=ej_all, phi_tier2=phi_t2, phi_all=phi_all,
               core=core, frac_sub_correlated=frac_sub, null_frac_correlated=float(np.mean(null_max >= .5)))
    run.macro("StabJOrig", f3(np.mean(j_orig))); run.macro("StabJPair", f3(np.mean(j_pair)))
    run.macro("StabExpJ", f"{ej_t2:.4f}"); run.macro("StabExpJAll", f"{ej_all:.4f}")
    run.macro("StabFoldEnrich", f"{np.mean(j_orig) / ej_t2:.0f}")
    run.macro("StabPhi", f3(phi_t2)); run.macro("StabPhiAll", f3(phi_all))
    run.macro("StabTierTwoN", f"{G_t2:,}".replace(",", "{,}"))
    run.macro("StabCoreN", str(len(core))); run.macro("StabCoreGenes", ", ".join(core) if core else "none")
    run.macro("StabSubN", str(len(sub_df)))
    run.macro("StabSubCorrPct", f"{100 * frac_sub:.0f}" if not np.isnan(frac_sub) else "n/a")
    run.macro("StabSubNullPct", f"{100 * out['null_frac_correlated']:.0f}")
    if not np.isnan(frac_sub) and frac_sub >= 0.3 and frac_sub >= 2 * out["null_frac_correlated"]:
        run.macro("StabSubVerdict", "most gene turnover is substitution among correlated genes")
    else:
        run.macro("StabSubVerdict", "gene turnover is only partly explained by substitution among correlated genes")
    if core:
        run.macro("StabCoreClause", f"a recurrent core of {len(core)} genes ({', '.join(core)}) selected in at least three of five resamples")
    else:
        run.macro("StabCoreClause", "no recurrent core, because no original gene was selected in at least three of five resamples")
    for i in range(M + 1):
        w = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}.get(i)
        if w:
            run.macro(f"StabRec{w}", str(int(((rec.n_folds == i) & rec.in_original).sum())))
    # performance of the core subset versus the full panel
    if len(core) >= 3:
        cmp_ = compare_panels(run, {"P27": P, f"core{len(core)}": core}, "P27", "R4_core_vs_P27")
        _panel_macros(run, cmp_, f"core{len(core)}", "StabCore")
    # evaluate every fold panel on the full cohort (Rashomon set)
    fp_auc = []
    for i, f in enumerate(fold_panels):
        g = [x for x in f if x in d.X.columns]
        o, fa = cv_oof(d.X[g].values, y, run.folds(), run.xgb)
        fp_auc.append(roc_auc_score(y, o))
    run.macro("StabFoldPanelAUCRange", f"{min(fp_auc):.4f}--{max(fp_auc):.4f}")
    pd.DataFrame(dict(fold=range(1, M + 1), n_genes=[len(f) for f in fold_panels], jaccard_vs_original=j_orig,
                      full_cohort_cv_auc=fp_auc)).to_csv(os.path.join(run.out, "tables", "R4_fold_panels.csv"), index=False)
    run.log(f"  mean Jaccard vs original {np.mean(j_orig):.3f} (random expectation {ej_t2:.4f}); "
            f"Nogueira phi={phi_t2:.3f}; core genes ({len(core)}): {core}")
    return out


# ========================================================================================
# R5  Contamination: panel-wide screen, QC-filtered panels, contamination-adjusted model
# ========================================================================================
def marker_scores(L: pd.DataFrame, sets: Dict[str, List[str]] = MARKER_SETS) -> pd.DataFrame:
    out = {}
    for k, genes in sets.items():
        g = [x for x in genes if x in L.columns]
        if len(g) == 0:
            continue
        Z = (L[g] - L[g].mean()) / (L[g].std() + 1e-9)
        out[k] = Z.mean(1)
    return pd.DataFrame(out, index=L.index)


def r5_contamination(run: Run) -> pd.DataFrame:
    d, cfg = run.d, run.cfg
    run.log("[R5] contamination sensitivity")
    P = run.panel_in(d.X)
    L = run.logX(); y = d.y.values; v = run.vumc
    S = marker_scores(L)
    run.macro("MarkerSetsUsed", ", ".join(f"{k} ({len([g for g in MARKER_SETS[k] if g in L.columns])})"
                                          for k in S.columns))
    # (1) panel-wide screen: within-class Spearman of each panel gene with each marker score
    rows = []
    for g in P:
        r = dict(gene=g)
        for k in S.columns:
            if k == "platelet":
                pass
            rh = []
            for cls in (0, 1):
                m = (y == cls)
                rh.append(stats.spearmanr(L.loc[m, g], S.loc[m, k]))
            r[f"rho_{k}_healthy"], r[f"p_{k}_healthy"] = rh[0][0], rh[0][1]
            r[f"rho_{k}_nsclc"], r[f"p_{k}_nsclc"] = rh[1][0], rh[1][1]
        rows.append(r)
    scr = pd.DataFrame(rows)
    pcols = [c for c in scr if c.startswith("p_") and "platelet" not in c]
    allp = scr[pcols].values.ravel()
    q = bh(allp).reshape(scr[pcols].shape)
    for j, c in enumerate(pcols):
        scr["q" + c[1:]] = q[:, j]
    flags = []
    for i, r in scr.iterrows():
        f = []
        for k in [c for c in S.columns if c != "platelet"]:
            for cls in ("healthy", "nsclc"):
                if abs(r[f"rho_{k}_{cls}"]) >= 0.3 and r[f"q_{k}_{cls}"] < 0.05:
                    f.append(f"{k}/{cls}")
        flags.append("; ".join(f))
    scr["contamination_flags"] = flags
    run.table(scr, "R5_contamination_screen_all27")
    flagged = scr.loc[scr["contamination_flags"] != "", "gene"].tolist()
    run.macro("ContamFlaggedN", str(len(flagged)))
    run.macro("ContamFlaggedGenes", ", ".join(flagged) if flagged else "none")
    for k in S.columns:
        nk = sum(k in f for f in flags)
        run.macro(f"ContamFlag{k.capitalize()}", str(nk))
    # (2) do the marker scores themselves differ by class inside Vumc? (pre-analytical imbalance)
    imb = []
    for k in S.columns:
        a_, b_ = S.loc[v & (y == 1), k], S.loc[v & (y == 0), k]
        u = stats.mannwhitneyu(a_, b_)
        imb.append(dict(score=k, auc_vumc=u.statistic / (len(a_) * len(b_)), p_vumc=u.pvalue,
                        median_nsclc=a_.median(), median_healthy=b_.median()))
    imb = pd.DataFrame(imb)
    run.table(imb, "R5_marker_scores_by_class_vumc")
    for _, r in imb.iterrows():
        run.macro(f"Score{r.score.capitalize()}AUC", f3(r.auc_vumc))
        run.macro(f"Score{r.score.capitalize()}P", pmac(r.p_vumc))
    # (3) QC-filtered panels (paired with P27, TOST)
    sus = [g for g in cfg["suspect_contamination"] if g in P]
    panels = {"P27": P, f"P{len(P) - 1}": [g for g in P if g != "KRTAP22-1"],
              f"P{len(P) - len(sus)}": [g for g in P if g not in sus]}
    extra = [g for g in flagged if g not in sus]
    keep_unflagged = [g for g in P if g not in set(sus) | set(flagged)]
    if extra and len(keep_unflagged) >= 5:
        panels[f"P{len(keep_unflagged)}unflagged"] = keep_unflagged
    cmp_ = compare_panels(run, panels, "P27", "R5_contamination_panels")
    for name, mname in [(f"P{len(P) - 1}", "QcOne"), (f"P{len(P) - len(sus)}", "QcThree")]:
        _panel_macros(run, cmp_, name, mname)
    # external + harmonised external for each panel
    ext_rows = []
    if d.X_ext is not None:
        for name, genes in panels.items():
            g = [x for x in genes if x in d.X_ext.columns]
            pe = fit_predict(d.X[g].values, y, d.X_ext[g].values, run.xgb)
            a, lo, hi = bootstrap_auc_ci(d.y_ext.values, pe, cfg["bootstrap_B"], cfg["seed"])
            ext_rows.append(dict(panel=name, n_genes_external=len(g), ext_auc=a, ext_lo=lo, ext_hi=hi))
        ext = pd.DataFrame(ext_rows)
        run.table(ext, "R5_contamination_panels_external")
        for name, mname in [(f"P{len(P) - 1}", "QcOne"), (f"P{len(P) - len(sus)}", "QcThree")]:
            r = ext[ext.panel == name].iloc[0]
            run.macro(f"{mname}ExtAUC", f4(r.ext_auc)); run.macro(f"{mname}ExtCI", ci(r.ext_lo, r.ext_hi))
            run.macro(f"{mname}ExtNgenes", str(int(r.n_genes_external)))
    # (4) contamination-adjusted model: panel genes residualised on marker scores (fit in training folds)
    Sc = S[[c for c in S.columns if c != "platelet"]].values
    XL = L[P].values

    class _Resid:
        def __init__(self, base): self.base = base
        def fit(self, X, yy):
            Z, Xg = X[:, :Sc.shape[1]], X[:, Sc.shape[1]:]
            A = np.c_[np.ones(len(Z)), Z]
            self.B = np.linalg.lstsq(A, Xg, rcond=None)[0]
            self.m = self.base().fit(Xg - A @ self.B, yy); return self
        def predict_proba(self, X):
            Z, Xg = X[:, :Sc.shape[1]], X[:, Sc.shape[1]:]
            A = np.c_[np.ones(len(Z)), Z]
            return self.m.predict_proba(Xg - A @ self.B)
    oof_adj, _ = cv_oof(np.c_[Sc, XL], y, run.folds(), lambda: _Resid(run.xgb))
    a, lo, hi, _ = delong_ci(y, oof_adj)
    base = run.cache.get("oof_P27")
    if base is None:
        base, _ = cv_oof(d.X[P].values, y, run.folds(), run.xgb)
    dp = delong_paired(y, base, oof_adj)
    run.macro("ContamAdjAUC", f4(a)); run.macro("ContamAdjCI", ci(lo, hi))
    run.macro("ContamAdjDelta", pp(dp["diff"])); run.macro("ContamAdjDeltaCI", f"{pp(dp['lo95'])} to {pp(dp['hi95'])}")
    # contamination scores alone as a classifier (how much signal is pre-analytical?)
    oof_s, _ = cv_oof(Sc, y, run.folds(), run.xgb)
    run.macro("ContamOnlyAUC", f4(roc_auc_score(y, oof_s)))
    oof_sv, _ = cv_oof(Sc[v], y[v], run.folds(y[v]), run.xgb)
    run.macro("ContamOnlyVumcAUC", f4(roc_auc_score(y[v], oof_sv)))
    # (5) updated stability for the filtered panels
    st = []
    for name, genes in panels.items():
        bf = [TABLE8[g][0] for g in genes if g in TABLE8]; sp = [TABLE8[g][1] for g in genes if g in TABLE8]
        row = dict(panel=name, mean_bootstrap_freq=np.mean(bf), min_bootstrap_freq=np.min(bf),
                   mean_stability_pi=np.mean(sp), n_pi_above_060=int(np.sum(np.array(sp) >= .6)))
        fp = None
        if cfg.get("fold_panels"):
            fp = json.load(open(cfg["fold_panels"]))
        elif "nested_six_stage" in run.cache:
            fp = run.cache["nested_six_stage"]["panels"]
        if fp:
            row["mean_nested_jaccard"] = np.mean([jaccard(genes, f) for f in fp])
        st.append(row)
    st = pd.DataFrame(st)
    run.table(st, "R5_contamination_panels_stability")
    r3 = st[st.panel == f"P{len(P) - len(sus)}"].iloc[0]
    run.macro("QcThreeMeanBoot", f3(r3.mean_bootstrap_freq)); run.macro("QcThreePiN", str(int(r3.n_pi_above_060)))
    if "mean_nested_jaccard" in st:
        run.macro("QcThreeJaccard", f3(r3.mean_nested_jaccard))
    # (6) optional: re-run selection with suspect genes removed from the candidate pool
    try:
        newp = call_select(run, d.X.drop(columns=[g for g in sus if g in d.X.columns]), y,
                           B=cfg["selection"]["boot_B_full"])
        repl = [g for g in newp if g not in P]
        run.macro("QcReselectN", str(len(newp))); run.macro("QcReselectNew", ", ".join(repl) if repl else "none")
        run.macro("QcReselectJaccard", f3(jaccard(P, newp)))
        pd.Series(newp, name="gene").to_csv(os.path.join(run.out, "tables", "R5_reselected_panel.csv"), index=False)
    except Exception as e:
        run.log("  re-selection skipped:", repr(e))
    try:
        trows = []
        for name in panels:
            ra = cmp_[(cmp_.panel == name) & (cmp_.scope == "all")].iloc[0]
            rv = cmp_[(cmp_.panel == name) & (cmp_.scope == "vumc")]
            rv = rv.iloc[0] if len(rv) else None
            ex = ext[ext.panel == name].iloc[0] if d.X_ext is not None else None
            dl = "reference" if name == "P27" else f"{pp(ra.delta_vs_ref)} ({pp(ra.delta_lo95)}, {pp(ra.delta_hi95)})"
            eq = "--" if name == "P27" else ("yes" if ra.equivalent else "no")
            trows.append([name.replace("unflagged", " (all flags removed)"), int(ra.n_genes),
                          f"{f4(ra.auc_rep_mean)} $\\pm$ {f4(ra.auc_rep_sd)}", dl, eq,
                          f4(rv.auc_rep_mean) if rv is not None else "--",
                          f"{f4(ex.ext_auc)} ({ci(ex.ext_lo, ex.ext_hi, 3)})" if ex is not None else "--"])
        write_tex_table(os.path.join(run.out, "tables", "R5_contamination_panels.tex"), trows,
                        ["Panel", "Genes", "CV AUC (mean $\\pm$ SD)", "$\\Delta$AUC vs 27, pp (95\\% CI)",
                         "Equivalent", "Vumc-only AUC", "GSE68086 AUC (95\\% CI)"], "lrrrcrr", "\\footnotesize")
    except Exception as e:
        run.log("  R5 tex table skipped:", repr(e))
    # recommended panel by the pre-specified rule
    eq = cmp_[(cmp_.panel == f"P{len(P) - len(sus)}") & (cmp_.scope == "all")].equivalent.iloc[0]
    rec_name = f"QC-filtered {len(P) - len(sus)}-gene panel" if eq else f"original {len(P)}-gene panel"
    run.macro("RecommendedPanel", rec_name)
    run.log(f"  flagged: {flagged}; recommended: {rec_name}")
    return cmp_


# ========================================================================================
# R6  Covariates: inferred sex, recorded covariates if present, smoking-gene check
# ========================================================================================
def infer_sex(L: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Sex from Y-linked transcripts (and XIST if present).  Returns (sex, score)."""
    yg = [g for g in SEX_Y_GENES if g in L.columns]
    xg = [g for g in SEX_X_GENES if g in L.columns]
    if not yg:
        raise ValueError("no Y-chromosome genes in the matrix")
    score = L[yg].mean(1) - (L[xg].mean(1) if xg else 0)
    # two-cluster split (1-D k-means)
    c = np.quantile(score, [.25, .75])
    for _ in range(50):
        lab = np.abs(score.values[:, None] - c[None, :]).argmin(1)
        c_new = np.array([score.values[lab == j].mean() if (lab == j).any() else c[j] for j in (0, 1)])
        if np.allclose(c, c_new):
            break
        c = c_new
    sex = pd.Series(np.where(lab == np.argmax(c), "M", "F"), index=L.index, name="inferred_sex")
    return sex, score.rename("sex_score")


def r6_covariates(run: Run) -> Dict:
    d, cfg = run.d, run.cfg
    run.log("[R6] covariates")
    L = run.logX(); y = d.y.values; v = run.vumc
    P = run.panel_in(d.X)
    out = {}
    try:
        sex, score = infer_sex(L)
    except ValueError as e:
        run.log("  sex inference skipped:", e); sex = None
    if sex is not None:
        ct = pd.crosstab(sex, d.y)
        chi = stats.chi2_contingency(ct.values)
        ctv = pd.crosstab(sex[v], d.y[v])
        chiv = stats.chi2_contingency(ctv.values) if ctv.shape == (2, 2) else (np.nan, np.nan)
        pm = lambda m: 100 * (sex[m] == "M").mean()
        run.macro("SexMaleNsclcPct", f"{pm(y == 1):.1f}"); run.macro("SexMaleHealthyPct", f"{pm(y == 0):.1f}")
        run.macro("SexChiP", pmac(chi[1])); run.macro("SexChiPVumc", pmac(chiv[1]))
        # panel genes associated with sex within class
        sx = []
        for g in P:
            ps = []
            for cls in (0, 1):
                m = (y == cls)
                a_, b_ = L.loc[m & (sex == "M").values, g], L.loc[m & (sex == "F").values, g]
                if len(a_) > 2 and len(b_) > 2:
                    u = stats.mannwhitneyu(a_, b_)
                    ps.append((u.pvalue, u.statistic / (len(a_) * len(b_))))
                else:
                    ps.append((np.nan, np.nan))
            sx.append(dict(gene=g, p_sex_healthy=ps[0][0], auc_sex_healthy=ps[0][1],
                           p_sex_nsclc=ps[1][0], auc_sex_nsclc=ps[1][1]))
        sx = pd.DataFrame(sx)
        sx["q_sex_healthy"] = bh(sx.p_sex_healthy.values); sx["q_sex_nsclc"] = bh(sx.p_sex_nsclc.values)
        sexg = sx[(sx.q_sex_healthy < .05) & (sx.q_sex_nsclc < .05)].gene.tolist()
        run.macro("SexAssocGenes", ", ".join(sexg) if sexg else "none")
        run.macro("SexAssocN", str(len(sexg)))
        run.table(sx, "R6_panel_genes_vs_inferred_sex")
        # sex-stratified and sex-adjusted performance (OOF panel score)
        oof = run.cache.get("oof_P27")
        if oof is None:
            oof, _ = cv_oof(d.X[P].values, y, run.folds(), run.xgb)
        for s_, nm in [("M", "Male"), ("F", "Female")]:
            m = (sex == s_).values
            if y[m].min() != y[m].max():
                a, lo, hi, _ = delong_ci(y[m], oof[m])
                run.macro(f"Sex{nm}AUC", f4(a)); run.macro(f"Sex{nm}CI", ci(lo, hi)); run.macro(f"Sex{nm}N", str(int(m.sum())))
        try:
            import statsmodels.api as sm
            lo_ = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / (1 - np.clip(oof, 1e-6, 1 - 1e-6)))
            Xm = sm.add_constant(np.c_[lo_, (sex == "M").astype(float).values])
            fit = sm.Logit(y, Xm).fit(disp=0)
            run.macro("SexAdjOR", f2(np.exp(fit.params[1]))); run.macro("SexAdjP", pmac(fit.pvalues[1]))
            run.macro("SexTermP", pmac(fit.pvalues[2]))
        except Exception as e:
            run.log("  sex-adjusted model skipped:", e)
        # balance check: sex-balanced subsample inside Vumc
        fig, ax = plt.subplots(figsize=(4, 2.6))
        ax.hist(score[y == 0], bins=40, alpha=.6, label="healthy"); ax.hist(score[y == 1], bins=40, alpha=.6, label="NSCLC")
        ax.set_xlabel("Y-chromosome transcript score"); ax.legend(frameon=False, fontsize=7)
        run.fig(fig, "R6_inferred_sex_score")
        pd.DataFrame({"inferred_sex": sex, "sex_score": score, "y": d.y, "site": d.site}).to_csv(
            os.path.join(run.out, "tables", "R6_inferred_sex.csv"))
        out["sex"] = sex
    # recorded covariates, if any column exists in the metadata
    for cov in ["age", "gender", "sex", "smoking", "smoking status", "smoking_status", "stage"]:
        col = [c for c in d.meta.columns if c.lower() == cov]
        if col:
            run.log(f"  recorded covariate found: {col[0]} ({d.meta[col[0]].notna().mean():.0%} complete)")
            out[f"has_{cov}"] = True
            if sex is not None and cov in ("gender", "sex"):
                rec = d.meta[col[0]].astype(str).str.upper().str[0]
                ok = rec.isin(["M", "F"])
                conc = (rec[ok] == sex[ok]).mean()
                run.macro("SexConcordance", f"{100 * conc:.1f}")
    # smoking-associated genes (exploratory)
    ov = [g for g in P if g in SMOKING_GENES]
    run.macro("SmokingPanelOverlap", ", ".join(ov) if ov else "none")
    sg = [g for g in SMOKING_GENES if g in L.columns]
    if sg:
        sm_score = ((L[sg] - L[sg].mean()) / (L[sg].std() + 1e-9)).mean(1)
        oof = run.cache.get("oof_P27")
        if oof is not None:
            rr = [stats.spearmanr(sm_score[y == c], oof[y == c])[0] for c in (0, 1)]
            run.macro("SmokingScoreRhoHealthy", f2(rr[0])); run.macro("SmokingScoreRhoNsclc", f2(rr[1]))
        u = stats.mannwhitneyu(sm_score[v & (y == 1)], sm_score[v & (y == 0)])
        run.macro("SmokingScoreAUCVumc", f3(u.statistic / ((v & (y == 1)).sum() * (v & (y == 0)).sum())))
        run.macro("SmokingGenesN", str(len(sg)))
    return out


# ========================================================================================
# R7  Biological interpretation: site-controlled pre-ranked GSEA + panel ORA
# ========================================================================================
def r7_enrichment(run: Run) -> Optional[pd.DataFrame]:
    """Pre-ranked GSEA on the within-Vumc (site-constant) Mann-Whitney statistics of ALL genes,
    and over-representation of the panel genes.  Needs `pip install gseapy` and internet
    (Enrichr libraries) or local .gmt files in cfg['gene_sets']."""
    d, cfg = run.d, run.cfg
    run.log("[R7] enrichment")
    try:
        import gseapy as gp
    except ImportError:
        run.log("  gseapy not installed - skipped"); return None
    L = run.logX(); y = d.y.values; v = run.vumc
    A, B = L.values[v & (y == 1)], L.values[v & (y == 0)]
    u = stats.mannwhitneyu(A, B, axis=0)
    auc = u.statistic / (A.shape[0] * B.shape[0])
    stat = -np.log10(np.clip(u.pvalue, 1e-300, 1)) * np.sign(auc - 0.5)
    all_genes = list(L.columns)
    del A, B, L                                   # free memory before the permutation runs
    import gc; gc.collect()
    rnk = pd.Series(stat, index=all_genes).replace([np.inf, -np.inf], np.nan).dropna()
    rnk = rnk[~rnk.index.duplicated()].sort_values(ascending=False)
    rnk.to_csv(os.path.join(run.out, "tables", "R7_vumc_ranked_genes.rnk"), sep="\t", header=False)
    res_all = []
    for gs in cfg["gene_sets"]:
        try:
            pre = gp.prerank(rnk=rnk, gene_sets=gs, permutation_num=1000, min_size=15, max_size=500,
                             seed=cfg["seed"], threads=1, outdir=None, verbose=False)
            r = pre.res2d.copy(); r["library"] = os.path.basename(str(gs))
            res_all.append(r)
        except Exception as e:
            run.log(f"  prerank failed for {gs}: {e!r}")
    out = None
    if res_all:
        out = pd.concat(res_all)
        fdr_col = [c for c in out.columns if "FDR" in c][0]
        out[fdr_col] = pd.to_numeric(out[fdr_col], errors="coerce")
        out["NES"] = pd.to_numeric(out["NES"], errors="coerce")
        out = out.sort_values(fdr_col)
        run.table(out, "R7_prerank_gsea_vumc")
        sig = out[out[fdr_col] < 0.05]
        run.macro("GseaNsig", str(len(sig)))
        up = sig[sig.NES > 0].head(3).Term.tolist(); dn = sig[sig.NES < 0].head(3).Term.tolist()
        run.macro("GseaTopUp", tex_escape("; ".join(up)) if up else "none")
        run.macro("GseaTopDown", tex_escape("; ".join(dn)) if dn else "none")
    # over-representation of the panel (background = all genes tested)
    try:
        P = run.panel_in(d.X)
        ora = []
        for gs in cfg["gene_sets"]:
            try:
                e = gp.enrich(gene_list=P, gene_sets=gs, background=all_genes, outdir=None, verbose=False)
                r = e.results.copy(); r["library"] = os.path.basename(str(gs)); ora.append(r)
            except Exception:
                e = gp.enrichr(gene_list=P, gene_sets=gs, background=all_genes, outdir=None)
                r = e.results.copy(); r["library"] = gs; ora.append(r)
        if ora:
            o = pd.concat(ora).sort_values("Adjusted P-value")
            run.table(o, "R7_panel_ORA")
            run.macro("OraNsig", str(int((o["Adjusted P-value"] < 0.05).sum())))
    except Exception as e:
        run.log("  ORA skipped:", repr(e))
    return out


# ========================================================================================
# R8  Head-to-head re-implementation of Sheng et al. (mRMR-IFS-SVM) on identical folds
# ========================================================================================
def r8_head_to_head(run: Run) -> pd.DataFrame:
    d, cfg = run.d, run.cfg
    run.log("[R8] head-to-head: six-stage + XGBoost vs mRMR-IFS-SVM (Sheng et al.)")
    y = d.y.values
    folds = run.folds()
    pd.DataFrame([(i, f) for f, (_, te) in enumerate(folds) for i in d.X.index[te]],
                 columns=["sample", "outer_fold"]).to_csv(os.path.join(run.out, "tables", "R8_outer_folds.csv"), index=False)
    # six-stage nested OOF: from your own run (cfg['nested_oof']) or re-run
    if cfg.get("nested_oof"):
        no = pd.read_csv(cfg["nested_oof"]).set_index("sample").reindex(d.X.index)
        six = dict(oof=no["oof_prob"].values, yhat=(no["oof_prob"].values >= .5).astype(int))
        run.log("  using your nested OOF predictions for the six-stage pipeline (check folds match!)")
    else:
        six = run.cache.get("nested_six_stage") or nested_cv(run, d.X, y, folds, "six_stage")
        run.cache["nested_six_stage"] = six
    she = nested_cv(run, d.X, y, folds, "sheng")
    run.cache["nested_sheng"] = she
    dp = delong_paired(y, six["oof"], she["oof"])
    rows = []
    for nm, r in [("Six-stage + XGBoost (nested)", six), ("mRMR-IFS-SVM, Sheng et al. (nested)", she)]:
        a, lo, hi, _ = delong_ci(y, r["oof"])
        m = binary_metrics(y, r["yhat"], 0.5)
        rows.append(dict(method=nm, auc=a, ci_lo=lo, ci_hi=hi, accuracy=m["accuracy"],
                         sensitivity=m["sensitivity"], specificity=m["specificity"], mcc=m["mcc"],
                         panel_sizes=";".join(str(len(p)) for p in r.get("panels", []))))
        run.register_auc("Head-to-head (identical outer folds)", nm, a, lo, hi, "DeLong", len(y))
    from statsmodels.stats.contingency_tables import mcnemar
    c1 = (six["yhat"] == y); c2 = (she["yhat"] == y)
    tb = [[np.sum(c1 & c2), np.sum(c1 & ~c2)], [np.sum(~c1 & c2), np.sum(~c1 & ~c2)]]
    p_mc = mcnemar(tb, exact=False, correction=True).pvalue
    h2h = pd.DataFrame(rows)
    h2h["delta_auc_six_minus_sheng"] = dp["diff"]; h2h["delta_lo95"] = dp["lo95"]; h2h["delta_hi95"] = dp["hi95"]
    h2h["p_delong"] = dp["p"]; h2h["p_mcnemar_accuracy"] = p_mc
    run.macro("HhSixAUC", f4(rows[0]["auc"])); run.macro("HhSixCI", ci(rows[0]["ci_lo"], rows[0]["ci_hi"]))
    run.macro("HhShengAUC", f4(rows[1]["auc"])); run.macro("HhShengCI", ci(rows[1]["ci_lo"], rows[1]["ci_hi"]))
    run.macro("HhSixMCC", f3(rows[0]["mcc"])); run.macro("HhShengMCC", f3(rows[1]["mcc"]))
    run.macro("HhSixAcc", f3(rows[0]["accuracy"])); run.macro("HhShengAcc", f3(rows[1]["accuracy"]))
    run.macro("HhDelta", pp(dp["diff"])); run.macro("HhDeltaCI", f"{pp(dp['lo95'])} to {pp(dp['hi95'])}")
    run.macro("HhP", pmac(dp["p"])); run.macro("HhMcnemarP", pmac(p_mc))
    run.macro("HhShengSizes", ", ".join(str(len(p)) for p in she["panels"]))
    verdict = ("higher than" if dp["p"] < .05 and dp["diff"] > 0 else
               "lower than" if dp["p"] < .05 and dp["diff"] < 0 else "not significantly different from")
    run.macro("HhVerdict", verdict)
    # Sheng-style replication on the full cohort (non-nested, as published)
    genes_s, mdl_s, curve = sheng_mrmr_ifs_svm(d.X, y, run, return_curve=True)
    curve.to_csv(os.path.join(run.out, "tables", "R8_sheng_ifs_curve.csv"), index=False)
    run.macro("HhShengFullK", str(len(genes_s)))
    Ls = run.logX()[genes_s].values
    if cfg["sheng"]["loocv"]:
        yh = np.zeros(len(y), int)
        for i in range(len(y)):
            tr = np.r_[0:i, i + 1:len(y)]
            mm = make_pipeline(StandardScaler(), SVC(kernel=cfg["sheng"]["svm_kernel"], C=cfg["sheng"]["svm_C"])).fit(Ls[tr], y[tr])
            yh[i] = mm.predict(Ls[i:i + 1])[0]
        m = binary_metrics(y, yh, .5)
        run.macro("HhShengLoocvAcc", f3(m["accuracy"])); run.macro("HhShengLoocvMCC", f3(m["mcc"]))
        run.macro("HhShengLoocvSens", f3(m["sensitivity"])); run.macro("HhShengLoocvSpec", f3(m["specificity"]))
    # site-constant comparison (Vumc only, fixed panels from the full cohort)
    v = run.vumc
    P = run.panel_in(d.X)
    o1, _ = cv_oof(d.X.loc[v, P].values, y[v], run.folds(y[v]), run.xgb)
    o2, _ = cv_oof(Ls[v], y[v], run.folds(y[v]),
                   lambda: make_pipeline(StandardScaler(), SVC(kernel=cfg["sheng"]["svm_kernel"], C=cfg["sheng"]["svm_C"], probability=False)))
    dv = delong_paired(y[v], o1, o2)
    run.macro("HhVumcSixAUC", f4(dv["auc1"])); run.macro("HhVumcShengAUC", f4(dv["auc2"]))
    run.macro("HhVumcDelta", pp(dv["diff"])); run.macro("HhVumcP", pmac(dv["p"]))
    # external
    if d.X_ext is not None:
        ye = d.y_ext.values
        pe6 = run.cache.get("p_ext_frozen")
        if pe6 is None:
            sh = [g for g in P if g in d.X_ext.columns]
            pe6 = fit_predict(d.X[sh].values, y, d.X_ext[sh].values, run.xgb)
        gs_ext = [g for g in genes_s if g in d.X_ext.columns]
        mdl = make_pipeline(StandardScaler(), SVC(kernel=cfg["sheng"]["svm_kernel"], C=cfg["sheng"]["svm_C"]))
        mdl.fit(run.logX()[gs_ext].values, y)
        pes = mdl.decision_function(run.logX(d.X_ext)[gs_ext].values)
        de = delong_paired(ye, pe6, pes)
        _, lo, hi = bootstrap_auc_ci(ye, pes, cfg["bootstrap_B"], cfg["seed"])
        run.macro("HhExtShengAUC", f4(de["auc2"])); run.macro("HhExtShengCI", ci(lo, hi))
        run.macro("HhExtShengNgenes", f"{len(gs_ext)}/{len(genes_s)}")
        run.macro("HhExtDelta", pp(de["diff"])); run.macro("HhExtP", pmac(de["p"]))
        run.register_auc("Independent platelet cohort", "mRMR-IFS-SVM (Sheng) frozen -> GSE68086",
                         de["auc2"], lo, hi, "bootstrap", len(ye))
    # published 48-gene list, if supplied
    if cfg.get("sheng48"):
        g48 = [l.strip() for l in open(cfg["sheng48"]) if l.strip()]
        g48p = [g for g in g48 if g in d.X.columns]
        run.macro("PubFortyEightFound", f"{len(g48p)}/{len(g48)}")
        o48, _ = cv_oof(run.logX()[g48p].values, y, folds,
                        lambda: make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)))
        o48x, _ = cv_oof(d.X[g48p].values, y, folds, run.xgb)
        run.macro("PubFortyEightSvmAUC", f4(roc_auc_score(y, o48)))
        run.macro("PubFortyEightXgbAUC", f4(roc_auc_score(y, o48x)))
        dd = delong_paired(y, run.cache.get("oof_P27", o48x), o48x)
        run.macro("PubFortyEightDelta", pp(dd["diff"])); run.macro("PubFortyEightP", pmac(dd["p"]))
        o48v, _ = cv_oof(d.X.loc[v, g48p].values, y[v], run.folds(y[v]), run.xgb)
        run.macro("PubFortyEightVumcAUC", f4(roc_auc_score(y[v], o48v)))
    run.table(h2h, "R8_head_to_head")
    write_r8_table(run.out, h2h)
    run.log(h2h.round(4).to_string(index=False))
    return h2h


def write_r8_table(out_dir: str, h2h: pd.DataFrame):
    """LaTeX table of the head-to-head comparison (compact layout)."""
    names = {"Six-stage + XGBoost (nested)": tex_stack("Six-stage selection", "+ XGBoost", align="l"),
             "mRMR-IFS-SVM, Sheng et al. (nested)": tex_stack("mRMR--IFS--SVM", "(Sheng et al.)", align="l")}
    trows = [[names.get(r.method, r.method), tex_auc_ci(r.auc, r.ci_lo, r.ci_hi, 4), f3(r.accuracy), f3(r.sensitivity),
              f3(r.specificity), f3(r.mcc), r.panel_sizes.replace(";", ", ")] for r in h2h.itertuples()]
    write_tex_table(os.path.join(out_dir, "tables", "R8_head_to_head.tex"), trows,
                    ["Method", tex_stack("AUC", "(95\\% CI)"), "Accuracy", tex_stack("Sensi-", "tivity"),
                     tex_stack("Speci-", "ficity"), "MCC", tex_stack("Genes per", "outer fold")], "lcccccl", "\\footnotesize")


# ========================================================================================
# R9  10,000 random and 10,000 expression/variance-matched random panels
# ========================================================================================
def _auc_of_panel(Xv: np.ndarray, y: np.ndarray, folds, params: Dict) -> Tuple[float, float]:
    oof = np.full(len(y), np.nan); fa = []
    for tr, te in folds:
        m = XGBClassifier(**params, n_jobs=1, verbosity=0).fit(Xv[tr], y[tr])
        p = m.predict_proba(Xv[te])[:, 1]; oof[te] = p; fa.append(roc_auc_score(y[te], p))
    return float(np.mean(fa)), float(roc_auc_score(y, oof))


def _matched_sampler(L: pd.DataFrame, panel: List[str], bins: int, rng):
    mu = L.mean(0); var = L.var(0)
    qb = lambda s: pd.qcut(s.rank(method="first"), bins, labels=False)
    key = qb(mu).astype(int) * bins + qb(var).astype(int)
    pool = key[~key.index.isin(panel)]
    groups = {k: list(v) for k, v in pool.groupby(pool).groups.items()}

    def draw():
        out = []
        for g in panel:
            k = key[g]
            cand = [x for x in groups.get(k, []) if x not in out]
            if not cand:  # nearest non-empty bin
                ks = sorted(groups, key=lambda z: abs(z - k))
                cand = [x for kk in ks for x in groups[kk] if x not in out][:50]
            out.append(cand[rng.integers(len(cand))])
        return out
    return draw


def r9_random_panels(run: Run, n_unmatched: Optional[int] = None, n_matched: Optional[int] = None) -> pd.DataFrame:
    """Null distributions with checkpointing (safe for Colab disconnects)."""
    from joblib import Parallel, delayed
    d, cfg = run.d, run.cfg
    n_unmatched = cfg["n_random_panels"] if n_unmatched is None else n_unmatched
    n_matched = cfg["n_matched_panels"] if n_matched is None else n_matched
    run.log(f"[R9] random panels: {n_unmatched} unmatched + {n_matched} matched")
    P = run.panel_in(d.X); y = d.y.values
    folds = run.folds()
    params = dict(cfg["xgb_params"])
    cur_mean, cur_pool = _auc_of_panel(d.X[P].values, y, folds, params)
    L = run.logX()
    rng = np.random.default_rng(cfg["seed"])
    pool = [g for g in d.X.columns if g not in set(P)]
    draw_m = _matched_sampler(L, P, cfg["matched_bins"], rng)
    njobs = cfg["n_jobs"] if cfg["n_jobs"] and cfg["n_jobs"] > 0 else (os.cpu_count() or 1)
    out = []
    for kind, n in [("unmatched", n_unmatched), ("matched", n_matched)]:
        ck = os.path.join(run.out, "cache", f"R9_{kind}.csv")
        done = pd.read_csv(ck) if os.path.exists(ck) else pd.DataFrame(columns=["kind", "rep", "auc_foldmean", "auc_pooled", "genes"])
        start = len(done)
        if start:
            run.log(f"  resuming {kind} from checkpoint at {start}")
            for _ in range(start):          # keep the random stream reproducible
                (rng.choice(pool, len(P), replace=False) if kind == "unmatched" else draw_m())
        chunk = cfg["random_checkpoint_every"]
        rep = start
        while rep < n:
            m = min(chunk, n - rep)
            sets = [list(rng.choice(pool, len(P), replace=False)) if kind == "unmatched" else draw_m() for _ in range(m)]
            res = Parallel(n_jobs=njobs)(delayed(_auc_of_panel)(d.X[s].values, y, folds, params) for s in sets)
            new = pd.DataFrame({"kind": kind, "rep": range(rep, rep + m), "auc_foldmean": [r[0] for r in res],
                                "auc_pooled": [r[1] for r in res], "genes": [";".join(s) for s in sets]})
            done = pd.concat([done, new], ignore_index=True)
            done.to_csv(ck, index=False)
            rep += m
            run.log(f"  {kind}: {rep}/{n}")
        out.append(done)
    allr = pd.concat(out, ignore_index=True)
    allr.to_csv(os.path.join(run.out, "tables", "R9_random_panels.csv"), index=False)
    rows = []
    for kind, nm in [("unmatched", "Rand"), ("matched", "Match")]:
        a = allr[allr.kind == kind].auc_foldmean.astype(float).values
        if len(a) == 0:
            continue
        b = int((a >= cur_mean).sum())
        p = (b + 1) / (len(a) + 1)
        rows.append(dict(kind=kind, n=len(a), mean=a.mean(), sd=a.std(ddof=1), q95=np.quantile(a, .95),
                         q999=np.quantile(a, .999), max=a.max(), curated=cur_mean, b=b, p_empirical=p,
                         z=(cur_mean - a.mean()) / a.std(ddof=1), advantage_pp=100 * (cur_mean - a.mean())))
        run.macro(f"{nm}N", f"{len(a):,}".replace(",", "{,}")); run.macro(f"{nm}Mean", f4(a.mean()))
        run.macro(f"{nm}SD", f4(a.std(ddof=1))); run.macro(f"{nm}Max", f4(a.max()))
        run.macro(f"{nm}QNineNineNine", f4(np.quantile(a, .999)))
        run.macro(f"{nm}B", str(b)); run.macro(f"{nm}P", pmac(p, None))
        run.macro(f"{nm}Exceeded", f"{len(a) - b:,}".replace(",", "{,}"))
        run.macro(f"{nm}PFrac", f"{b + 1}/{len(a) + 1:,}".replace(",", "{,}"))
        run.macro(f"{nm}Z", f"{(cur_mean - a.mean()) / a.std(ddof=1):.1f}")
        run.macro(f"{nm}Adv", f"{100 * (cur_mean - a.mean()):.2f}")
    run.macro("RandCurated", f4(cur_mean))
    summ = pd.DataFrame(rows)
    run.table(summ, "R9_random_panels_summary")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=False)
    for ax, kind, ttl in [(axs[0], "unmatched", "A  Random 27-gene panels"),
                          (axs[1], "matched", "B  Expression/variance-matched panels")]:
        a = allr[allr.kind == kind].auc_foldmean.astype(float)
        if len(a) == 0:
            continue
        ax.hist(a, bins=60, color="#B8B8B8")
        ax.axvline(cur_mean, c="#C0392B", lw=2)
        ax.set_title(f"{ttl} (n = {len(a):,})", fontsize=9, loc="left")
        ax.set_xlabel("Five-fold CV AUC"); ax.text(cur_mean, ax.get_ylim()[1] * .9, " 27-gene panel", color="#C0392B", fontsize=8)
    run.fig(fig, "R9_random_panels")
    run.log(summ.round(4).to_string(index=False))
    return summ


# ========================================================================================
# R10 Tissue transfer: is the failure about platform or about biology?
# ========================================================================================
def r10_tissue(run: Run) -> Optional[pd.DataFrame]:
    d = run.d
    if not d.tissue:
        run.log("[R10] no tissue cohorts supplied - skipped"); return None
    run.log("[R10] tissue-transfer controls")
    P = run.panel_in(d.X)
    L = run.logX(); y = d.y.values; v = run.vumc
    R = run.rawX()
    lfc_pl = np.log2(R.loc[v & (y == 1), P].mean() + 1) - np.log2(R.loc[v & (y == 0), P].mean() + 1)
    rows = []
    names = list(d.tissue)
    for nm in names:
        Xt, yt = d.tissue[nm]
        Xt = Xt if Xt.max().max() < 50 else np.log2(Xt.clip(lower=0) + 1)     # microarray already log2
        g = [x for x in P if x in Xt.columns]
        # (a) within-tissue CV with the panel genes
        o, _ = cv_oof(Xt[g].values, yt.values, run.folds(yt.values), run.xgb)
        a, lo, hi, _ = delong_ci(yt.values, o)
        # (b) direction concordance with platelets (Vumc, site constant)
        lfc_t = Xt.loc[yt.values == 1, g].mean() - Xt.loc[yt.values == 0, g].mean()
        conc = int((np.sign(lfc_t) == np.sign(lfc_pl[g])).sum())
        pb = stats.binomtest(conc, len(g), 0.5).pvalue
        # (c) frozen platelet model with single-sample ranks (platform-agnostic)
        common = [x for x in L.columns if x in Xt.columns]
        Rd, Rt = L[common].rank(axis=1, pct=True), Xt[common].rank(axis=1, pct=True)
        pr = fit_predict(Rd[g].values, y, Rt[g].values, run.xgb)
        ar = roc_auc_score(yt.values, pr)
        rows.append(dict(cohort=nm, n=len(yt), n_genes=len(g), within_tissue_cv_auc=a, ci_lo=lo, ci_hi=hi,
                         direction_concordant=conc, binom_p=pb, platelet_rank_model_auc=ar))
        run.register_auc("Lung tissue", f"{nm}: panel genes trained within tissue (5-fold CV)", a, lo, hi, "DeLong", len(yt))
    # (d) tissue -> tissue positive control
    if len(names) >= 2:
        for a_, b_ in [(names[0], names[1]), (names[1], names[0])]:
            Xa, ya = d.tissue[a_]; Xb, yb = d.tissue[b_]
            Xa = Xa if Xa.max().max() < 50 else np.log2(Xa.clip(lower=0) + 1)
            Xb = Xb if Xb.max().max() < 50 else np.log2(Xb.clip(lower=0) + 1)
            g = [x for x in P if x in Xa.columns and x in Xb.columns]
            Za = (Xa[g] - Xa[g].mean()) / Xa[g].std(); Zb = (Xb[g] - Xb[g].mean()) / Xb[g].std()
            p = fit_predict(Za.values, ya.values, Zb.values, run.xgb)
            rows.append(dict(cohort=f"{a_} -> {b_} (tissue-to-tissue)", n=len(yb), n_genes=len(g),
                             within_tissue_cv_auc=np.nan, platelet_rank_model_auc=np.nan,
                             tissue_to_tissue_auc=roc_auc_score(yb.values, p)))
    df = pd.DataFrame(rows)
    run.table(df, "R10_tissue_controls")
    for i, r in df.iterrows():
        tag = re.sub(r"[^A-Za-z]", "", {0: "One", 1: "Two", 2: "OneToTwo", 3: "TwoToOne"}.get(i, "X"))
        if pd.notna(r.get("within_tissue_cv_auc", np.nan)):
            run.macro(f"Tissue{tag}CvAUC", f4(r.within_tissue_cv_auc))
            run.macro(f"Tissue{tag}Conc", f"{int(r.direction_concordant)}/{int(r.n_genes)}")
            run.macro(f"Tissue{tag}RankAUC", f4(r.platelet_rank_model_auc))
        if pd.notna(r.get("tissue_to_tissue_auc", np.nan)):
            run.macro(f"Tissue{tag}AUC", f4(r.tissue_to_tissue_auc))
    run.log(df.round(4).to_string(index=False))
    return df


# ========================================================================================
# Minor points: out-of-fold SHAP, AUC summary with CIs, simplified figures
# ========================================================================================
def shap_out_of_fold(run: Run) -> pd.DataFrame:
    """SHAP computed on held-out folds only (reduces post-selection optimism of the ranking)."""
    import shap
    d = run.d
    run.log("[minor] out-of-fold SHAP")
    P = run.panel_in(d.X); X = d.X[P].values; y = d.y.values
    S = np.zeros_like(X, dtype=float)
    for tr, te in run.folds():
        m = run.xgb().fit(X[tr], y[tr])
        S[te] = shap.TreeExplainer(m).shap_values(X[te])
    oof_imp = np.abs(S).mean(0)
    m_all = run.xgb().fit(X, y)
    ins = np.abs(shap.TreeExplainer(m_all).shap_values(X)).mean(0)
    rng = np.random.default_rng(run.cfg["seed"]); ranks = []
    for _ in range(500):
        i = rng.integers(0, len(y), len(y))
        ranks.append(stats.rankdata(-np.abs(S[i]).mean(0)))
    ranks = np.array(ranks)
    df = pd.DataFrame(dict(gene=P, mean_abs_shap_oof=oof_imp, mean_abs_shap_insample=ins,
                           rank_oof=stats.rankdata(-oof_imp), rank_insample=stats.rankdata(-ins),
                           rank_ci_lo=np.quantile(ranks, .025, axis=0), rank_ci_hi=np.quantile(ranks, .975, axis=0)))
    df = df.sort_values("rank_oof")
    run.table(df, "minor_shap_out_of_fold")
    rho = stats.spearmanr(df.rank_oof, df.rank_insample)[0]
    run.macro("ShapRho", f2(rho)); run.macro("ShapTopFive", ", ".join(df.gene.head(5)))
    missing = run.cache.get("ext_missing") or []
    if missing:
        run.macro("ShapMissingShare", f"{100 * df.set_index('gene').loc[[g for g in missing if g in P], 'mean_abs_shap_oof'].sum() / df.mean_abs_shap_oof.sum():.1f}")
    fig, ax = plt.subplots(figsize=(4.2, 5.2))
    dd = df.iloc[::-1]
    ax.barh(dd.gene, dd.mean_abs_shap_oof, color="#5D6D7E")
    ax.set_xlabel("Mean |SHAP| on held-out folds"); ax.tick_params(axis="y", labelsize=7)
    run.fig(fig, "minor_shap_out_of_fold")
    return df


def auc_summary(run: Run) -> pd.DataFrame:
    df = pd.DataFrame(run.aucs)
    if df.empty:
        return df
    run.table(df, "ALL_AUCs_with_95CI")
    trows = []
    for lv in dict.fromkeys(df.level):
        trows.append([f"\\multicolumn{{4}}{{l}}{{\\textit{{{tex_escape(lv)}}}}}"] )
        for r in df[df.level == lv].itertuples():
            trows.append([tex_escape(r.analysis).replace("->", "$\\to$"), f"{f4(r.auc)} ({ci(r.ci_lo, r.ci_hi)})",
                          r.ci_method, str(r.n)])
    path = os.path.join(run.out, "tables", "ALL_AUCs_with_95CI.tex")
    write_tex_table(path, trows, ["Analysis", "AUC (95\\% CI)", "CI method", "$n$"], "p{9.2cm}rlr", "\\footnotesize")
    with open(path) as fh:
        txt = fh.read()
    with open(path, "w") as fh:        # multicolumn rows have only one cell
        fh.write(txt)
    fig_auc_forest(run, df)
    return df


def fig_auc_forest(run: Run, df: pd.DataFrame, name: str = "forest_all_aucs"):
    """Simplified Figure 16: every AUC with its 95% CI, grouped by evaluation level."""
    levels = list(dict.fromkeys(df.level))
    fig, ax = plt.subplots(figsize=(7.2, 0.33 * len(df) + 1.2))
    ypos = 0; yt, yl = [], []
    cols = {"Within cohort": "#1F618D", "Site-aware audit": "#7D3C98", "Independent platelet cohort": "#C0392B",
            "Head-to-head (identical outer folds)": "#117A65", "Lung tissue": "#7F8C8D"}
    for lv in levels:
        sub = df[df.level == lv]
        ax.text(0.405, ypos - .45, lv, fontsize=8, fontweight="bold", va="bottom")
        for _, r in sub.iterrows():
            c = cols.get(lv, "k")
            ax.plot([r.ci_lo, r.ci_hi], [ypos, ypos], c=c, lw=1.6)
            ax.plot(r.auc, ypos, "o", c=c, ms=4.5)
            ax.text(1.005, ypos, f"{r.auc:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})", fontsize=6.5, va="center")
            yt.append(ypos); yl.append(r.analysis[:62]); ypos += 1
        ypos += 1.0
    ax.axvline(.5, ls=":", c="k", lw=.8)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlim(.4, 1.0); ax.set_xlabel("AUC (95% CI)")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    run.fig(fig, name)


def fig_nested_summary(fold_aucs: Sequence[float], fold_jacc: Sequence[float], fixed: float, nested: float,
                       external: float, out_path: str):
    """Simplified Figure 10: one panel, AUC (left axis) and Jaccard (right axis) per fold."""
    f = np.arange(1, len(fold_aucs) + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(f, fold_aucs, "o", c="#1F618D", ms=7, label="held-out AUC")
    ax.axhline(fixed, c="#1F618D", ls="--", lw=1); ax.text(5.3, fixed, f"fixed {fixed:.3f}", fontsize=7, va="center", color="#1F618D")
    ax.axhline(nested, c="#1F618D", ls=":", lw=1); ax.text(5.3, nested, f"nested {nested:.3f}", fontsize=7, va="center", color="#1F618D")
    ax.set_ylim(0, 1.02); ax.set_xlabel("Nested outer fold"); ax.set_ylabel("AUC", color="#1F618D")
    ax2 = ax.twinx()
    ax2.bar(f, fold_jacc, width=.35, color="#C0392B", alpha=.55, label="Jaccard vs original")
    ax2.set_ylim(0, 1.02); ax2.set_ylabel("Jaccard with 27-gene panel", color="#C0392B")
    ax.set_xlim(.4, 6.2); ax.set_xticks(f)
    for s in ["top"]:
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def fig_rfecv(curve_csv: str, out_path: str, panel_n: int = 27, panel_auc: Optional[float] = None):
    """Simplified Figure 15 from your RFECV curve (columns: n_features, mean_auc[, sd_auc])."""
    c = pd.read_csv(curve_csv)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(c.iloc[:, 0], c.iloc[:, 1], c="#1F618D", lw=1.5)
    if c.shape[1] > 2:
        ax.fill_between(c.iloc[:, 0], c.iloc[:, 1] - c.iloc[:, 2], c.iloc[:, 1] + c.iloc[:, 2], alpha=.15)
    best = c.iloc[c.iloc[:, 1].idxmax()]
    ax.plot(best.iloc[0], best.iloc[1], "o", c="k"); ax.annotate(f"optimum: {int(best.iloc[0])} genes", (best.iloc[0], best.iloc[1]),
                                                               xytext=(5, -12), textcoords="offset points", fontsize=7)
    if panel_auc:
        ax.plot(panel_n, panel_auc, "D", c="#C0392B"); ax.annotate(f"27-gene panel {panel_auc:.3f}", (panel_n, panel_auc),
                                                                   xytext=(5, 5), textcoords="offset points", fontsize=7, color="#C0392B")
    ax.set_xscale("log"); ax.set_xlabel("Number of genes (log scale)"); ax.set_ylabel("CV AUC")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


# ========================================================================================
# Run everything
# ========================================================================================
ALL_STEPS = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "shap"]


def run_all(cfg: Dict, steps: Sequence[str] = ALL_STEPS, select_fn: Optional[Callable] = None) -> Run:
    data = load_data(cfg)
    run = Run(cfg, data, select_fn)
    t0 = time.time()

    def step(name, fn, *a, **k):
        if name.split(".")[0] not in steps:
            return None
        t = time.time()
        try:
            r = fn(*a, **k)
        except Exception as e:  # keep going; the log says what failed
            import traceback
            run.log(f"!! step {name} failed: {e!r}\n{traceback.format_exc()}")
            r = None
        run.log(f"   ({name} took {time.time() - t:.0f}s)")
        run.write_macros()
        return r

    step("r0", r0_core_estimates, run)
    step("r1.table", r1_site_table, run)
    step("r1.genes", r1_gene_effects, run)
    step("r1.vumc", r1_vumc_rederivation, run)
    step("r1.nested", r1_vumc_nested, run)
    step("r2.equiv", r2_panel_equivalence, run)
    step("r2.ops", r2_external_operating, run)
    step("r3", r3_harmonisation_audit, run)
    step("r8", r8_head_to_head, run)          # before R4: provides the nested six-stage panels
    step("r4", r4_stability, run)
    step("r5", r5_contamination, run)
    step("r6", r6_covariates, run)
    step("r7", r7_enrichment, run)
    step("r9", r9_random_panels, run)
    step("r10", r10_tissue, run)
    step("shap", shap_out_of_fold, run)
    auc_summary(run)
    run.write_macros()
    run.log(f"=== finished in {(time.time() - t0) / 60:.1f} min ===")
    return run
