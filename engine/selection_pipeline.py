#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selection_pipeline.py
=====================
Seeded, deterministic implementation of the six-stage selection pipeline. It follows the development
notebook (nsclc_phase_1.ipynb, cells 1, 5 and 6) step by step and is used for every analysis that
re-runs the selection (nested CV, Vumc-only re-derivation, stability, head-to-head).

Stage 1  Mann-Whitney U on the processed matrix + BH; Tier 2 = padj < 0.05 and |log2FC| > 0.5,
         log2FC = log2((mean_NSCLC + 1e-6) / (mean_healthy + 1e-6))                [as in the notebook]
Stage 2  LassoCV  intersect  ElasticNetCV(l1_ratio 0.5/0.7/0.9) on standardised Tier-2 genes,
         StratifiedKFold(5, shuffle, seed 42), 50 alphas, max_iter 10,000, tol 1e-3   [as in the notebook]
Stage 3  protein-coding / artefact name filter                                     [as in the notebook]
Stage 4  MRMR (mutual information, top 30) - in the development notebook this only decided which
         genes received a STRING PPI score                                         [as in the notebook]
Stage 5  B bootstrap resamples x LassoCV(cv=3, 20 alphas, max_iter 3,000, tol 1e-2);
         stability = selection frequency; stable = >= 0.60
         CHANGE: resample b uses np.random.RandomState(b) (the notebook was unseeded, so two
         runs of it gave panels that differ by two genes)
Stage 6  composite = .20 (1-padj) + .15 min(|log2FC|,3)/max + .15 |LASSO coef|/max
                     + .30 stability + .10 PPI + .10 G106 prior (1.0 if in G106 else 0.2);
         top 30 by composite, then artefact QC (is_clean_gene)                     [as in the notebook]

`use_ppi` / `use_prior` switch the two terms that cannot be recomputed inside a training fold:
the STRING query (no internet here; for the full-cohort run the panel's PPI scores can be
passed in) and membership of G106, the 106-gene consensus that was itself derived from all 633
labelled samples (using it inside nested CV would leak test-fold labels).  When a term is off its
weight is dropped; the ranking is unaffected by re-normalising the remaining weights.
"""
from __future__ import annotations

import re
import time
import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

warnings.filterwarnings("ignore")

# ---- the G106 list exactly as in nsclc_phase_1.ipynb (list(set([...]))) ----
G106 = sorted(set([
    'TPH1', 'HRASLS2', 'RIMBP2', 'MYO3B', 'ACOT7', 'PLEKHA3', 'PGM1', 'BCAT1',
    'RPS6KA6', 'GGT1', 'CNNM2', 'TRHDE', 'NCKAP1', 'ACADVL', 'JOSD1', 'PRSS50',
    'ST3GAL6', 'PICALM', 'MYH9', 'LGALS3BP', 'ZNF296', 'ZMYND12', 'ATP2A3',
    'CACNA1I', 'P2RY10', 'NOMO2', 'SP100', 'TNPO1', 'ZFYVE21', 'PCDH9', 'TTLL7',
    'DHX29', 'PPIE', 'RNF24', 'MACC1', 'C11orf24', 'RPL31', 'WBP11', 'TBL1X',
    'MAOB', 'TMC1', 'WDR1', 'GCKR', 'ZC3H12B', 'ZNF385D', 'DAZAP1', 'CD59',
    'VWA8', 'ADIPOR2', 'MS4A1', 'PREP', 'CTCF', 'RAB12', 'COQ9', 'ACD',
    'ITGB3BP', 'TNRC6C', 'CDIP1', 'COMT', 'KATNAL1', 'NME4', 'TMEM63B',
    'DUOXA1', 'HSD17B3', 'ISYNA1', 'USP27X', 'JAK3', 'STAP1', 'SKIV2L2',
    'POLR2B', 'ANO2', 'VPS9D1', 'MGAT5', 'TPM2', 'DLK1', 'FCER1A', 'WFDC3',
    'TBPL1', 'TREML4', 'CRYM', 'UBE2W', 'LIPC', 'CPQ', 'HCFC1R1', 'ULK1',
    'ISOC2', 'FAM214B', 'FAM76A', 'CAPN1', 'BID', 'RNH1', 'EHD2', 'RRAGD',
    'IFNGR1', 'ANK1', 'FAM184B',
]))

NON_CODING_PREFIXES = ('MIR', 'RNU', 'RNY', 'RNR', 'LOC', 'LINC', 'SNOR', 'SCARNA',
                       'NOMO1.1', 'HSPA1B.1', 'MPIG6B.1')
DEFAULT_WEIGHTS = dict(stat=0.20, effect=0.15, lasso=0.15, boot=0.30, ppi=0.10, prior=0.10)


def is_protein_coding(g: str) -> bool:
    """Stage-3 filter of the original code."""
    return not any(g.upper().startswith(p) for p in NON_CODING_PREFIXES)


def is_clean_gene(g: str) -> bool:
    """Final artefact QC of the original code (removed KIR2DS2.2, GOLGA8J.1, CNOT3.3)."""
    nc = ('MIR', 'RNU', 'RNY', 'RNR', 'LOC', 'LINC', 'SNOR', 'SCARNA', 'NOMO1.', 'MPIG6B.')
    if any(g.upper().startswith(p) for p in nc):
        return False
    if re.search(r'-AS\d*$', g, re.IGNORECASE):
        return False
    m = re.search(r'\.(\d+)$', g)
    if m and int(m.group(1)) >= 2:
        return False
    if g.startswith('KIR') and m:
        return False
    if re.search(r'GOLGA\d+[A-Z]\.\d+$', g):
        return False
    return True


def stage1_mw(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Vectorised version of the original per-gene loop (identical numbers)."""
    y = np.asarray(y).astype(int)
    V = X.values
    A, B = V[y == 1], V[y == 0]
    keep = ~((A.std(0) == 0) & (B.std(0) == 0))
    genes = X.columns[keep]
    p = stats.mannwhitneyu(A[:, keep], B[:, keep], axis=0, alternative='two-sided').pvalue
    lfc = np.log2((A[:, keep].mean(0) + 1e-6) / (B[:, keep].mean(0) + 1e-6))
    padj = multipletests(p, method='fdr_bh')[1]
    mw = pd.DataFrame({'pvalue': p, 'padj': padj, 'log2FC': lfc}, index=genes)
    return mw.sort_values('padj')


def mrmr_select(X: np.ndarray, y: np.ndarray, names: Sequence[str], n: int = 30,
                random_state: int = 42) -> List[str]:
    """MRMR exactly as in the original code, with the pairwise MI values cached
    (mutual_info_regression with a fixed random_state is deterministic, so caching does not
    change the result, only the run time)."""
    relevance = mutual_info_classif(X, y, random_state=random_state)
    selected, remaining = [], list(range(X.shape[1]))
    cache: Dict = {}

    def mi(i, j):
        k = (i, j)
        if k not in cache:
            cache[k] = mutual_info_regression(X[:, i].reshape(-1, 1), X[:, j],
                                              random_state=random_state)[0]
        return cache[k]

    for _ in range(min(n, len(remaining))):
        if not selected:
            best = remaining[int(np.argmax([relevance[i] for i in remaining]))]
        else:
            scores = [relevance[i] - np.mean([mi(i, j) for j in selected]) for i in remaining]
            best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return [names[i] for i in selected]


_STAGE12_CACHE: Dict = {}


def _data_key(X: pd.DataFrame, y: np.ndarray, seed: int) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update("|".join(map(str, X.index)).encode()); h.update("|".join(map(str, X.columns)).encode())
    h.update(np.asarray(y).astype(np.int8).tobytes()); h.update(str(seed).encode())
    h.update(str(X.shape).encode())
    return h.hexdigest()


def _boot_one(b: int, Xb: np.ndarray, yb: np.ndarray) -> np.ndarray:
    idx = np.random.RandomState(b).choice(len(yb), len(yb), replace=True)
    Xs = StandardScaler().fit_transform(Xb[idx])
    m = LassoCV(cv=3, max_iter=3000, tol=1e-2, alphas=20, random_state=b)
    m.fit(Xs, yb[idx])
    return m.coef_ != 0


def phase1_select(X: pd.DataFrame, y, run=None, B: Optional[int] = None,
                  use_ppi: bool = False, ppi_scores: Optional[Dict[str, float]] = None,
                  use_prior: bool = False, prior_genes: Sequence[str] = G106,
                  run_mrmr: Optional[bool] = None, weights: Optional[Dict[str, float]] = None,
                  top_n: int = 30, stable_thr: float = 0.60, seed: int = 42,
                  n_jobs: int = 2, return_details: bool = False, verbose: bool = False):
    """Run Stages 1-6 on (X, y) and return the panel (list of genes).
    X: samples x genes on the processed scale used in the paper (NOT log-transformed)."""
    t0 = time.time()
    B = B or 500
    y = np.asarray(y).astype(int)
    W = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if not use_ppi:
        W['ppi'] = 0.0
    if not use_prior:
        W['prior'] = 0.0
    log = {}
    key = _data_key(X, y, seed)
    if key in _STAGE12_CACHE:                     # identical data: Stages 1-2 are deterministic
        mw, t2, tier2, consensus, lasso_coef, l2 = _STAGE12_CACHE[key]
        log.update(l2)
    else:
        # ---- Stage 1
        mw = stage1_mw(X, y)
        t2 = mw[(mw.padj < 0.05) & (mw.log2FC.abs() > 0.5)]
        tier2 = t2.index.tolist()
        # ---- Stage 2
        Z = StandardScaler().fit_transform(X[tier2].values)
        cv5 = StratifiedKFold(5, shuffle=True, random_state=seed)
        lasso = LassoCV(cv=cv5, max_iter=10000, tol=1e-3, alphas=50, n_jobs=n_jobs, random_state=seed)
        lasso.fit(Z, y)
        enet = ElasticNetCV(cv=cv5, max_iter=10000, tol=1e-3, alphas=50, l1_ratio=[0.5, 0.7, 0.9],
                            n_jobs=n_jobs, random_state=seed)
        enet.fit(Z, y)
        lasso_nz = set(np.array(tier2)[lasso.coef_ != 0])
        enet_nz = set(np.array(tier2)[enet.coef_ != 0])
        consensus = [g for g in tier2 if g in lasso_nz and g in enet_nz]   # Tier-2 (padj) order
        lasso_coef = dict(zip(tier2, lasso.coef_))
        l2 = dict(stage1_n=len(tier2), stage2_lasso_n=len(lasso_nz), stage2_enet_n=len(enet_nz),
                  stage2_n=len(consensus), lasso_alpha=float(lasso.alpha_), enet_alpha=float(enet.alpha_),
                  enet_l1_ratio=float(enet.l1_ratio_), stage2_seconds=round(time.time() - t0, 1))
        log.update(l2)
        _STAGE12_CACHE[key] = (mw, t2, tier2, consensus, lasso_coef, l2)
    # ---- Stage 3
    coding = [g for g in consensus if is_protein_coding(g)]
    log['stage3_n'] = len(coding)
    if len(coding) < 5:
        raise RuntimeError(f"Stage 3 left only {len(coding)} genes")
    Xc = X[coding].values
    # ---- Stage 4 (MRMR; only feeds the PPI term)
    do_mrmr = use_ppi if run_mrmr is None else run_mrmr
    mrmr = mrmr_select(Xc, y, coding, n=min(30, len(coding))) if do_mrmr else []
    log['mrmr'] = mrmr
    # ---- Stage 5 (bootstrap stability, seeded)
    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        sel = Parallel(n_jobs=n_jobs)(delayed(_boot_one)(b, Xc, y) for b in range(B))
    else:
        sel = [_boot_one(b, Xc, y) for b in range(B)]
    freq = np.vstack(sel).mean(0)
    stability = pd.Series(freq, index=coding)
    stable = [g for g in coding if stability[g] >= stable_thr]
    log['stage5_n'] = len(stable)
    # ---- Stage 6 (composite)
    working = [g for g in stable if g in t2.index]
    if len(working) < 10:
        working = mrmr if mrmr else coding
    sc = pd.DataFrame(index=working)
    sc['stat_score'] = (1 - mw.loc[working, 'padj'].fillna(1.0)).clip(0, 1)
    fc = mw.loc[working, 'log2FC'].abs().clip(upper=3)
    sc['effect_score'] = fc / fc.max()
    lv = pd.Series({g: abs(lasso_coef.get(g, 0.0)) for g in working})
    sc['lasso_score'] = lv / (lv.max() + 1e-9)
    sc['boot_score'] = stability.reindex(working).fillna(0)
    if use_ppi:
        pp = pd.Series(ppi_scores or {}, dtype=float)
        sc['ppi_score'] = [float(pp.get(g, 0.0)) if g in mrmr else 0.0 for g in working]
    else:
        sc['ppi_score'] = 0.0
    sc['ml_prior'] = [1.0 if g in set(prior_genes) else 0.2 for g in working] if use_prior else 0.0
    sc['composite'] = (W['stat'] * sc.stat_score + W['effect'] * sc.effect_score +
                       W['lasso'] * sc.lasso_score + W['boot'] * sc.boot_score +
                       W['ppi'] * sc.ppi_score + W['prior'] * sc.ml_prior)
    sc['log2FC'] = mw.loc[working, 'log2FC']
    sc['padj'] = mw.loc[working, 'padj']
    sc = sc.assign(_g=sc.index).sort_values(['composite', '_g'], ascending=[False, True]).drop(columns='_g')
    top = sc.head(top_n) if len(sc) >= 20 else sc[sc.composite >= 0.35]
    panel = [g for g in top.index if is_clean_gene(g)]
    log['final_n'] = len(panel)
    log['seconds'] = round(time.time() - t0, 1)
    if verbose or (run is not None and hasattr(run, 'log')):
        msg = (f"      phase1_select: Tier2={log['stage1_n']} -> LASSO {log['stage2_lasso_n']} / "
               f"ENet {log['stage2_enet_n']} -> consensus {log['stage2_n']} -> coding "
               f"{log['stage3_n']} -> stable {log['stage5_n']} -> panel {log['final_n']} "
               f"(B={B}, {log['seconds']}s)")
        (run.log if run is not None and hasattr(run, 'log') else print)(msg)
    if return_details:
        return panel, dict(log=log, mw=mw, scores=sc, stability=stability, consensus=consensus,
                           coding=coding, lasso_coef=lasso_coef)
    return panel


def make_select_fn(B_default: int = 500, **kw):
    """Adapter with the signature expected by tep_analysis.Run: f(X, y, run, B=None)."""
    def f(X, y, run=None, B=None):
        return phase1_select(X, y, run=run, B=B or B_default, **kw)
    f.__name__ = "phase1_select"
    return f
