"""Gene-panel selection procedures behind one interface.

Every selector receives the training data only (samples x genes on the processed scale) and returns an
ordered gene list. ``fit_predict`` trains the selector's classifier on the training samples and scores
the held-out samples; ``inner_threshold`` derives a decision threshold from training data only (Youden
point of five-fold out-of-fold scores), so that no held-out label is used for any choice.

Selectors:
    six_stage        the paper's procedure (Stages 1-6, four-term composite; engine ``phase1_select``)
    mw_topk          univariate filter: Mann-Whitney + BH, top k by adjusted p (then |log2FC|)
    l1_topk          L1-penalised logistic regression on the Tier-2 pool, C by inner CV among the values that
                     keep at least k genes, top k by |coef|
    stability        stability selection (Meinshausen & Buehlmann 2010): L1-logistic on half-subsamples of
                     the training data, genes ranked by selection probability, top k
    mrmr_ifs_svm     re-implementation of Sheng et al. (2018) (engine ``sheng_mrmr_ifs_svm``), linear SVM
    random           k genes drawn uniformly from the Tier-2 pool (null reference)
    fixed            a given panel (no selection; used for the panel-fixed arm)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .engine import S, T
from .metrics.predictive import youden_threshold


# --------------------------------------------------------------------------- shared helpers
def tier2(X: pd.DataFrame, y: np.ndarray):
    """Stage-1 statistics of the engine and the Tier-2 pool (padj < 0.05 and |log2FC| > 0.5)."""
    mw = S.stage1_mw(X, y)
    pool = mw[(mw.padj < 0.05) & (mw.log2FC.abs() > 0.5)]
    keep = [g for g in pool.index if S.is_protein_coding(g) and S.is_clean_gene(g)]
    return mw, pool.loc[keep]


def _clean(genes: Sequence[str]) -> List[str]:
    return [g for g in genes if S.is_protein_coding(g) and S.is_clean_gene(g)]


class Selector:
    name = "base"
    classifier = "xgb"          # 'xgb' on processed values, or 'svm' (selector supplies the model)

    def __init__(self, k: int = 27, seed: int = 42, n_jobs: int = 2, **kw):
        self.k, self.seed, self.n_jobs, self.kw = k, seed, n_jobs, kw
        self.info: Dict = {}

    def select(self, X: pd.DataFrame, y: np.ndarray, run) -> List[str]:
        raise NotImplementedError

    # ---- classifier
    def fit_predict(self, Xtr, ytr, Xte, genes, run) -> np.ndarray:
        m = run.xgb(n_jobs=self.n_jobs).fit(Xtr[genes].values, ytr)
        return m.predict_proba(Xte[genes].values)[:, 1]

    def inner_threshold(self, Xtr, ytr, genes, run, folds: int = 5) -> float:
        oof = np.zeros(len(ytr))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=self.seed).split(Xtr, ytr):
            oof[te] = self.fit_predict(Xtr.iloc[tr], ytr[tr], Xtr.iloc[te], genes, run)
        return youden_threshold(ytr, oof)


class SixStage(Selector):
    name = "six_stage"

    def select(self, X, y, run):
        panel, det = S.phase1_select(X, y, B=self.kw.get("B", 500), n_jobs=self.n_jobs, seed=self.seed,
                                     return_details=True)
        self.info = {k: v for k, v in det["log"].items() if k.endswith("_n")}
        return list(panel)


class MWTopK(Selector):
    name = "mw_topk"

    def select(self, X, y, run):
        mw, _ = tier2(X, y)
        mw = mw.assign(_a=-mw.log2FC.abs(), _g=mw.index).sort_values(["padj", "_a", "_g"])
        return _clean(mw.index)[: self.k]


def _standardise(X: pd.DataFrame, genes: Sequence[str]) -> np.ndarray:
    return StandardScaler().fit_transform(X[list(genes)].values)


class L1TopK(Selector):
    name = "l1_topk"
    GRID = np.logspace(-2.5, 0.5, 7)

    def select(self, X, y, run):
        _, pool = tier2(X, y)
        genes = list(pool.index); Z = _standardise(X, genes)
        cv = list(StratifiedKFold(3, shuffle=True, random_state=self.seed).split(Z, y))
        fit = lambda C, A, b: LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000,
                                                 random_state=self.seed).fit(A, b)
        cvauc, nnz = [], []
        for C in self.GRID:
            cvauc.append(np.mean([roc_auc_score(y[te], fit(C, Z[tr], y[tr]).decision_function(Z[te])) for tr, te in cv]))
            nnz.append(int((fit(C, Z, y).coef_ != 0).sum()))
        # best inner-CV AUC among the C values that keep at least k genes (else the least sparse C)
        ok = [i for i in range(len(self.GRID)) if nnz[i] >= min(self.k, len(genes))]
        i = max(ok, key=lambda j: (cvauc[j], -j)) if ok else int(np.argmax(nnz))
        best = (float(cvauc[i]), float(self.GRID[i]))
        m = fit(best[1], Z, y)
        w = np.abs(m.coef_.ravel())
        order = sorted([i for i in range(len(genes)) if w[i] > 0], key=lambda i: (-w[i], genes[i]))
        self.info = {"C": best[1], "nonzero": int((w > 0).sum())}
        return [genes[i] for i in order[: self.k]]


class StabilitySelection(Selector):
    name = "stability"
    GRID = np.logspace(-3, 0, 13)

    def select(self, X, y, run):
        _, pool = tier2(X, y)
        genes = list(pool.index); Z = _standardise(X, genes)
        q = self.kw.get("q", 2 * self.k); n_sub = self.kw.get("n_sub", 50)
        nnz = []
        for C in self.GRID:          # regularisation giving about q selected genes on the training data
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000,
                                   random_state=self.seed).fit(Z, y)
            nnz.append(int((m.coef_ != 0).sum()))
        C = float(self.GRID[int(np.argmin([abs(v - q) for v in nnz]))])
        rng = np.random.default_rng(self.seed)
        i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        sel = np.zeros(len(genes)); mag = np.zeros(len(genes))
        for _ in range(n_sub):
            idx = np.r_[rng.choice(i1, len(i1) // 2, replace=False), rng.choice(i0, len(i0) // 2, replace=False)]
            Zs = StandardScaler().fit_transform(X.iloc[idx][genes].values)
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000,
                                   random_state=self.seed).fit(Zs, y[idx])
            c = np.abs(m.coef_.ravel()); sel += c > 0; mag += c
        pi = sel / n_sub
        order = sorted(range(len(genes)), key=lambda i: (-pi[i], -mag[i], genes[i]))
        self.info = {"C": C, "q": q, "n_pi_ge_0.6": int((pi >= 0.6).sum())}
        return [genes[i] for i in order[: self.k]]


class MRMRIFSSVM(Selector):
    name = "mrmr_ifs_svm"
    classifier = "svm"

    def select(self, X, y, run):
        genes, _ = T.sheng_mrmr_ifs_svm(X, y, run)
        return list(genes)

    def fit_predict(self, Xtr, ytr, Xte, genes, run):
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import SVC
        sc = run.cfg["sheng"]
        m = make_pipeline(StandardScaler(), SVC(kernel=sc["svm_kernel"], C=sc["svm_C"]))
        m.fit(run.logX(Xtr)[genes].values, ytr)
        return m.decision_function(run.logX(Xte)[genes].values)


class RandomPanel(Selector):
    name = "random"

    def select(self, X, y, run):
        _, pool = tier2(X, y)
        rng = np.random.default_rng(self.seed)
        return list(rng.choice(list(pool.index), self.k, replace=False))


class Fixed(Selector):
    name = "fixed"

    def __init__(self, genes: Sequence[str], **kw):
        super().__init__(**kw); self.genes = list(genes)

    def select(self, X, y, run):
        missing = [g for g in self.genes if g not in X.columns]
        if missing:
            raise KeyError(f"fixed panel genes not in the matrix: {missing}")
        return list(self.genes)


REGISTRY = {c.name: c for c in (SixStage, MWTopK, L1TopK, StabilitySelection, MRMRIFSSVM, RandomPanel)}


def make(name: str, **kw) -> Selector:
    if name not in REGISTRY:
        raise KeyError(f"unknown selector '{name}'; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
