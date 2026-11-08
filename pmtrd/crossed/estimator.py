"""Crossed pick-freeze estimator of the components of re-derivation variance.

Notation (docs/crossed_estimator_derivation.md). Draw r = (S_r, xi_r) is an independent re-derivation input:
a training subsample S_r of the training set T and a selection seed xi_r. Cell (r, q), r != q, is the model
trained on S_r with the panel A(S_q, xi_q) and a classifier seed that is unique to the fit; its held-out
prediction (log-odds) for patient p is z[r, q, p]. The diagonal z[r, r, p] is the real (tied) pipeline.

For r != q the inputs U = S_r, V = (S_q, xi_q) and W = seed are independent given T, so the Hoeffding
decomposition of h(U, V, W) is exact and

    D_col  = 1/2 E (z_rq - z_r'q)^2   (share V; redraw U, W) = s2_S + s2_SQ + s2_E
    D_row  = 1/2 E (z_rq - z_rq')^2   (share U; redraw V, W) = s2_Q + s2_SQ + s2_E
    D_both = Var z_rq                 (share nothing)        = s2_S + s2_Q + s2_SQ + s2_E
    D_seed = 1/2 E (z_rqm - z_rqm')^2 (share U, V; redraw W) = s2_E
    V_R    = Var z_rr                 (tied pipeline)        = D_both + gamma

where s2_E is the sum of every Hoeffding term that involves the classifier seed. All moments are computed as
weighted U-statistics over pairs of cells whose non-shared draws are distinct, so that the same code serves
complete grids, incomplete designs (missing cells as NaN) and bootstrap resamples (integer weights).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

COMPONENTS = ("S", "Q", "SQ", "E", "gamma")


@dataclass
class CrossedData:
    """Predicted log-odds of one crossed design.

    grid:       (R, R, P) array; grid[r, q, p] for r != q (first seed of the cell) and the tied pipeline on
                the diagonal. Missing cells are NaN (the missingness pattern must not depend on p).
    seed_cells: (C, 2) int array of off-diagonal cells (r, q) that were refitted with extra seeds.
    seed_vals:  (C, M, P) array of their predictions, M >= 2 independent seeds per cell (seed 0 may repeat
                the grid value).
    eq3_tied, eq3_fixed: optional (R, P) arrays for the published estimator (Eq. 3): the tied pipeline and
                the fixed-panel arm, both with one fixed classifier seed.
    """

    grid: np.ndarray
    seed_cells: Optional[np.ndarray] = None
    seed_vals: Optional[np.ndarray] = None
    eq3_tied: Optional[np.ndarray] = None
    eq3_fixed: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        g = np.asarray(self.grid)
        if g.dtype not in (np.float32, np.float64):
            g = g.astype(float)
        if g.ndim != 3 or g.shape[0] != g.shape[1]:
            raise ValueError("grid must have shape (R, R, P)")
        self.grid = g

    def patient_slice(self, a: int, b: int) -> "CrossedData":
        """View on patients a:b (used to bound memory for large designs)."""
        sv = None if self.seed_vals is None else self.seed_vals[:, :, a:b]
        return CrossedData(grid=self.grid[:, :, a:b], seed_cells=self.seed_cells, seed_vals=sv)

    @property
    def R(self) -> int:
        return self.grid.shape[0]

    @property
    def P(self) -> int:
        return self.grid.shape[2]


@dataclass
class Components:
    """Patient-averaged moments and variance components (weighted means over patients)."""

    D_col: float
    D_row: float
    D_both: float
    D_seed: float
    V_R: float
    s2: Dict[str, float] = field(default_factory=dict)

    def shares(self) -> Dict[str, float]:
        """Each component divided by V_R; they sum to 1 because gamma is defined as the remainder."""
        out = {c: self.s2[c] / self.V_R for c in COMPONENTS}
        out["Q+gamma"] = out["Q"] + out["gamma"]
        return out


def _pair_moment(Z: np.ndarray, A: np.ndarray, outer: np.ndarray) -> tuple:
    """Sum over groups g (axis 1 of A) of outer[g] * sum_{i<j} A[i,g] A[j,g] (Z[i,g]-Z[j,g])^2 / 2, and the
    matching sum of pair weights. Z is (n, G, P) with zeros where A is 0."""
    S0 = A.sum(0)                                   # (G,)
    S1 = np.einsum("ig,igp->gp", A, Z)
    S2 = np.einsum("ig,igp->gp", A, Z * Z)
    A2 = (A * A).sum(0)
    num = (outer[:, None] * 0.5 * (S0[:, None] * S2 - S1 * S1)).sum(0)   # (P,)
    den = float((outer * 0.5 * (S0 * S0 - A2)).sum())
    return num, den


def moments(data: CrossedData, v: Optional[np.ndarray] = None, chunk: int = 0) -> Dict[str, np.ndarray]:
    """Per-patient moments D_col, D_row, D_both, D_seed and V_R (arrays of length P).

    v: draw weights (length R); bootstrap multiplicities or ones. Pairs are formed between distinct draw
    indices only, so a resampled draw never forms a pair with itself.
    chunk: if > 0, process patients in slices of this size (identical result, bounded memory).
    """
    if chunk and data.P > chunk:
        parts = [moments(data.patient_slice(a, a + chunk), v) for a in range(0, data.P, chunk)]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    R, P = data.R, data.P
    v = np.ones(R) if v is None else np.asarray(v, float)
    g = np.asarray(data.grid, dtype=float)
    off = ~np.eye(R, dtype=bool)
    present = ~np.isnan(g[:, :, 0])
    O = (present & off).astype(float)
    Z = np.where((present & off)[:, :, None], g, 0.0)

    # D_col: pairs (r, q), (r', q) with r != r'; groups are columns q, weight of row r is v_r
    num_c, den_c = _pair_moment(Z, v[:, None] * O, v)
    # D_row: pairs (r, q), (r, q'); groups are rows r -> transpose
    Zt = np.transpose(Z, (1, 0, 2))
    num_r, den_r = _pair_moment(Zt, v[:, None] * O.T, v)

    # D_both = E z^2 - E z z' over pairs sharing no draw (inclusion-exclusion over the four coincidences
    # r = r', q = q', r = q', q = r'; see the derivation document, section 3.3)
    w = (v[:, None] * v[None, :]) * O                      # cell weights
    u = w[:, :, None] * Z
    m2 = (u * Z).sum((0, 1)) / w.sum()

    def disjoint_sum(x: np.ndarray) -> np.ndarray:        # x: (R, R, ...) weighted values
        rows = x.sum(1)
        cols = x.sum(0)
        tot = x.sum((0, 1))
        return (tot * tot - (rows * rows).sum(0) - (cols * cols).sum(0) - 2 * (rows * cols).sum(0)
                + (x * x).sum((0, 1)) + (x * np.swapaxes(x, 0, 1)).sum((0, 1)))

    num_b = disjoint_sum(u)
    den_b = float(disjoint_sum(w))
    D_both = m2 - num_b / den_b if den_b > 0 else np.full(P, np.nan)

    # D_seed: pairs of seeds within the same cell; cell weight v_r v_q
    if data.seed_cells is not None and data.seed_vals is not None and len(data.seed_cells):
        sv = np.asarray(data.seed_vals, float)
        M = sv.shape[1]
        cw = v[data.seed_cells[:, 0]] * v[data.seed_cells[:, 1]]
        dev = sv - sv[:, :1]                         # shift by the first seed: exact zeros when all seeds agree
        within = 0.5 * (M * (dev * dev).sum(1) - dev.sum(1) ** 2)          # sum over pairs of (zm - zm')^2 / 2
        D_seed = (cw[:, None] * within).sum(0) / (cw.sum() * M * (M - 1) / 2)
    else:
        D_seed = np.full(P, np.nan)

    # V_R: pairs of diagonal (tied) predictions from distinct draws
    d = np.diagonal(g, axis1=0, axis2=1).T                 # (R, P)
    ok = ~np.isnan(d[:, 0])
    vv = v * ok
    dd = np.where(ok[:, None], d, 0.0)
    s0, s1, s2 = vv.sum(), (vv[:, None] * dd).sum(0), (vv[:, None] * dd * dd).sum(0)
    V_R = 0.5 * (s0 * s2 - s1 * s1) / (0.5 * (s0 * s0 - (vv * vv).sum()))

    return {"D_col": num_c / den_c if den_c > 0 else np.full(P, np.nan),
            "D_row": num_r / den_r if den_r > 0 else np.full(P, np.nan),
            "D_both": D_both, "D_seed": D_seed, "V_R": V_R}


def components(data: CrossedData, v: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None,
               chunk: int = 0) -> Components:
    """Patient-weighted means of the moments and the derived components.

    w: patient weights (length P). Ratios of means (not means of per-patient ratios) are reported, as in Eq. 3.
    If no cell has two seeds, s2_E is not identified: it is returned as NaN and s2_SQ then holds SQ + E.
    """
    m = moments(data, v, chunk)
    w = np.ones(data.P) if w is None else np.asarray(w, float)
    a = {k: float(np.sum(w * x) / np.sum(w)) for k, x in m.items()}
    seed = a["D_seed"] if np.isfinite(a["D_seed"]) else 0.0
    s2 = {"S": a["D_both"] - a["D_row"], "Q": a["D_both"] - a["D_col"],
          "SQ": a["D_row"] + a["D_col"] - a["D_both"] - seed,
          "E": a["D_seed"], "gamma": a["V_R"] - a["D_both"]}
    return Components(a["D_col"], a["D_row"], a["D_both"], a["D_seed"], a["V_R"], s2)


def eq3_psi(data: CrossedData, v: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None) -> float:
    """The published estimator (Eq. 3): 1 - sum_p Var_r(fixed arm) / sum_p Var_r(tied pipeline), with the
    variances as weighted U-statistics over distinct draws."""
    if data.eq3_tied is None or data.eq3_fixed is None:
        return float("nan")
    R = data.eq3_tied.shape[0]
    v = np.ones(R) if v is None else np.asarray(v, float)
    w = np.ones(data.eq3_tied.shape[1]) if w is None else np.asarray(w, float)

    def var(x: np.ndarray) -> np.ndarray:
        s0, s1, s2 = v.sum(), (v[:, None] * x).sum(0), (v[:, None] * x * x).sum(0)
        return 0.5 * (s0 * s2 - s1 * s1) / (0.5 * (s0 * s0 - (v * v).sum()))

    return float(1 - np.sum(w * var(data.eq3_fixed)) / np.sum(w * var(data.eq3_tied)))


def _stats(data: CrossedData, v=None, w=None) -> np.ndarray:
    c = components(data, v, w)
    sh = c.shares()
    return np.array([sh[k] for k in STAT_NAMES[:-1]] + [eq3_psi(data, v, w)])


STAT_NAMES = ("S", "Q", "SQ", "E", "gamma", "Q+gamma", "eq3")


def _stats_batch(data: CrossedData, V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Vectorised statistics for a batch of draw weights V (B, R) and patient weights W (B, P); identical to
    calling _stats row by row (checked in the tests)."""
    R = data.R
    g = np.asarray(data.grid, dtype=float)
    off = ~np.eye(R, dtype=bool)
    present = ~np.isnan(g[:, :, 0]) & off
    O = present.astype(float)
    Z = np.where(present[:, :, None], g, 0.0)
    Z2 = Z * Z

    def pair(Zx, Z2x, Ox):                                # groups along axis 1 of Zx
        A = V[:, :, None] * Ox[None]                      # (B, n, G)
        S0, A2 = A.sum(1), (A * A).sum(1)
        S1 = np.einsum("bng,ngp->bgp", A, Zx)
        S2 = np.einsum("bng,ngp->bgp", A, Z2x)
        num = np.einsum("bg,bgp->bp", V, 0.5 * (S0[:, :, None] * S2 - S1 * S1))
        den = (V * 0.5 * (S0 * S0 - A2)).sum(1)
        return num / den[:, None]

    D_col = pair(Z, Z2, O)
    D_row = pair(np.transpose(Z, (1, 0, 2)), np.transpose(Z2, (1, 0, 2)), O.T)
    Wc = V[:, :, None] * V[:, None, :] * O[None]           # (B, R, R) cell weights
    m2 = np.einsum("brq,rqp->bp", Wc, Z2) / Wc.sum((1, 2))[:, None]

    def disjoint(x, xsq, xT):
        rows, cols = x.sum(2), x.sum(1)
        tot = x.sum((1, 2))
        return (tot * tot - (rows * rows).sum(1) - (cols * cols).sum(1) - 2 * (rows * cols).sum(1)
                + xsq + xT)

    u_tot = np.einsum("brq,rqp->bp", Wc, Z)
    u_rows = np.einsum("brq,rqp->brp", Wc, Z)
    u_cols = np.einsum("brq,rqp->bqp", Wc, Z)
    u_sq = np.einsum("brq,rqp->bp", Wc * Wc, Z2)
    u_T = np.einsum("brq,bqr,rqp,qrp->bp", Wc, Wc, Z, Z)
    num_b = (u_tot * u_tot - (u_rows * u_rows).sum(1) - (u_cols * u_cols).sum(1) - 2 * (u_rows * u_cols).sum(1)
             + u_sq + u_T)
    den_b = disjoint(Wc, (Wc * Wc).sum((1, 2)), (Wc * np.swapaxes(Wc, 1, 2)).sum((1, 2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        D_both = m2 - num_b / den_b[:, None]
    if data.seed_cells is not None and data.seed_vals is not None and len(data.seed_cells):
        sv = np.asarray(data.seed_vals, float)
        M = sv.shape[1]
        dev = sv - sv[:, :1]
        within = 0.5 * (M * (dev * dev).sum(1) - dev.sum(1) ** 2)
        cw = V[:, data.seed_cells[:, 0]] * V[:, data.seed_cells[:, 1]]
        with np.errstate(invalid="ignore", divide="ignore"):
            D_seed = (cw @ within) / (cw.sum(1) * M * (M - 1) / 2)[:, None]
        has_seed = True
    else:
        D_seed, has_seed = np.full_like(D_col, np.nan), False
    d = np.diagonal(g, axis1=0, axis2=1).T
    ok = ~np.isnan(d[:, 0])
    vv = V * ok
    dd = np.where(ok[:, None], d, 0.0)

    def wvar(x, wt):
        s0, s1, s2 = wt.sum(1), wt @ x, wt @ (x * x)
        return 0.5 * (s0[:, None] * s2 - s1 * s1) / (0.5 * (s0 * s0 - (wt * wt).sum(1)))[:, None]

    V_R = wvar(dd, vv)
    mean = lambda x: (W * x).sum(1) / W.sum(1)  # noqa: E731
    a = {k: mean(x) for k, x in (("col", D_col), ("row", D_row), ("both", D_both), ("seed", D_seed), ("VR", V_R))}
    seed = a["seed"] if has_seed else 0.0
    s2 = {"S": a["both"] - a["row"], "Q": a["both"] - a["col"], "SQ": a["row"] + a["col"] - a["both"] - seed,
          "E": a["seed"], "gamma": a["VR"] - a["both"]}
    out = [s2[c] / a["VR"] for c in COMPONENTS]
    out.append(out[1] + out[4])
    if data.eq3_tied is not None and data.eq3_fixed is not None:
        out.append(1 - mean(wvar(np.asarray(data.eq3_fixed, float), V)) / mean(wvar(np.asarray(data.eq3_tied, float), V)))
    else:
        out.append(np.full(len(V), np.nan))
    return np.column_stack(out)


def bootstrap(data: CrossedData, B: int, rng: np.random.Generator, mode: str = "pigeonhole",
              batch: int = 50) -> np.ndarray:
    """Bootstrap replicates of the statistics in STAT_NAMES (array (B, 7); rows with fewer than four distinct
    draws are NaN because D_both is then undefined).

    mode: 'pigeonhole' resamples draws and patients independently (Owen 2007); 'draws' resamples draws only
    (target: the given patients); 'patients' resamples patients only (the published interval for psi).
    Draw resampling uses multiplicity weights, and pairs are formed only between distinct original draws.
    """
    R, P = data.R, data.P
    V = (rng.multinomial(R, np.full(R, 1 / R), size=B).astype(float) if mode in ("pigeonhole", "draws")
         else np.ones((B, R)))
    W = (rng.multinomial(P, np.full(P, 1 / P), size=B).astype(float) if mode in ("pigeonhole", "patients")
         else np.ones((B, P)))
    out = np.full((B, len(STAT_NAMES)), np.nan)
    batch = max(4, min(batch, int(1.5e7 // (R * R * P))))     # bound the (batch, R, R, P) intermediates
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(0, B, batch):
            out[a:a + batch] = _stats_batch(data, V[a:a + batch], W[a:a + batch])
    out[(V > 0).sum(1) < 4] = np.nan
    return out


def percentile_ci(reps: np.ndarray, level: float = 0.95) -> np.ndarray:
    """(2, k) array of percentile limits, ignoring NaN replicates."""
    a = (1 - level) / 2
    return np.nanquantile(reps, [a, 1 - a], axis=0)
