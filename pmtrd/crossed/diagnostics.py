"""POST-HOC diagnostic of the pre-registered pigeonhole interval (not part of the E1-E4 verdict).

Written on 8 Nov 2026 after the interim E2 results showed pigeonhole coverage of 1.00 with intervals several
times wider than the replicate SD for every statistic that involves D_both. Nothing here changes the
pre-registered thresholds, interval or verdict; it only tests an explanation and a candidate remedy, which
would need its own pre-registered confirmation run (fresh seeds) before use.

Hypothesis. With uniform draw weights the panel-only part of the grid cancels exactly between D_both and
D_row (and the sample-only part between D_both and D_col): every ordered pair of distinct panels q, q' gets the
same weight in both moments. Under multiplicity weights v the pair (q, q') gets weight v_q v_q' (W - v_q - v_q')
in D_row and v_q v_q' [(W - v_q - v_q')^2 - sum_{r not in {q, q'}} v_r^2] in D_both, so the cancellation fails
and the panel variance leaks into the bootstrap replicates of S, SQ and gamma (and the sample variance into Q).

Candidate remedy. The delete-one-draw jackknife: dropping draw j leaves a design with uniform weights on the
remaining R - 1 draws, so the cancellation is preserved. Interval: estimate +/- t_{R-1, 0.975} * SE_jk.

Run:  python -m pmtrd.crossed.diagnostics --out runs/crossed_sim/diagnostics_posthoc
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
from scipy import stats as st

from .estimator import STAT_NAMES, CrossedData, _stats, _stats_batch, bootstrap, percentile_ci
from .synthetic import AnalyticModel, make_model

ROOT_SEED = 20261109


def leak_demo(R: int = 10, B: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Pure panel effect z_rq = phi_q (exact S = SQ = 0). Returns the estimate at uniform weights and the
    spread of the bootstrap replicates of the S and SQ variance components."""
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal(R)
    grid = np.broadcast_to(phi[None, :, None], (R, R, 1)).astype(float).copy()
    d = CrossedData(grid=grid)
    c0 = _stats(d)
    V = rng.multinomial(R, np.full(R, 1 / R), size=B).astype(float)
    V = V[(V > 0).sum(1) >= 4]
    reps = _stats_batch(d, V, np.ones((len(V), 1)))
    return {"S_uniform": float(c0[0]), "SQ_uniform": float(c0[2]),
            "S_boot_sd": float(np.nanstd(reps[:, 0])), "SQ_boot_sd": float(np.nanstd(reps[:, 2])),
            "S_boot_q025": float(np.nanquantile(reps[:, 0], .025)),
            "S_boot_q975": float(np.nanquantile(reps[:, 0], .975))}


def jackknife(d: CrossedData) -> tuple:
    """Delete-one-draw jackknife: (estimate, SE) arrays over STAT_NAMES."""
    R = d.R
    V = 1.0 - np.eye(R)
    with np.errstate(invalid="ignore", divide="ignore"):
        th = _stats_batch(d, V, np.ones((R, d.P)))
    est = _stats(d)
    se = np.sqrt((R - 1) / R * np.nansum((th - np.nanmean(th, 0)) ** 2, 0))
    return est, se


def panel_dominant(P: int = 40, seed: int = 0) -> AnalyticModel:
    """Mimics the random-selector setting: small S, large Q (kappa = 0)."""
    rng = np.random.default_rng(seed)
    u = lambda lo, hi: rng.uniform(lo, hi, P)  # noqa: E731
    return AnalyticModel(alpha=u(.05, .15), beta=u(.8, 1.0), lam=u(.2, .4), tau=u(.1, .2),
                         eta=np.zeros(P), kappa=0.0)


def calibration(model: AnalyticModel, R: int, N: int, B: int, seed: int) -> Dict:
    """N independent designs: replicate SD versus bootstrap SD and jackknife SE; coverage of the draws-only
    percentile interval and of the jackknife-t interval, against the exact shares of the analytic model."""
    rng = np.random.default_rng(seed)
    truth = model.truth()
    names = [s for s in STAT_NAMES if s != "eq3"]
    tv = np.array([truth[k] if k != "Q+gamma" else truth["Q"] + truth["gamma"] for k in names])
    tq = st.t.ppf(0.975, R - 1)
    est, bsd, jse, cov_b, cov_j, wid_b, wid_j = ([] for _ in range(7))
    for _ in range(N):
        d = model.simulate(R, rng, n_seed_cells=R, M=3)
        e, se = jackknife(d)
        reps = bootstrap(d, B, rng, "draws")
        ci = percentile_ci(reps)
        e, se, ci, reps = e[:6], se[:6], ci[:, :6], reps[:, :6]
        est.append(e)
        bsd.append(np.nanstd(reps, 0, ddof=1))
        jse.append(se)
        cov_b.append((ci[0] <= tv) & (tv <= ci[1]))
        wid_b.append(ci[1] - ci[0])
        cov_j.append((e - tq * se <= tv) & (tv <= e + tq * se))
        wid_j.append(2 * tq * se)
    est = np.array(est)
    out = {}
    for i, k in enumerate(names):
        out[k] = {"truth": float(tv[i]), "mean_est": float(est[:, i].mean()), "replicate_sd": float(est[:, i].std(ddof=1)),
                  "mean_boot_sd": float(np.mean(np.array(bsd)[:, i])), "mean_jk_se": float(np.mean(np.array(jse)[:, i])),
                  "coverage_boot_draws": float(np.mean(np.array(cov_b)[:, i])),
                  "coverage_jackknife_t": float(np.mean(np.array(cov_j)[:, i])),
                  "width_boot_draws": float(np.mean(np.array(wid_b)[:, i])),
                  "width_jackknife_t": float(np.mean(np.array(wid_j)[:, i]))}
    return out


def crossed_jackknife(d: CrossedData, G: int = 10) -> tuple:
    """Delete-one-draw jackknife plus a grouped delete-patients jackknife (G groups); the two variances are
    added, mirroring the two margins the pigeonhole bootstrap resamples. Returns (estimate, SE)."""
    est, se_d = jackknife(d)
    groups = np.array_split(np.arange(d.P), G)
    Wg = np.ones((G, d.P))
    for i, g in enumerate(groups):
        Wg[i, g] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        th = _stats_batch(d, np.ones((G, d.R)), Wg)
    se_p = np.sqrt((G - 1) / G * np.nansum((th - np.nanmean(th, 0)) ** 2, 0))
    return est, np.sqrt(se_d ** 2 + se_p ** 2)


def pipeline_pilot(cfg_path: str, truth_dir: str, out: str, clf: str, n_T: int, n_rep: int, R: int = 10) -> None:
    """Fresh-seed replicates of the base configuration (draw ids from 90000, patient seeds rep >= 500, so no
    overlap with the pre-registered replicates). Each unit stores the point estimates, the pre-registered
    pigeonhole percentile interval (to check that the over-coverage reproduces on fresh seeds) and the
    candidate crossed-jackknife t interval. Resumable: one JSON per unit."""
    from .simulate import PanelCache, Setting, build_design, seed_for

    cfg = json.load(open(cfg_path))
    cfg = {**cfg, "rep_draw_base": {**cfg["rep_draw_base"], str(R): 90000 - 500 * R}}
    os.makedirs(os.path.join(out, "units"), exist_ok=True)
    tq = st.t.ppf(0.975, R - 1)
    for sel in cfg["selectors"]:
        stg = Setting(cfg, cfg["base"], sel, clf)
        for t in range(n_T):
            # one panel cache per training set: panels (and P0, key -1) depend on T
            cache = PanelCache(os.path.join(out, f"panels_{sel}_T{t}.npz"))
            X, y = stg.T(t)
            for i in range(n_rep):
                rep = 500 + i
                path = os.path.join(out, "units", f"base__{sel}__{clf}__T{t}__R{R}_{rep}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                Xh, _ = stg.gen.sample(stg.data["n"], np.random.default_rng(seed_for(stg.key, t, 5, R, rep)),
                                       stratified=False)
                ids = [cfg["rep_draw_base"][str(R)] + rep * R + j for j in range(R)]
                cells = np.argwhere(~np.eye(R, dtype=bool))
                seed_cells = np.array([(r, (r + 1) % R) for r in range(R)])
                d, n_fits = build_design(stg, t, X, y, ids, cells, seed_cells, 2, Xh, cache, tag=R)
                est, se = crossed_jackknife(d)
                rng = np.random.default_rng(seed_for(stg.key, t, 8, stg.sel_code, stg.clf_code, R, rep))
                ci = percentile_ci(bootstrap(d, cfg["bootstrap_B"], rng, "pigeonhole"))
                res = {"selector": sel, "clf": clf, "T": t, "R": R, "rep": rep, "n_fits": n_fits,
                       "est": dict(zip(STAT_NAMES, est.tolist())), "se_jk": dict(zip(STAT_NAMES, se.tolist())),
                       "ci_jk_t": {k: [est[j] - tq * se[j], est[j] + tq * se[j]] for j, k in enumerate(STAT_NAMES)},
                       "ci_pigeonhole": {k: [float(ci[0, j]), float(ci[1, j])] for j, k in enumerate(STAT_NAMES)},
                       "seconds": time.time() - t0}
                cache.save()
                with open(path + ".tmp", "w") as f:
                    json.dump(res, f, default=float)
                os.replace(path + ".tmp", path)
            cache.save()
    summarise_pilot(truth_dir, out)


def summarise_pilot(truth_dir: str, out: str) -> None:
    """Coverage of both intervals against the pre-registered truths (runs/crossed_sim/v1/units/*truth*)."""
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "units", "*.json"))):
        u = json.load(open(p))
        tp = os.path.join(truth_dir, f"base__{u['selector']}__{u['clf']}__T{u['T']}__truth_R200_000.json")
        if not os.path.exists(tp):
            continue
        tr = json.load(open(tp))
        truth = {**tr["share"], "eq3": tr["eq3"]}
        for k in STAT_NAMES:
            if u["clf"] == "logistic" and k == "E":
                continue
            lo_j, hi_j = u["ci_jk_t"][k]
            lo_b, hi_b = u["ci_pigeonhole"][k]
            rows.append({"selector": u["selector"], "clf": u["clf"], "stat": k, "T": u["T"],
                         "err": u["est"][k] - truth[k], "cov_jk": lo_j <= truth[k] <= hi_j,
                         "cov_pig": lo_b <= truth[k] <= hi_b, "w_jk": hi_j - lo_j, "w_pig": hi_b - lo_b})
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return
    s = (df.groupby(["clf", "selector", "stat"])
         .agg(n=("err", "size"), bias=("err", "mean"), rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
              coverage_pigeonhole=("cov_pig", "mean"), coverage_jackknife_t=("cov_jk", "mean"),
              width_pigeonhole=("w_pig", "mean"), width_jackknife_t=("w_jk", "mean")).reset_index())
    s.to_csv(os.path.join(out, "pilot_summary.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/crossed_sim/diagnostics_posthoc")
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--pilot", choices=["logistic", "xgboost"], help="run the fresh-seed pipeline pilot instead")
    ap.add_argument("--config", default="configs/crossed_sim.json")
    ap.add_argument("--truth-dir", default="runs/crossed_sim/v1/units")
    ap.add_argument("--n-T", type=int, default=4)
    ap.add_argument("--n-rep", type=int, default=25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.pilot:
        pipeline_pilot(a.config, a.truth_dir, os.path.join(a.out, f"pilot_{a.pilot}"), a.pilot, a.n_T, a.n_rep)
        return
    t0 = time.time()
    res = {"_note": "POST-HOC diagnostic; not part of the pre-registered E1-E4 verdict.",
           "leak_demo_R10": leak_demo(10), "leak_demo_R50": leak_demo(50), "calibration": {}}
    models = {"panel_dominant": panel_dominant(), "interaction": make_model("interaction"),
              "coupling": make_model("coupling")}
    for i, (name, m) in enumerate(models.items()):
        for R in (10, 20):
            res["calibration"][f"{name}_R{R}"] = calibration(m, R, a.N, a.B, ROOT_SEED + 100 * i + R)
            print(f"{name} R={R} done ({time.time() - t0:.0f}s)", flush=True)
    res["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "diagnostics.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
