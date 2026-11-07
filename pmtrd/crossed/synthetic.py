"""Analytic test functions with known Hoeffding components, used to validate the estimator code.

Each draw r has a latent sample effect s_r ~ N(0, 1) and a selection seed xi_r ~ N(0, 1). The panel latent
of draw q is phi_q = a s_q + b xi_q with a = kappa / sqrt(1 + kappa^2), b = 1 / sqrt(1 + kappa^2), so that
phi ~ N(0, 1) and kappa controls how strongly the panel depends on the subsample it was selected on (the
source of coupling). Every fit has its own seed effect w ~ N(0, 1). For patient p,

    h(r, q, w) = mu_p + alpha_p s_r + beta_p phi_q + lambda_p s_r phi_q + tau_p w + eta_p w s_r.

Because s_r, phi_q (q != r) and w are independent standard normals and the products are uncorrelated with
the main terms, the exact components are
    s2_S = alpha^2, s2_Q = beta^2, s2_SQ = lambda^2, s2_E = tau^2 + eta^2,
and for the tied pipeline (q = r) Var h = alpha^2 + beta^2 + 2 alpha beta a + lambda^2 (1 + a^2) + tau^2 + eta^2,
so gamma = 2 alpha beta a + lambda^2 a^2 (derivation document, section 5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .estimator import COMPONENTS, CrossedData


@dataclass
class AnalyticModel:
    alpha: np.ndarray
    beta: np.ndarray
    lam: np.ndarray
    tau: np.ndarray
    eta: np.ndarray
    kappa: float = 0.0
    mu: float = 0.0

    @property
    def a(self) -> float:
        return self.kappa / np.sqrt(1 + self.kappa ** 2)

    def truth(self) -> Dict[str, float]:
        """Exact components averaged over patients (ratio-of-means scale, as the estimator reports)."""
        a = self.a
        s2 = {"S": np.mean(self.alpha ** 2), "Q": np.mean(self.beta ** 2), "SQ": np.mean(self.lam ** 2),
              "E": np.mean(self.tau ** 2 + self.eta ** 2),
              "gamma": np.mean(2 * self.alpha * self.beta * a + self.lam ** 2 * a ** 2)}
        V_R = sum(s2.values())
        return {"V_R": V_R, **{f"s2_{k}": v for k, v in s2.items()}, **{k: s2[k] / V_R for k in COMPONENTS}}

    def simulate(self, R: int, rng: np.random.Generator, n_seed_cells: int = 0, M: int = 2,
                 missing: float = 0.0) -> CrossedData:
        """One crossed design with R draws; optional seed cells and cells missing completely at random."""
        a, b = self.a, 1 / np.sqrt(1 + self.kappa ** 2)
        s = rng.standard_normal(R)
        phi = a * s + b * rng.standard_normal(R)
        al, be, la, ta, et = (x[None, None, :] for x in (self.alpha, self.beta, self.lam, self.tau, self.eta))

        def h(si, ph, w):
            return self.mu + al * si + be * ph + la * si * ph + ta * w + et * w * si

        W = rng.standard_normal((R, R, 1))
        grid = h(s[:, None, None], phi[None, :, None], W)
        if missing > 0:
            drop = (rng.random((R, R)) < missing) & ~np.eye(R, dtype=bool)
            grid[drop] = np.nan
        cells = vals = None
        if n_seed_cells:
            off = np.argwhere(~np.eye(R, dtype=bool))
            cells = off[rng.choice(len(off), n_seed_cells, replace=False)]
            Wm = rng.standard_normal((n_seed_cells, M, 1))
            vals = h(s[cells[:, 0]][:, None, None], phi[cells[:, 1]][:, None, None], Wm)
        return CrossedData(grid=grid, seed_cells=cells, seed_vals=vals)


def make_model(kind: str, P: int = 40, seed: int = 0) -> AnalyticModel:
    """Named test cases: additive, interaction, random_panel, fixed_panel, deterministic, seed, coupling."""
    rng = np.random.default_rng(seed)
    u = lambda lo, hi: rng.uniform(lo, hi, P)  # noqa: E731
    z = np.zeros(P)
    cases = {
        "additive": dict(alpha=u(.5, 1), beta=u(.5, 1), lam=z, tau=u(.2, .4), eta=z, kappa=0.0),
        "interaction": dict(alpha=u(.5, 1), beta=u(.5, 1), lam=u(.3, .6), tau=u(.2, .4), eta=u(.1, .3), kappa=0.0),
        "random_panel": dict(alpha=u(.5, 1), beta=u(.5, 1), lam=u(.2, .4), tau=u(.2, .4), eta=z, kappa=0.0),
        "fixed_panel": dict(alpha=u(.5, 1), beta=z, lam=z, tau=u(.2, .4), eta=u(.1, .2), kappa=1.0),
        "deterministic": dict(alpha=u(.5, 1), beta=u(.5, 1), lam=u(.2, .4), tau=z, eta=z, kappa=1.0),
        "seed": dict(alpha=u(.3, .6), beta=u(.3, .6), lam=z, tau=u(.6, 1), eta=u(.3, .5), kappa=0.0),
        "coupling": dict(alpha=u(.5, 1), beta=u(.5, 1), lam=u(.3, .5), tau=u(.2, .3), eta=z, kappa=1.5),
    }
    return AnalyticModel(**cases[kind])
