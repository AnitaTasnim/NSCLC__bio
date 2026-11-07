"""Crossed pick-freeze decomposition of re-derivation variance (methods development; not yet applied to the
NSCLC data). See docs/crossed_estimator_derivation.md for the mathematics and docs/crossed_simulation.md for
the simulation study."""
from .estimator import Components, CrossedData, bootstrap, components, eq3_psi, moments  # noqa: F401
