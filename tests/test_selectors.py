import numpy as np
import pytest

from pmtrd import selectors as SEL
from pmtrd.engine import validate_inputs


def test_mw_topk_finds_shifted_genes_and_is_deterministic(toy):
    X, y = toy
    a = SEL.make("mw_topk", k=10).select(X, y, None)
    b = SEL.make("mw_topk", k=10).select(X, y, None)
    assert a == b and len(set(a) & {f"GENE{i:03d}" for i in range(12)}) >= 9


def test_random_panel_is_seeded(toy):
    X, y = toy
    a = SEL.make("random", k=5, seed=3).select(X, y, None)
    b = SEL.make("random", k=5, seed=3).select(X, y, None)
    c = SEL.make("random", k=5, seed=4).select(X, y, None)
    assert a == b and a != c


def test_l1_and_stability_selectors(toy):
    X, y = toy
    l1 = SEL.make("l1_topk", k=8).select(X, y, None)
    st = SEL.make("stability", k=8, n_sub=10).select(X, y, None)
    informative = {f"GENE{i:03d}" for i in range(12)}
    assert len(l1) == 8 and len(set(l1) & informative) >= 6 and len(set(st) & informative) >= 6
    assert st == SEL.make("stability", k=8, n_sub=10).select(X, y, None)


def test_fixed_and_registry(toy):
    X, y = toy
    assert SEL.Fixed(["GENE001"]).select(X, y, None) == ["GENE001"]
    with pytest.raises(KeyError):
        SEL.Fixed(["NOPE"]).select(X, y, None)
    with pytest.raises(KeyError):
        SEL.make("unknown")


def test_input_validation(toy):
    X, y = toy
    validate_inputs(X, y)
    with pytest.raises(ValueError):
        validate_inputs(X.iloc[:10], y)
    Xn = X.copy(); Xn.iloc[0, 0] = -1
    with pytest.raises(ValueError):
        validate_inputs(Xn, y)
