import numpy as np
import pandas as pd
import pytest

from pmtrd.pathway import centred_ranks, filter_sets, set_scores, top_sets


def _sets(genes):
    return {"SET_A": genes[:20], "SET_B": genes[20:40], "SET_C": genes[40:60]}


def test_scores_are_invariant_to_library_size(toy):
    X, y = toy
    s1 = set_scores(X, _sets(list(X.columns)))
    s2 = set_scores(X.mul(np.linspace(0.2, 5, len(X)), axis=0), _sets(list(X.columns)))
    assert np.allclose(s1.values, s2.values)


def test_single_sample_scoring_does_not_depend_on_other_samples(toy):
    X, y = toy
    full = set_scores(X, _sets(list(X.columns)))
    one = set_scores(X.iloc[[5]], _sets(list(X.columns)))
    assert np.allclose(full.iloc[[5]].values, one.values)


def test_centred_ranks_range(toy):
    X, _ = toy
    R = centred_ranks(X)
    assert R.min() > -0.5 and R.max() <= 0.5 + 1e-12


def test_top_sets_finds_the_shifted_set(toy):
    X, y = toy
    genes = list(X.columns)
    sets = {"SHIFTED": genes[:12], "NULL1": genes[100:130], "NULL2": genes[200:230]}
    assert top_sets(set_scores(X, sets), y, 1) == ["SHIFTED"]


def test_filter_sets_and_errors(toy):
    X, _ = toy
    out = filter_sets({"small": ["GENE001"], "ok": list(X.columns[:20]), "big": list(X.columns)}, X.columns, 15, 100)
    assert list(out) == ["ok"]
    with pytest.raises(ValueError):
        set_scores(X, {"none": ["NOT_A_GENE"]})
