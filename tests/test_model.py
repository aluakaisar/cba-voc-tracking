"""Checks on the modelling helpers that do not need the review data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model  # noqa: E402


def _terms_with_all_anchors():
    return [[anchor, "filler"] for anchor in model.TOPIC_ANCHORS]


def test_labels_follow_anchor_words_not_topic_order():
    terms = _terms_with_all_anchors()[::-1]
    labels = model.label_topics(terms)
    assert labels[0] == list(model.TOPIC_ANCHORS.values())[-1]


def test_missing_anchor_fails_loudly():
    terms = _terms_with_all_anchors()
    terms[0] = ["nothing", "relevant"]
    with pytest.raises(ValueError):
        model.label_topics(terms)


def test_prevalence_rows_sum_to_one_and_empty_rows_dropped():
    w = np.array([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]])
    tp, keep = model.to_prevalence(w, ["a", "b"])
    assert keep.tolist() == [True, False, True]
    assert np.allclose(tp.sum(axis=1), 1)


def test_topic_sentiment_weights_by_prevalence():
    tp = pd.DataFrame({"a": [1.0, 0.5], "b": [0.0, 0.5]})
    s = model.topic_sentiment(tp, np.array([1.0, -1.0]))
    # a: (1*1 + 0.5*-1) / 1.5 ; b: (0.5*-1) / 0.5
    assert s["a"] == pytest.approx(1 / 3)
    assert s["b"] == pytest.approx(-1.0)
