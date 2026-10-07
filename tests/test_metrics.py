import numpy as np

from anomaly.metrics import evaluate, point_adjust, prf


def test_prf_counts():
    pred = np.array([1, 1, 0, 0], dtype=bool)
    label = np.array([1, 0, 1, 0], dtype=bool)
    result = prf(pred, label)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["precision"] == 0.5 and result["recall"] == 0.5 and result["f1"] == 0.5


def test_prf_with_no_predictions_is_zero_not_an_error():
    result = prf(np.zeros(4, dtype=bool), np.array([1, 0, 0, 0], dtype=bool))
    assert result["f1"] == 0.0


def test_point_adjust_fills_a_segment_that_was_hit_once():
    label = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    adjusted = point_adjust(pred, label)
    # First segment had one hit, so all three points count; second had none.
    assert adjusted.tolist() == [False, True, True, True, False, False, False, False]


def test_point_adjust_leaves_false_positives_alone():
    label = np.array([0, 0, 1, 1], dtype=bool)
    pred = np.array([1, 0, 0, 0], dtype=bool)
    assert point_adjust(pred, label).tolist() == [True, False, False, False]


def test_point_adjusted_f1_is_never_below_plain_f1():
    rng = np.random.default_rng(0)
    label = np.zeros(500, dtype=bool)
    label[100:140] = True
    label[300:310] = True
    scores = rng.random(500) + label * 0.3
    result = evaluate(scores, label, threshold=0.9)
    assert result["point_adjusted"]["f1"] >= result["plain"]["f1"]
    assert result["best_plain"]["f1"] >= result["plain"]["f1"]
