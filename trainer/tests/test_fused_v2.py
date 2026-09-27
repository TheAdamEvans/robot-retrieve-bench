from pathlib import Path

import numpy as np

from alloy_index.models.fused import Head, load, save
from alloy_trainer.learn import importance
from alloy_trainer.learn.fused_v2 import probabilities
from alloy_trainer.learn.fused_supervision import programs


def test_interval_counts_deduplicate_an_intent_and_add_distinct_answers():
    times = np.array([0, 1, 2, 3, 4], np.float64)
    intervals = [("answer_a", 0.5, 2.5), ("answer_a", 1.5, 3.5),
                 ("answer_b", 2.0, 3.0)]
    np.testing.assert_array_equal(importance.counts(times, intervals), [0, 1, 2, 2, 0])


def test_targets_use_only_positive_train_episodes(monkeypatch):
    rows = {
        "a": {"recordingId": "Train_A", "intentGroupId": "answer_a",
              "startS": 1, "endS": 2, "grade": 2},
        "b": {"recordingId": "Train_A", "intentGroupId": "answer_a",
              "startS": 1, "endS": 2, "grade": 2},
        "c": {"recordingId": "Train_A", "intentGroupId": "answer_b",
              "startS": 1, "endS": 2, "grade": 0},
        "d": {"recordingId": "Val_B", "intentGroupId": "answer_c",
              "startS": 1, "endS": 2, "grade": 2},
    }
    monkeypatch.setattr(importance, "load_labels", lambda *args: rows)
    monkeypatch.setattr(importance, "held_out", lambda: {"Val_B"})
    assert importance.targets(Path("unused"), {"Train_A", "Val_B"}) == {
        "Train_A": [("answer_a", 1.0, 2.0)]}


def test_crossfit_never_fits_on_the_scored_recording(monkeypatch):
    def fake_fit(bundle, labels, recs):
        return recs

    def fake_predict(bundle, fitted_recs, rec):
        assert rec not in fitted_recs
        return {f"{rec}:0004": 0.5}

    monkeypatch.setattr(importance, "fit", fake_fit)
    monkeypatch.setattr(importance, "predict_windows", fake_predict)
    scores = importance.crossfit(None, {}, ["A", "B", "C", "D"], groups=2)
    assert set(scores) == {"A:0004", "B:0004", "C:0004", "D:0004"}


def test_expansion_excludes_reserved_composition_features():
    expanded = programs()
    assert len(expanded) >= 100
    assert all("lateral_clearance_right_m" not in str(p.program) for p in expanded)
    assert all("speed-up" not in p.text for p in expanded)


def test_configured_single_hidden_layer_survives_save_and_load(tmp_path):
    model = Head(46, "mlp", hidden_dim=32, dropout=0.1)
    save(tmp_path, "mlp", model, np.zeros(23), np.ones(23), {"recipe": "test"})
    kind, restored, _, _, meta = load(tmp_path)
    assert kind == "mlp"
    assert restored.net[1].out_features == 32
    assert meta["dropout"] == 0.1


def test_importance_sampling_is_bounded_and_keeps_uniform_support():
    wins = [f"Train_A:{i:04d}" for i in range(4, 104)]
    scores = {w: 0.0 for w in wins}
    scores[wins[-1]] = 100.0
    uniform = probabilities(wins, scores, "uniform")
    sampled = probabilities(wins, scores, "importance")
    np.testing.assert_allclose(uniform.sum(), 1)
    np.testing.assert_allclose(sampled.sum(), 1)
    assert sampled.min() > 0
    assert sampled.max() <= 4 / len(wins) + 1e-9
    assert sampled[-1] > uniform[-1]
