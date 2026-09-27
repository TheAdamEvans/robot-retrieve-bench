from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alloy_index.models.fused import Head, load, save
from alloy_trainer.eval.fused_development import ranked_metrics, require_out_of_fold
from alloy_trainer.learn import importance
from alloy_trainer.learn.fused_v2 import probabilities
from alloy_trainer.learn.fused_supervision import development_programs, programs


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
    assert len(expanded) >= 140
    assert all("lateral_clearance_right_m" not in str(p.program) for p in expanded)
    assert all("speed-up" not in p.text for p in expanded)


def test_development_queries_hold_out_programs_and_texts():
    import json
    train, development = programs(), development_programs()
    assert not {p.text for p in train} & {p.text for p in development}
    assert not {json.dumps(p.program, sort_keys=True) for p in train} & {
        json.dumps(p.program, sort_keys=True) for p in development}


def test_ranked_metrics_expose_unknown_top_results_without_calling_them_negative():
    result = ranked_metrics(["unknown", "positive", "negative"], {"positive"}, {"negative"})
    assert result["hit_at_10"] == 1
    assert result["judged_at_10"] == 2 / 3
    assert result["precision_at_10_judged"] == 0.5
    assert result["recall_at_50"] == 1
    assert result["ndcg_at_10_judged"] == 1 / np.log2(3)


def test_development_measurement_rejects_in_sample_train_vectors(tmp_path):
    from types import SimpleNamespace
    import pytest

    (tmp_path / "index").mkdir()
    path = tmp_path / "index" / "candidate_windows.parquet"
    pq.write_table(pa.table({"recording_id": ["Train_A", "Val_B"],
                             "source": ["full_model", "full_model"]}), path)
    with pytest.raises(ValueError, match="not out of fold"):
        require_out_of_fold(SimpleNamespace(root=tmp_path), "candidate", ["Train_A"])


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
