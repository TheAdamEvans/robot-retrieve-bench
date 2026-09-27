from alloy_trainer.eval.report import build, html_page
from alloy_trainer.eval.run import generation_costs


def test_replayed_report_preserves_original_generation_cost_and_marks_its_source(tmp_path):
    import json
    row = {
        "query_id": "q", "intent_group_id": "intent", "query_set": "demo5", "config": "PROGRAM_LUNA",
        "windows": [], "generated": [], "status": "INSUFFICIENT_EVIDENCE", "expected_status": "INSUFFICIENT_EVIDENCE",
        "program_state": "GENERATED", "program_attempts": 0, "program_cache_hit": True,
        "program": None, "stages": [], "cost": {}, "wall_ms": 25,
        "generation_cost": {"wall_ms": 5000, "tokens": 1234}, "generation_cost_source": "v7",
    }
    (tmp_path / "runs.jsonl").write_text(json.dumps(row) + "\n")
    report = build(tmp_path, tmp_path / "labels")
    result = report["tables"]["demo5"][0]
    assert result["wall_p50"] == 5025 and result["wall_cached_p50"] == 25
    assert result["tokens"]["mean"] == 1234
    assert report["generation_cost_sources"] == ["v7"]
    assert "Uncached latency is an estimate" in html_page(report)


def test_measured_generation_overrides_a_replay_reference():
    reference = {"query_id": "q", "config": "HYBRID_LUNA", "program_cache_hit": True,
                 "generation_cost": {"wall_ms": 5000, "tokens": 1234}}
    measured = {"query_id": "q", "config": "PROGRAM_LUNA", "program_cache_hit": False,
                "stages": [{"stage_id": "program_generation", "wall_ms": 6000, "tokens": 1500}]}
    expected = {"q": {"wall_ms": 6000, "tokens": 1500}}
    assert generation_costs([reference, measured]) == expected
    assert generation_costs([measured, reference]) == expected
