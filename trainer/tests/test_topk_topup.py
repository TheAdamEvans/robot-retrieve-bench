from types import SimpleNamespace

from alloy_trainer.annotate.topk_topup import assignments, brief_for


def _query(query_id, intent, abstain=False):
    return SimpleNamespace(query_id=query_id, intent_group_id=intent, intent=intent,
                           expect_abstain=abstain, scope=SimpleNamespace(recording_ids=[]))


def test_assignments_are_blind_deduplicated_and_skip_existing_or_abstain():
    rows = [
        {"config": "candidate", "query_id": "q1", "windows": ["A:0004", "A:0005", "B:0004"]},
        {"config": "candidate", "query_id": "q2", "windows": ["A:0005", "B:0004", "B:0005"]},
        {"config": "other", "query_id": "q1", "windows": ["C:0004"]},
        {"config": "candidate", "query_id": "battery", "windows": ["D:0004"]},
    ]
    queries = [_query("q1", "intent"), _query("q2", "intent"), _query("battery", "abstain", True)]
    work = assignments(rows, queries, {("intent", "A:0004")}, "candidate", 2)
    assert list(work) == ["intent"]
    assert work["intent"]["windows"] == ["A:0005", "B:0004"]
    brief = brief_for("intent", "opaque-job", work["intent"])
    assert "blinded evaluation pool" in brief
    assert "several retrieval systems" not in brief
    assert "A:0005" in brief and "B:0004" in brief
