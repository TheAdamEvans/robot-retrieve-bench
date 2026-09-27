import json
import os

import pytest

from alloy_index.annotate.store import load_labels, use_for
from alloy_trainer.eval.report import load_judgments


def write_shard(root, use, campaign, grade, intent="intent"):
    path = root / use / "judgments" / f"{campaign}.job.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"intentGroupId": intent, "windowId": "recording:0004", "grade": grade,
                                "campaign": campaign}) + "\n")
    return path


def set_priority(root, campaign, priority):
    d = root / "metadata" / campaign
    d.mkdir(parents=True, exist_ok=True)
    (d / "campaign.json").write_text(json.dumps({"campaign": campaign, "priority": priority}))


def test_campaign_priority_survives_reversed_and_equal_file_times(tmp_path):
    base = write_shard(tmp_path, "eval", "base", 1)
    revision = write_shard(tmp_path, "eval", "revision", 2)
    set_priority(tmp_path, "revision", 1)
    for base_time, revision_time in [(100, 200), (200, 100), (100, 100)]:
        os.utime(base, (base_time, base_time))
        os.utime(revision, (revision_time, revision_time))
        assert load_judgments(tmp_path) == {"intent": {"recording:0004": 2}}
        assert load_labels("judgment", ("eval",), tmp_path)["intent|recording:0004"]["grade"] == 2


def test_ambiguous_cross_campaign_revision_requires_explicit_priority(tmp_path):
    write_shard(tmp_path, "eval", "base", 1)
    write_shard(tmp_path, "eval", "revision", 2)
    with pytest.raises(ValueError, match="higher priority"):
        load_judgments(tmp_path)


def test_a_label_has_exactly_one_use(tmp_path):
    write_shard(tmp_path, "train", "a", 1)
    write_shard(tmp_path, "eval", "b", 1)
    with pytest.raises(ValueError, match="has one use"):
        load_labels("judgment", ("train", "eval"), tmp_path)


def test_agreement_judgments_never_reach_scoring(tmp_path):
    write_shard(tmp_path, "eval", "first", 2)
    write_shard(tmp_path, "agreement", "recheck", 0)
    assert load_judgments(tmp_path) == {"intent": {"recording:0004": 2}}
    assert load_judgments(tmp_path, uses=("agreement",)) == {"intent": {"recording:0004": 0}}


def test_new_records_are_routed_by_use(tmp_path):
    set_priority(tmp_path, "recheck", 0)
    (tmp_path / "metadata" / "recheck" / "campaign.json").write_text(json.dumps({"campaign": "recheck", "use": "agreement"}))
    assert use_for("attributes", {"segmentId": "R:0004"}, "l1-x", tmp_path) == "train"
    assert use_for("judgment", {"intentGroupId": "crowd_hesitation"}, "l2-x", tmp_path) == "eval"
    assert use_for("judgment", {"intentGroupId": "l1x_dream_steer_or_brake"}, "ep-x", tmp_path) == "train"
    assert use_for("judgment", {"intentGroupId": "l1x_dream_test_stairs_vs_people"}, "ep-x", tmp_path) == "eval"
    assert use_for("judgment", {"intentGroupId": "crowd_hesitation"}, "recheck", tmp_path) == "agreement"
