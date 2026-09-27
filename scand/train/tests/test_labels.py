import json
import os

import pytest

from alloy_train.annotate.store import load_labels
from alloy_train.eval.report import load_judgments


def write_shard(ann, name, grade):
    path = ann / "labels" / "judgment" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"intentGroupId": "intent", "windowId": "recording:0004", "grade": grade}) + "\n")
    return path


def test_revision_precedence_survives_reversed_and_equal_file_times(tmp_path):
    base = write_shard(tmp_path, "base.jsonl", 1)
    revision = write_shard(tmp_path, "revision.jsonl", 2)
    (tmp_path / "labels" / "precedence.json").write_text(json.dumps({"judgment": {"revision.jsonl": 1}}))
    for base_time, revision_time in [(100, 200), (200, 100), (100, 100)]:
        os.utime(base, (base_time, base_time))
        os.utime(revision, (revision_time, revision_time))
        assert load_judgments(tmp_path) == {"intent": {"recording:0004": 2}}
        assert load_labels(tmp_path, "judgment")["intent|recording:0004"]["grade"] == 2


def test_ambiguous_cross_shard_revision_requires_explicit_precedence(tmp_path):
    write_shard(tmp_path, "base.jsonl", 1)
    write_shard(tmp_path, "revision.jsonl", 2)
    with pytest.raises(ValueError, match="set explicit shard priorities"):
        load_judgments(tmp_path)
