"""Contract tests: causality, validation, pipeline invariants, lazy program, accounting, isolation."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from google.protobuf import json_format

from alloy_server.gen.alloy.v1 import answer_pb2 as a
from alloy_server.gen.alloy.v1 import common_pb2 as c
from alloy_server.gen.alloy.v1 import pipeline_pb2 as pp
from alloy_server.gen.alloy.v1 import query_pb2 as q
from alloy_server.programs.validator import validate
from alloy_server.timeline.store import NO_HEADER, TopicTimeline
from alloy_server.verify.receipts import causal_last

SERVER = Path(__file__).resolve().parents[1] / "src" / "alloy_server"
BUNDLE = Path(__file__).resolve().parents[2] / "bundles" / "dev"
MS = 1_000_000


class FakeRec:
    def __init__(self, log_ms, header_ms):
        log = np.array(log_ms, dtype=np.int64) * MS
        hdr = np.array([NO_HEADER if h is None else h * MS for h in header_ms], dtype=np.int64)
        n = len(log)
        self.topics = {"/t": TopicTimeline("/t", log, hdr, np.arange(n), np.zeros((n, 16), np.uint8),
                                           *(np.zeros(n, np.int64) for _ in range(4)))}

    def message_id(self, topic, i):
        return c.MessageId(recording_id="fake", topic=topic, topic_ordinal=i)


# ---------------- causal receipts ----------------

def test_strict_before_excludes_message_at_cutoff_inclusive_includes_it():
    rec = FakeRec([100, 200], [95, 195])
    strict = causal_last(rec, "/t", 200 * MS, 1000 * MS, c.STRICT_BEFORE)
    incl = causal_last(rec, "/t", 200 * MS, 1000 * MS, c.INCLUSIVE)
    assert strict.message.topic_ordinal == 0 and incl.message.topic_ordinal == 1


def test_future_stamped_message_cannot_hide_earlier_valid_one():
    # message 1 claims to be measured 500 ms after it was received: invalid, masked before selection
    rec = FakeRec([100, 150], [90, 650])
    it = causal_last(rec, "/t", 200 * MS, 1000 * MS, c.STRICT_BEFORE)
    assert it.message.topic_ordinal == 0 and it.ok == c.TRUTH_TRUE


def test_negative_age_is_unknown_not_accepted():
    rec = FakeRec([100], [110])  # stamped 10 ms after receipt: within skew tolerance, but after the cutoff
    it = causal_last(rec, "/t", 105 * MS, 1000 * MS, c.STRICT_BEFORE)
    assert it.ok == c.TRUTH_UNKNOWN and it.reason == c.FUTURE_MEASUREMENT


def test_stale_is_unknown_with_no_fallback():
    rec = FakeRec([100], [100])
    it = causal_last(rec, "/t", 900 * MS, 500 * MS, c.STRICT_BEFORE)
    assert it.ok == c.TRUTH_UNKNOWN and it.reason == c.STALE and it.message.topic_ordinal == 0


def test_missing_header_is_unknown():
    it = causal_last(FakeRec([100], [None]), "/t", 200 * MS, 500 * MS, c.STRICT_BEFORE)
    assert it.reason == c.HEADER_MISSING


# ---------------- semantic validation ----------------

ROBOTS = {"Butler": "spot", "Sanjac": "jackal"}


def prog(d):
    base = {"primaryEvent": "e", "selection": {"quantifier": "ALL"}, "contextBefore": {"value": 4, "unit": "S"},
            "contextAfter": {"value": 4, "unit": "S"}}
    return json_format.ParseDict({**base, **d}, q.QueryProgram())


def codes(p):
    return {e.code for e in validate(p, ROBOTS)}


def test_unknown_feature_rejected():
    p = prog({"events": [{"name": "e", "kind": "THRESHOLD", "feature": "battery", "comparator": "LT",
                          "threshold": {"value": 1, "unit": "RATIO"}}]})
    assert "SEM_UNKNOWN_FEATURE" in codes(p)


def test_unit_dimension_mismatch_rejected_and_conversion_applied():
    bad = prog({"events": [{"name": "e", "kind": "THRESHOLD", "feature": "speed_mps", "comparator": "GT",
                            "threshold": {"value": 1, "unit": "M"}}]})
    assert "SEM_UNIT_MISMATCH" in codes(bad)
    ok = prog({"events": [{"name": "e", "kind": "THRESHOLD", "feature": "speed_mps", "comparator": "GT",
                           "threshold": {"value": 3.6, "unit": "KMH"}}]})
    assert not validate(ok, ROBOTS) and abs(ok.events[0].threshold.value - 1.0) < 1e-9


def test_relation_cycle_and_orphans_rejected():
    ev = lambda n: {"name": n, "kind": "THRESHOLD", "feature": "speed_mps", "comparator": "GT",
                    "threshold": {"value": 1, "unit": "MPS"}}
    r = lambda p_, ch: {"parent": {"event": p_, "point": "START"}, "child": {"event": ch, "point": "START"},
                        "kind": "WITHIN", "maxGap": {"value": 1, "unit": "S"}}
    p = prog({"events": [ev("e"), ev("x"), ev("y")], "relations": [r("x", "y"), r("y", "x")]})
    assert {"SEM_RELATION_CYCLE", "SEM_NOT_TREE"} & codes(p)


def test_track_kind_needs_track_feature():
    p = prog({"events": [{"name": "e", "kind": "TRACK_APPEAR", "feature": "speed_mps"}]})
    assert "SEM_BAD_KIND_FEATURE" in codes(p)


# ---------------- enums never emitted unset ----------------

def test_answer_status_and_truth_enums_have_unset_zero():
    for enum in (a.AnswerStatus, c.Truth, pp.Completeness, pp.StageOutcome, c.Boundary):
        assert enum.Name(0).endswith("UNSPECIFIED")


# ---------------- isolation and no dispatch ----------------

def test_server_never_imports_train_or_offline_deps():
    bad = re.compile(r"^\s*(from|import)\s+(alloy_train|alloy_index|rosbags|scipy|duckdb|mcap\b)", re.M)
    offenders = [str(p) for p in SERVER.rglob("*.py") if "gen" not in p.parts and bad.search(p.read_text())]
    assert not offenders, offenders


def test_no_puzzle_dispatch_in_server():
    pat = re.compile(r"\bP\d\d\b|crowd_hesitation|doorway_crossing|last_safe_evidence|chained_turn_person")
    offenders = [str(p) for p in SERVER.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".textproto")
                 and "gen" not in p.parts and pat.search(p.read_text())]
    assert not offenders, offenders


def test_no_direct_bundle_io_outside_io_package():
    pat = re.compile(r"pq\.read_table\(|np\.load\(|safe_open\(|mmap\.")
    offenders = [str(p) for p in SERVER.rglob("*.py") if "io" not in p.parts and "gen" not in p.parts
                 and pat.search(p.read_text())]
    assert not offenders, offenders


# ---------------- pipeline (needs the dev bundle) ----------------

needs_bundle = pytest.mark.skipif(not (BUNDLE / "intake").exists(), reason="dev bundle not built")


@pytest.fixture(scope="module")
def bundle():
    from alloy_server.bundle import Bundle
    return Bundle(BUNDLE)


@needs_bundle
def test_tags_never_triggers_generation(bundle):
    from alloy_server.pipeline.runner import run
    calls = []
    resp = run(bundle, bundle.specs["TAGS"], a.SearchRequest(utterance="crowd", pipeline_id="TAGS"),
               generator=lambda u, b: calls.append(u))
    assert not calls and resp.status == a.ANSWERED_UNVERIFIED


@needs_bundle
def test_provided_program_is_never_regenerated_and_unknown_survives(bundle):
    from alloy_server.pipeline.runner import run
    calls = []
    p = json_format.ParseDict({
        "primaryEvent": "e", "selection": {"quantifier": "ALL"}, "contextBefore": {"value": 4, "unit": "S"},
        "contextAfter": {"value": 4, "unit": "S"},
        "events": [{"name": "e", "kind": "THRESHOLD", "feature": "persons_in_corridor", "comparator": "GTE",
                    "threshold": {"value": 6, "unit": "DIMENSIONLESS"}, "required": True}]}, q.QueryProgram())
    req = a.SearchRequest(utterance="x", pipeline_id="PROGRAM", k=5, program=p)
    req.scope.recording_ids.append("Butler")
    resp = run(bundle, bundle.specs["PROGRAM"], req, generator=lambda u, b: calls.append(u))
    assert not calls and resp.diagnostics.state == a.PROVIDED
    # >=6 in corridor is never provable under the ESTIMATED band: candidates survive as UNKNOWN, none filtered
    assert resp.results and all(r.candidate.clauses[0].truth == c.TRUTH_UNKNOWN for r in resp.results)
    assert resp.status == a.ANSWERED_PARTIAL


@needs_bundle
def test_undeclared_program_access_raises(bundle):
    from alloy_server.pipeline.context import QueryContext, UndeclaredProgramAccess
    from alloy_server.pipeline.stages import BM25Ranker
    ctx = QueryContext(bundle, a.SearchRequest(utterance="x"), generator=None)
    with pytest.raises(UndeclaredProgramAccess):
        ctx.require_program(BM25Ranker("bm25", {}))


@needs_bundle
def test_mcap_bytes_counted_equal_chunk_length(bundle):
    from alloy_server.io import cost
    rec = bundle.recordings["RLM"]
    tl = rec.topics["/odom"]
    with cost.scope("t") as s:
        rec.read("/odom", 10)
    assert s.totals()["bytes_read"] == int(tl.chunk_len[10])


def test_onset_not_delayed_by_gait_oscillation():
    """Regression: the labeller found onsets ~0.6 s late because early-stride raw speed dips below threshold."""
    from alloy_server.catalog.features import Series
    from alloy_server.catalog.registry import REGISTRY
    from alloy_server.verify.executor import onset_instances
    t = (np.arange(0, 6, 0.06) * 1e9).astype(np.int64)
    raw = np.where(t < 3e9, 0.0, 0.8 + 0.75 * np.sin(2 * np.pi * 3.3 * (t / 1e9 - 3)))  # starts at 3.0 s, dips < 0.1
    smooth = np.convolve(raw, np.ones(15) / 15)[: len(raw)]
    s = Series(REGISTRY["speed_mps"], t, smooth, None, None, t, 16.0, True, [])
    ev = json_format.ParseDict({"name": "go", "kind": "ONSET", "feature": "speed_mps", "fromBelow": {"value": 0.05, "unit": "MPS"},
                                "minDuration": {"value": 1, "unit": "S"}, "threshold": {"value": 0.1, "unit": "MPS"},
                                "sustain": {"value": 0.5, "unit": "S"}}, q.EventSpec())
    (inst,) = onset_instances(s, ev, raw)
    assert abs(inst.start / 1e9 - 3.0) < 0.1


@needs_bundle
def test_unexpressible_only_program_abstains_or_is_unverified(bundle):
    """A purely visual question ('legs close up') has no expressible requirement: PROGRAM abstains, HYBRID ranks by
    similarity and says it is unverified — never a loosely related proxy feature."""
    from alloy_server.pipeline.runner import run
    p = json_format.ParseDict({"primaryEvent": "", "events": [], "unexpressible": ["legs close up"],
                               "selection": {"quantifier": "ALL"}, "contextBefore": {"value": 4, "unit": "S"},
                               "contextAfter": {"value": 4, "unit": "S"}}, q.QueryProgram())
    assert not validate(p, bundle.robots)
    prog = run(bundle, bundle.specs["PROGRAM"], a.SearchRequest(utterance="legs close up", pipeline_id="PROGRAM", program=p))
    hyb = run(bundle, bundle.specs["HYBRID"], a.SearchRequest(utterance="legs close up", pipeline_id="HYBRID", k=3, program=p))
    assert prog.status == a.INSUFFICIENT_EVIDENCE and not prog.results
    assert hyb.status == a.ANSWERED_UNVERIFIED and hyb.results and any("not expressible" in n for n in hyb.notes)


@needs_bundle
def test_detector_silence_never_yields_none_found_exhaustive(bundle):
    """A clause on a learned-detector feature that matches nothing is not proof of absence."""
    from alloy_server.pipeline.runner import run
    p = json_format.ParseDict({"primaryEvent": "e", "selection": {"quantifier": "ALL"},
                               "contextBefore": {"value": 4, "unit": "S"}, "contextAfter": {"value": 4, "unit": "S"},
                               "events": [{"name": "e", "kind": "THRESHOLD", "feature": "persons_visible_front",
                                           "comparator": "GTE", "threshold": {"value": 500, "unit": "DIMENSIONLESS"},
                                           "required": True}]}, q.QueryProgram())
    resp = run(bundle, bundle.specs["PROGRAM"], a.SearchRequest(utterance="x", pipeline_id="PROGRAM", program=p))
    assert not resp.results and resp.status == a.INSUFFICIENT_EVIDENCE


@needs_bundle
@pytest.mark.parametrize("requested,default,expected", [(5, 10, 5), (20, 10, 20), (50, 10, 50), (0, 20, 20), (0, 0, 10)])
def test_search_result_count_uses_request_then_pipeline_default(bundle, requested, default, expected):
    from alloy_server.pipeline.runner import run
    spec = pp.PipelineSpec()
    spec.CopyFrom(bundle.specs["TAGS"])
    spec.final_k = default
    req = a.SearchRequest(utterance="doorway", pipeline_id="TAGS", k=requested)
    resp = run(bundle, spec, req)
    assert len(resp.results) == expected


@needs_bundle
def test_score_all_ignores_search_result_limits(bundle):
    from alloy_server.pipeline.runner import run
    wins = next(ws[:25] for ws in bundle.windows.values() if len(ws) >= 25)
    req = a.SearchRequest(utterance="doorway", pipeline_id="TAGS", k=5, mode=pp.SCORE_ALL, window_ids=wins)
    resp = run(bundle, bundle.specs["TAGS"], req)
    assert {r.candidate.window_id for r in resp.results} == set(wins)


@needs_bundle
@pytest.mark.parametrize("failure", ["timeout", "rate_limit", "connection"])
def test_program_api_failure_is_reported_and_next_search_can_recover(bundle, tmp_path, monkeypatch, failure):
    import httpx
    from openai import APIConnectionError, APITimeoutError, RateLimitError
    from unittest.mock import Mock
    from alloy_server.pipeline.runner import run
    from alloy_server.programs.openai_generator import OpenAIProgramGenerator

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ALLOY_PROGRAM_TIMEOUT_S", "20")
    gen = OpenAIProgramGenerator(bundle, [], cache_dir=tmp_path)
    assert gen.client.timeout == 20 and gen.client.max_retries == 0
    request = httpx.Request("POST", "https://example.invalid")
    errors = {
        "timeout": APITimeoutError(request=request),
        "connection": APIConnectionError(request=request),
        "rate_limit": RateLimitError("test", response=httpx.Response(429, request=request), body=None),
    }
    call = Mock(side_effect=errors[failure])
    monkeypatch.setattr(gen.client.chat.completions, "create", call)
    try:
        for pid, expected in [("PROGRAM", a.INSUFFICIENT_EVIDENCE), ("HYBRID", a.ANSWERED_UNVERIFIED)]:
            req = a.SearchRequest(utterance="the robot moves", pipeline_id=pid, k=3)
            resp = run(bundle, bundle.specs[pid], req, generator=gen)
            assert resp.status == expected and resp.diagnostics.state == a.FAILED
            assert resp.cost.llm_calls == 1 and resp.diagnostics.attempts == 1
            assert any("Program generation unavailable" in note for note in resp.notes)
            assert bool(resp.results) == (pid == "HYBRID")
        assert call.call_count == 2 and not list(tmp_path.iterdir())

        program = {"primaryEvent": "moving", "events": [{"name": "moving", "kind": "THRESHOLD",
                   "feature": "speed_mps", "comparator": "GT", "required": True,
                   "threshold": {"value": 0.1, "unit": "MPS"}}], "selection": {"quantifier": "ALL"},
                   "contextBefore": {"value": 4, "unit": "S"}, "contextAfter": {"value": 4, "unit": "S"}}
        import json
        call.side_effect = None
        call.return_value = SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(program)))])
        recovered = run(bundle, bundle.specs["PROGRAM"], a.SearchRequest(utterance="the robot moves", pipeline_id="PROGRAM"),
                        generator=gen)
        assert recovered.diagnostics.state == a.GENERATED and recovered.results
        assert call.call_count == 3 and list(tmp_path.glob("*.json"))
    finally:
        gen.client.close()


@needs_bundle
def test_cache_only_generation_fails_on_miss_without_an_api_client(bundle, tmp_path, monkeypatch):
    from alloy_server.programs.openai_generator import OpenAIProgramGenerator
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen = OpenAIProgramGenerator(bundle, [], cache_dir=tmp_path, cache_only=True)
    assert gen.client is None
    with pytest.raises(RuntimeError, match="refusing a new API call"):
        gen("a query with no cached program", bundle)


@needs_bundle
def test_merge_is_display_only_and_accounts_for_every_member(bundle):
    from alloy_server.pipeline.runner import run
    spec = bundle.specs["TAGS"]
    assert any(st.impl == "MergeAdjacent" for st in spec.rankers)
    bare = pp.PipelineSpec()
    bare.CopyFrom(spec)
    del bare.rankers[:]
    bare.rankers.extend(st for st in spec.rankers if st.impl != "MergeAdjacent")
    ids = lambda r: [x.candidate.candidate_id for x in r.results]
    req = a.SearchRequest(utterance="doorway", pipeline_id="TAGS", k=10)
    assert ids(run(bundle, spec, req)) == ids(run(bundle, bare, req))  # eval requests: identical to no merge stage
    req.presentation = True
    shown = run(bundle, spec, req)
    merged = [x.candidate for x in shown.results if x.candidate.kind == pp.MERGED]
    assert merged and len(shown.results) == 10
    into = {f.candidate_id: f.detail for f in shown.filtered if f.reason == pp.MERGED_INTO}
    for m in merged:
        assert all(into[i] == m.candidate_id for i in m.member_ids)
        assert (m.seed.end_ns - m.seed.start_ns) <= 20 * 10**9
