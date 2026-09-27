"""Offline checks for the candidate pack; does not judge relevance or call models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HELD_OUT = {"Rec_Tent_129", "Bass_Garage_134"}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        require(sha(HERE / name) == digest, f"Pack checksum changed: {name}")
    for name, digest in manifest["protectedFiles"].items():
        require(sha(ROOT / name) == digest, f"Protected query file changed: {name}")

    # Load source rows once. Audit is append-only; validate referenced lines rather
    # than rejecting legitimate later tool calls elsewhere in the project.
    audit = read_rows(ROOT / "annotations/audit.jsonl")
    label_cache: dict[str, list[dict]] = {}
    seen_ids: set[str] = set()
    seen_intents: set[str] = set()
    seen_utterances: set[str] = set()
    source_count = 0
    budget_by_split: dict[str, int] = {}
    for split in ("dev", "test"):
        queries = read_rows(HERE / f"queries_{split}.jsonl")
        requests = read_rows(HERE / f"requests_{split}.jsonl")
        sources = read_rows(HERE / f"sources_{split}.jsonl")
        require(len(queries) == manifest["counts"][f"l1_compositions_{split}"], "Count mismatch")
        require(len(queries) == len(requests) == len(sources), "Missing query/request/source row")
        budget_by_split[split] = 0
        for q, request, source in zip(queries, requests, sources, strict=True):
            qid = q["queryId"]
            require(qid not in seen_ids, f"Duplicate query: {qid}")
            require(q["intentGroupId"] not in seen_intents, f"Duplicate canonical intent: {qid}")
            require(q["utterance"] not in seen_utterances, f"Duplicate utterance: {qid}")
            seen_ids.add(qid)
            seen_intents.add(q["intentGroupId"])
            seen_utterances.add(q["utterance"])
            require(q["status"] == "NEEDS_INDEPENDENT_JUDGMENT", f"Unsupported gold claim: {qid}")
            require("behavior_review" in q["tags"], f"Outside behavior focus: {qid}")
            require(bool(q["behaviorValue"]), f"Missing behavior value: {qid}")
            cap = q["judgingPlan"]["maxInitialEpisodeIntentJudgments"]
            require(cap == manifest["initialJudgingPlan"]["maxEpisodesPerIntent"], f"Wrong judging cap: {qid}")
            require(2 <= len(q["candidateContrasts"]) <= cap, f"Unbounded initial candidates: {qid}")
            budget_by_split[split] += cap
            require(q["querySet"] == f"l1_compositions_{split}", f"Wrong query set: {qid}")
            require(source["queryId"] == qid == request["queryId"], f"Mismatched source: {qid}")
            keys = {"queryId", "intentGroupId", "querySet", "utterance", "scope"}
            require(set(request) == keys, f"Author evidence leaked into request: {qid}")
            require(request == {k: q[k] for k in keys}, f"Request differs from query: {qid}")
            scope = set(q["scope"]["recordingIds"])
            require(bool(scope), f"Unbounded scope: {qid}")
            require(bool(scope & HELD_OUT) == (split == "test"), f"Split violation: {qid}")
            # This release's test cases are entirely within the held-out recordings.
            if split == "test":
                require(scope <= HELD_OUT, f"Unexpected cross-split test scope: {qid}")
            require(len(q["requiredClauses"]) >= 3, f"Underspecified clauses: {qid}")
            for key in ("hardNegativeRecipe", "capabilityGaps", "returnRequirements"):
                require(bool(q[key]), f"Missing {key}: {qid}")
            for span in q["discoverySpans"]:
                require(span["recordingId"] in scope, f"Seed outside scope: {qid}")
                require(0 <= span["startS"] < span["endS"], f"Invalid span: {qid}")
                require(span["role"] == "discovery_only_not_gold", f"Seed asserted as gold: {qid}")
            for contrast in q["candidateContrasts"]:
                require(contrast["windowId"].split(":")[0] in scope, f"Contrast outside scope: {qid}")
                require(contrast["hypothesis"] in {"likely_match", "near_miss", "partial", "uncertain", "mixed"},
                        f"Unknown contrast role: {qid}")
            require(bool(source["labelSources"]), f"No evidence leads: {qid}")
            cited_segments = {s["segmentId"] for s in source["labelSources"]}
            require(all(c["windowId"] in cited_segments for c in q["candidateContrasts"]),
                    f"Initial candidate lacks cited label source: {qid}")
            for s in source["labelSources"]:
                path = s["path"]
                if path not in label_cache:
                    label_cache[path] = read_rows(ROOT / path)
                original = label_cache[path][s["line"] - 1]
                require(original["segmentId"] == s["segmentId"], f"Label line drift: {qid}")
                require(original["caption"] == s["caption"], f"Caption drift: {qid}")
                require(original.get("refs", []) == s["refs"], f"Sensor-ref drift: {qid}")
                require(original["recordingId"] in scope, f"Source outside scope: {qid}")
                source_count += 1
            for s in source["reportSources"]:
                path = ROOT / s["path"]
                require(sha(path) == s["sha256"], f"Report changed: {qid}")
                require(json.loads(path.read_text())["session_id"] == s["sessionId"], f"Wrong session: {qid}")
            for s in source["reviewedRenders"]:
                require(sha(ROOT / s["path"]) == s["sha256"], f"Inspected render changed: {qid}")
            require(bool(source["auditLines"]), f"No audit provenance: {qid}")
            for line in source["auditLines"]:
                require(1 <= line <= len(audit), f"Audit line missing: {qid}")
                args = audit[line - 1].get("args", {})
                rec = args.get("rec") or args.get("window_id", "").split(":")[0]
                require(rec in scope, f"Audit evidence outside scope: {qid}")
        # No held-out scene text can enter any development-facing data artifact.
        if split == "dev":
            for name in ("queries_dev.jsonl", "requests_dev.jsonl", "sources_dev.jsonl", "catalog_dev.md", "attention_dev.json"):
                body = (HERE / name).read_text()
                require(not any(rec in body for rec in HELD_OUT), f"Held-out reference in {name}")
    plan = manifest["initialJudgingPlan"]
    require(budget_by_split["dev"] == plan["devMaximum"], "Dev judging cap mismatch")
    require(budget_by_split["test"] == plan["testMaximum"], "Test judging cap mismatch")
    require(sum(budget_by_split.values()) == plan["maxEpisodeIntentJudgments"], "Total judging cap mismatch")
    print(f"PASS: {len(seen_ids)} unique behavior intents; {source_count} label citations;")
    print(f"initial judging cap: {budget_by_split['dev']} dev + {budget_by_split['test']} test episode–intent judgments;")
    print("request isolation, local evidence, split boundaries and protected query fingerprints verified.")
    print("This validates artifact integrity, not relevance, numeric clauses or retrieval quality.")


if __name__ == "__main__":
    main()
