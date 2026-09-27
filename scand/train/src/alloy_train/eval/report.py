"""EvalReport: one table per query set x config, stage diagnostics, program metrics, cost; JSON + a static HTML page."""
from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

from alloy_train.eval import metrics as M
from alloy_train.eval.run import CONFIGS, EVAL_SETS
from alloy_train.recordings import SCAND_ROOT

ACCEPTABLE = {  # status outcomes that answer the intent honestly (the puzzle allows either for P12)
    "vehicle_interaction_gdc": {"ANSWERED", "ANSWERED_PARTIAL", "INSUFFICIENT_EVIDENCE"},
}


def load_judgments(ann: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    d = ann / "labels" / "judgment"
    for p in sorted(d.glob("*.jsonl"), key=lambda x: x.stat().st_mtime) if d.exists() else []:
        for line in p.read_text().splitlines():
            r = json.loads(line)
            qrels[r["intentGroupId"]][r["windowId"]] = int(r.get("grade", 0))
    return qrels


def clause_set(prog: dict | None) -> set[tuple]:
    """Canonical clause tuples for program comparison (units already canonical after validation)."""
    if not prog:
        return set()
    out = set()
    for e in prog.get("events", []):
        thr = e.get("threshold", {}).get("value") or e.get("change", {}).get("value")
        out.add(("event", e["kind"], e["feature"], e.get("comparator"), e.get("direction"),
                 None if thr is None else round(float(thr), 2)))
    for r in prog.get("relations", []):
        out.add(("relation", r["kind"], r.get("maxGap", {}).get("value") and round(r["maxGap"]["value"], 1)))
    if prog.get("receipt"):
        out.add(("receipt", tuple(sorted(prog["receipt"].get("sensors", [])))))
    out.add(("select", prog.get("selection", {}).get("quantifier")))
    out |= {("unexpressible",)} if prog.get("unexpressible") else set()
    return out


def build(run_dir: Path, ann: Path) -> dict:
    runs = [json.loads(x) for x in (run_dir / "runs.jsonl").read_text().splitlines()]
    sa_path = run_dir / "score_all.jsonl"
    score_all = [json.loads(x) for x in sa_path.read_text().splitlines()] if sa_path.exists() else []
    qrels = load_judgments(ann)
    group_of = {r["query_id"]: r["intent_group_id"] for r in runs}
    by = defaultdict(dict)
    for r in runs:
        by[(r["query_set"], r["config"])][r["query_id"]] = r
    sa = {(r["query_id"], r["config"]): r for r in score_all}
    oracle_windows = {(r["query_id"], r["config"].split("_")[0]): r["windows"][:10] for r in runs
                      if r["config"].endswith("ORACLE")}
    tables = {}
    for qset in EVAL_SETS:
        rows = []
        for cfg in CONFIGS:
            rs = by.get((qset, cfg), {})
            if not rs:
                continue
            per = defaultdict(dict)
            for qid, r in rs.items():
                qr = qrels.get(r["intent_group_id"], {})
                judged = {w: g for w, g in qr.items() if g >= 0}
                npos = sum(g >= M.POS for g in judged.values())
                nneg = sum(g < 1 for g in judged.values())
                auc_eligible = npos > 0 and nneg > 0
                if npos:
                    per["ndcg10"][qid] = M.ndcg_at(r["windows"], judged, 10)
                    per["ndcg10_unj0"][qid] = M.ndcg_at(r["windows"], judged, 10, unjudged_as_zero=True)
                    per["r10"][qid] = M.recall_at(r["windows"], judged, 10)
                    per["r50"][qid] = M.recall_at(r["windows"], judged, 50)
                    pos = {w for w, g in judged.items() if g >= M.POS}
                    per["gen_recall"][qid] = len(pos & set(r["generated"])) / len(pos)
                    gj = [w for w in r["generated"] if w in judged]
                    per["gen_precision"][qid] = (sum(judged[w] >= M.POS for w in gj) / len(gj)) if gj else None
                if judged:
                    per["judged10"][qid] = M.judged_at(r["windows"], judged, 10)
                s = sa.get((qid, cfg))
                if auc_eligible and s is not None:
                    wins = sorted(judged)
                    labels = [int(judged[w] >= M.POS) for w in wins]
                    prevalence = sum(labels) / len(labels)
                    if s["abstained"]:
                        per["coverage"][qid] = 0.0
                        per["roc_pen"][qid], per["pr_pen"][qid] = 0.5, prevalence
                    else:
                        scores = M.score_all_scores(s["order"], wins)
                        roc, pr = M.roc_auc(scores, labels), M.pr_auc(scores, labels)
                        per["coverage"][qid] = 1.0
                        per["roc_cond"][qid], per["pr_cond"][qid] = roc, pr
                        per["roc_pen"][qid], per["pr_pen"][qid] = roc, pr
                    fl = [w for w in s["filtered"] if w in judged]
                    if fl:
                        per["filter_precision"][qid] = sum(judged[w] == 0 for w in fl) / len(fl)
                ok = ACCEPTABLE.get(r["intent_group_id"], {r["expected_status"]})
                per["status_ok"][qid] = float(r["status"] in ok)
                per["wall_ms"][qid] = r["wall_ms"]
                c = r["cost"]
                per["raw_ratio"][qid] = float(c.get("rawRatio", 0.0))
                gen = r.get("generation_cost") or {}
                per["tokens"][qid] = float(c.get("promptTokens", 0)) + float(c.get("completionTokens", 0)) + \
                    (gen.get("tokens", 0) if c.get("promptTokens", 0) == 0 else 0)
                if CONFIGS[cfg][1] == "luna":
                    per["program_valid"][qid] = float(r["program_state"] == "GENERATED")
                    per["program_attempts"][qid] = float(r["program_attempts"])
                    oracle = next((x["program"] for x in by.get((qset, cfg.replace("LUNA", "ORACLE")), {}).values()
                                   if x["query_id"] == qid), None)
                    got, want = clause_set(r["program"]), clause_set(oracle)
                    if want:
                        per["clause_p"][qid] = len(got & want) / len(got) if got else 0.0
                        per["clause_r"][qid] = len(got & want) / len(want)
                    ow = oracle_windows.get((qid, cfg.split("_")[0]), [])
                    if ow or r["windows"]:
                        a_, b_ = set(r["windows"][:10]), set(ow)
                        per["exec_jaccard"][qid] = len(a_ & b_) / len(a_ | b_) if a_ | b_ else 1.0
            row = {"config": cfg, "n_queries": len(rs), "n_groups": len({group_of[q] for q in rs})}
            for k, v in per.items():
                row[k] = M.summarise(v, group_of)
            walls = sorted(per["wall_ms"].values())
            row["wall_p50"] = statistics.median(walls) if walls else None
            row["wall_p95"] = walls[min(len(walls) - 1, int(0.95 * len(walls)))] if walls else None
            rows.append(row)
        tables[qset] = rows
    return {"tables": tables, "n_judged": {k: len(v) for k, v in qrels.items()},
            "n_positive": {k: sum(g >= M.POS for g in v.values()) for k, v in qrels.items()}}


def fmt(x, pct=False, digits=2):
    if not isinstance(x, dict):
        return "–" if x is None else f"{x:.0f}"
    m, ci = x.get("mean"), x.get("ci")
    if m is None:
        return "–"
    s = f"{100 * m:.0f}%" if pct else f"{m:.{digits}f}"
    if ci:
        s += f" <span class=ci>[{(100 * ci[0]):.0f}–{(100 * ci[1]):.0f}]</span>" if pct else \
            f" <span class=ci>[{ci[0]:.{digits}f}–{ci[1]:.{digits}f}]</span>"
    return s


COLS = [("roc_pen", "ROC-AUC (penalised)", False), ("coverage", "coverage", True), ("roc_cond", "ROC-AUC (cond.)", False),
        ("pr_pen", "PR-AUC (pen.)", False), ("ndcg10", "nDCG@10", False), ("ndcg10_unj0", "nDCG@10 unj=0", False),
        ("r10", "R@10", False), ("judged10", "judged@10", True), ("gen_recall", "gen. recall", True),
        ("gen_precision", "gen. precision", True), ("filter_precision", "filter precision", True),
        ("status_ok", "status ok", True), ("tokens", "tokens/query", None), ("raw_ratio", "bytes / raw", None)]


def html_page(rep: dict) -> str:
    out = ["<!doctype html><meta charset=utf-8><title>SCAND search eval</title><style>",
           ":root{--bg:#fff;--fg:#1a1a1a;--mute:#666;--line:#e5e5e5}",
           "@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mute:#999;--line:#333}}",
           "body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1400px;padding:0 16px}",
           "table{border-collapse:collapse;margin:8px 0 24px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}",
           "th,td{border-bottom:1px solid var(--line);padding:5px 9px;text-align:right;white-space:nowrap}",
           "th:first-child,td:first-child{text-align:left}.ci{color:var(--mute);font-size:11px}</style>",
           "<h1>SCAND search — evaluation</h1>",
           "<p>Judgments are single-judge, agent-provisional (Opus labeller with audited tools). Unjudged windows are never counted "
           "as non-relevant except in the labelled <i>unj=0</i> column. AUC is macro over queries; <b>penalised</b> scores an "
           "abstaining config at chance. CIs: bootstrap over intent groups; <b>n is small</b> — read differences as directional.</p>"]
    for qset, rows in rep["tables"].items():
        if not rows:
            continue
        out.append(f"<h2>{html.escape(qset)} <span class=ci>({rows[0]['n_queries']} queries, {rows[0]['n_groups']} intent groups)</span></h2>")
        out.append("<table><tr><th>config</th>" + "".join(f"<th>{html.escape(c[1])}</th>" for c in COLS) + "<th>p50 ms</th><th>p95 ms</th></tr>")
        for r in rows:
            cells = []
            for key, _, pct in COLS:
                v = r.get(key)
                if key == "raw_ratio":
                    cells.append(f"{v['mean']:.1e}" if isinstance(v, dict) and v.get("mean") is not None else "–")
                elif key == "tokens":
                    cells.append(f"{v['mean']:.0f}" if isinstance(v, dict) and v.get("mean") is not None else "–")
                else:
                    cells.append(fmt(v, pct=bool(pct)))
            p50 = f"{r['wall_p50']:.0f}" if r.get("wall_p50") is not None else "–"
            p95 = f"{r['wall_p95']:.0f}" if r.get("wall_p95") is not None else "–"
            out.append(f"<tr><td>{r['config']}</td>" + "".join(f"<td>{c}</td>" for c in cells) + f"<td>{p50}</td><td>{p95}</td></tr>")
        out.append("</table>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a_ = ap.parse_args()
    run_dir = SCAND_ROOT / "results" / "eval" / a_.run
    rep = build(run_dir, SCAND_ROOT / "annotations")
    (run_dir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (run_dir / "report.html").write_text(html_page(rep))
    print(run_dir / "report.html")


if __name__ == "__main__":
    main()
