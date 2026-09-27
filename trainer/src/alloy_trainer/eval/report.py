"""EvalReport: one table per query set x config, stage diagnostics, program metrics, cost; JSON + a static HTML page."""
from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

from alloy_trainer.eval import metrics as M
from alloy_index.annotate.store import LABELS, campaign, load_labels
from alloy_trainer.eval.run import CONFIGS, EVAL_SETS, generation_costs
from alloy_index.recordings import held_out
from alloy_index.recordings import SCAND_ROOT

ACCEPTABLE = {  # status outcomes that answer the intent honestly (the puzzle allows either for P12)
    "vehicle_interaction_gdc": {"ANSWERED", "ANSWERED_PARTIAL", "INSUFFICIENT_EVIDENCE"},
    "test_body_cam_person": {"ANSWERED_PARTIAL", "INSUFFICIENT_EVIDENCE"},  # the decisive feature is not indexed
}


def load_judgments(root: Path = LABELS, uses: tuple[str, ...] = ("train", "eval")) -> dict[str, dict[str, int]]:
    """Window grades per intent. Agreement re-judgments are excluded unless asked for."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for r in load_labels("judgment", uses, root).values():
        qrels[r["intentGroupId"]][r["windowId"]] = int(r.get("grade", 0))
    return qrels


def ground_truth(root: Path = LABELS) -> dict[str, dict]:
    """Complete ground truth from exhaustive campaigns: {intent: {"episodes": [...], "complete": bool}}.

    Complete means every sweep chunk assigned to the intent's jobs (labels/metadata/<campaign>/jobs/) has at least
    one episode record; only then is recall reported."""
    gt: dict[str, dict] = {}
    rows = [r for r in load_labels("episode", ("train", "eval"), root).values()
            if campaign(r.get("campaign", ""), root).get("exhaustive")]
    for intent in {r["intentGroupId"] for r in rows}:
        mine = [r for r in rows if r["intentGroupId"] == intent]
        want = set()
        for p in {r["campaign"] for r in mine}:
            for j in (root / "metadata" / p / "jobs").glob(f"ex-{intent}-j*.json"):
                want |= {c["chunk_id"] for c in json.loads(j.read_text()).get("chunks", [])}
        got = {r.get("chunkId") for r in mine}
        gt[intent] = {"episodes": [r for r in mine if r.get("grade", 0) >= 1], "complete": bool(want) and want <= got,
                      "missing_chunks": sorted(want - got)}
    for p in (root / "metadata").glob("*/sweep/*.json"):  # nothing passes the numeric screen: complete, with no episodes
        sw = json.loads(p.read_text())
        if sw.get("sweep_recall_ok") and all(m["n"] == 0 for m in sw["members"].values()):
            gt[sw["intent"]] = {"episodes": [], "complete": True, "missing_chunks": [], "proof": "numeric screen"}
    return gt


def complete_qrels(qrels: dict[str, dict[str, int]], gt: dict[str, dict], bundle_dir: Path) -> None:
    """For an intent with complete ground truth, every in-scope window that overlaps no relevant episode is a real 0
    (not unjudged). Windows inside relevant episodes keep the judge's own window grades."""
    from alloy_server.catalog.windows import windows
    from alloy_trainer.eval import querysets
    scopes = {q.intent_group_id: list(q.scope.recording_ids)
              for name in querysets.challenge_sets() for q in querysets.load_challenge(name)}
    for intent, g in gt.items():
        if not g["complete"] or intent not in scopes:
            continue
        qr = qrels.setdefault(intent, {})
        for rec in scopes[intent]:
            info = json.loads((bundle_dir / "mcap" / f"{rec}.json").read_text())
            for w in windows(rec, (info["log_end_ns"] - info["log_start_ns"]) / 1e9):
                end = int(w.split(":")[1])
                inside = any(e["recordingId"] == rec and end - 4 < e["endS"] and end > e["startS"] for e in g["episodes"])
                if w not in qr and not inside:
                    qr[w] = 0


def episode_recall(ranked: list[str], episodes: list[dict], k: int) -> float | None:
    """Fraction of ground-truth episodes overlapped by at least one of the top-k windows (Rec:EEEE = [E-4, E] s)."""
    if not episodes:
        return None
    top = [(w.split(":")[0], int(w.split(":")[1])) for w in ranked[:k]]
    hit = sum(any(r == e["recordingId"] and end - 4 < e["endS"] and end > e["startS"] for r, end in top)
              for e in episodes)
    return hit / len(episodes)


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


def build(run_dir: Path, root: Path = LABELS, bundle_dir: Path = SCAND_ROOT / "bundles" / "dev") -> dict:
    runs = [json.loads(x) for x in (run_dir / "runs.jsonl").read_text().splitlines()]
    sa_path = run_dir / "score_all.jsonl"
    score_all = [json.loads(x) for x in sa_path.read_text().splitlines()] if sa_path.exists() else []
    qrels = load_judgments(root)
    gt = ground_truth(root)
    complete_qrels(qrels, gt, bundle_dir)
    group_of = {r["query_id"]: r["intent_group_id"] for r in runs}
    by = defaultdict(dict)
    for r in runs:
        by[(r["query_set"], r["config"])][r["query_id"]] = r
    sa = {(r["query_id"], r["config"]): r for r in score_all}
    oracle_windows = {(r["query_id"], r["config"].split("_")[0]): r["windows"][:10] for r in runs
                      if r["config"].endswith("ORACLE")}
    oracle_prog = {r["query_id"]: r["program"] for r in runs if r["config"] == "PROGRAM_ORACLE"}
    # program generation as paid by the first (uncached) LUNA config for each query: every LUNA config would pay it
    gen_first = {qid: {"ms": c["wall_ms"], "tokens": c["tokens"]} for qid, c in generation_costs(runs).items()}
    tables = {}
    held = held_out()
    slices = [(q, None) for q in EVAL_SETS] + [(q, sl) for q in ("compose_test",) for sl in ("val", "train")]
    for qset, sl in slices:
        keep = (lambda w: True) if sl is None else (lambda w, v=(sl == "val"): (w.split(":")[0] in held) == v)
        rows = []
        for cfg in CONFIGS:
            rs = {qid: {**r, "windows": [w for w in r["windows"] if keep(w)],
                        "generated": [w for w in r["generated"] if keep(w)]}
                  for qid, r in by.get((qset, cfg), {}).items()}
            if not rs:
                continue
            per = defaultdict(dict)
            for qid, r in rs.items():
                qr = qrels.get(r["intent_group_id"], {})
                judged = {w: g for w, g in qr.items() if g >= 0 and keep(w)}
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
                g = gt.get(r["intent_group_id"])
                if g and g["complete"]:
                    full = [e for e in g["episodes"] if e.get("grade", 0) == 2 and keep(f"{e['recordingId']}:0")]
                    part = [e for e in g["episodes"] if keep(f"{e['recordingId']}:0")]
                    for k_ in (10, 50):
                        per[f"ep_recall{k_}"][qid] = episode_recall(r["windows"], full, k_)
                        per[f"ep_recall{k_}_g1"][qid] = episode_recall(r["windows"], part, k_)
                if judged:
                    per["judged10"][qid] = M.judged_at(r["windows"], judged, 10)
                    per["judged50"][qid] = M.judged_at(r["windows"], judged, 50)
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
                if judged and npos == 0 and not any(g == 1 for g in judged.values()):
                    ok = {"NONE_FOUND_EXHAUSTIVE", "INSUFFICIENT_EVIDENCE"}  # judgments say nothing qualifies
                if r["query_set"].startswith("abstain"):  # every config: returning results here is a failure to abstain
                    per["abstain_ok"][qid] = float(r["status"] == "NONE_FOUND_EXHAUSTIVE" and not r["windows"])
                if r["status"] != "ANSWERED_UNVERIFIED" and r["expected_status"] != "ANSWER_STATUS_UNSPECIFIED":
                    per["status_ok"][qid] = float(r["status"] in ok)
                c = r["cost"]
                cold_model = float(c.get("encoderForwardMs", 0)) > 2000  # one-off model load: reported, not in p50/p95
                per["raw_ratio"][qid] = float(c.get("rawRatio", 0.0))
                gen = gen_first.get(qid, {}) if CONFIGS[cfg][1] == "luna" else {}
                own_gen = any(s["stage_id"] == "program_generation" for s in r["stages"]) and not r["program_cache_hit"]
                per["tokens"][qid] = float(c.get("promptTokens", 0)) + float(c.get("completionTokens", 0)) + \
                    (0 if own_gen else gen.get("tokens", 0))
                cached_ms = r["wall_ms"] - (gen.get("ms", 0) if own_gen else 0)
                if not cold_model:
                    per["wall_ms"][qid] = cached_ms + gen.get("ms", 0)  # as if the program were generated now
                    per["wall_ms_cached"][qid] = cached_ms
                if CONFIGS[cfg][1] == "luna":
                    per["program_valid"][qid] = float(r["program_state"] == "GENERATED")
                    per["program_attempts"][qid] = float(r["program_attempts"])
                    oracle = oracle_prog.get(qid)
                    got, want = clause_set(r["program"]), clause_set(oracle)
                    if want:
                        per["clause_p"][qid] = len(got & want) / len(got) if got else 0.0
                        per["clause_r"][qid] = len(got & want) / len(want)
                    ow = oracle_windows.get((qid, cfg.split("_")[0]), None)
                    if ow is not None:
                        a_, b_ = set(r["windows"][:10]), set(ow)
                        per["exec_jaccard"][qid] = len(a_ & b_) / len(a_ | b_) if a_ | b_ else 1.0
            row = {"config": cfg, "n_queries": len(rs), "n_groups": len({group_of[q] for q in rs})}
            for k, v in per.items():
                row[k] = M.summarise(v, group_of)
            for key, name in (("wall_ms", "wall"), ("wall_ms_cached", "wall_cached")):
                walls = sorted(per[key].values())
                row[f"{name}_p50"] = statistics.median(walls) if walls else None
                row[f"{name}_p95"] = walls[min(len(walls) - 1, int(0.95 * len(walls)))] if walls else None
            rows.append(row)
        tables[qset if sl is None else f"{qset}@{sl}"] = rows
    return {"tables": tables, "generation_cost_sources": sorted({r["generation_cost_source"] for r in runs
                                                                 if r.get("generation_cost_source")}),
            "ground_truth": {k: {"complete": v["complete"], "missing_chunks": v["missing_chunks"],
                                 "grade2": sum(e.get("grade", 0) == 2 for e in v["episodes"]),
                                 "grade1": sum(e.get("grade", 0) == 1 for e in v["episodes"])} for k, v in gt.items()},
            "n_judged": {k: sum(g >= 0 for g in v.values()) for k, v in qrels.items()},
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
        ("r10", "R@10", False), ("r50", "R@50", False), ("judged10", "judged@10", True),
        ("judged50", "judged@50", True), ("gen_recall", "gen. recall", True),
        ("gen_precision", "gen. precision", True), ("filter_precision", "filter precision", True),
        ("status_ok", "status ok", True), ("tokens", "tokens/query", None), ("raw_ratio", "bytes / raw", None)]


FAMILY = {"TAGS": 1, "EMBED": 1, "FUSED_CONCAT": 1, "PROGRAM_ORACLE": 2, "PROGRAM_LUNA": 2, "HYBRID_ORACLE": 2,
          "HYBRID_LUNA": 2, "FUSED_V_LUNA": 2, "FUSED": 3, "FUSED_LINEAR": 3,
          "FUSED_V2_UNIFORM_OOF": 3, "FUSED_V2_IMPORTANCE_OOF": 3}
FAMILY_NAME = {1: "baselines", 2: "program-based (LLM or oracle)", 3: "learned single-vector (Stage C)"}


def frontier_svg(rows: list[dict], title: str) -> str:
    """Quality (penalised macro ROC-AUC, CI) vs median latency (log). Colour = family; hollow = oracle ceiling."""
    import math
    pts = [(r["config"], r.get("wall_p50"), r.get("roc_pen") or {}) for r in rows]
    pts = [(c, x, m) for c, x, m in pts if x and m.get("mean") is not None]
    if not pts:
        return ""
    W, H, L, R, T, B = 900, 400, 64, 250, 24, 48
    xmin, xmax = math.log10(max(1, min(x for _, x, _ in pts) / 1.5)), math.log10(max(x for _, x, _ in pts) * 1.5)
    X = lambda v: L + (math.log10(max(v, 1)) - xmin) / (xmax - xmin) * (W - L - R)
    Y = lambda v: T + (1 - (v - 0.3) / 0.7) * (H - T - B)
    g = [f'<svg class=viz viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for yv in (0.3, 0.5, 0.7, 0.9, 1.0):
        g.append(f'<line x1={L} x2={W - R} y1={Y(yv):.1f} y2={Y(yv):.1f} class=grid /><text x={L - 8} y={Y(yv) + 4:.1f} class=tick text-anchor=end>{yv:.1f}</text>')
    g.append(f'<text x={W - R - 6} y={Y(0.5) - 5:.1f} class=tick text-anchor=end>chance</text>')
    for xv in (1, 10, 100, 1000, 10000, 100000):
        if xmin <= math.log10(xv) <= xmax:
            lab = f"{xv / 1000:g} s" if xv >= 1000 else f"{xv} ms"
            g.append(f'<line x1={X(xv):.1f} x2={X(xv):.1f} y1={T} y2={H - B} class=grid /><text x={X(xv):.1f} y={H - B + 18} class=tick text-anchor=middle>{lab}</text>')
    g.append(f'<text x={(L + W - R) / 2} y={H - 8} class=axis text-anchor=middle>median latency per query (log; includes program generation)</text>')
    g.append(f'<text transform="translate(16,{(T + H - B) / 2}) rotate(-90)" class=axis text-anchor=middle>ROC-AUC, penalised (macro)</text>')
    placed: list[tuple[float, float]] = []  # greedy label nudging: no two labels within 11 px vertically nearby

    def label_y(cx: float, cy: float) -> float:
        y = cy + 4
        for _ in range(12):
            clash = [py for px, py in placed if abs(px - cx) < 110 and abs(py - y) < 11]
            if not clash:
                break
            y = max(clash) + 11
        placed.append((cx, y))
        return y

    for c, x, m in sorted(pts, key=lambda p: (-p[2]["mean"], p[1])):
        fam = FAMILY.get(c, 1)
        cx, cy = X(x), Y(m["mean"])
        ci = m.get("ci")
        tip = f"{c}: ROC-AUC {m['mean']:.2f}" + (f" [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else "") + f", p50 {x:.0f} ms, {m.get('n_groups', 0)} intent groups"
        if ci:
            g.append(f'<line x1={cx:.1f} x2={cx:.1f} y1={Y(ci[0]):.1f} y2={Y(ci[1]):.1f} class="ci s{fam}" />')
        hollow = c.endswith("ORACLE")
        g.append(f'<circle cx={cx:.1f} cy={cy:.1f} r=6 class="pt s{fam}{" hollow" if hollow else ""}"><title>{html.escape(tip)}</title></circle>')
        ly_ = label_y(cx, cy)
        if abs(ly_ - (cy + 4)) > 1:  # leader line to a nudged label
            g.append(f'<line x1={cx + 6:.1f} y1={cy:.1f} x2={cx + 9:.1f} y2={ly_ - 4:.1f} class=grid />')
        g.append(f'<text x={cx + 10:.1f} y={ly_:.1f} class=lbl>{c}</text>')
    ly = T
    for fam, name in FAMILY_NAME.items():
        g.append(f'<circle cx={W - R + 24} cy={ly + 6} r=5 class="pt s{fam}" /><text x={W - R + 36} y={ly + 10} class=lbl>{name}</text>')
        ly += 20
    g.append(f'<circle cx={W - R + 24} cy={ly + 6} r=5 class="pt s2 hollow" /><text x={W - R + 36} y={ly + 10} class=lbl>oracle program (ceiling)</text>')
    g.append("</svg>")
    return "".join(g)


def html_page(rep: dict) -> str:
    out = ["<!doctype html><meta charset=utf-8><title>SCAND search eval</title><style>",
           ":root{--bg:#fff;--fg:#1a1a1a;--mute:#666;--line:#e5e5e5}",
           "@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mute:#999;--line:#333}}",
           "body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1400px;padding:0 16px}",
           "table{border-collapse:collapse;margin:8px 0 24px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}",
           "th,td{border-bottom:1px solid var(--line);padding:5px 9px;text-align:right;white-space:nowrap}",
           "th:first-child,td:first-child{text-align:left}.ci{color:var(--mute);font-size:11px}",
           ":root{--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--grid:#e8e7e3}",
           "@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--s1:#3987e5;--s2:#d95926;--s3:#199e70;--grid:#2c2c2a}}",
           ".viz{width:100%;max-width:900px;height:auto;margin:8px 0 20px}.viz .grid{stroke:var(--grid);stroke-width:1}",
           ".viz .tick,.viz .lbl{fill:var(--mute);font-size:11px}.viz .axis{fill:var(--fg);font-size:12px}",
           ".viz .pt{stroke:var(--bg);stroke-width:2}.viz .s1{fill:var(--s1);stroke:var(--s1)}.viz .s2{fill:var(--s2);stroke:var(--s2)}",
           ".viz .s3{fill:var(--s3);stroke:var(--s3)}.viz .pt.s1,.viz .pt.s2,.viz .pt.s3{stroke:var(--bg)}",
           ".viz .hollow{fill:var(--bg)!important;stroke-width:2.5}.viz .hollow.s2{stroke:var(--s2)}.viz line.ci{stroke-width:2;opacity:.55}",
           "</style>",
           "<h1>SCAND search — evaluation</h1>",
           "<p>Judgments are single-judge, agent-provisional (Opus labeller with audited tools). Unjudged windows are never counted "
           "as non-relevant except in the labelled <i>unj=0</i> column. AUC is macro over queries; <b>penalised</b> scores an "
           "abstaining config at chance. CIs: bootstrap over intent groups; <b>n is small</b> — read differences as directional.</p>"]
    out.append(f"<p><b>{sum(rep['n_judged'].values()):,} judged intent/window pairs.</b> Recall is against known positives in "
               "the judgment pool; judged@50 shows how much of each returned top-50 list has been graded. "
               "ORACLE rows use supplied programs; LUNA rows use model-generated programs.</p>")
    if rep.get("generation_cost_sources"):
        sources = html.escape(", ".join(rep["generation_cost_sources"]))
        out.append(f"<p><b>Cached-program replay.</b> Programs were reused from {sources} without new API calls. "
                   "Uncached latency is an estimate: measured replay execution time plus the original measured generation "
                   "time. Token counts retain the original generation usage. Cached-program latency measures this replay.</p>")
    ag = SCAND_ROOT / "results" / "eval" / "agreement.json"
    if ag.exists():
        g = json.loads(ag.read_text())
        out.append(f"<p><b>Label consistency.</b> A fresh labeller instance, blind to earlier grades, re-graded {g['n']} stratified "
                   f"windows: exact agreement {100 * g['exact']:.0f}%, within one grade {100 * g['within_one']:.0f}%, relevant-or-not "
                   f"{100 * g['binary_pos']:.0f}%, Cohen's κ {g['kappa_graded']:.2f}. Same model family and tools: this measures "
                   f"consistency, not independence.</p>")
    for qset, rows in rep["tables"].items():
        if not rows:
            continue
        out.append(f"<h2>{html.escape(qset)} <span class=ci>({rows[0]['n_queries']} queries, {rows[0]['n_groups']} intent groups)</span></h2>")
        out.append(frontier_svg(rows, f"{qset}: quality vs latency"))
        out.append("<table><tr><th>config</th>" + "".join(f"<th>{html.escape(c[1])}</th>" for c in COLS) + "<th>p50 ms</th><th>p95 ms</th><th>p50 ms (cached program)</th></tr>")
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
            p50c = f"{r['wall_cached_p50']:.0f}" if r.get("wall_cached_p50") is not None else "–"
            out.append(f"<tr><td>{r['config']}</td>" + "".join(f"<td>{c}</td>" for c in cells) + f"<td>{p50}</td><td>{p95}</td><td>{p50c}</td></tr>")
        out.append("</table>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a_ = ap.parse_args()
    run_dir = SCAND_ROOT / "results" / "eval" / a_.run
    rep = build(run_dir)
    (run_dir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (run_dir / "report.html").write_text(html_page(rep))
    print(run_dir / "report.html")


if __name__ == "__main__":
    main()
