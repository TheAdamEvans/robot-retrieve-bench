"""Export real media and publication-ready plots for the offline presentation (eval v9). Run from the repository root."""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from google.protobuf import json_format

from alloy_server.bundle import Bundle
from alloy_server.catalog import embodiment as E
from alloy_server.gen.alloy.v1 import answer_pb2 as a, query_pb2 as q
from alloy_server.io.clips import render
from alloy_server.io.ros1 import compressed_image
from alloy_server.pipeline.runner import run

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "assets"
RUN = ROOT / "benchmark/results/v9"
RECHECK = ROOT / "benchmark/results/v8-correctness"  # the blind 80-window label re-check
COLORS = {"TAGS": "#777a78", "EMBED": "#3475a5", "FUSED": "#087f72", "PROGRAM_LUNA": "#c45b36",
          "HYBRID_LUNA": "#ad8640", "FUSED_V_LUNA": "#725a9c", "PROGRAM_ORACLE": "#292e2c"}
NAMES = {"TAGS": "Tags", "EMBED": "Embeddings", "FUSED": "Fused", "PROGRAM_LUNA": "Program",
         "HYBRID_LUNA": "Embed + verify", "FUSED_V_LUNA": "Fused + verify", "PROGRAM_ORACLE": "Oracle program"}


def opus_judgments() -> int:
    """Window-level relevance judgments written by the Opus judges (not the derived zeros of complete ground truth)."""
    from alloy_index.annotate.store import load_labels
    return len(load_labels(ROOT / "annotations", "judgment"))


def save_image(rec, topic, t, name, rotation=0):
    tl = rec.topics[topic]
    _, payload = compressed_image(rec.read(topic, tl.nearest(rec.t_abs(t))).data)
    im = Image.open(io.BytesIO(payload)).convert("RGB")
    if rotation:
        im = im.rotate(rotation, expand=True)
    im.thumbnail((1120, 800))
    im.save(OUT / name, quality=85, optimize=True)


SLICE_CONFIGS = ["EMBED", "FUSED", "FUSED_V_LUNA", "PROGRAM_LUNA", "PROGRAM_ORACLE"]
ORIGINAL = {"Brackenridge", "Butler", "GDC", "JCL", "Library_MLK", "RLM", "Sanjac"}


def generalization() -> dict:
    """compose_test nDCG@10 and ROC-AUC on the original 7 recordings, the 9 new ones, and the 2 held-out Val ones."""
    import alloy_train.eval.report as R
    from alloy_index.recordings import held_out
    new = {p.stem for p in (ROOT / "bundles/dev/mcap").glob("*.json")} - ORIGINAL
    out = {}
    for name, rec_set in (("original 7", ORIGINAL), ("new 9", new), ("held-out 2", set(held_out()))):
        R.held_out = lambda s=rec_set: s  # the report's "@val" slice becomes this recording set
        tab = R.build(RUN, ROOT / "annotations")["tables"]["compose_test@val"]
        out[name] = {r["config"]: {"ndcg10": r["ndcg10"]["mean"], "roc_pen": r["roc_pen"]["mean"]}
                     for r in tab if r["config"] in SLICE_CONFIGS}
    fig, ax = plt.subplots(figsize=(9.6, 4.2), facecolor="#f8f8f2")
    ax.set_facecolor("#f8f8f2")
    w = .16
    for k, cfg in enumerate(SLICE_CONFIGS):
        vals = [out[s][cfg]["ndcg10"] for s in out]
        xs = np.arange(len(out)) + (k - 2) * w
        ax.bar(xs, vals, width=w * .92, color=COLORS[cfg], alpha=.9 if cfg != "PROGRAM_ORACLE" else .35, label=NAMES[cfg])
        for x, v in zip(xs, vals):
            ax.text(x, v + .015, f"{v:.2f}", ha="center", fontsize=8, color="#4b5752")
    ax.set_xticks(np.arange(len(out)), ["Original 7 recordings", "9 new recordings", "2 held-out (SCAND Val)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("nDCG@10 on compositions")
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.12))
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "generalization.svg", bbox_inches="tight")
    plt.close(fig)
    return out


def dream(rows: list[dict]) -> dict:
    """Exhaustive ground truth for the six dream questions, and how many ground-truth episodes each config finds."""
    from alloy_train.eval.report import episode_recall, ground_truth
    gt = ground_truth(ROOT / "annotations")
    titles = {json.loads(l)["intentGroupId"]: json.loads(l)["title"]
              for l in (ROOT / "benchmark/challenges/l1_compositions_v1/queries_dev.jsonl").read_text().splitlines() if l}
    out = {"questions": [], "configs": [c for c in COLORS if c not in ("PROGRAM_ORACLE",)]}
    for intent, title in titles.items():
        g = gt[intent]
        full = [e for e in g["episodes"] if e.get("grade", 0) == 2]
        found = {}
        for cfg in out["configs"]:
            r = next((x for x in rows if x["intent_group_id"] == intent and x["config"] == cfg), None)
            if r is None:
                continue
            n = lambda es, k: round(episode_recall(r["windows"], es, k) * len(es)) if es else 0
            found[cfg] = {"full10": n(full, 10), "full50": n(full, 50), "rel50": n(g["episodes"], 50), "status": r["status"]}
        out["questions"].append({"intent": intent, "title": title, "complete": g["complete"], "full": len(full),
                                 "partial": len(g["episodes"]) - len(full), "proof": g.get("proof", "judged"),
                                 "found": found})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads((RUN / "report.json").read_text())
    rows = [json.loads(x) for x in (RUN / "runs.jsonl").read_text().splitlines()]
    judgments = json.loads((RUN / "judgments.json").read_text())
    bundle = Bundle(ROOT / "bundles/dev")
    topic = "/image_raw/compressed"
    rec = bundle.recordings["Library_MLK"]
    for name, start, end in [("doorway-embed", 114, 118), ("doorway-fused", 25, 29)]:
        save_image(rec, topic, (start + end) / 2, name + ".jpg")
        if not (OUT / (name + ".mp4")).exists():
            render(rec, topic, rec.t_abs(start), rec.t_abs(end), str(OUT / (name + ".mp4")), width=640, label=rec.id)
    save_image(rec, topic, 30, "hero.jpg")
    save_image(bundle.recordings["Butler"], topic, 50, "butler.jpg")
    jackal_topic = E.sensor(E.profile(bundle.robots["Sanjac"]), "front_camera").topics[0]
    save_image(bundle.recordings["Sanjac"], jackal_topic, 70, "sanjac.jpg")

    # The evaluated v9 program, replayed on Library_MLK (the clip shown throughout) rather than the whole corpus.
    original = next(r for r in rows if r["query_id"] == "last_safe_evidence:canonical" and r["config"] == "PROGRAM_LUNA")
    program = json_format.ParseDict(original["program"], q.QueryProgram())
    req = a.SearchRequest(utterance="evidence before acceleration", program=program, k=1)
    req.scope.recording_ids.append(rec.id)
    response = run(bundle, bundle.specs["PROGRAM"], req)
    item = response.results[0]
    anchor = item.receipt.causal_cutoff_ns
    event_s = rec.t_rel(anchor)
    save_image(rec, topic, event_s, "receipt.jpg")
    if not (OUT / "receipt.mp4").exists():
        render(rec, topic, anchor - 2_000_000_000, anchor + 2_000_000_000, str(OUT / "receipt.mp4"), anchor,
               width=640, label=rec.id)
    receipt = json_format.MessageToDict(item.receipt)
    receipt["recording"] = rec.id
    receipt["event_s"] = event_s
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    # Actual measured speed around the displayed event, not an illustrative curve.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#d4d7d0",
                         "text.color": "#252e2b", "axes.labelcolor": "#4b5752", "xtick.color": "#66716a",
                         "ytick.color": "#66716a", "savefig.facecolor": "#f8f8f2"})
    s = bundle.features.series("speed_mps", rec.id)
    t = (s.t_ns - rec.start_ns) / 1e9
    mask = (t >= 22) & (t <= 33)
    fig, ax = plt.subplots(figsize=(7.8, 2.6), facecolor="#f8f8f2")
    ax.set_facecolor("#f8f8f2")
    ax.fill_between(t[mask], s.value[mask], color="#087f72", alpha=.12)
    ax.plot(t[mask], s.value[mask], color="#087f72", linewidth=2)
    ax.axvline(event_s, color="#c45b36", linestyle="--", linewidth=1.5)
    ax.text(event_s + .15, ax.get_ylim()[1] * .92, f"cutoff {event_s:.3f} s", color="#c45b36", fontsize=9)
    ax.set(xlabel="Recording time (s)", ylabel="Speed (m/s)", xlim=(22, 33))
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(OUT / "motion.svg", bbox_inches="tight")
    plt.close(fig)

    # Static SVG exports remain useful outside the HTML; controls select among these plots.
    for split, table in report["tables"].items():
        if "@" in split:  # recording slices are summarised by generalization() instead
            continue
        selected = [r for cfg in COLORS for r in table if r["config"] == cfg]
        for metric, title in [("ndcg10", "nDCG@10 · higher is better"), ("roc_pen", "Penalised ROC-AUC · higher is better")]:
            for mode, key in [("fresh", "wall_p50"), ("cached", "wall_cached_p50")]:
                fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor="#f8f8f2")
                ax.set_facecolor("#f8f8f2")
                xs = np.arange(len(selected))
                for i, r in enumerate(selected):
                    cfg, value = r["config"], r[metric]
                    m = value["mean"]
                    ci = value.get("ci")
                    ax.barh(i, m, height=.52, color=COLORS[cfg], alpha=.9 if cfg != "PROGRAM_ORACLE" else .2)
                    if ci:
                        ax.errorbar(m, i, xerr=[[m-ci[0]], [ci[1]-m]], fmt="none", color="#28342e", capsize=3, linewidth=1)
                    latency = r[key]
                    text = "<1 ms" if latency < 1 else f"{latency:.0f} ms" if latency < 1000 else f"{latency/1000:.2f} s"
                    ax.text(1.035, i, f"{m:.2f}  /  {text}", va="center", fontsize=10, transform=ax.get_yaxis_transform())
                ax.set_yticks(xs, [NAMES[r["config"]] for r in selected])
                ax.invert_yaxis()
                ax.set_xlim(0, 1)
                ax.set_xlabel(title)
                ax.grid(axis="x", alpha=.16)
                ax.set_axisbelow(True)
                for spine in ["left", "bottom"]:
                    ax.spines[spine].set_visible(False)
                ax.tick_params(axis="y", length=0, pad=10)
                ax.text(1.035, 1.045, "quality / median latency", transform=ax.transAxes, fontsize=9, color="#66716a")
                fig.subplots_adjust(left=.2, right=.78, top=.88, bottom=.15)
                fig.savefig(OUT / f"quality-{split}-{metric}-{mode}.svg")
                plt.close(fig)

    data = {"report": report, "receipt": receipt, "names": NAMES, "colors": COLORS,
            "agreement": json.loads((RECHECK / "agreement.json").read_text()),
            "generalization": generalization(), "dream": dream(rows),
            "examples": [{"query_id": r["query_id"], "config": r["config"], "status": r["status"],
                          "windows": [{"id": w, "grade": judgments.get(r["intent_group_id"], {}).get(w)} for w in r["windows"][:3]],
                          "program": r["program"], "unsupported": r["unsupported_top"]}
                         for r in rows if r["query_id"] in ["doorway_crossing:para3", "test_turn_then_brake:canonical",
                                                             "test_battery_abstain:canonical", "test_body_cam_person:canonical"]
                         and r["config"] in COLORS],
            "corpus": {"recordings": len(bundle.recordings), "raw_bytes": sum(bundle.raw_bytes.values()),
                       "judgments": opus_judgments(),
                       "seconds": sum((r.end_ns-r.start_ns)/1e9 for r in bundle.recordings.values()),
                       "session_bytes": bundle.session["bytes_read"]}}
    (OUT / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"Exported media, plots and benchmark data to {OUT}")


if __name__ == "__main__":
    main()
