"""Benchmark query sets and oracle QueryPrograms (authored by Claude as the oracle, ProgramState=PROVIDED).

Frozen: `write()` emits benchmark/queries/<set>.json plus a sha256 manifest. Paraphrases were written from the
intent text only. compose_dev is the ONLY source of few-shot examples; compose_test uses feature x operator
pairings that do not appear in compose_dev.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from google.protobuf import json_format

from alloy_server.gen.alloy.v1 import eval_pb2

OUT = Path(__file__).resolve().parents[4] / "benchmark" / "queries"


# ---- tiny builders (proto-JSON field names) ----
def qty(v, u):
    return {"value": v, "unit": u}


def ev(name, kind, feature, doc, required=True, **kw):
    d = {"name": name, "kind": kind, "feature": feature, "doc": doc, "required": required}
    d.update(kw)
    return d


def rel(parent, ppoint, child, cpoint, kind, max_gap=None, min_gap=None, doc="", required=True):
    d = {"parent": {"event": parent, "point": ppoint}, "child": {"event": child, "point": cpoint}, "kind": kind,
         "required": required, "doc": doc}
    if max_gap is not None:
        d["maxGap"] = qty(max_gap, "S")
    if min_gap is not None:
        d["minGap"] = qty(min_gap, "S")
    return d


def anchor(e, p="START"):
    return {"event": e, "point": p}


def program(primary, events, relations=(), recs=(), quant="ALL", by=None, anchors=(), receipt=None,
            before=4, after=4, unexpressible=(), abstain=False):
    d = {"scope": {"recordingIds": list(recs)}, "primaryEvent": primary, "events": list(events),
         "relations": list(relations), "selection": {"quantifier": quant}, "returnAnchors": list(anchors),
         "contextBefore": qty(before, "S"), "contextAfter": qty(after, "S"),
         "unexpressible": list(unexpressible), "abstainIfInsufficient": abstain}
    if by:
        d["selection"].update({"byFeature": by[0], "byEvent": by[1]})
    if receipt:
        d["receipt"] = receipt
    return d


STOP_TO_GO = ev("go", "ONSET", "speed_mps", "the robot accelerates from a stop",
                fromBelow=qty(0.05, "MPS"), minDuration=qty(1, "S"), threshold=qty(0.1, "MPS"), sustain=qty(0.5, "S"))

# ---------------- intents ----------------
INTENTS = {
    "crowd_hesitation": {
        "set": "demo5", "scope": ["Butler"], "expected": "ANSWERED_PARTIAL",
        "intent": "In the Butler run: the first time at least six people occupy the robot's forward travel corridor "
                  "and the robot loses at least 0.7 m/s within 3 s; anchors: just before deceleration, minimum "
                  "speed, recovery to 90% of the pre-event speed.",
        "canonical": "In the Butler-to-LBJ run, find the first interval in which at least six people occupy the "
                     "forward travel corridor and Spot loses at least 0.7 m/s within three seconds. Return the frame "
                     "just before deceleration, the minimum-speed time, and the first recovery to 90% of pre-event speed.",
        "para": ["During the Butler walk, when does a crowd of six or more directly ahead first make Spot brake by "
                 "0.7 m/s or more inside three seconds, and when is it back to 90% of its earlier speed?",
                 "Butler recording: first moment Spot is forced to slow hard (>=0.7 m/s within 3 s) by a dense group "
                 "of people in its path. Give the pre-braking frame, the slowest point and the recovery.",
                 "Show me where a big crowd in front of the robot on the Butler route makes it hesitate — at least "
                 "six people in its lane and a 0.7 m/s speed drop within 3 seconds — and when it picks back up."],
        "program": program(
            "crowd",
            [ev("crowd", "THRESHOLD", "persons_in_corridor", "at least six people in the forward corridor",
                comparator="GTE", threshold=qty(6, "DIMENSIONLESS")),
             ev("slow", "CHANGE", "speed_mps", "loses at least 0.7 m/s within 3 s", change=qty(0.7, "MPS"),
                direction="DOWN", within=qty(3, "S")),
             ev("recover", "RETURN_TO", "speed_mps", "recovers to 90% of pre-event speed", fraction=0.9,
                reference=anchor("slow"))],
            [rel("crowd", "START", "slow", "START", "WITHIN", max_gap=3, doc="the slowdown coincides with the crowd"),
             rel("slow", "EXTREMUM_POINT", "recover", "START", "AFTER", max_gap=20)],
            recs=["Butler"], quant="FIRST",
            anchors=[anchor("slow"), anchor("slow", "EXTREMUM_POINT"), anchor("recover")], before=4, after=20),
    },
    "doorway_crossing": {
        "set": "demo5", "scope": ["Library_MLK"], "expected": "ANSWERED",
        "intent": "In the Library_MLK run: the narrowest doorway the robot actually passes through (not beside); "
                  "approach, threshold crossing and full-body-clear times, minimum left/right clearance.",
        "canonical": "In the Library-to-MLK run, find the narrowest doorway actually traversed. Return approach, "
                     "threshold crossing, and full-body-clear times, plus minimum left and right clearance in the "
                     "doorway frame.",
        "para": ["Which doorway in the Library_MLK recording is the tightest one Spot actually walks through, and "
                 "how much room did it have on each side?",
                 "Library to MLK: find the narrowest door Spot goes through (not one it passes next to). When does it "
                 "approach, cross and fully clear it?",
                 "Tightest doorway traversal on the library route — give the crossing times and the left and right gaps."],
        "program": program(
            "door", [ev("door", "THRESHOLD", "doorway_active", "passes through a doorway-like gap",
                        comparator="EQ", threshold=qty(1, "DIMENSIONLESS"), minDuration=qty(0.3, "S"))],
            recs=["Library_MLK"], quant="ARGMIN", by=("gap_width_m", "door"),
            anchors=[anchor("door"), anchor("door", "EXTREMUM_POINT"), anchor("door", "END")], before=4, after=4),
    },
    "vehicle_interaction_gdc": {
        "set": "demo5", "scope": ["GDC"], "expected": "INSUFFICIENT_EVIDENCE",
        "intent": "In the GDC run: the strongest localized vehicle interaction (a vehicle near the robot with a "
                  "motion response by the robot), or insufficient evidence. The recording-level tag is not evidence.",
        "canonical": "In the GDC-to-AHG run, either localize the strongest vehicle interaction with supporting frames "
                     "and motion response or return insufficient evidence. A recording-level Vehicle Interaction tag "
                     "is not sufficient evidence.",
        "para": ["GDC recording: is there a specific moment where Spot reacts to a car or truck near it? Show the "
                 "strongest one, or say there isn't enough evidence.",
                 "Where in the GDC walk does a vehicle come close enough that Spot slows for it? Don't just trust the "
                 "recording's tags.",
                 "Find the clearest vehicle encounter in the GDC to AHG route where the robot changes its motion "
                 "because of the vehicle; abstain if none can be localized."],
        "program": program(
            "veh",
            [ev("veh", "THRESHOLD", "vehicles_visible_front", "a vehicle is visible ahead", comparator="GTE",
                threshold=qty(1, "DIMENSIONLESS"), minDuration=qty(0.5, "S")),
             ev("resp", "CHANGE", "speed_mps", "the robot slows in response", change=qty(20, "PERCENT"),
                direction="DOWN", within=qty(3, "S"))],
            [rel("veh", "START", "resp", "START", "WITHIN", max_gap=4, doc="the slowdown coincides with the vehicle")],
            recs=["GDC"], quant="ARGMAX", by=("vehicle_box_frac", "veh"), anchors=[anchor("resp")],
            before=4, after=6, abstain=True),
    },
    "last_safe_evidence": {
        "set": "demo5", "scope": [], "expected": "ANSWERED",
        "intent": "Immediately before the robot accelerates from a stop: the latest available front image, side/rear "
                  "images, lidar scan, pose and transform that could have informed the action, with each age.",
        "canonical": "Immediately before the robot accelerates from a stop, return the latest available front image, "
                     "side/rear images, lidar scan, pose, and transform that could have informed the action. Report "
                     "each message's age.",
        "para": ["Right before the robot starts moving again after standing still, what was the freshest sensor data "
                 "it could actually have used, and how old was each piece?",
                 "For each stop-then-go, list the last camera frames, lidar, odometry and transform available before "
                 "the acceleration, with their ages — nothing from after the start.",
                 "What did the robot know when it set off from a standstill? Give the latest pre-departure messages "
                 "per sensor and their staleness."],
        "program": program(
            "go", [STOP_TO_GO], quant="ALL", anchors=[anchor("go")], before=4, after=4,
            receipt={"cutoff": anchor("go"), "sensors": ["front_camera", "body_cameras", "lidar", "odom", "tf"],
                     "maxAge": qty(500, "MS"), "boundary": "STRICT_BEFORE"}),
    },
    "chained_turn_person": {
        "set": "demo5", "scope": [], "expected": "ANSWERED_PARTIAL",
        "intent": "Earliest event across recordings: a person becomes visible only after a turn; the robot then slows "
                  ">=30%; lidar clearance reaches a local minimum within 2 s of the speed minimum; the robot returns to "
                  "90% of pre-event speed before the person disappears from every camera view.",
        "canonical": "Find the earliest event, across all complete recordings, in which a person or group becomes "
                     "relevant only after a turn; the robot then reduces speed by at least 30%; lidar clearance reaches "
                     "a local minimum within two seconds of the speed minimum; and the robot returns to 90% of its "
                     "pre-event speed before the person disappears from every camera view.",
        "para": ["Across all runs, what's the first time the robot rounds a turn, a person appears, it slows by 30% or "
                 "more with the lidar gap bottoming out near the slowest point, and it's back up to speed while the "
                 "person is still in view?",
                 "Earliest case of: turn, then a newly visible pedestrian, then a >=30% slowdown with closest lidar "
                 "clearance within 2 s of minimum speed, then recovery to 90% before the pedestrian leaves all cameras.",
                 "Find the first turn-reveal-brake-recover sequence in the whole corpus: person shows up after a turn, "
                 "robot brakes at least 30%, nearest obstacle distance dips around the braking, robot recovers before "
                 "losing sight of them."],
        "program": program(
            "turn",
            [ev("turn", "CHANGE", "heading_deg", "the robot turns", change=qty(45, "DEG"), within=qty(3, "S")),
             ev("person", "TRACK_APPEAR", "person_tracks_front", "a person becomes visible after the turn"),
             ev("slow", "CHANGE", "speed_mps", "the robot slows by at least 30%", change=qty(30, "PERCENT"),
                direction="DOWN", within=qty(4, "S")),
             ev("clear", "EXTREMUM", "min_clearance_front_m", "lidar clearance reaches a local minimum",
                direction="DOWN"),
             ev("recover", "RETURN_TO", "speed_mps", "back to 90% of pre-event speed", fraction=0.9,
                reference=anchor("slow")),
             ev("gone", "TRACK_DISAPPEAR", "person_tracks_all_cameras",
                "the person disappears from every camera view", reference=anchor("person"))],
            [rel("turn", "START", "person", "START", "AFTER", min_gap=0, max_gap=5),
             rel("person", "START", "slow", "START", "AFTER", max_gap=5),
             rel("slow", "EXTREMUM_POINT", "clear", "EXTREMUM_POINT", "WITHIN", max_gap=2),
             rel("slow", "EXTREMUM_POINT", "recover", "START", "AFTER", max_gap=15),
             rel("recover", "START", "gone", "START", "AFTER", max_gap=30, doc="recovery happens before the person is gone")],
            quant="FIRST",
            anchors=[anchor("person"), anchor("slow"), anchor("slow", "EXTREMUM_POINT"),
                     anchor("clear", "EXTREMUM_POINT"), anchor("recover")], before=6, after=20),
    },
    # ---------------- compose_dev: the few-shot pool ----------------
    "dev_long_stop": {
        "set": "compose_dev", "scope": [], "expected": "ANSWERED",
        "intent": "Moments where the robot stands still for more than 2 seconds.",
        "canonical": "Where does the robot stop and stand still for more than two seconds?",
        "program": program("stop", [ev("stop", "THRESHOLD", "speed_mps", "standing still for more than 2 s",
                                       comparator="LTE", threshold=qty(0.05, "MPS"), minDuration=qty(2, "S"))],
                           anchors=[anchor("stop"), anchor("stop", "END")]),
    },
    "dev_bicycle_ahead": {
        "set": "compose_dev", "scope": [], "expected": "ANSWERED",
        "intent": "A bicycle is visible ahead while the robot is moving faster than 1 m/s.",
        "canonical": "Show me a bike in front of the robot while it's moving at over 1 m/s.",
        "program": program("bike",
                           [ev("bike", "THRESHOLD", "bicycles_visible_front", "a bicycle is visible ahead",
                               comparator="GTE", threshold=qty(1, "DIMENSIONLESS")),
                            ev("fast", "THRESHOLD", "speed_mps", "moving faster than 1 m/s", comparator="GT",
                               threshold=qty(1, "MPS"))],
                           [rel("bike", "START", "fast", "START", "WITHIN", max_gap=1)], anchors=[anchor("bike")]),
    },
    "dev_left_turn_narrow_indoor": {
        "set": "compose_dev", "scope": [], "expected": "ANSWERED_PARTIAL",
        "intent": "A sharp left turn (more than 60 degrees within 4 s) in a narrow passage (under 2 m wide), indoors.",
        "canonical": "Find a sharp left turn, more than 60 degrees in 4 seconds, taken in a narrow indoor corridor "
                     "under 2 m wide.",
        "program": program("turn",
                           [ev("turn", "CHANGE", "heading_deg", "sharp left turn: >60 deg within 4 s",
                               change=qty(60, "DEG"), direction="UP", within=qty(4, "S")),
                            ev("narrow", "THRESHOLD", "gap_width_m", "passage narrower than 2 m", comparator="LT",
                               threshold=qty(200, "CM"))],
                           [rel("turn", "START", "narrow", "START", "WITHIN", max_gap=2)],
                           anchors=[anchor("turn"), anchor("turn", "END")],
                           unexpressible=["indoors (no indoor/outdoor feature in the registry)"]),
    },
    "dev_evidence_before_stop": {
        "set": "compose_dev", "scope": [], "expected": "ANSWERED",
        "intent": "The lidar scan and odometry available just before the robot comes to a stop, with ages.",
        "canonical": "What lidar and odometry did the robot have just before it came to a halt? Include their ages.",
        "program": program("halt", [ev("halt", "THRESHOLD", "speed_mps", "the robot comes to a stop",
                                       comparator="LTE", threshold=qty(0.05, "MPS"), minDuration=qty(1, "S"))],
                           anchors=[anchor("halt")],
                           receipt={"cutoff": anchor("halt"), "sensors": ["lidar", "odom"], "maxAge": qty(0.3, "S"),
                                    "boundary": "STRICT_BEFORE"}),
    },
    # ---------------- compose_test: held-out pairings ----------------
    "test_speedup_after_person": {
        "set": "compose_test", "scope": [], "expected": "ANSWERED",
        "intent": "The robot speeds up (by at least 0.3 m/s within 2 s) shortly after a new person appears in front "
                  "of it.",
        "canonical": "When does the robot accelerate by 0.3 m/s or more within a couple of seconds of a new person "
                     "showing up in front of it?",
        "program": program("person",
                           [ev("person", "TRACK_APPEAR", "person_tracks_front", "a new person appears ahead"),
                            ev("faster", "CHANGE", "speed_mps", "speeds up by >=0.3 m/s within 2 s",
                               change=qty(0.3, "MPS"), direction="UP", within=qty(2, "S"))],
                           [rel("person", "START", "faster", "START", "AFTER", max_gap=3)],
                           anchors=[anchor("faster")]),
    },
    "test_tight_right": {
        "set": "compose_test", "scope": [], "expected": "ANSWERED",
        "intent": "The robot squeezes past something with less than 40 cm of room on its right side.",
        "canonical": "Where does the robot squeeze past an obstacle with less than 40 cm to spare on its right?",
        "program": program("tight", [ev("tight", "THRESHOLD", "lateral_clearance_right_m",
                                        "less than 40 cm of room on the right", comparator="LT",
                                        threshold=qty(40, "CM"))],
                           quant="ARGMIN", by=("lateral_clearance_right_m", "tight"), anchors=[anchor("tight")]),
    },
    "test_turn_then_brake": {
        "set": "compose_test", "scope": [], "expected": "ANSWERED",
        "intent": "Ordered: the robot turns (>=45 degrees within 3 s) and then brakes (>=30% speed drop) within 5 s "
                  "after the turn starts.",
        "canonical": "Find where the robot turns and then brakes: a turn of at least 45 degrees followed within five "
                     "seconds by a drop of 30% or more in speed.",
        "program": program("turn",
                           [ev("turn", "CHANGE", "heading_deg", "turn of >=45 deg within 3 s", change=qty(45, "DEG"),
                               within=qty(3, "S")),
                            ev("brake", "CHANGE", "speed_mps", "speed drops by >=30%", change=qty(0.3, "RATIO"),
                               direction="DOWN", within=qty(3, "S"))],
                           [rel("turn", "START", "brake", "START", "AFTER", min_gap=0, max_gap=5)],
                           anchors=[anchor("turn"), anchor("brake")]),
    },
    "test_car_close_fast": {
        "set": "compose_test", "scope": [], "expected": "ANSWERED",
        "intent": "A vehicle close in front (large in the image) while the robot moves faster than 1.5 m/s.",
        "canonical": "Show a car close in front of the robot while it's going faster than 1.5 metres per second.",
        "program": program("car",
                           [ev("car", "THRESHOLD", "vehicle_box_frac", "a vehicle close ahead (>=5% of the image)",
                               comparator="GTE", threshold=qty(0.05, "RATIO")),
                            ev("fast", "THRESHOLD", "speed_mps", "faster than 1.5 m/s", comparator="GT",
                               threshold=qty(1.5, "MPS"))],
                           [rel("car", "START", "fast", "START", "WITHIN", max_gap=1)], anchors=[anchor("car")]),
    },
    "test_battery_abstain": {
        "set": "compose_test", "scope": [], "expected": "INSUFFICIENT_EVIDENCE", "abstain": True,
        "intent": "When the robot's battery level drops below 20% (not recorded: the system should abstain).",
        "canonical": "When did the robot's battery drop below 20 percent?",
        "program": program("moving", [ev("moving", "THRESHOLD", "speed_mps", "placeholder: any time the robot moves",
                                         comparator="GT", threshold=qty(0.05, "MPS"))],
                           unexpressible=["battery level below 20% (battery state is not recorded)"], abstain=True),
    },
    "test_body_cam_person": {
        "set": "compose_test", "scope": [], "expected": "ANSWERED_PARTIAL",
        "intent": "Spot's body cameras see a person while the front camera sees nobody.",
        "canonical": "Find moments where Spot's side or rear cameras see a person but its front camera sees nobody.",
        "program": program("empty_front",
                           [ev("empty_front", "THRESHOLD", "persons_visible_front", "the front camera sees nobody",
                               comparator="EQ", threshold=qty(0, "DIMENSIONLESS"), minDuration=qty(1, "S")),
                            ev("body", "THRESHOLD", "persons_visible_body_cameras", "a body camera sees a person",
                               comparator="GTE", threshold=qty(1, "DIMENSIONLESS"))],
                           [rel("empty_front", "START", "body", "START", "DURING")],
                           recs=["Butler", "JCL", "Library_MLK", "GDC", "RLM"], anchors=[anchor("empty_front")]),
    },
}


def queries() -> list[eval_pb2.EvalQuery]:
    out = []
    for gid, it in INTENTS.items():
        utts = [("canonical", it["set"], it["canonical"])]
        if it["set"] == "demo5":
            utts += [(f"para{k + 1}", "demo5_para", u) for k, u in enumerate(it["para"])]
        for tag, qset, utt in utts:
            q = eval_pb2.EvalQuery(query_id=f"{gid}:{tag}", intent_group_id=gid, query_set=qset, utterance=utt,
                                   intent=it["intent"], expect_abstain=bool(it.get("abstain", False)))
            q.scope.recording_ids.extend(it["scope"])
            json_format.ParseDict(it["program"], q.oracle_program)
            q.expected_status = eval_pb2.EvalQuery.DESCRIPTOR.fields_by_name["expected_status"].enum_type \
                .values_by_name[it["expected"]].number
            out.append(q)
    return out


def write() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    by_set: dict[str, list] = {}
    for q in queries():
        by_set.setdefault(q.query_set, []).append(json_format.MessageToDict(q))
    manifest = {}
    for name, rows in sorted(by_set.items()):
        p = OUT / f"{name}.json"
        blob = json.dumps(rows, indent=1, sort_keys=True)
        p.write_text(blob)
        manifest[name] = {"queries": len(rows), "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def normalise_utt(u: str) -> str:
    import re
    return re.sub(r"\s+", " ", u.strip().lower())


def load(sets: list[str] | None = None) -> list[eval_pb2.EvalQuery]:
    out = []
    for p in sorted(OUT.glob("*.json")):
        if p.name == "MANIFEST.json" or (sets and p.stem not in sets):
            continue
        out += [json_format.ParseDict(d, eval_pb2.EvalQuery()) for d in json.loads(p.read_text())]
    return out


if __name__ == "__main__":
    print(json.dumps(write(), indent=1))
