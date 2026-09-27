"""Program-grounded positives and explicit near misses for FUSED v2.

Every verdict is computed on Train recordings. A false verdict is used only when
the executor confirms coverage; unknowns never become negatives. The fixed
grammar excludes the feature pairings reserved by compose_test.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass

from google.protobuf import json_format

from alloy_server.catalog.windows import window_span_s
from alloy_server.gen.alloy.v1 import common_pb2 as c
from alloy_server.gen.alloy.v1 import query_pb2 as q
from alloy_trainer.learn.fused import PSEUDO


@dataclass(frozen=True)
class ProgramText:
    family: str
    text: str
    program: dict


@dataclass
class Supervision:
    spec: ProgramText
    positive: set[str]
    hard: set[str]
    easy: set[str]


def programs() -> list[ProgramText]:
    out = [ProgramText(f"base_{i}", text, copy.deepcopy(program))
           for i, (texts, program) in enumerate(PSEUDO) for text in texts]

    # Numeric variants use literal units in their text. They do not introduce
    # turn->brake, person->speed-up, right-side-room, or close-car-while-fast.
    variants = [
        (0, "slow", "change", [(10, "at least 10 percent"), (15, "at least 15 percent"),
                                (25, "at least 25 percent"), (35, "at least 35 percent"),
                                (40, "at least 40 percent"), (45, "at least 45 percent"),
                                (55, "at least 55 percent"), (60, "at least 60 percent"),
                                (80, "at least 80 percent")], "the robot slows by {} within three seconds"),
        (2, "turn_left", "change", [(10, "10 degrees"), (15, "15 degrees"), (25, "25 degrees"),
                                    (35, "35 degrees"), (40, "40 degrees"), (55, "55 degrees"),
                                    (65, "65 degrees"), (70, "70 degrees"), (90, "90 degrees")],
         "the robot turns left by {}"),
        (3, "turn_right", "change", [(10, "10 degrees"), (15, "15 degrees"), (25, "25 degrees"),
                                     (35, "35 degrees"), (40, "40 degrees"), (55, "55 degrees"),
                                     (65, "65 degrees"), (70, "70 degrees"), (90, "90 degrees")],
         "the robot turns right by {}"),
        (7, "crowd", "threshold", [(2, "two"), (3, "three"), (4, "four"), (5, "five"), (6, "six")],
         "at least {} people are visible in front of the robot"),
        (8, "clearance", "threshold", [(0.4, "40 centimetres"), (0.5, "half a metre"),
                                       (0.75, "75 centimetres"), (0.9, "90 centimetres"),
                                       (1.0, "one metre"), (1.25, "1.25 metres"),
                                       (1.5, "1.5 metres"), (1.75, "1.75 metres")],
         "an obstacle is closer than {} in front of the robot"),
        (12, "speed", "threshold", [(0.4, "0.4 metres per second"), (0.6, "0.6 metres per second"),
                                   (0.9, "0.9 metres per second"), (1.0, "1.0 metres per second"),
                                   (1.2, "1.2 metres per second"), (1.4, "1.4 metres per second"),
                                   (1.5, "1.5 metres per second"), (1.8, "1.8 metres per second"),
                                   (2.0, "2.0 metres per second")], "the robot moves faster than {}"),
        (13, "gap", "threshold", [(1.0, "one metre"), (1.25, "1.25 metres"),
                                 (1.5, "1.5 metres"), (1.6, "1.6 metres"),
                                 (1.75, "1.75 metres"), (2.0, "two metres"),
                                 (2.2, "2.2 metres"), (2.5, "2.5 metres"),
                                 (2.75, "2.75 metres"), (3.0, "three metres")],
         "the passage is narrower than {}"),
    ]
    for source, family, field, values, template in variants:
        base = PSEUDO[source][1]
        for value, words in values:
            p = copy.deepcopy(base)
            p["events"][0][field]["value"] = value
            out.append(ProgramText(family, template.format(words), p))

    # For compositions, move one boundary at a time while keeping the other
    # condition and the temporal relation unchanged. The text states observable
    # co-occurrence, rather than asserting that one event caused another.
    for source, family, field, values, template in [
        (14, "crowd_slow", ("crowd", "threshold"), [3, 4, 5, 6],
         "the robot slows while at least {} people are visible ahead"),
        (15, "person_slow", ("slow", "change"), [10, 20, 25, 35, 45, 55, 65, 75],
         "a person appears ahead and the robot slows by at least {} percent"),
        (16, "stop_obstacle", ("close", "threshold"), [0.5, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6],
         "the robot stops with an obstacle closer than {} metres ahead"),
        (17, "turn_person", ("turn", "change"), [10, 15, 25, 35, 45, 60, 70, 80],
         "a person appears ahead during a robot turn of at least {} degrees"),
        (18, "turn_gap", ("narrow", "threshold"), [1.0, 1.25, 1.5, 1.75, 2.0, 2.2, 2.5, 2.75],
         "the robot turns in a passage narrower than {} metres"),
        (19, "stop_car", ("stop", "threshold"), [0.03, 0.05, 0.08, 0.12],
         "the robot moves slower than {} metres per second with a car in view"),
    ]:
        for value in values:
            p = copy.deepcopy(PSEUDO[source][1])
            event, key = field
            next(e for e in p["events"] if e["name"] == event)[key]["value"] = value
            out.append(ProgramText(family, template.format(value), p))
    return out


def development_programs() -> list[ProgramText]:
    """Train-only retrieval probes; numeric settings and utterances are absent from programs()."""
    probes = [
        (0, "slow", "slow", "change", [(20, "20"), (50, "50"), (70, "70")],
         "Find a moment when the robot loses at least {} percent of its speed."),
        (2, "left_turn", "turn", "change", [(20, "20"), (50, "50"), (80, "80")],
         "Where does the robot turn left by at least {} degrees?"),
        (3, "right_turn", "turn", "change", [(20, "20"), (50, "50"), (80, "80")],
         "Show a right turn of {} degrees or more."),
        (8, "front_obstacle", "close", "threshold", [(0.65, "0.65"), (1.1, "1.1"), (1.35, "1.35")],
         "When is something less than {} metres in front of the robot?"),
        (12, "fast", "fast", "threshold", [(0.75, "0.75"), (1.1, "1.1"), (1.65, "1.65")],
         "Find the robot travelling faster than {} metres per second."),
        (13, "narrow", "narrow", "threshold", [(1.4, "1.4"), (1.9, "1.9"), (2.25, "2.25")],
         "Show a passage narrower than {} metres."),
        (15, "person_then_slow", "slow", "change", [(40, "40"), (50, "50")],
         "Where does a person appear before the robot slows by {} percent?"),
        (16, "stop_near_obstacle", "close", "threshold", [(0.7, "0.7"), (1.1, "1.1")],
         "Find the robot stopped with an obstacle within {} metres."),
        (17, "turn_then_person", "turn", "change", [(20, "20"), (50, "50")],
         "Where does the robot turn by {} degrees before a person appears?"),
        (18, "turn_in_gap", "narrow", "threshold", [(1.4, "1.4"), (1.9, "1.9")],
         "Find a robot turn inside a passage narrower than {} metres."),
    ]
    out = []
    train_programs = {json.dumps(spec.program, sort_keys=True) for spec in programs()}
    train_texts = {spec.text for spec in programs()}
    for source, family, event, field, values, template in probes:
        for value, wording in values:
            program = copy.deepcopy(PSEUDO[source][1])
            next(e for e in program["events"] if e["name"] == event)[field]["value"] = value
            spec = ProgramText(family, template.format(wording), program)
            if spec.text in train_texts or json.dumps(program, sort_keys=True) in train_programs:
                raise ValueError(f"development probe overlaps training: {spec.text}")
            out.append(spec)
    return out


def build(bundle, recs: list[str], specs: list[ProgramText] | None = None) -> list[Supervision]:
    """Execute fixed programs on the requested recordings, caching repeated text renderings."""
    by_rec = defaultdict(list)
    for wid in bundle.embedding_index("siglip2").ids:
        rec, start, end = window_span_s(wid)
        if rec in recs:
            by_rec[rec].append((wid, bundle.recordings[rec].t_abs(start), bundle.recordings[rec].t_abs(end)))
    out = []
    cache: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for spec in (programs() if specs is None else specs):
        key = json.dumps(spec.program, sort_keys=True)
        if key in cache:
            pos, hard, easy = cache[key]
            out.append(Supervision(spec, pos.copy(), hard.copy(), easy.copy()))
            continue
        program = json_format.ParseDict(spec.program, q.QueryProgram())
        if bundle.validate(program):
            raise ValueError(f"invalid generated program: {spec.text}")
        pos, hard, easy = set(), set(), set()
        for rec in recs:
            evals = bundle.executor.evals(program, rec)
            for wid, lo, hi in by_rec[rec]:
                match, primary = bundle.executor.best_near(program, rec, (lo, hi), 0, evals)
                if match is None:
                    if primary and primary.truth == c.TRUTH_FALSE:
                        easy.add(wid)
                    continue
                clauses = [cl.truth for cl in match.clauses if cl.required]
                # A four-second document must show each bound event. For a
                # missing child, the executor has already checked coverage
                # over the full temporal search interval before saying FALSE.
                visible = all(b is None or (b.start <= hi and b.end >= lo)
                              for b in match.bindings.values())
                if not visible or c.TRUTH_UNKNOWN in clauses:
                    continue
                if all(v == c.TRUTH_TRUE for v in clauses):
                    pos.add(wid)
                elif clauses.count(c.TRUTH_FALSE) == 1 and clauses.count(c.TRUTH_TRUE) == len(clauses) - 1:
                    hard.add(wid)
                elif c.TRUTH_FALSE in clauses:
                    easy.add(wid)
        hard.difference_update(pos)
        easy.difference_update(pos | hard)
        cache[key] = pos, hard, easy
        out.append(Supervision(spec, pos, hard, easy))
    return out
