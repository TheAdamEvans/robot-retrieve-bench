"""EmbodimentContext geometry and normalisation, per robot. Seeded by the labeller's follower-behind finding: a person
walking directly behind Spot was read as lateral room on the right."""
import numpy as np

from alloy_server.catalog.embodiment import EmbodimentContext, profiles
from alloy_server.evalkit import Dataset, Gate, abs_error

DATASETS = [Dataset("geometry", "cases.jsonl")]
SCORERS = [abs_error("left_m"), abs_error("right_m"), abs_error("front_m"), abs_error("speed_frac")]
GATES = [Gate(f"{f}.abs_error.max", "<=", 0.01) for f in ("left_m", "right_m", "front_m", "speed_frac")]


def run_case(case, ctx):
    i = case["input"]
    em = EmbodimentContext(profiles()[i["robot"]])
    if i["op"] == "body_side_room":
        left, right = em.body_side_room(np.array(i["xy"], dtype=float).reshape(-1, 2))
        return {"left_m": left, "right_m": right}
    if i["op"] == "front_margin":
        return {"front_m": em.front_margin(np.array(i["xy"], dtype=float).reshape(-1, 2))}
    return {"speed_frac": float(em.normalize_speed(i["speed_mps"]))}
