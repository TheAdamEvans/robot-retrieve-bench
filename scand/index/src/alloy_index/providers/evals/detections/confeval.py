"""RT-DETR detector smoke: the pinned model loads and re-detects known frames with the counts the provider stored.
Would have caught the refactor that deleted the module's model cache (NameError at first frame)."""
from alloy_server.catalog.embodiment import EmbodimentContext
from alloy_server.evalkit import Dataset, Gate, abs_error, exact
from alloy_server.timeline.store import Recording
from alloy_index.providers import detections as D

DATASETS = [Dataset("frames", "cases.jsonl", requires=["bundle"])]
SCORERS = [abs_error("persons"), abs_error("vehicles"), exact("frame_size")]
GATES = [Gate("persons.abs_error.max", "<=", 1), Gate("vehicles.abs_error.max", "<=", 1),
         Gate("frame_size.accuracy", ">=", 1.0)]


def run_case(case, ctx):
    i = case["input"]
    rec = Recording(ctx.bundle_root, i["recording"])
    topic = EmbodimentContext.for_recording(ctx.bundle_root, i["recording"]).topic("front_camera")
    k = rec.topics[topic].nearest(int(i["t_ns"]))
    img = D.frame(rec, topic, k)
    ((names, _, _),) = D.detect([img])
    return {"persons": names.count("person"), "vehicles": sum(n in D.VEHICLES for n in names),
            "frame_size": [img.width, img.height] == [D.W, D.H]}
