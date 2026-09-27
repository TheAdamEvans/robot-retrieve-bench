"""Source -> Stages -> Sink. Walks the IndexSpec in order; builds only artifacts whose key the sink doesn't have.

A stage's key = sha256(stage@VERSION, params, its inputs' content keys, recording). Existing outputs built by the same
version with the same params can be *adopted* (registered without rebuilding) with --adopt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from alloy_server.gen.alloy.v1 import index_pb2
from alloy_index import recordings as R
from alloy_index.sink import LocalBundleSink
from alloy_index.sources import LocalBagSource
from alloy_index.stages import REGISTRY, IndexContext
from alloy_index.stages.base import key_of, params_sha


@dataclass
class Step:
    stage: str
    recording: str
    status: str            # built | adopted | ran | todo | blocked | n/a | gated
    detail: str = ""


@dataclass
class Report:
    steps: list[Step] = field(default_factory=list)

    def add(self, *a, **k):
        self.steps.append(Step(*a, **k))


def flag_for(stage_name: str) -> str:
    return stage_name.split(".")[0]  # annotate.l1 -> --annotate


class Runner:
    def __init__(self, spec: index_pb2.IndexSpec | None = None, flags: set[str] | None = None,
                 recordings: list[str] | None = None, stages: list[str] | None = None, adopt: bool = False,
                 dry: bool = False, log=print):
        self.spec = spec or R.spec()
        self.flags, self.adopt, self.dry, self.log = flags or set(), adopt, dry, log
        self.only_stages = set(stages or [])
        allrecs = LocalBagSource(self.spec).recordings()
        self.recs = {k: v for k, v in allrecs.items() if not recordings or k in recordings}
        self.bundle = R.SCAND_ROOT / self.spec.sink.path
        self.sink = LocalBundleSink(self.bundle, R.SCAND_ROOT)

    def _record(self, stage, impl, rid, key, params, input_keys, paths, info, wall, ctx, rec) -> None:
        r = index_pb2.ArtifactRecord(key=key, stage=stage.name, version=impl.VERSION, recording_id=rid,
                                     params_sha256=params_sha(params), input_keys=sorted(input_keys), paths=paths,
                                     wall_s=wall, created_at=datetime.now(timezone.utc).isoformat())
        for k, v in (info or {}).items():
            r.info[k] = str(v)
        if rec is not None:
            r.content_key = impl.content_key(ctx, rec, paths)
        self.sink.put(r)

    def run(self) -> Report:
        rep = Report()
        enabled = [s for s in self.spec.stages if s.enabled]
        name_of = {s.impl: s.name for s in enabled}
        for stage in enabled:
            impl = REGISTRY[stage.impl]()
            params = dict(stage.params)
            ctx = IndexContext(self.bundle, R.SCAND_ROOT, self.sink, params, self.flags, self.log)
            gated = stage.requires_flag and flag_for(stage.name) not in self.flags
            selected = not self.only_stages or stage.name in self.only_stages
            deps = [name_of[d] for d in impl.DEPENDS if d in name_of]
            if impl.SCOPE == "PER_RECORDING":
                todo = []
                for rid, rec in sorted(self.recs.items()):
                    upstream = [self.sink.latest(d, rid) for d in deps]
                    missing = [d for d, u in zip(deps, upstream) if u is None]
                    if missing:
                        rep.add(stage.name, rid, "blocked", f"after {', '.join(missing)}")
                        continue
                    ok, why = impl.applies(ctx, rec)
                    if not ok:
                        rep.add(stage.name, rid, "n/a", why)
                        continue
                    inputs = [u.content_key or u.key for u in upstream]
                    key = key_of(stage.name, impl.VERSION, params, inputs, rid,
                                 extra=rec.fingerprint if not deps else "")
                    if self.sink.has(key):
                        rep.add(stage.name, rid, "built")
                        continue
                    if gated:
                        rep.add(stage.name, rid, "gated", f"needs --{flag_for(stage.name)}")
                        continue
                    if self.adopt and (paths := impl.adopt(ctx, rec)):
                        if not self.dry:
                            self._record(stage, impl, rid, key, params, inputs, paths, {"adopted": "true"}, 0.0, ctx, rec)
                        rep.add(stage.name, rid, "adopted")
                        continue
                    todo.append((rid, rec, key, inputs))
                if not todo:
                    continue
                if self.dry or not selected:
                    for rid, *_ in todo:
                        rep.add(stage.name, rid, "todo", "" if selected else "stage not selected")
                    continue
                t0 = time.perf_counter()
                if hasattr(impl, "run_many"):
                    results = impl.run_many(ctx, [rec for _, rec, _, _ in todo])
                    for rid, rec, key, inputs in todo:
                        paths, info = results[rid]
                        self._record(stage, impl, rid, key, params, inputs, paths, info,
                                     time.perf_counter() - t0, ctx, rec)
                        rep.add(stage.name, rid, "ran", ", ".join(f"{k}={v}" for k, v in info.items())[:160])
                else:
                    for rid, rec, key, inputs in todo:
                        t1 = time.perf_counter()
                        self.log(f"[{stage.name}] {rid} ...")
                        paths, info = impl.run(ctx, rec)
                        self._record(stage, impl, rid, key, params, inputs, paths, info, time.perf_counter() - t1, ctx, rec)
                        rep.add(stage.name, rid, "ran", ", ".join(f"{k}={v}" for k, v in info.items())[:160])
                        self.log(f"[{stage.name}] {rid} done in {time.perf_counter() - t1:.1f}s")
            else:  # CORPUS: keyed by every input recording's upstream key
                upstream = [r for d in deps for rid in sorted(self.recs) if (r := self.sink.latest(d, rid))]
                recs_in = sorted({u.recording_id for u in upstream} or
                                 {p.stem for p in (self.bundle / "mcap").glob("*.json")})
                if deps and not upstream:
                    rep.add(stage.name, "*", "blocked", f"after {', '.join(deps)}")
                    continue
                inputs = [u.content_key or u.key for u in upstream]
                key = key_of(stage.name, impl.VERSION, params, inputs, "", extra=",".join(recs_in))
                if self.sink.has(key):
                    rep.add(stage.name, "*", "built")
                    continue
                if self.dry or not selected:
                    rep.add(stage.name, "*", "todo", f"{len(recs_in)} recordings")
                    continue
                t0 = time.perf_counter()
                self.log(f"[{stage.name}] corpus ({len(recs_in)} recordings) ...")
                paths, info = impl.run(ctx, recs_in)
                self._record(stage, impl, "", key, params, inputs, paths, info, time.perf_counter() - t0, ctx, None)
                rep.add(stage.name, "*", "ran", ", ".join(f"{k}={v}" for k, v in info.items())[:160])
        return rep
