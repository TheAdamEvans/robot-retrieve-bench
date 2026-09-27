"""Embodiment profiles (declared data) and intake reports (measured per log).

Profiles live in alloy_server/embodiments/*.textproto and are the only place robot-specific facts are written down.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

from google.protobuf import json_format, text_format

from ..gen.alloy.v1 import embodiment_pb2 as e

PROFILE_DIR = Path(__file__).resolve().parents[1] / "embodiments"


@functools.lru_cache(maxsize=None)
def profiles() -> dict[str, e.EmbodimentProfile]:
    out = {}
    for p in sorted(PROFILE_DIR.glob("*.textproto")):
        prof = text_format.Parse(p.read_text(), e.EmbodimentProfile())
        out[prof.embodiment_id] = prof
    return out


def profile(embodiment_id: str) -> e.EmbodimentProfile:
    return profiles()[embodiment_id]


def match(topics: set[str]) -> tuple[str | None, dict[str, list[str]]]:
    """→ (matching embodiment_id or None, {embodiment_id: missing signature topics})."""
    missing = {pid: [t for t in p.signature_topics if t not in topics] for pid, p in profiles().items()}
    hits = [pid for pid, m in missing.items() if not m]
    return (hits[0] if len(hits) == 1 else None), missing


def sensor(prof: e.EmbodimentProfile, name: str) -> e.SensorSpec | None:
    return next((s for s in prof.sensors if s.name == name), None)


def topic_display(prof: e.EmbodimentProfile) -> dict[str, tuple[int, str]]:
    """topic → (display rotation, display name)."""
    out = {}
    for s in prof.sensors:
        for k, t in enumerate(s.topics):
            rot = s.display_rotation_deg[k] if k < len(s.display_rotation_deg) else 0
            name = s.display_names[k] if k < len(s.display_names) else s.name
            out[t] = (rot, name)
    return out


def sensors_by_name(prof: e.EmbodimentProfile) -> dict[str, list[str]]:
    return {s.name: list(s.topics) for s in prof.sensors}


def load_intake(bundle: Path, rec: str) -> e.IntakeReport | None:
    p = bundle / "intake" / f"{rec}.json"
    return json_format.Parse(p.read_text(), e.IntakeReport()) if p.exists() else None


def save_intake(bundle: Path, report: e.IntakeReport) -> Path:
    p = bundle / "intake" / f"{report.recording_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_format.MessageToDict(report), indent=1))
    return p


def speed_window_s(prof: e.EmbodimentProfile, intake: e.IntakeReport | None) -> tuple[str, float]:
    sm = prof.speed_smoothing
    if sm.method == e.GAIT_SYNC_MEAN and intake is not None and intake.HasField("gait"):
        return "mean", sm.gait_cycles * intake.gait.period_s
    return ("mean" if sm.method == e.GAIT_SYNC_MEAN else "median"), sm.window_s
