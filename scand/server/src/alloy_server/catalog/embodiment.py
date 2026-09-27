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
    missing = {pid: [t for t in (p.identity_topics or p.signature_topics) if t not in topics]
               for pid, p in profiles().items()}
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


class EmbodimentContext:
    """Every robot-relative question a provider, view or verifier needs answered, from profile + intake.

    Providers receive one of these instead of reading profiles; it is the single place measurements are normalised
    to the embodiment (body-side room, speed as a fraction of the robot's maximum, gait-synchronous smoothing, ...).
    """

    def __init__(self, prof: e.EmbodimentProfile, intake: e.IntakeReport | None = None):
        self.profile, self.intake = prof, intake

    @classmethod
    def for_recording(cls, bundle: Path, rec_id: str) -> "EmbodimentContext":
        intake = load_intake(bundle, rec_id)
        if intake is None:
            raise RuntimeError(f"{rec_id}: no intake report")
        return cls(profile(intake.embodiment_id), intake)

    @property
    def robot(self) -> str:
        return self.profile.embodiment_id

    # ---- sensors ----
    def available(self, name: str) -> bool:
        return sensor(self.profile, name) is not None and not (self.intake and name in self.intake.absent_sensors)

    def topics(self, name: str) -> list[str]:
        s = sensor(self.profile, name)
        return list(s.topics) if s is not None and self.available(name) else []

    def topic(self, name: str) -> str | None:
        t = self.topics(name)
        return t[0] if t else None

    @property
    def clearance(self) -> e.SensorSpec:
        return sensor(self.profile, self.profile.clearance_sensor)

    def stale_leading(self, topic: str) -> int:
        return int(self.intake.stale_leading_frames.get(topic, 0)) if self.intake else 0

    # ---- geometry (NOMINAL footprint, ESTIMATED corridor/camera) ----
    @property
    def footprint(self) -> e.Footprint:
        return self.profile.footprint

    @property
    def corridor(self) -> e.Corridor:
        return self.profile.corridor

    def camera_band(self, name: str = "front_camera") -> dict:
        cm = sensor(self.profile, name).camera
        band = lambda b: (b.nominal, b.lo, b.hi)
        return {"hfov": band(cm.hfov_deg), "h": band(cm.height_m), "pitch": band(cm.pitch_down_deg)}

    def body_side_room(self, xy) -> tuple[float, float]:
        """(left, right) room from each body side to the nearest return strictly alongside the body (a follower
        directly behind is not 'on the right'). 5.0 m when nothing is within 5 m."""
        import numpy as np
        hl, hw = self.footprint.length_m / 2, self.footprint.width_m / 2
        beside = (np.abs(xy[:, 0]) <= hl) & (np.abs(xy[:, 1]) >= hw) & (np.abs(xy[:, 1]) < 5.0)
        lp = xy[beside & (xy[:, 1] > 0), 1] - hw
        rp = -xy[beside & (xy[:, 1] < 0), 1] - hw
        return (float(lp.min()) if len(lp) else 5.0), (float(rp.min()) if len(rp) else 5.0)

    def front_margin(self, xy, horizon_m: float = 10.0) -> float:
        """Free distance ahead of the front edge within the body's width."""
        import numpy as np
        hl, hw = self.footprint.length_m / 2, self.footprint.width_m / 2
        ahead = (xy[:, 0] > hl) & (np.abs(xy[:, 1]) <= hw) & (xy[:, 0] < hl + horizon_m)
        return float(xy[ahead, 0].min() - hl) if ahead.any() else horizon_m

    # ---- motion ----
    def speed_smoother(self) -> tuple[str, float]:
        return speed_window_s(self.profile, self.intake)

    def normalize_speed(self, v):
        return v / self.profile.max_speed_mps.nominal

    @property
    def gait_period_s(self) -> float | None:
        return self.intake.gait.period_s if self.intake and self.intake.HasField("gait") else None
