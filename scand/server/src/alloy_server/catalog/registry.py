"""FeatureRegistry: every feature a QueryProgram may reference, with unit dimension, applicability and provenance.

`indexed=False` features are real concepts the grammar can express but no provider computes yet; clauses over them
resolve to UNKNOWN(NOT_INDEXED), which is how the system gives an honest partial answer instead of a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# unit dimension per Unit enum name
DIMENSION = {
    "DIMENSIONLESS": "count", "RATIO": "ratio", "PERCENT": "ratio",
    "M": "length", "CM": "length", "MPS": "speed", "KMH": "speed", "MPS2": "accel",
    "DEG": "angle", "DEG_PER_S": "angular_rate", "S": "time", "MS": "time",
}
TO_CANONICAL = {"PERCENT": 0.01, "CM": 0.01, "KMH": 1 / 3.6, "MS": 0.001}  # → RATIO, M, MPS, S


@dataclass(frozen=True)
class Feature:
    name: str
    provider: str
    unit: str                    # canonical Unit enum name
    kind: str                    # continuous | count | bool | track
    doc: str
    robots: tuple[str, ...] = ("spot", "jackal")
    indexed: bool = True
    causal: bool = True
    spatial_basis: str = "SPATIAL_NONE"
    column: str = ""             # column in the provider table (default: name)
    uncertain: bool = False      # provider also writes <column>_lo / <column>_hi

    @property
    def dimension(self) -> str:
        return DIMENSION[self.unit]


FEATURES: list[Feature] = [
    Feature("speed_mps", "motion", "MPS", "continuous",
            "Robot planar speed from odometry, causal trailing 1 s median (Spot's raw speed oscillates with gait)."),
    Feature("yaw_rate_dps", "motion", "DEG_PER_S", "continuous",
            "Signed yaw rate from odometry, trailing 0.5 s mean; positive = turning left."),
    Feature("heading_deg", "motion", "DEG", "continuous",
            "Heading integrated from yaw rate since recording start (relative, drifts slowly). Use CHANGE with no "
            "direction for 'turns by N degrees' in either direction."),
    Feature("accel_mps2", "motion", "MPS2", "continuous",
            "Longitudinal acceleration: backward difference of smoothed speed over 0.5 s."),
    Feature("min_clearance_front_m", "clearance", "M", "continuous",
            "Minimum lidar range within +/-30 deg of forward, from the sensor origin (not the footprint edge)."),
    Feature("min_clearance_any_m", "clearance", "M", "continuous",
            "Minimum lidar range in any direction beyond 0.5 m of the sensor."),
    Feature("lateral_clearance_left_m", "clearance", "M", "continuous",
            "Distance to the nearest obstacle beside the robot on the left (points alongside the footprint).",
            spatial_basis="NOMINAL"),
    Feature("lateral_clearance_right_m", "clearance", "M", "continuous",
            "Distance to the nearest obstacle beside the robot on the right.", spatial_basis="NOMINAL"),
    Feature("gap_width_m", "clearance", "M", "continuous",
            "Width of the free gap the robot is passing through: left + right lateral clearance.",
            spatial_basis="NOMINAL"),
    Feature("doorway_active", "clearance", "DIMENSIONLESS", "bool",
            "1 while the robot passes through a doorway-like gap (gap_width_m < 1.6 m for 0.3-5 s). Geometric, "
            "not a semantic door detector.", spatial_basis="NOMINAL"),
    Feature("persons_visible_front", "detections", "DIMENSIONLESS", "count",
            "People detected in the front camera (RT-DETRv2, 10 Hz)."),
    Feature("persons_in_corridor", "detections", "DIMENSIONLESS", "count",
            "People whose feet project into the robot's forward travel corridor (~1.5 m wide, 5 m ahead) using a "
            "NOMINAL camera model; reported as a [lo, hi] range, so thresholds can be UNKNOWN.",
            spatial_basis="ESTIMATED", uncertain=True),
    Feature("vehicles_visible_front", "detections", "DIMENSIONLESS", "count",
            "Motor vehicles (car, truck, bus, motorcycle) detected in the front camera."),
    Feature("vehicle_box_frac", "detections", "RATIO", "continuous",
            "Image area fraction of the largest detected vehicle: a proximity proxy (bigger = closer)."),
    Feature("bicycles_visible_front", "detections", "DIMENSIONLESS", "count",
            "Bicycles detected in the front camera."),
    Feature("person_tracks_front", "detections", "DIMENSIONLESS", "track",
            "Person tracks in the front camera (causal IoU tracker at 10 Hz). Use with TRACK_APPEAR / "
            "TRACK_DISAPPEAR."),
    Feature("person_tracks_all_cameras", "detections", "DIMENSIONLESS", "track",
            "Person tracks fused across every camera (front + body). NOT INDEXED: body cameras are not run through "
            "the detector.", indexed=False),
    Feature("persons_visible_body_cameras", "detections", "DIMENSIONLESS", "count",
            "People visible in Spot's five body cameras. NOT INDEXED.", robots=("spot",), indexed=False),
    Feature("imu_vibration_rms", "imu", "MPS2", "continuous",
            "High-frequency IMU vibration (terrain roughness). Jackal only. NOT INDEXED.", robots=("jackal",),
            indexed=False),
]

REGISTRY: dict[str, Feature] = {f.name: f for f in FEATURES}

# Sensors a ReceiptSpec may name → topics per robot, straight from the embodiment profiles.
from .embodiment import profiles as _profiles, sensors_by_name as _sensors_by_name

def _with_aliases(d: dict[str, list[str]]) -> dict[str, list[str]]:
    # "lidar" means the best lidar the robot has (Jackal only has the flattened 2D scan)
    if "lidar" not in d and "lidar_2d" in d:
        d = {**d, "lidar": d["lidar_2d"]}
    return d


SENSORS = {pid: _with_aliases(_sensors_by_name(p)) for pid, p in _profiles().items()}
ALL_SENSORS = sorted({s for m in SENSORS.values() for s in m})


def registry_hash() -> str:
    import hashlib
    import json
    blob = json.dumps([f.__dict__ for f in FEATURES], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
