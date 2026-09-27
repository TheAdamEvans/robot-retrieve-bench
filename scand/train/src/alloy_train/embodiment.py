"""Train-side conveniences over the embodiment profiles (alloy_server/embodiments/*.textproto).

No robot facts are written here; everything is read from the profile.
"""
from __future__ import annotations

from google.protobuf import json_format

from alloy_server.catalog import embodiment as E


def prof(robot: str):
    return E.profile(robot)


def front_camera_topic(robot: str) -> str:
    return E.sensor(prof(robot), "front_camera").topics[0]


def odom_topic(robot: str) -> str:
    return E.sensor(prof(robot), "odom").topics[0]


def scan_topic(robot: str) -> str:
    return E.sensor(prof(robot), "lidar_2d").topics[0]


def lidar3d_topic(robot: str) -> str | None:
    s = E.sensor(prof(robot), "lidar")
    return s.topics[0] if s else None


def body_camera_topics(robot: str) -> list[str]:
    s = E.sensor(prof(robot), "body_cameras")
    return list(s.topics) if s else []


def body_camera_names(robot: str) -> list[str]:
    s = E.sensor(prof(robot), "body_cameras")
    return [t.split("/")[3] for t in s.topics] if s else []


def footprint(robot: str):
    return prof(robot).footprint


def corridor(robot: str):
    return prof(robot).corridor


def camera_model(robot: str):
    return E.sensor(prof(robot), "front_camera").camera


def cards() -> dict:
    return {pid: json_format.MessageToDict(p) for pid, p in E.profiles().items()}
