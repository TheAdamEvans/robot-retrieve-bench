"""Embodiment cards. Every number carries a basis label; none of these are measured on the recorded units."""

SPOT_BODY_CAMS = ["frontleft", "frontright", "left", "right", "back"]

CARDS = {
    "spot": {
        "robot": "Boston Dynamics Spot", "locomotion": "legged",
        "footprint_m": {"length": 1.10, "width": 0.50, "height": 0.61, "basis": "DATASHEET_NOMINAL"},
        "max_speed_mps": {"value": 1.6, "basis": "DATASHEET_NOMINAL"},
        "sensors": {
            "front_camera": {"topic": "/image_raw/compressed", "hz": 30, "res": "1280x720 RGB",
                             "intrinsics": "none recorded; nominal HFOV 90 deg", "basis": "ESTIMATED"},
            "body_cameras": {"topics": [f"/spot/camera/{c}/image/compressed" for c in SPOT_BODY_CAMS], "hz": 4.5,
                             "res": "640x480 mono", "intrinsics": "camera_info recorded",
                             "extrinsics": "URDF_NOMINAL"},
            "lidar": {"topics": ["/velodyne_points", "/scan"], "hz": 10, "model": "Velodyne VLP-16",
                      "mount": "RECORDED_TF base_link->velodyne"},
            "odom": {"topic": "/odom", "hz": 16.4, "note": "twist expressed in odom frame"},
            "tf": {"topic": "/tf"},
            "joystick": {"topic": "/joystick", "hz": 60, "note": "axes are control inputs, not m/s"},
        },
        "absent": ["imu", "camera_info for front camera", "cmd_vel (all zeros: teleoperated)"],
    },
    "jackal": {
        "robot": "Clearpath Jackal", "locomotion": "wheeled (skid-steer)",
        "footprint_m": {"length": 0.508, "width": 0.430, "height": 0.250, "basis": "DATASHEET_NOMINAL"},
        "max_speed_mps": {"value": 2.0, "basis": "DATASHEET_NOMINAL"},
        "sensors": {
            "front_camera": {"topic": "/camera/rgb/image_raw/compressed", "hz": 30, "res": "1280x720 RGB (Kinect)",
                             "intrinsics": "none recorded; nominal", "basis": "ESTIMATED"},
            "lidar": {"topics": ["/velodyne_2dscan"], "hz": 10, "model": "Velodyne VLP-16 flattened to 2D",
                      "mount": "URDF_NOMINAL (no /tf)"},
            "odom": {"topic": "/jackal_velocity_controller/odom", "hz": 50},
            "imu": {"topic": "/imu/data_raw", "hz": 70},
            "joystick": {"topic": "/bluetooth_teleop/joy", "hz": 65},
        },
        "absent": ["body cameras", "/tf", "3D point cloud (raw packets only)"],
    },
}

# Nominal travel corridor for overlays and the corridor-count provider (ESTIMATED).
CORRIDOR = {"spot": {"half_width_m": 0.75, "length_m": 5.0}, "jackal": {"half_width_m": 0.6, "length_m": 5.0}}


def front_camera_topic(robot: str) -> str:
    return CARDS[robot]["sensors"]["front_camera"]["topic"]


def odom_topic(robot: str) -> str:
    return CARDS[robot]["sensors"]["odom"]["topic"]


def scan_topic(robot: str) -> str:
    return "/scan" if robot == "spot" else "/velodyne_2dscan"


def body_camera_topics(robot: str) -> list[str]:
    return CARDS[robot]["sensors"]["body_cameras"]["topics"] if robot == "spot" else []
