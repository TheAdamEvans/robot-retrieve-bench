#!/usr/bin/env python3
"""Build a visual inventory of locally downloaded SCAND recordings."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import av
import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader


COLORS = ["#66d9ef", "#ffb454", "#a78bfa", "#7ee787", "#ff7b72", "#f778ba"]


def esc(value: object) -> str:
    return html.escape(str(value))


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{number:.1f} GiB"


def percentile(values: list[float], q: float) -> float | None:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    return float(np.percentile(finite, q)) if finite.size else None


def header_stamp_ns(raw: bytes) -> int | None:
    """Read the ROS1 std_msgs/Header prefix without decoding the payload."""
    try:
        _, sec, nsec, frame_len = struct.unpack_from("<IIII", raw, 0)
        if frame_len > len(raw) - 16:
            return None
        return sec * 1_000_000_000 + nsec
    except (struct.error, ValueError):
        return None


def parse_ros1_imu(raw: bytes) -> tuple[float, float] | None:
    """Return acceleration and angular-rate magnitudes from a ROS1 Imu message."""
    try:
        _, _, _, frame_len = struct.unpack_from("<IIII", raw, 0)
        offset = 16 + frame_len
        values = struct.unpack_from("<37d", raw, offset)
        angular = values[13:16]
        acceleration = values[25:28]
        return math.sqrt(sum(v * v for v in acceleration)), math.sqrt(sum(v * v for v in angular))
    except (struct.error, ValueError):
        return None


def downsample(points: list[tuple[float, float]], limit: int = 600) -> list[tuple[float, float]]:
    if len(points) <= limit:
        return points
    step = math.ceil(len(points) / limit)
    return points[::step]


def line_svg(series: dict[str, list[tuple[float, float]]], title: str, unit: str = "") -> str:
    width, height = 620, 190
    left, right, top, bottom = 48, 14, 26, 30
    clean = {name: downsample([(x, y) for x, y in pts if math.isfinite(x) and math.isfinite(y)]) for name, pts in series.items() if pts}
    all_points = [point for points in clean.values() for point in points]
    if not all_points:
        return f'<div class="empty-plot">No {esc(title.lower())} data</div>'
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    margin = (ymax - ymin) * 0.08
    ymin -= margin
    ymax += margin

    def xy(x: float, y: float) -> tuple[float, float]:
        px = left + (x - xmin) / max(xmax - xmin, 1e-9) * (width - left - right)
        py = top + (ymax - y) / max(ymax - ymin, 1e-9) * (height - top - bottom)
        return px, py

    grid = []
    for i in range(5):
        y = top + i * (height - top - bottom) / 4
        value = ymax - i * (ymax - ymin) / 4
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/><text x="{left-6}" y="{y+4:.1f}" text-anchor="end">{value:.2g}</text>')
    paths = []
    legends = []
    for index, (name, points) in enumerate(clean.items()):
        color = COLORS[index % len(COLORS)]
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(*point) for point in points))
        paths.append(f'<polyline points="{coords}" stroke="{color}"/>')
        legends.append(f'<span><i style="background:{color}"></i>{esc(name)}</span>')
    return f'''<div class="plot"><div class="plot-title">{esc(title)} <small>{esc(unit)}</small></div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">
        <g class="grid">{''.join(grid)}</g><g class="lines">{''.join(paths)}</g>
        <text x="{left}" y="{height-7}">{xmin:.0f}s</text><text x="{width-right}" y="{height-7}" text-anchor="end">{xmax:.0f}s</text>
      </svg><div class="legend">{''.join(legends)}</div></div>'''


def trajectory_svg(points: list[tuple[float, float, float]]) -> str:
    width, height, pad = 620, 190, 18
    finite = [(t, x, y) for t, x, y in points if math.isfinite(x) and math.isfinite(y)]
    if len(finite) < 2:
        return '<div class="empty-plot">No trajectory data</div>'
    finite = finite[:: max(1, len(finite) // 800)]
    xs, ys = [p[1] for p in finite], [p[2] for p in finite]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    scale = min((width - 2 * pad) / max(xmax - xmin, 1e-9), (height - 2 * pad) / max(ymax - ymin, 1e-9))

    def xy(x: float, y: float) -> tuple[float, float]:
        return pad + (x - xmin) * scale, height - pad - (y - ymin) * scale

    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(x, y) for _, x, y in finite))
    sx, sy = xy(finite[0][1], finite[0][2])
    ex, ey = xy(finite[-1][1], finite[-1][2])
    distance = sum(math.hypot(finite[i][1] - finite[i - 1][1], finite[i][2] - finite[i - 1][2]) for i in range(1, len(finite)))
    return f'''<div class="plot"><div class="plot-title">Trajectory <small>sampled path · {distance:.0f} m</small></div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Robot trajectory"><polyline class="route" points="{coords}"/>
      <circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" class="start"/><circle cx="{ex:.1f}" cy="{ey:.1f}" r="5" class="end"/></svg>
      <div class="legend"><span><i class="start-key"></i>start</span><span><i class="end-key"></i>end</span></div></div>'''


def rate_bars(topics: list[dict]) -> str:
    selected = sorted(topics, key=lambda item: item["rate"], reverse=True)[:10]
    maximum = max((item["rate"] for item in selected), default=1)
    bars = []
    for item in selected:
        width = 100 * item["rate"] / maximum
        bars.append(f'''<div class="rate-row"><span title="{esc(item['type'])}">{esc(item['topic'])}</span>
          <div><i style="width:{width:.1f}%"></i></div><b>{item['rate']:.1f} Hz</b></div>''')
    return '<div class="rate-bars">' + "".join(bars) + '</div>'


def inspect_avi(path: Path) -> dict:
    result = {"size": path.stat().st_size, "duration": None, "frames": None, "rate": None, "codec": None, "width": None, "height": None}
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            result.update(
                duration=float(stream.duration * stream.time_base) if stream.duration else (float(container.duration / av.time_base) if container.duration else None),
                frames=stream.frames or None,
                rate=float(stream.average_rate) if stream.average_rate else None,
                codec=stream.codec_context.name,
                width=stream.width,
                height=stream.height,
            )
    except av.AVError as error:
        result["error"] = str(error)
    return result


def extract_avi_frames(path: Path, output_dir: Path, count: int = 6) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total = stream.frames or 0
        if not total:
            return frames
        targets = {round(value): index for index, value in enumerate(np.linspace(0, total - 1, count))}
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index not in targets:
                continue
            slot = targets[frame_index]
            target = output_dir / f"avi_{slot + 1:02d}.jpg"
            Image.fromarray(frame.to_ndarray(format="rgb24")).save(target, quality=84)
            timestamp = float(frame.time) if frame.time is not None else None
            frames.append({"path": target, "t": timestamp, "topic": "standalone AVI", "slot": slot})
    return sorted(frames, key=lambda item: item["slot"])


def inspect_bag(path: Path, metadata: dict, asset_root: Path) -> dict:
    result = {
        "stem": path.stem,
        "bag_path": path,
        "bag_size": path.stat().st_size,
        "metadata": metadata,
        "signals": defaultdict(list),
        "trajectory": [],
        "topics": [],
        "keyframes": [],
        "surround": [],
        "alerts": [],
    }
    assets = asset_root / path.stem
    assets.mkdir(parents=True, exist_ok=True)
    with AnyReader([path]) as reader:
        start = reader.start_time
        duration = reader.duration / 1e9
        result["start_ns"] = start
        result["duration"] = duration
        result["message_count"] = sum(c.msgcount for c in reader.connections)
        robot = metadata.get("Robot", "")
        main_camera = "/camera/rgb/image_raw/compressed" if robot == "Jackal" else "/image_raw/compressed"
        body_cameras = {c.topic for c in reader.connections if c.msgtype == "sensor_msgs/msg/CompressedImage" and "/spot/camera/" in c.topic}
        targets = [start + int(duration * fraction * 1e9) for fraction in (0.05, 0.23, 0.41, 0.59, 0.77, 0.95)]
        candidates: list[tuple[int, int, bytes, object] | None] = [None] * len(targets)
        surround_target = start + int(duration * 0.5 * 1e9)
        surround_candidates: dict[str, tuple[int, int, bytes, object]] = {}
        times: dict[str, list[int]] = defaultdict(list)
        camera_skews_ms: list[float] = []
        decode_errors = Counter()

        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            relative = (timestamp - start) / 1e9
            times[topic].append(timestamp)
            raw = bytes(rawdata)

            if topic == main_camera:
                stamp = header_stamp_ns(raw)
                if stamp:
                    camera_skews_ms.append((timestamp - stamp) / 1e6)
                for index, target in enumerate(targets):
                    delta = abs(timestamp - target)
                    if delta <= 2_000_000_000 and (candidates[index] is None or delta < candidates[index][0]):
                        candidates[index] = (delta, timestamp, raw, connection)
                continue
            if topic in body_cameras:
                delta = abs(timestamp - surround_target)
                if delta <= 2_000_000_000 and (topic not in surround_candidates or delta < surround_candidates[topic][0]):
                    surround_candidates[topic] = (delta, timestamp, raw, connection)
                continue
            if connection.msgtype in {"sensor_msgs/msg/PointCloud2", "velodyne_msgs/msg/VelodyneScan", "sensor_msgs/msg/CameraInfo", "tf2_msgs/msg/TFMessage"}:
                continue

            try:
                if connection.msgtype == "sensor_msgs/msg/Imu":
                    parsed = parse_ros1_imu(raw)
                    if parsed:
                        result["signals"]["acceleration magnitude"].append((relative, parsed[0]))
                        result["signals"]["angular-rate magnitude"].append((relative, parsed[1]))
                    else:
                        decode_errors[topic] += 1
                    continue
                message = reader.deserialize(raw, connection.msgtype)
            except Exception:
                decode_errors[topic] += 1
                continue

            if connection.msgtype == "nav_msgs/msg/Odometry":
                linear = message.twist.twist.linear
                angular = message.twist.twist.angular
                speed = math.sqrt(linear.x ** 2 + linear.y ** 2 + linear.z ** 2)
                result["signals"]["observed speed"].append((relative, speed))
                result["signals"]["observed yaw rate"].append((relative, angular.z))
                position = message.pose.pose.position
                if robot == "Jackal":
                    result["trajectory"].append((relative, position.x, position.y))
            elif connection.msgtype == "geometry_msgs/msg/Twist" and topic == "/navigation/cmd_vel":
                result["signals"]["commanded speed"].append((relative, message.linear.x))
                result["signals"]["commanded yaw rate"].append((relative, message.angular.z))
            elif connection.msgtype == "amrl_msgs/msg/Localization2DMsg":
                result["trajectory"].append((relative, message.pose.x, message.pose.y))
            elif connection.msgtype == "sensor_msgs/msg/LaserScan" and topic in {"/scan", "/velodyne_2dscan"}:
                scan = np.asarray(message.ranges, dtype=float)
                scan = scan[np.isfinite(scan) & (scan >= max(float(message.range_min), 0.05)) & (scan <= float(message.range_max))]
                if scan.size:
                    result["signals"]["clearance p05"].append((relative, float(np.percentile(scan, 5))))
                    result["signals"]["range median"].append((relative, float(np.median(scan))))
            elif connection.msgtype == "sensor_msgs/msg/JointState" and len(message.velocity):
                velocities = np.asarray(message.velocity, dtype=float)
                efforts = np.asarray(message.effort, dtype=float)
                result["signals"]["joint velocity RMS"].append((relative, float(np.sqrt(np.mean(velocities ** 2)))))
                if efforts.size:
                    result["signals"]["joint effort RMS"].append((relative, float(np.sqrt(np.mean(efforts ** 2)))))

        # Aggregate duplicate connections under a single topic.
        connection_types: dict[str, set[str]] = defaultdict(set)
        connection_counts: Counter[str] = Counter()
        for connection in reader.connections:
            connection_types[connection.topic].add(connection.msgtype)
            connection_counts[connection.topic] += connection.msgcount
        for topic, timestamps in sorted(times.items()):
            span = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0
            gaps = np.diff(np.asarray(timestamps, dtype=np.int64)) / 1e9 if len(timestamps) > 1 else np.asarray([])
            result["topics"].append({
                "topic": topic,
                "type": ", ".join(sorted(connection_types[topic])),
                "count": connection_counts[topic],
                "rate": (len(timestamps) - 1) / span if span else 0,
                "span": span,
                "max_gap": float(gaps.max()) if gaps.size else None,
                "p95_gap": float(np.percentile(gaps, 95)) if gaps.size else None,
            })

        for index, candidate in enumerate(candidates):
            if not candidate:
                continue
            _, timestamp, raw, connection = candidate
            try:
                message = reader.deserialize(raw, connection.msgtype)
                extension = ".png" if "png" in message.format.lower() else ".jpg"
                target = assets / f"keyframe_{index + 1:02d}{extension}"
                target.write_bytes(bytes(message.data))
                result["keyframes"].append({"path": target, "t": (timestamp - start) / 1e9, "topic": connection.topic})
            except Exception:
                decode_errors[connection.topic] += 1
        for slot, (topic, candidate) in enumerate(sorted(surround_candidates.items())):
            _, timestamp, raw, connection = candidate
            try:
                message = reader.deserialize(raw, connection.msgtype)
                extension = ".png" if "png" in message.format.lower() else ".jpg"
                target = assets / f"surround_{slot + 1:02d}{extension}"
                target.write_bytes(bytes(message.data))
                result["surround"].append({"path": target, "t": (timestamp - start) / 1e9, "topic": topic})
            except Exception:
                decode_errors[topic] += 1

        if camera_skews_ms:
            result["camera_skew"] = {
                "median": median(camera_skews_ms),
                "p95_abs": percentile([abs(value) for value in camera_skews_ms], 95),
                "min": min(camera_skews_ms),
                "max": max(camera_skews_ms),
            }
        result["decode_errors"] = dict(decode_errors)
        if decode_errors:
            detail = ", ".join(f"{topic} ({count:,})" for topic, count in decode_errors.items())
            result["alerts"].append(f"Custom payload decoding skipped for {detail}; topic timing and counts remain available")
    return result


def frame_gallery(frames: list[dict], output_html: Path, title: str) -> str:
    if not frames:
        return '<div class="empty-plot">No visual frames extracted</div>'
    items = []
    for frame in frames:
        relative = Path(frame["path"]).relative_to(output_html.parent)
        label = frame["topic"].replace("/spot/camera/", "").replace("/image/compressed", "").replace("/camera/rgb/image_raw/compressed", "front RGB")
        time_text = f'{frame["t"]:.1f}s' if frame["t"] is not None else "encoded timeline"
        items.append(f'<figure><img src="{esc(relative.as_posix())}" loading="lazy"><figcaption>{esc(label)} · {time_text}</figcaption></figure>')
    return f'<div class="gallery-title">{esc(title)}</div><div class="gallery">{"".join(items)}</div>'


def file_section(record: dict, output_html: Path) -> str:
    metadata = record["metadata"]
    tags = [tag.strip() for tag in metadata.get("Tags", "").split(",") if tag.strip()]
    tag_html = "".join(f'<span class="tag">{esc(tag)}</span>' for tag in tags)
    avi = record.get("avi")
    paired = avi is not None
    ratio = avi["duration"] / record["duration"] if paired and avi.get("duration") and record.get("duration") else None
    if ratio is not None and ratio < 0.5:
        record["alerts"].append(f"AVI encoded duration is {ratio:.1%} of bag duration; use bag timestamps")
    alerts = "".join(f'<li>{esc(alert)}</li>' for alert in record["alerts"])
    skew = record.get("camera_skew", {})
    summary = [
        ("bag duration", f'{record["duration"]:.1f}s'),
        ("messages", f'{record["message_count"]:,}'),
        ("topics", str(len(record["topics"]))),
        ("bag size", human_bytes(record["bag_size"])),
        ("camera skew p95", f'{skew.get("p95_abs", float("nan")):.1f}ms' if skew else "n/a"),
        ("AVI", f'{avi["frames"]:,}f / {avi["duration"]:.1f}s' if paired and avi.get("frames") and avi.get("duration") else "missing"),
    ]
    speed_values = [value for name in ("observed speed", "commanded speed") for _, value in record["signals"].get(name, [])]
    clearance_values = [value for _, value in record["signals"].get("clearance p05", [])]
    if speed_values:
        summary.append(("maximum speed", f"{max(speed_values):.2f}m/s"))
    if clearance_values:
        summary.append(("low clearance", f"{percentile(clearance_values, 5):.2f}m"))
    stats = "".join(f'<div><b>{esc(value)}</b><span>{esc(label)}</span></div>' for label, value in summary)
    speed_series = {name: record["signals"][name] for name in ("observed speed", "commanded speed") if record["signals"].get(name)}
    turn_series = {name: record["signals"][name] for name in ("observed yaw rate", "commanded yaw rate") if record["signals"].get(name)}
    clearance_series = {"clearance p05": record["signals"]["clearance p05"]} if record["signals"].get("clearance p05") else {}
    if metadata.get("Robot") == "Spot":
        embodiment = line_svg({"joint velocity RMS": record["signals"]["joint velocity RMS"]} if record["signals"].get("joint velocity RMS") else {}, "Joint motion", "rad/s RMS")
        embodiment += line_svg({"joint effort RMS": record["signals"]["joint effort RMS"]} if record["signals"].get("joint effort RMS") else {}, "Joint effort", "RMS")
    else:
        embodiment = line_svg({"acceleration magnitude": record["signals"]["acceleration magnitude"]} if record["signals"].get("acceleration magnitude") else {}, "IMU acceleration", "m/s²")
        embodiment += line_svg({"angular-rate magnitude": record["signals"]["angular-rate magnitude"]} if record["signals"].get("angular-rate magnitude") else {}, "IMU angular rate", "rad/s")
    topic_rows = "".join(
        f'<tr><td>{esc(t["topic"])}</td><td>{esc(t["type"])}</td><td>{t["count"]:,}</td><td>{t["rate"]:.2f}</td><td>{t["max_gap"]:.3f}</td></tr>'
        if t["max_gap"] is not None
        else f'<tr><td>{esc(t["topic"])}</td><td>{esc(t["type"])}</td><td>{t["count"]:,}</td><td>{t["rate"]:.2f}</td><td>n/a</td></tr>'
        for t in sorted(record["topics"], key=lambda item: item["rate"], reverse=True)
    )
    return f'''<section class="recording" id="{esc(record['stem'])}">
      <header><div><div class="eyebrow">{esc(metadata.get('Robot',''))} · {esc(metadata.get('Train/Val',''))} · index {esc(metadata.get('fr','?'))}</div>
      <h2>{esc(record['stem'])}</h2><div class="tags">{tag_html}</div></div><div class="route-name">{esc(metadata.get('Start Location','?'))} → {esc(metadata.get('End Location','?'))}</div></header>
      <div class="stats">{stats}</div>{f'<ul class="alerts">{alerts}</ul>' if alerts else ''}
      {frame_gallery(record['keyframes'], output_html, 'Timestamped front-camera keyframes')}
      {frame_gallery(record['surround'], output_html, 'Spot surround view at the temporal midpoint') if record['surround'] else ''}
      <div class="multiples">
        {trajectory_svg(record['trajectory'])}
        {line_svg(speed_series, 'Linear speed', 'm/s')}
        {line_svg(turn_series, 'Yaw rate', 'rad/s')}
        {line_svg(clearance_series, 'Nearby lidar clearance', 'p05 · m')}
        {embodiment}
        <div class="plot"><div class="plot-title">Topic frequencies</div>{rate_bars(record['topics'])}</div>
      </div>
      <details><summary>All topics, message counts and cadence</summary><div class="table-wrap"><table><thead><tr><th>Topic</th><th>Type</th><th>Messages</th><th>Hz</th><th>Max gap (s)</th></tr></thead><tbody>{topic_rows}</tbody></table></div></details>
    </section>'''


def video_only_section(record: dict, output_html: Path) -> str:
    metadata = record["metadata"]
    avi = record["avi"]
    tags = "".join(f'<span class="tag">{esc(tag.strip())}</span>' for tag in metadata.get("Tags", "").split(",") if tag.strip())
    return f'''<section class="recording incomplete" id="{esc(record['stem'])}"><header><div><div class="eyebrow">{esc(metadata.get('Robot',''))} · video only</div><h2>{esc(record['stem'])}</h2><div class="tags">{tags}</div></div></header>
      <div class="stats"><div><b>{human_bytes(avi['size'])}</b><span>AVI size</span></div><div><b>{avi.get('frames') or 'n/a'}</b><span>frames</span></div><div><b>{avi.get('duration'):.1f}s</b><span>encoded duration</span></div><div><b>{avi.get('rate'):.1f} fps</b><span>encoded rate</span></div></div>
      <ul class="alerts"><li>The matching bag has not arrived. These AVI timestamps are encoded playback time and cannot yet support source-accurate annotations.</li></ul>
      {frame_gallery(record['keyframes'], output_html, 'AVI preview only')}
    </section>'''


def build(root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "assets"
    assets.mkdir(exist_ok=True)
    index_path = root / "SCAND_index.csv"
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        index = {row["FileName"].strip(): row for row in csv.DictReader(handle)}
    avi_paths = {path.stem: path for path in (root / "raw").glob("*.avi")}
    bag_paths = {path.stem: path for path in (root / "raw").glob("*.bag")}
    records = []
    for stem in sorted(set(avi_paths) | set(bag_paths)):
        metadata = index.get(stem, {"FileName": stem})
        avi = inspect_avi(avi_paths[stem]) if stem in avi_paths else None
        if stem in bag_paths:
            record = inspect_bag(bag_paths[stem], metadata, assets)
            record["avi"] = avi
        else:
            frames = extract_avi_frames(avi_paths[stem], assets / stem)
            record = {"stem": stem, "metadata": metadata, "avi": avi, "keyframes": frames}
        records.append(record)

    complete = [record for record in records if "bag_path" in record]
    bag_duration = sum(record["duration"] for record in complete)
    bag_messages = sum(record["message_count"] for record in complete)
    total_bytes = sum(path.stat().st_size for path in (root / "raw").glob("*"))
    robots = Counter(record["metadata"].get("Robot", "unknown") for record in records)
    tags = Counter(tag.strip() for record in records for tag in record["metadata"].get("Tags", "").split(",") if tag.strip())
    nav = "".join(f'<a href="#{esc(r["stem"])}"><span>{esc(r["metadata"].get("Robot", "?"))}</span>{esc(r["metadata"].get("fr", "?"))} · {esc(r["stem"].split("_", 2)[-1])}</a>' for r in records)
    tag_rows = "".join(f'<div><span>{esc(tag)}</span><b>{count}</b></div>' for tag, count in tags.most_common())
    topic_universe = sorted({topic["topic"] for record in complete for topic in record["topics"]})
    important = [topic for topic in topic_universe if any(key in topic for key in ("image", "odom", "scan", "joint", "imu", "cmd_vel", "localization"))]
    matrix_header = "".join(f'<th title="{esc(record["stem"])}">{esc(record["metadata"].get("fr", "?"))}</th>' for record in complete)
    matrix_rows = []
    for topic in important:
        cells = []
        for record in complete:
            match = next((item for item in record["topics"] if item["topic"] == topic), None)
            cells.append(f'<td class="{"yes" if match else "no"}" title="{f"{match["rate"]:.1f} Hz" if match else "absent"}">{f"{match["rate"]:.0f}" if match else "·"}</td>')
        matrix_rows.append(f'<tr><td>{esc(topic)}</td>{"".join(cells)}</tr>')

    html_path = output_dir / "scand_dashboard.html"
    sections = "".join(file_section(record, html_path) if "bag_path" in record else video_only_section(record, html_path) for record in records)
    html_doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SCAND local data atlas</title>
<style>
:root{{--bg:#080b10;--panel:#10151d;--panel2:#151c26;--ink:#e6edf3;--muted:#8996a8;--line:#263140;--cyan:#66d9ef;--green:#7ee787;--amber:#ffb454;--red:#ff7b72}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}}a{{color:inherit}}.layout{{display:grid;grid-template-columns:minmax(300px,1fr) minmax(0,2fr);max-width:1800px;margin:auto}}aside{{position:sticky;top:0;height:100vh;overflow:auto;border-right:1px solid var(--line);padding:28px 24px;background:#0b0f15}}main{{min-width:0;padding:28px}}
h1{{font-size:28px;line-height:1.05;margin:0 0 8px;letter-spacing:-.03em}}h2{{font-size:20px;margin:2px 0 7px;letter-spacing:-.02em}}h3{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:24px 0 10px}}p{{color:var(--muted)}}.kpis{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:20px 0}}.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px}}.kpi b{{display:block;font-size:20px}}.kpi span{{color:var(--muted);font-size:11px}}.taxonomy>div{{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1b2430}}.taxonomy span{{color:var(--muted)}}.nav{{display:grid;gap:4px}}.nav a{{text-decoration:none;padding:7px 9px;border-radius:6px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.nav a:hover{{background:var(--panel2);color:var(--ink)}}.nav a span{{color:var(--cyan);font-size:10px;margin-right:6px;text-transform:uppercase}}
.overview,.recording{{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin-bottom:20px;padding:20px;box-shadow:0 12px 35px #0003}}.overview h2{{font-size:24px}}.recording header{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}}.eyebrow{{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--cyan)}}.route-name{{color:var(--muted);font-weight:600}}.tags{{display:flex;gap:5px;flex-wrap:wrap}}.tag{{background:#1b2633;color:#b9d7ef;border:1px solid #2b3b4d;border-radius:999px;padding:3px 7px;font-size:10px}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:8px;margin:16px 0}}.stats>div{{background:var(--panel2);border-radius:8px;padding:10px}}.stats b,.stats span{{display:block}}.stats b{{font-size:15px}}.stats span{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}.alerts{{background:#2b2113;border-left:3px solid var(--amber);padding:9px 10px 9px 27px;color:#f4c97b;border-radius:4px}}.incomplete{{border-color:#6c5328}}
.gallery-title,.plot-title{{font-weight:700;margin:14px 0 8px}}.plot-title small{{color:var(--muted);font-weight:400;margin-left:5px}}.gallery{{display:grid;grid-template-columns:repeat(6,1fr);gap:6px}}figure{{margin:0;background:#090c11;border-radius:7px;overflow:hidden;border:1px solid #202a37}}figure img{{display:block;width:100%;aspect-ratio:16/9;object-fit:cover}}figcaption{{padding:5px 6px;color:var(--muted);font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.multiples{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px}}.plot,.empty-plot{{background:var(--panel2);border:1px solid #202b39;border-radius:10px;padding:9px;min-height:130px;overflow:hidden}}.plot svg{{width:100%;height:auto;display:block;background:#0e141c;border-radius:6px}}.grid line{{stroke:#263241;stroke-width:1}}.grid text,svg>text{{fill:#758196;font-size:10px}}.lines polyline{{fill:none;stroke-width:1.5;vector-effect:non-scaling-stroke}}.route{{fill:none;stroke:var(--cyan);stroke-width:3;vector-effect:non-scaling-stroke}}.start{{fill:var(--green)}}.end{{fill:var(--red)}}.legend{{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:10px;margin-top:5px}}.legend i{{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px}}.start-key{{background:var(--green)}}.end-key{{background:var(--red)}}
.rate-bars{{display:grid;gap:5px}}.rate-row{{display:grid;grid-template-columns:minmax(110px,1.3fr) 1fr 48px;gap:6px;align-items:center;font-size:9px}}.rate-row>span{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--muted)}}.rate-row>div{{height:6px;background:#263140;border-radius:5px;overflow:hidden}}.rate-row i{{display:block;height:100%;background:var(--cyan)}}.rate-row b{{text-align:right}}details{{margin-top:12px}}summary{{cursor:pointer;color:var(--muted)}}.table-wrap{{overflow:auto;max-height:400px;margin-top:8px}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:6px 8px;border-bottom:1px solid #212b38;text-align:left}}th{{color:var(--muted);font-size:10px;text-transform:uppercase;position:sticky;top:0;background:var(--panel)}}.matrix th:not(:first-child),.matrix td:not(:first-child){{text-align:center}}.matrix .yes{{color:var(--green);background:#102419}}.matrix .no{{color:#465363}}.note{{padding:10px 12px;border-left:3px solid var(--cyan);background:#101e28;color:#a8c7d8;margin:12px 0}}
@media(max-width:1000px){{.layout{{grid-template-columns:1fr}}aside{{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--line)}}.nav{{display:none}}.stats{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:700px){{main{{padding:12px}}.multiples{{grid-template-columns:1fr}}.gallery{{grid-template-columns:repeat(2,1fr)}}.stats{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="layout"><aside><div class="eyebrow">Local data atlas</div><h1>SCAND<br>retrieval corpus</h1><p>Raw recordings inspected directly. Bag clocks are canonical; standalone AVI time is presentation time.</p>
<div class="kpis"><div class="kpi"><b>{len(complete)}</b><span>complete bags</span></div><div class="kpi"><b>{bag_duration/60:.1f}m</b><span>bag duration</span></div><div class="kpi"><b>{bag_messages:,}</b><span>messages</span></div><div class="kpi"><b>{human_bytes(total_bytes)}</b><span>local bytes</span></div></div>
<h3>Embodiments</h3><div class="taxonomy">{''.join(f'<div><span>{esc(k)}</span><b>{v}</b></div>' for k,v in robots.items())}</div><h3>Provided labels</h3><div class="taxonomy">{tag_rows}</div><h3>Recordings</h3><nav class="nav">{nav}</nav></aside><main>
<section class="overview"><div class="eyebrow">Schema coverage</div><h2>Seven complete recordings, one awaiting its bag</h2><p>The cells show approximate topic frequency in Hz. The two embodiments share lidar, front RGB and motion state, while Spot adds five body cameras and joint state and Jackal adds raw IMU and platform status.</p><div class="note">The AVI files are accelerated, downsampled presentation videos. Use timestamped compressed images inside each bag for annotation and evaluation.</div><div class="table-wrap"><table class="matrix"><thead><tr><th>Topic</th>{matrix_header}</tr></thead><tbody>{''.join(matrix_rows)}</tbody></table></div></section>
{sections}</main></div></body></html>'''
    html_path.write_text(html_doc, encoding="utf-8")
    summary_path = output_dir / "dashboard_summary.json"
    summary_path.write_text(json.dumps({
        "recordings": [{"stem": record["stem"], "metadata": record["metadata"], "duration": record.get("duration"), "messages": record.get("message_count"), "topics": record.get("topics"), "camera_skew": record.get("camera_skew"), "alerts": record.get("alerts", [])} for record in records]
    }, indent=2), encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "dashboard")
    args = parser.parse_args()
    print(build(args.root, args.output).resolve())


if __name__ == "__main__":
    main()
