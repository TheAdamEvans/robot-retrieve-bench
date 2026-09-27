"""Causal evidence receipts: the latest message per sensor that could have informed an action at cutoff c.

available_at = log_time (receipt by the recorder). measured_at = header stamp. A message is invalid for measurement
if it is stamped more than `skew_tol` after its receipt; invalid messages are masked BEFORE selection, so a
future-stamped message can never hide an earlier valid one. Among valid messages available before c (per the
declared boundary), pick the largest measured_at; require 0 <= age <= max_age. No fallback to another message.
"""
from __future__ import annotations

import numpy as np

from ..gen.alloy.v1 import answer_pb2 as a
from ..gen.alloy.v1 import common_pb2 as c
from ..timeline.store import NO_HEADER

SKEW_TOL_NS = 25_000_000  # tf is future-stamped by up to ~20 ms on these robots (intake reports it)


def causal_last(rec, topic: str, cutoff_ns: int, max_age_ns: int, boundary: int,
                skew_tol_ns: int = SKEW_TOL_NS) -> a.ReceiptItem:
    item = a.ReceiptItem(topic=topic)
    tl = rec.topics.get(topic)
    if tl is None or not len(tl):  # the profile lists this topic but the log never recorded it
        item.ok, item.reason = c.TRUTH_UNKNOWN, c.SENSOR_ABSENT_IN_LOG
        return item
    n = int(np.searchsorted(tl.log_ns, cutoff_ns, side="left" if boundary == c.STRICT_BEFORE else "right"))
    if n == 0:
        item.ok, item.reason = c.TRUTH_UNKNOWN, c.OUTSIDE_COVERAGE
        return item
    h, lg = tl.header_ns[:n], tl.log_ns[:n]
    if (h == NO_HEADER).all():
        item.ok, item.reason = c.TRUTH_UNKNOWN, c.HEADER_MISSING
        return item
    valid = (h != NO_HEADER) & (h <= lg + skew_tol_ns)
    if not valid.any():
        item.ok, item.reason = c.TRUTH_UNKNOWN, c.CLOCK_INCOMPATIBLE
        return item
    i = int(np.flatnonzero(valid)[np.argmax(h[valid])])
    item.message.CopyFrom(rec.message_id(topic, i))
    age = cutoff_ns - int(h[i])
    item.age_ns = age
    if age < 0:
        item.ok, item.reason = c.TRUTH_UNKNOWN, c.FUTURE_MEASUREMENT
    elif age > max_age_ns:
        item.ok, item.reason = c.TRUTH_UNKNOWN, c.STALE
    else:
        item.ok = c.TRUTH_TRUE
    return item


def receipt(rec, sensors: dict[str, list[str]], names: list[str], cutoff_ns: int, max_age_ns: int,
            boundary: int) -> a.EvidenceReceipt:
    out = a.EvidenceReceipt(causal_cutoff_ns=cutoff_ns, boundary=boundary, max_age_ns=max_age_ns,
                            discovery_mode="RETROSPECTIVE")
    for name in names:
        topics = sensors.get(name)
        if not topics:
            out.items.add(sensor=name, ok=c.TRUTH_UNKNOWN, reason=c.NOT_APPLICABLE)
            continue
        for t in topics:
            it = causal_last(rec, t, cutoff_ns, max_age_ns, boundary)
            it.sensor = name
            out.items.append(it)
    return out
