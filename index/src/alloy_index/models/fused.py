"""FUSED document-side encoder: [SigLIP2 image mean ; standardised window signals + masks] -> head -> SigLIP2 text space.

Owned by the indexing package because *applying* a trained head to windows is indexing; alloy-trainer imports this to
train it. Saved models live in bundle/models/<name>/ (safetensors weights + JSON stats/spec).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from alloy_server.catalog.windows import window_span_s
from alloy_server.models.siglip import SPEC

SIGNALS = [  # (feature, reducer) over the window
    ("speed_mps", "mean"), ("speed_mps", "min"), ("speed_mps", "max"), ("speed_mps", "delta"), ("speed_mps", "drop"),
    ("yaw_rate_dps", "absmax"), ("heading_deg", "delta"), ("accel_mps2", "min"), ("accel_mps2", "max"),
    ("min_clearance_front_m", "min"), ("min_clearance_any_m", "min"), ("lateral_clearance_left_m", "min"),
    ("lateral_clearance_right_m", "min"), ("gap_width_m", "min"), ("doorway_active", "max"),
    ("persons_visible_front", "mean"), ("persons_visible_front", "max"), ("persons_in_corridor", "max"),
    ("vehicles_visible_front", "max"), ("vehicle_box_frac", "max"), ("bicycles_visible_front", "max"),
]
N_SIGNALS = len(SIGNALS) + 2  # + new person tracks + robot flag


def window_signals(bundle, wid: str) -> tuple[np.ndarray, np.ndarray]:
    rec_id, t0, t1 = window_span_s(wid)
    rec = bundle.recordings[rec_id]
    lo, hi = rec.t_abs(t0), rec.t_abs(t1)
    vals, mask = [], []
    for name, red in SIGNALS:
        s = bundle.features.series(name, rec_id)
        if s is None:
            vals.append(0.0); mask.append(0.0)
            continue
        m = (s.t_ns >= lo) & (s.t_ns <= hi)
        y = s.value[m]
        y = y[np.isfinite(y)]
        if not len(y):
            vals.append(0.0); mask.append(0.0)
            continue
        v = {"mean": y.mean(), "min": y.min(), "max": y.max(), "delta": y[-1] - y[0], "absmax": np.abs(y).max(),
             "drop": (np.maximum.accumulate(y) - y).max()}[red]
        vals.append(float(v)); mask.append(1.0)
    tr = bundle.features.tracks("person_tracks_front", rec_id)
    new = float(((tr.first_ns >= lo) & (tr.first_ns <= hi)).sum()) if tr is not None else 0.0
    vals.append(new); mask.append(float(tr is not None))
    vals.append(float(bundle.robots[rec_id] == "spot")); mask.append(1.0)
    return np.array(vals, np.float32), np.array(mask, np.float32)


def image_vectors(bundle) -> dict[str, np.ndarray]:
    idx = bundle.embedding_index("siglip2")
    return {w: idx.vector(w) for w in idx.ids}


def standardise(raw: tuple[np.ndarray, np.ndarray], mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x, m = raw
    return np.concatenate([(x - mu) / sd * m, m]).astype(np.float32)


class Head(nn.Module):
    def __init__(self, d_sig: int, kind: str, dim: int = SPEC["dim"]):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(dim + d_sig, dim)
            last = self.net
        else:
            self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim + d_sig, 512), nn.GELU(), nn.Dropout(0.2),
                                     nn.Linear(512, dim))
            last = self.net[-1]
        nn.init.zeros_(last.weight)  # start exactly at the image vector (= EMBED) and learn a correction
        nn.init.zeros_(last.bias)
        self.t = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(-10.0))

    def forward(self, img, sig):
        x = self.net(torch.cat([img, sig], -1)) + img  # residual: start from the image vector
        return x / x.norm(dim=-1, keepdim=True)


def encode(kind: str, head: Head | None, img: np.ndarray, sig: np.ndarray) -> np.ndarray:
    if kind == "concat":  # C1a control: a text query [q; 0] only sees the image part of [img; sig] / ||[img; sig]||
        v = np.concatenate([img, sig], 1)
        return (img / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    with torch.no_grad():
        head.eval()
        return head(torch.tensor(img, dtype=torch.float32), torch.tensor(sig, dtype=torch.float32)).numpy()


def save(dir_: Path, kind: str, head: Head | None, mu: np.ndarray, sd: np.ndarray, info: dict) -> None:
    from safetensors.torch import save_file
    dir_.mkdir(parents=True, exist_ok=True)
    if head is not None:
        save_file(head.state_dict(), str(dir_ / "head.safetensors"))
    (dir_ / "model.json").write_text(json.dumps({"kind": kind, "mu": mu.tolist(), "sd": sd.tolist(),
                                                 "n_signals": N_SIGNALS, "dim": SPEC["dim"], **info}, indent=1))


def load(dir_: Path) -> tuple[str, Head | None, np.ndarray, np.ndarray, dict]:
    meta = json.loads((dir_ / "model.json").read_text())
    kind = meta["kind"]
    head = None
    if kind != "concat":
        from safetensors.torch import load_file
        head = Head(2 * meta["n_signals"], kind)
        head.load_state_dict(load_file(str(dir_ / "head.safetensors")))
    return kind, head, np.array(meta["mu"], np.float32), np.array(meta["sd"], np.float32), meta
