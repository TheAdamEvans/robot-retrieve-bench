"""MP4 clips of a result's span, rendered from the source MCAP (every frame at its true timestamp).

Frames are decoded, downscaled, overlaid with the recording time, a progress bar and an anchor marker, then encoded
as H.264 (libx264, yuv420p, faststart) in-process with PyAV — no system ffmpeg needed. Reads go through the
instrumented MCAP reader, so a clip's bytes are counted like any other evidence read.
"""
from __future__ import annotations

import io
from fractions import Fraction

import av
from PIL import Image, ImageDraw, ImageFont

from .ros1 import compressed_image

FONT = ImageFont.load_default(size=16)
MAX_SPAN_NS = 12_000_000_000


def render(rec, topic: str, t0_ns: int, t1_ns: int, out_path: str, anchor_ns: int | None = None, rotation: int = 0,
           width: int = 640, label: str = "") -> None:
    """Write the clip to out_path (faststart needs a real file: the muxer rewrites it to put the index first)."""
    if t1_ns - t0_ns > MAX_SPAN_NS:
        t1_ns = t0_ns + MAX_SPAN_NS
    tl = rec.topics[topic]
    sl = tl.range(t0_ns, t1_ns)
    idx = list(range(sl.start, sl.stop))
    if not idx:
        raise ValueError("no frames in span")
    anchor_i = None
    if anchor_ns is not None:
        after = [i for i in idx if tl.log_ns[i] >= anchor_ns]
        anchor_i = after[0] if after else idx[-1]
    out = av.open(out_path, mode="w", format="mp4", options={"movflags": "faststart"})
    stream = None
    for i in idx:
        _, data = compressed_image(rec.read(topic, i).data)
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if rotation:
            im = im.rotate(rotation, expand=True)
        h = int(round(im.height * width / im.width / 2) * 2)
        im = im.resize((width, h))
        d = ImageDraw.Draw(im)
        t_rel = (int(tl.log_ns[i]) - rec.start_ns) / 1e9
        text = f"{label}  t={t_rel:6.2f}s"
        d.rectangle([0, 0, 10 + d.textlength(text, font=FONT), 24], fill=(0, 0, 0))
        d.text((5, 3), text, fill=(255, 255, 0), font=FONT)
        frac = (int(tl.log_ns[i]) - t0_ns) / max(1, t1_ns - t0_ns)
        d.rectangle([0, h - 5, int(width * frac), h], fill=(255, 200, 0))
        if anchor_ns is not None:
            ax = int(width * (anchor_ns - t0_ns) / max(1, t1_ns - t0_ns))
            d.rectangle([ax - 1, h - 9, ax + 1, h], fill=(255, 60, 60))
            if anchor_i is not None and 0 <= i - anchor_i < 6:  # ~0.2 s flash on the anchor frame
                d.rectangle([0, 0, width - 1, h - 1], outline=(255, 60, 60), width=4)
                d.text((width - 80, 3), "ANCHOR", fill=(255, 80, 80), font=FONT)
        if stream is None:
            stream = out.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = width, h, "yuv420p"
            stream.time_base = Fraction(1, 1000)
            stream.options = {"crf": "28", "preset": "veryfast"}
        frame = av.VideoFrame.from_image(im)
        frame.pts = int((int(tl.log_ns[i]) - int(tl.log_ns[idx[0]])) / 1e6)  # true timing, in ms
        frame.time_base = Fraction(1, 1000)
        for pkt in stream.encode(frame):
            out.mux(pkt)
    for pkt in stream.encode():
        out.mux(pkt)
    out.close()
