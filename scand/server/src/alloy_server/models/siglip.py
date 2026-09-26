"""SigLIP2 encoder: one recipe shared by the document side (indexing) and the query side (server).

The recipe (model revision, text padding/length, image preprocessing via the model's own processor, L2 norm) is
pinned in SPEC; the index records the spec it was built with and the server refuses a mismatch.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import torch
from PIL import Image

from ..io import cost

SPEC = {
    "model_id": "google/siglip2-so400m-patch14-384",
    "revision": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    "dim": 1152,
    "normalization": "L2",
    "dtype": "float16",
    "text_recipe": "padding=max_length,max_length=64,lowercase=false",
    "image_recipe": "processor default (384px)",
}
SPACE_ID = "siglip2:" + hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:12]


class SiglipEncoder:
    def __init__(self, device: str | None = None):
        from transformers import AutoModel, AutoProcessor

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(SPEC["model_id"], revision=SPEC["revision"],
                                               dtype=torch.float16).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(SPEC["model_id"], revision=SPEC["revision"])

    @staticmethod
    def _norm(x: torch.Tensor) -> np.ndarray:
        x = x.float()
        return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy()

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        x = self.proc(images=images, return_tensors="pt").to(self.device, torch.float16)
        out = self.model.get_image_features(**x)
        return self._norm(getattr(out, "pooler_output", out))

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        t0 = time.perf_counter()
        x = self.proc(text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
        out = self.model.get_text_features(**x)
        v = self._norm(getattr(out, "pooler_output", out))
        s = cost.current()
        if s is not None:
            s.encoder_forward_ms += (time.perf_counter() - t0) * 1000
        return v
