"""A0 smoke: SigLIP2 load on MPS, image/text throughput, text-image sanity."""
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL = "google/siglip2-so400m-patch14-384"


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t = time.perf_counter()
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    print(f"load {time.perf_counter() - t:.1f}s device={device} rev={model.config._commit_hash}")

    imgs = [Image.new("RGB", (640, 480), (i * 8 % 255, 100, 150)) for i in range(32)]
    with torch.no_grad():
        for bs in (8, 32):
            x = proc(images=imgs[:bs], return_tensors="pt").to(device, torch.float16)
            model.get_image_features(**x)
            torch.mps.synchronize() if device == "mps" else None
            t = time.perf_counter()
            for _ in range(3):
                f = model.get_image_features(**x)
                f = getattr(f, "pooler_output", f)
            torch.mps.synchronize() if device == "mps" else None
            dt = (time.perf_counter() - t) / 3
            print(f"image bs={bs}: {bs / dt:.1f} img/s dim={f.shape[-1]}")
        texts = ["a robot walking through a crowd of people"] * 16
        x = proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(device)
        t = time.perf_counter()
        f = model.get_text_features(**x)
        f = getattr(f, "pooler_output", f)
        print(f"text bs=16: {16 / (time.perf_counter() - t):.1f} q/s dim={f.shape[-1]}")


if __name__ == "__main__":
    main()
