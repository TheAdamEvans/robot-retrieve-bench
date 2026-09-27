"""alloy-train providers: build provider tables for every recording."""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

from alloy_server.timeline.store import Recording
from alloy_index.recordings import RECORDINGS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--providers", default="motion,clearance")
    ap.add_argument("recs", nargs="*", default=list(RECORDINGS))
    a = ap.parse_args()
    mods = [importlib.import_module(f"alloy_index.providers.{p}") for p in a.providers.split(",")]
    for rec_id in a.recs:
        rec = Recording(a.bundle, rec_id)
        for m in mods:
            t0 = time.perf_counter()
            info = m.build(a.bundle, rec)
            print(json.dumps({"recording": rec_id, **info, "s": round(time.perf_counter() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
