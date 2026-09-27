"""Blind re-check agreement: a fresh labeller instance re-grades a stratified sample without seeing prior grades.

Reports exact-grade agreement, binary agreement on the positive class (grade 2) and Cohen's kappa. Both passes are the
same model family with the same tools, so this measures consistency, not independence from shared blind spots.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from alloy_train.eval.report import load_judgments
from alloy_index.recordings import SCAND_ROOT


def kappa(pairs: list[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main() -> dict:
    first = load_judgments(SCAND_ROOT / "annotations")
    second = load_judgments(SCAND_ROOT / "annotations_recheck")
    pairs, per = [], defaultdict(list)
    for intent, wg in second.items():
        for w, g2 in wg.items():
            g1 = first.get(intent, {}).get(w)
            if g1 is None or g1 < 0 or g2 < 0:
                continue
            pairs.append((g1, g2))
            per[intent].append((g1, g2))
    binp = [(int(a >= 2), int(b >= 2)) for a, b in pairs]
    out = {"n": len(pairs), "exact": sum(a == b for a, b in pairs) / max(1, len(pairs)),
           "within_one": sum(abs(a - b) <= 1 for a, b in pairs) / max(1, len(pairs)),
           "binary_pos": sum(a == b for a, b in binp) / max(1, len(binp)), "kappa_graded": kappa(pairs),
           "kappa_binary": kappa(binp),
           "per_intent": {k: {"n": len(v), "exact": round(sum(a == b for a, b in v) / len(v), 2)} for k, v in per.items()},
           "confusion": {f"{a}->{b}": c for (a, b), c in sorted(Counter(pairs).items())}}
    (SCAND_ROOT / "results" / "eval" / "agreement.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
