"""alloy-server <- alloy-index <- alloy-train: the indexing package never imports the trainer."""
import re
from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "src" / "alloy_index"


def test_index_never_imports_train():
    bad = re.compile(r"^\s*(from|import)\s+alloy_train\b", re.M)
    offenders = [str(p) for p in INDEX.rglob("*.py") if bad.search(p.read_text())]
    assert not offenders, offenders
