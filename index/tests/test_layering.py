"""alloy-server <- alloy-index <- alloy-trainer: the indexing package never imports the trainer."""
import re
from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "src" / "alloy_index"


def test_index_never_imports_train():
    bad = re.compile(r"^\s*(from|import)\s+alloy_trainer\b", re.M)
    offenders = [str(p) for p in INDEX.rglob("*.py") if bad.search(p.read_text())]
    assert not offenders, offenders


def test_providers_use_the_embodiment_context():
    """Providers get robot facts from EmbodimentContext, never by reading profiles or helpers directly."""
    bad = re.compile(r"from alloy_index import embodiment|E\.profile\(|emb\.(prof|corridor|camera_model|footprint)\(")
    offenders = [str(p) for p in (INDEX / "providers").glob("*.py") if bad.search(p.read_text())]
    assert not offenders, offenders


def test_body_side_room_ignores_followers_and_measures_from_the_side():
    import numpy as np
    from alloy_server.catalog.embodiment import EmbodimentContext, profile
    c = EmbodimentContext(profile("spot"))
    xy = np.array([[0.0, 0.60], [0.2, -0.40], [-0.7, -0.05]])  # left wall, right wall, a follower directly behind
    left, right = c.body_side_room(xy)
    assert abs(left - 0.35) < 1e-6 and abs(right - 0.15) < 1e-6
