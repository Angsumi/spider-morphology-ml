import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_key import OUT, build, parse_range  # noqa: E402


def test_parse_range():
    assert parse_range("5.03–7.25") == [5.03, 7.25]
    assert parse_range("2–4") == [2.0, 4.0]
    assert parse_range("3") == [3.0, 3.0]
    assert parse_range(None) is None


def test_key_data_is_consistent_and_current():
    data = build()
    assert len(data["species"]) == 45
    assert data["problems"] == []
    for q in data["questions"]:
        if q["kind"] == "choice":
            assert all(n > 0 for n in q["n_species"]), q["text"]
            assert all(0 <= i < len(q["options"]) for v in q["states"].values() for i in v)
        else:
            assert all(r[0] <= r[1] for r in q["ranges"].values())
    # The committed docs/key-data.json must match a fresh build.
    assert json.loads(OUT.read_text()) == json.loads(json.dumps(data, ensure_ascii=False))
