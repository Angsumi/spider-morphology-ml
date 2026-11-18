import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from prepare_data import (ID_COL, ROOT, TARGET, build, clean_species,  # noqa: E402
                          feature_columns, load_processed, parse_header, snake)
from reproduce_notebook import midpoint  # noqa: E402


@pytest.mark.parametrize("header, expected", [
    ("Body Length (BL, mm)", ("Body Length", "BL", "mm")),
    ("Prosoma Shape (PrS)", ("Prosoma Shape", "PrS", "")),
    ("Relative Abdomen Length (RAL, %)", ("Relative Abdomen Length", "RAL", "%")),
    ("Leg I/III Length Ratio (Leg I/III)", ("Leg I/III Length Ratio", "Leg I/III", "")),
    ("Species", ("Species", "", "")),
])
def test_parse_header(header, expected):
    p = parse_header(header)
    assert (p["label"], p["abbreviation"], p["unit"]) == expected


def test_snake_names():
    assert snake("Body Length", "mm") == "body_length_mm"
    assert snake("Relative Abdomen Length", "%") == "relative_abdomen_length_pct"
    assert snake("Leg I/III Length Ratio") == "leg_i_iii_length_ratio"


def test_clean_species_strips_whitespace():
    assert clean_species("  Hasarius   adansoni ") == "Hasarius adansoni"


@pytest.mark.parametrize("raw, expected", [
    ("10.3–10.7", 10.5), ("3-5", 4.0), ("7.2", 7.2), ("NR", np.nan), (None, np.nan), ("abc", np.nan),
])
def test_midpoint(raw, expected):
    out = midpoint(raw)
    if np.isnan(expected):
        assert np.isnan(out)
    else:
        assert out == pytest.approx(expected)


def _toy_raw():
    return pd.DataFrame({
        "m": ["S1", "S2", "S3"],
        "Species": [" Aus bus", "Aus bus", "Cus dus "],
        "Sternum Shape (StS)": ["Cordate", " Cordate ", "NR"],
        "Body Length (BL, mm)": [5.0, 5.5, 9.0],
    })


def test_build_renames_cleans_and_documents():
    df, dd = build(_toy_raw())
    assert list(df.columns) == [ID_COL, TARGET, "sternum_shape", "body_length_mm"]
    assert df[TARGET].tolist() == ["Aus bus", "Aus bus", "Cus dus"]
    assert df["sternum_shape"].iloc[1] == "Cordate"
    assert pd.isna(df["sternum_shape"].iloc[2])          # "NR" -> missing
    row = dd.set_index("column").loc["body_length_mm"]
    assert row["abbreviation"] == "BL" and row["unit"] == "mm" and row["type"] == "numeric"
    assert dd.set_index("column").loc[ID_COL, "role"].startswith("identifier")


def test_specimen_id_is_never_a_feature():
    df, _ = build(_toy_raw())
    num, cat = feature_columns(df)
    assert ID_COL not in num + cat and TARGET not in num + cat
    assert num == ["body_length_mm"] and cat == ["sternum_shape"]


def test_duplicate_ids_rejected():
    raw = _toy_raw()
    raw.loc[2, "m"] = "S1"
    with pytest.raises(ValueError):
        build(raw)


PROCESSED = ROOT / "data" / "processed" / "spiders.csv"


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data not built")
def test_processed_table_shape_and_dictionary():
    df = load_processed()
    dd = pd.read_csv(ROOT / "data" / "data_dictionary.csv")
    assert df.shape == (1721, 57)
    assert df[TARGET].nunique() == 45
    assert df[ID_COL].is_unique
    assert list(dd["column"]) == list(df.columns)
    assert (df[TARGET] == df[TARGET].str.strip()).all()
