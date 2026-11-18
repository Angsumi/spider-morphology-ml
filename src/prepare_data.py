"""Build the tidy specimen table and data dictionary from the canonical workbook.

Input : data/raw/spider_dataset_v3_merged.xlsx  (sheet "Sheet1", 1721 rows x 57 columns)
Output: data/processed/spiders.csv
        data/data_dictionary.csv

Column headers in the workbook look like "Body Length (BL, mm)" or
"Prosoma Shape (PrS)". They are split into a snake_case name, the original
abbreviation and the unit. The specimen ID column "m" is kept as
`specimen_id` for joining (e.g. with the identification key) but is NOT a
predictor: IDs are assigned in species order and would leak the label.

Usage:
    python src/prepare_data.py [--raw PATH] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "data" / "raw" / "spider_dataset_v3_merged.xlsx"
SHEET = "Sheet1"

ID_COL = "specimen_id"
TARGET = "species"
# Columns that must never be used as predictors.
NON_FEATURES = [ID_COL, TARGET]

_HEADER_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<inner>[^()]*)\)\s*$")


def parse_header(header: str) -> dict:
    """Split 'Body Length (BL, mm)' -> name/abbreviation/unit."""
    header = str(header).strip()
    m = _HEADER_RE.match(header)
    if not m:
        return {"label": header, "abbreviation": "", "unit": ""}
    label = m.group("name").strip()
    parts = [p.strip() for p in m.group("inner").split(",")]
    abbr = parts[0]
    unit = parts[1] if len(parts) > 1 else ""
    return {"label": label, "abbreviation": abbr, "unit": unit}


def snake(label: str, unit: str = "") -> str:
    s = label.lower()
    s = s.replace("/", " ").replace("%", "pct")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if unit:
        u = {"mm": "mm", "%": "pct"}.get(unit, re.sub(r"[^a-z0-9]+", "_", unit.lower()))
        if not s.endswith("_" + u):
            s = f"{s}_{u}"
    return s


def clean_species(name: str) -> str:
    """Normalise whitespace in species names (e.g. leading spaces)."""
    return re.sub(r"\s+", " ", str(name)).strip()


def clean_text(value):
    if pd.isna(value):
        return np.nan
    v = re.sub(r"\s+", " ", str(value)).strip()
    if v == "" or v.upper() in {"NR", "NA", "N/A"}:
        return np.nan
    return v


def build(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (tidy table, data dictionary) from the raw Sheet1 frame."""
    rename, meta = {}, []
    for col in raw.columns:
        if str(col).strip() == "m":
            new, info = ID_COL, {"label": "Specimen ID", "abbreviation": "m", "unit": ""}
        elif str(col).strip() == "Species":
            new, info = TARGET, {"label": "Species", "abbreviation": "", "unit": ""}
        else:
            info = parse_header(col)
            new = snake(info["label"], info["unit"])
        if new in rename.values():
            raise ValueError(f"duplicate column name after cleaning: {new}")
        rename[col] = new
        meta.append({"column": new, "original_header": str(col), **info})

    df = raw.rename(columns=rename).copy()
    df[TARGET] = df[TARGET].map(clean_species)
    for c in df.columns:
        if c in (ID_COL, TARGET):
            continue
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(clean_text).astype(object)

    if df[ID_COL].duplicated().any():
        raise ValueError("specimen IDs are not unique")

    rows = []
    for m in meta:
        c = m["column"]
        s = df[c]
        if c == ID_COL:
            role, kind = "identifier (not a predictor)", "id"
        elif c == TARGET:
            role, kind = "target", "categorical"
        else:
            role = "predictor"
            kind = "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
        # Share of species in which this column takes a single value.
        if c in (ID_COL, TARGET):
            const_share = pd.NA
        else:
            const_share = round(float((df.groupby(TARGET)[c].nunique(dropna=True) <= 1).mean()), 3)
        rows.append({
            "column": c,
            "original_header": m["original_header"],
            "label": m["label"],
            "abbreviation": m["abbreviation"],
            "unit": m["unit"],
            "type": kind,
            "role": role,
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": int(s.isna().sum()),
            "share_species_constant": const_share,
            "min": s.min() if kind == "numeric" else pd.NA,
            "max": s.max() if kind == "numeric" else pd.NA,
        })
    return df, pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric and categorical predictor columns (excludes ID and target)."""
    feats = [c for c in df.columns if c not in NON_FEATURES]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]
    return num, cat


def load_processed(path: Path | None = None) -> pd.DataFrame:
    path = path or ROOT / "data" / "processed" / "spiders.csv"
    # Categorical columns stay object dtype with np.nan for missing values,
    # which is what scikit-learn's imputers expect (pd.NA is not supported).
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    raw = pd.read_excel(args.raw, sheet_name=SHEET)
    df, dd = build(raw)
    (args.out_dir / "processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "processed" / "spiders.csv", index=False)
    dd.to_csv(args.out_dir / "data_dictionary.csv", index=False)
    num, cat = feature_columns(df)
    print(f"rows={len(df)} species={df[TARGET].nunique()} numeric={len(num)} categorical={len(cat)}")
    print(f"wrote {args.out_dir / 'processed' / 'spiders.csv'} and {args.out_dir / 'data_dictionary.csv'}")


if __name__ == "__main__":
    main()
