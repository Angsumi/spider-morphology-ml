"""Diagnostics on how the specimen table relates to species-level descriptions.

Answers three questions and writes results/data_diagnostics.json:
1. How many predictors take one single value within every species, and how many
   of the 45 species can be told apart by those species-constant predictors alone?
2. How much do size measurements vary within species compared with between them?
3. (legacy v1 table) For each character, what share of the 1721 rows repeat the
   text of the 45-row species-description table word for word?

Usage:
    python src/data_diagnostics.py
"""
from __future__ import annotations

import json

import pandas as pd

from prepare_data import ROOT, TARGET, feature_columns, load_processed

LEGACY_DIR = ROOT / "data" / "raw" / "legacy"


def main() -> None:
    df = load_processed()
    num, cat = feature_columns(df)
    feats = num + cat
    n_per_sp = df.groupby(TARGET)[feats].nunique(dropna=False)
    constant = [c for c in feats if (n_per_sp[c] <= 1).all()]
    profiles = df.groupby(TARGET)[constant].first().astype(str)
    n_distinct_profiles = int(profiles.drop_duplicates().shape[0])
    sizes = ["body_length_mm", "prosoma_length_mm", "abdomen_length_mm", "total_leg_length_i_mm"]
    size_stats = {}
    for c in sizes:
        g = df.groupby(TARGET)[c]
        within_cv = (g.std() / g.mean()).median()
        size_stats[c] = {
            "median_within_species_cv": round(float(within_cv), 4),
            "cv_of_species_means": round(float(g.mean().std() / g.mean().mean()), 4),
        }

    legacy = {}
    v0 = pd.read_excel(LEGACY_DIR / "spider_species_descriptions_v0.xlsx")
    v1 = pd.read_excel(LEGACY_DIR / "spider_dataset_for_ml_v1.xlsx")
    v0.columns = v1.columns
    for d in (v0, v1):
        d["Species"] = d["Species"].astype(str).str.strip()
    text_cols = [c for c in v1.columns if c != "Species" and v1[c].dtype == object
                 and "(mm)" not in c]
    for c in text_cols:
        ref = dict(zip(v0["Species"], v0[c].astype(str).str.strip()))
        legacy[c] = round(float((v1[c].astype(str).str.strip() == v1["Species"].map(ref)).mean()), 3)

    out = {
        "canonical_v3": {
            "n_predictors": len(feats),
            "predictors_constant_within_every_species": constant,
            "n_predictors_constant_within_every_species": len(constant),
            "n_species": int(df[TARGET].nunique()),
            "n_species_with_distinct_profile_on_constant_predictors": n_distinct_profiles,
            "size_variation": size_stats,
        },
        "legacy_v1_share_rows_matching_species_description_text": legacy,
        "legacy_v1_n_rows": int(len(v1)),
        "legacy_v0_n_rows": int(len(v0)),
    }
    path = ROOT / "results" / "data_diagnostics.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["canonical_v3"], indent=2))
    exact = sum(v == 1.0 for v in legacy.values())
    print(f"legacy v1: {exact}/{len(legacy)} text characters copy the species description in every row")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
