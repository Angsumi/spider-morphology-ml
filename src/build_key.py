"""Export the interactive identification key data to docs/key-data.json.

Questions and answer options come from the `Website_Questions` sheet of the
canonical workbook. For each question, the species that can show each answer
are taken from the sheet the question's "Data column" lives in:

  - `Species_Profiles` (questions 1-39): one row per species, states separated
    by " | ", measurements as "min–max" ranges
  - `Sheet1` (questions 40-62): per-specimen values, aggregated to the set of
    states / the min–max range observed in each species

"Calculated automatically" questions (ratios) are not asked in the app.
Options without a label (blank cells) and "Not sure / skip" are dropped; the
app always offers its own skip.

Usage:
    python src/build_key.py
"""
from __future__ import annotations

import json
import re

import pandas as pd

from prepare_data import RAW_DEFAULT, ROOT

OUT = ROOT / "docs" / "key-data.json"
SKIP = "Not sure / skip"


def parse_range(v):
    if pd.isna(v):
        return None
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", str(v))]
    if not nums:
        return None
    return [min(nums), max(nums)]


def unit_of(answer_type: str) -> str:
    m = re.search(r"\(([^)]+)\)", str(answer_type))
    return m.group(1) if m else ""


def build() -> dict:
    w = pd.read_excel(RAW_DEFAULT, sheet_name="Website_Questions")
    w["Q#"] = w["Q#"].ffill()
    prof = pd.read_excel(RAW_DEFAULT, sheet_name="Species_Profiles")
    prof["Species"] = prof["Species"].str.strip()
    prof = prof.set_index("Species")
    spec = pd.read_excel(RAW_DEFAULT, sheet_name="Sheet1")
    spec["Species"] = spec["Species"].str.strip()
    species = sorted(prof.index)
    if sorted(spec["Species"].unique()) != species:
        raise ValueError("species differ between Species_Profiles and Sheet1")

    questions, problems = [], []
    for q, g in w.groupby("Q#", sort=False):
        h = g.iloc[0]
        col, atype = str(h["Data column"]), str(h["Answer type"])
        if atype.startswith("Calculated"):
            continue
        if col in prof.columns:
            source = "Species_Profiles"
        elif col in spec.columns:
            source = "Sheet1"
        else:
            problems.append(f"Q{int(q)}: column {col!r} not found")
            continue
        item = {"id": int(q), "group": str(h["Group"]), "text": str(h["Question"]),
                "level": str(h["Level"]), "source": source}

        if atype == "Choose one":
            labels = [str(o).strip() for o in g["Option"] if pd.notna(o) and str(o).strip() != SKIP]
            if source == "Species_Profiles":
                sets = {sp: {s.strip() for s in str(v).split("|")} if pd.notna(v) else set()
                        for sp, v in prof[col].items()}
            else:
                sets = spec.dropna(subset=[col]).groupby("Species")[col].apply(
                    lambda s: {str(v).strip() for v in s}).to_dict()
            states = {sp: [i for i, lab in enumerate(labels) if lab in sets.get(sp, set())]
                      for sp in species}
            unmatched = [lab for i, lab in enumerate(labels)
                         if not any(i in v for v in states.values())]
            if unmatched:
                problems.append(f"Q{int(q)}: options with no species: {unmatched}")
            item.update(kind="choice", options=labels,
                        n_species=[sum(i in v for v in states.values()) for i in range(len(labels))],
                        states={sp: v for sp, v in states.items() if v})
        else:
            if source == "Species_Profiles":
                ranges = {sp: parse_range(v) for sp, v in prof[col].items()}
            else:
                agg = spec.groupby("Species")[col].agg(["min", "max"])
                ranges = {sp: [float(r["min"]), float(r["max"])] for sp, r in agg.iterrows()
                          if pd.notna(r["min"])}
            ranges = {sp: r for sp, r in ranges.items() if r is not None}
            lo = min(r[0] for r in ranges.values())
            hi = max(r[1] for r in ranges.values())
            item.update(kind="number", unit=unit_of(atype), min=lo, max=hi,
                        integer=atype == "Number" and all(float(v).is_integer()
                                                          for r in ranges.values() for v in r),
                        ranges=ranges)
        questions.append(item)

    return {
        "title": "Spider identification key",
        "credit": "Morphological data measured by Mimangsa Darshan Chakraborty (Mizoram University)",
        "license": "Data CC BY 4.0",
        "species": species,
        "questions": questions,
        "problems": problems,
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    n_choice = sum(q["kind"] == "choice" for q in data["questions"])
    print(f"{len(data['species'])} species, {len(data['questions'])} questions "
          f"({n_choice} choice, {len(data['questions']) - n_choice} numeric) -> {OUT}")
    for p in data["problems"]:
        print("WARN", p)


if __name__ == "__main__":
    main()
