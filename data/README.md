# Data

Morphological data collected by MD <!-- TODO: full name and affiliation -->. Licensed CC BY 4.0 (see `LICENSE`).

## Canonical source

`raw/spider_dataset_v3_merged.xlsx` is the single source for all analyses. It is a byte-identical copy of the
collaborator's `spider_dataset 3 merged.xlsx`. Its `Sheet1` is identical to the collaborator's `character data.xlsx`.

| Sheet | Rows × cols | Content |
|---|---|---|
| `Sheet1` | 1721 × 57 | Specimen × character matrix (input to `src/prepare_data.py`) |
| `Website_Questions` | 435 × 9 | Question/option table for an interactive identification key (one question per block; options on following rows) |
| `Species_Profiles` | 45 × 40 | Per-species ranges and character states |
| `Split_Characters` | 1721 × 20 | Compound characters split into atomic states per specimen |
| `*_Key` sheets | 14–45 rows | Per-character species summaries (prosoma, carapace, abdomen, folium, chelicera, eyes, legs, scopula, sternum, epigyne) |
| `Measurement_Sources` | 45 × 7 | Per-species mean/SD of body and prosoma size |

`Website_Questions_ADDED.xlsx` (107 extra questions) from the collaborator's folder is **not** included yet.
It belongs to the identification-key work.

## Derived files

- `processed/spiders.csv`: `Sheet1` with snake_case column names, whitespace-normalised text, "NR"/blank → missing.
  `specimen_id` is an identifier, never a predictor.
- `data_dictionary.csv`: one row per column. `share_species_constant` is the fraction of the 45 species in which
  the column takes a single value. 1.0 means it is a species-level trait in this table.

Rebuild with `python src/prepare_data.py`.

## Legacy

- `raw/legacy/spider_dataset_for_ml_v1.xlsx`: the 1721 × 38 table used by the original notebook.
- `raw/legacy/spider_species_descriptions_v0.xlsx`: 45 × 38 species-level descriptions. v1 is an expansion of this table (see `results/data_diagnostics.json`).
- `raw/legacy/earlier_drafts/`: earlier, differently structured drafts (1885–2003 rows with male/female columns; a 46-row taxonomy table). They are not used by any script and are kept for provenance.
