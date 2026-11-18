"""Re-run the original notebook's evaluation on the legacy workbook.

The notebook (notebooks/Spider_ML_complete_with_RF_interpretability.ipynb) used
data/raw/legacy/spider_dataset_for_ml_v1.xlsx with:
  - range strings such as "10.3–10.7" converted to midpoints (3 size columns)
  - the constant column "Tarsal Claw Number" dropped
  - a stratified 80/20 train/test split (random_state=42)
  - hold-out test accuracy, plus 5-fold out-of-fold metrics on the training part

This script repeats that procedure in the pinned environment and writes
results/notebook_reproduction.csv/json next to the values printed in the
notebook's saved outputs, so differences are visible.

Usage:
    python src/reproduce_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from models import MODEL_NAMES, SEED, make_model
from prepare_data import ROOT

LEGACY = ROOT / "data" / "raw" / "legacy" / "spider_dataset_for_ml_v1.xlsx"
RANGE_COLS = ["Body length (mm)", "Prosoma Length (mm)", "Prosoma width (mm)"]

# Values printed in the notebook's saved cell outputs (cells 15 and 17).
NOTEBOOK_REPORTED = {
    "Decision Tree":       {"test_accuracy": 0.9826, "oof_accuracy": 0.9891, "oof_macro_f1": 0.9884},
    "KNN":                 {"test_accuracy": 0.8145, "oof_accuracy": 0.8321, "oof_macro_f1": 0.8207},
    "Logistic Regression": {"test_accuracy": 0.9884, "oof_accuracy": 0.9651, "oof_macro_f1": 0.9630},
    "Random Forest":       {"test_accuracy": 0.9884, "oof_accuracy": 0.9840, "oof_macro_f1": 0.9835},
    "SVM":                 {"test_accuracy": 0.9536, "oof_accuracy": 0.9397, "oof_macro_f1": 0.9281},
}


def midpoint(x):
    """Notebook's conversion of 'a–b' range strings to (a+b)/2."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace("–", "-").replace("—", "-")
    if x.upper() == "NR" or x == "":
        return np.nan
    if "-" in x:
        a, b = x.split("-", 1)
        try:
            return (float(a.strip()) + float(b.strip())) / 2
        except ValueError:
            return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def load_legacy(path: Path = LEGACY) -> pd.DataFrame:
    df = pd.read_excel(path)
    for c in RANGE_COLS:
        df[c] = df[c].apply(midpoint)
    if df["Tarsal Claw Number"].nunique(dropna=True) <= 1:
        df = df.drop(columns=["Tarsal Claw Number"])
    return df


def main() -> None:
    df = load_legacy()
    X, y = df.drop(columns=["Species"]), df["Species"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    num = X_tr.select_dtypes(include=np.number).columns.tolist()
    cat = X_tr.select_dtypes(exclude=np.number).columns.tolist()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows = []
    for name in MODEL_NAMES:
        # The notebook's GridSearchCV used cv=5 (unshuffled folds).
        m = make_model(name, num, cat, n_jobs=-1, inner_shuffle=False).fit(X_tr, y_tr)
        test_acc = accuracy_score(y_te, m.predict(X_te))
        est = m.best_estimator_ if hasattr(m, "best_estimator_") else make_model(name, num, cat, n_jobs=-1)
        oof = cross_val_predict(est, X_tr, y_tr, cv=cv, n_jobs=-1)
        rep = NOTEBOOK_REPORTED[name]
        rows.append({
            "model": name,
            "test_accuracy": round(test_acc, 4),
            "notebook_test_accuracy": rep["test_accuracy"],
            "oof_accuracy": round(accuracy_score(y_tr, oof), 4),
            "notebook_oof_accuracy": rep["oof_accuracy"],
            "oof_macro_f1": round(f1_score(y_tr, oof, average="macro", zero_division=0), 4),
            "notebook_oof_macro_f1": rep["oof_macro_f1"],
        })
        print(rows[-1], flush=True)

    # Diagnostic: how much do the three range-midpoint size columns alone reveal?
    size_only = make_model("Random Forest", RANGE_COLS, [], n_jobs=-1).fit(X_tr, y_tr)
    size_acc = accuracy_score(y_te, size_only.predict(X_te))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "notebook_reproduction.csv", index=False)
    (out / "notebook_reproduction.json").write_text(json.dumps({
        "data": str(LEGACY.relative_to(ROOT)),
        "n_specimens": int(len(df)), "n_species": int(y.nunique()),
        "split": "train_test_split(test_size=0.2, stratify=y, random_state=42); OOF = 5-fold on training part",
        "results": rows,
        "diagnostic_rf_three_size_midpoints_only_test_accuracy": round(size_acc, 4),
    }, indent=2))
    print(res.to_string(index=False))
    print(f"RF on the 3 size midpoints only, hold-out accuracy: {size_acc:.4f}")


if __name__ == "__main__":
    main()
