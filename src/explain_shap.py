"""SHAP importance of morphological characters for random forest and logistic regression.

Both models are fitted on all specimens with all characters. SHAP values are
computed for a stratified subsample (default 5 specimens per species) to bound
memory: values have shape samples x one-hot columns x 45 classes.
|SHAP| is averaged over classes, summed over the one-hot columns that belong
to each original character, then averaged over specimens.

KNN and SVM need model-agnostic KernelExplainer, which is slow on 45 classes;
they are left out here (the original notebook computed them on 100 samples).

Writes results/shap_importance.csv and figures/shap_importance.png.

Usage:
    python src/explain_shap.py [--per-species 5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from models import SEED, make_model
from prepare_data import ROOT, TARGET, feature_columns, load_processed
from train_evaluate import sample_per_species


def column_groups(pre, numeric: list[str], categorical: list[str]) -> list[str]:
    """Original character name for every transformed column, in order."""
    owners = list(numeric)
    if categorical:
        enc = pre.named_transformers_["cat"].named_steps["encoder"]
        for col, cats in zip(categorical, enc.categories_):
            owners.extend([col] * len(cats))
    return owners


def to_3d(values, n_classes: int) -> np.ndarray:
    """Normalise SHAP output to (samples, features, classes)."""
    if isinstance(values, list):
        return np.stack(values, axis=-1)
    v = np.asarray(values)
    if v.ndim == 3 and v.shape[1] == n_classes and v.shape[2] != n_classes:
        v = v.transpose(0, 2, 1)
    return v


def aggregate(values: np.ndarray, owners: list[str]) -> pd.Series:
    per_feat = np.abs(values).mean(axis=2)          # samples x transformed columns
    frame = pd.DataFrame(per_feat, columns=owners)
    return frame.T.groupby(level=0, sort=False).sum().T.mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "spiders.csv")
    ap.add_argument("--per-species", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    args = ap.parse_args()

    df = load_processed(args.data)
    num, cat = feature_columns(df)
    X, y = df[num + cat], df[TARGET]
    sub = sample_per_species(df, args.per_species, SEED)
    importances = {}

    for name in ["Random Forest", "Logistic Regression"]:
        model = make_model(name, num, cat, n_jobs=-1).fit(X, y)
        pre, clf = model.named_steps["preprocessor"], model.named_steps["classifier"]
        owners = column_groups(pre, num, cat)
        Xs = pre.transform(sub[num + cat])
        Xs = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
        if name == "Random Forest":
            sv = shap.TreeExplainer(clf).shap_values(Xs, check_additivity=False)
        else:
            Xb = pre.transform(X)
            Xb = Xb.toarray() if hasattr(Xb, "toarray") else np.asarray(Xb)
            sv = shap.LinearExplainer(clf, shap.maskers.Independent(Xb, max_samples=200)).shap_values(Xs)
        importances[name] = aggregate(to_3d(sv, len(clf.classes_)), owners)
        print(f"{name}: SHAP done on {len(sub)} specimens, {Xs.shape[1]} transformed columns", flush=True)

    imp = pd.DataFrame(importances)
    imp.index.name = "character"
    norm = imp / imp.sum()
    out = imp.join(norm.add_suffix(" (share)")).sort_values("Random Forest (share)", ascending=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(args.out_dir / "shap_importance.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = norm.loc[out.index[:20]][::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = np.arange(len(top))
    ax.barh(pos + 0.2, top["Random Forest"], height=0.4, label="Random forest")
    ax.barh(pos - 0.2, top["Logistic Regression"], height=0.4, label="Logistic regression")
    ax.set_yticks(pos, top.index.str.replace("_", " "), fontsize=8)
    ax.set_xlabel("Share of total mean |SHAP| (one-hot columns summed per character)")
    ax.set_title(f"Top 20 characters by SHAP importance\n(subsample: {args.per_species} specimens per species)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_dir / "shap_importance.png", dpi=200)
    print(out.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
