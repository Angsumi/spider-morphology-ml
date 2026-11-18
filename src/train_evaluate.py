"""Stratified k-fold evaluation of five classifiers on the specimen table.

For each feature set (all characters, categorical only, numeric only) and each
model, out-of-fold predictions are collected with StratifiedKFold
(shuffle, fixed seed). Writes to results/:

  metrics_summary.csv        mean and SD over folds: accuracy, balanced accuracy, macro-F1
  metrics_folds.csv          per-fold scores
  per_species_f1.csv         out-of-fold F1 per species (feature set "all")
  confusion_<model>.csv      out-of-fold confusion counts (feature set "all")
  metrics.json               everything above plus run settings
and figures/:
  confusion_<best model>.png, per_species_f1.png

Usage:
    python src/train_evaluate.py                 # full run
    python src/train_evaluate.py --sample 8 --folds 3 --out-dir /tmp/x   # quick check (CI)
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from models import MODEL_NAMES, SEED, make_model
from prepare_data import ROOT, TARGET, feature_columns, load_processed


def sample_per_species(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return (df.groupby(TARGET, group_keys=False)
              .apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
              .reset_index(drop=True))


def evaluate(df: pd.DataFrame, models: list[str], feature_sets: dict[str, tuple[list, list]],
             folds: int, seed: int, n_jobs: int):
    y = df[TARGET].to_numpy()
    labels = np.sort(np.unique(y))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    splits = list(cv.split(df, y))
    fold_rows, oof = [], {}
    for fs_name, (num, cat) in feature_sets.items():
        X = df[num + cat]
        for name in models:
            t0 = time.time()
            pred = np.empty(len(y), dtype=object)
            for k, (tr, te) in enumerate(splits):
                model = make_model(name, num, cat, seed=seed, n_jobs=n_jobs,
                                   inner_folds=min(5, folds))
                model.fit(X.iloc[tr], y[tr])
                p = model.predict(X.iloc[te])
                pred[te] = p
                fold_rows.append({
                    "feature_set": fs_name, "model": name, "fold": k,
                    "accuracy": accuracy_score(y[te], p),
                    "balanced_accuracy": balanced_accuracy_score(y[te], p),
                    "macro_f1": f1_score(y[te], p, average="macro", labels=labels, zero_division=0),
                })
            oof[(fs_name, name)] = pred
            print(f"[{fs_name:11s}] {name:20s} macro-F1(OOF)="
                  f"{f1_score(y, pred, average='macro', zero_division=0):.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    return pd.DataFrame(fold_rows), oof, labels, y


def plot_confusion(cm: np.ndarray, labels, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(13, 11.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7, fontstyle="italic")
    ax.set_yticks(range(len(labels)), labels, fontsize=7, fontstyle="italic")
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Proportion of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_per_species(per_sp: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = per_sp.mean(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(8, 12))
    for col, marker in zip(per_sp.columns, "osD^v"):
        ax.scatter(per_sp.loc[order, col], range(len(order)), label=col, marker=marker, s=22, alpha=0.8)
    ax.set_yticks(range(len(order)), order, fontsize=7, fontstyle="italic")
    ax.set_xlabel("Out-of-fold F1")
    ax.set_xlim(per_sp.min().min() - 0.02, 1.01)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=5, fontsize=8, frameon=False)
    ax.set_title("Per-species F1, all characters, stratified CV")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "spiders.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sample", type=int, default=0, help="max specimens per species (0 = all)")
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    args = ap.parse_args()

    df = load_processed(args.data)
    if args.sample:
        df = sample_per_species(df, args.sample, args.seed)
    num, cat = feature_columns(df)
    feature_sets = {"all": (num, cat), "categorical": ([], cat), "numeric": (num, [])}

    folds_df, oof, labels, y = evaluate(df, args.models, feature_sets, args.folds, args.seed, args.n_jobs)

    out, figs = args.out_dir, args.fig_dir
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    summary = (folds_df.groupby(["feature_set", "model"])
               [["accuracy", "balanced_accuracy", "macro_f1"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    fs_order = {k: i for i, k in enumerate(feature_sets)}
    summary = summary.sort_values(["feature_set", "macro_f1_mean"],
                                  key=lambda s: s.map(fs_order) if s.name == "feature_set" else -s,
                                  ).reset_index(drop=True)
    summary.round(4).to_csv(out / "metrics_summary.csv", index=False)
    folds_df.round(4).to_csv(out / "metrics_folds.csv", index=False)

    per_sp = pd.DataFrame({
        name: f1_score(y, oof[("all", name)], average=None, labels=labels, zero_division=0)
        for name in args.models
    }, index=pd.Index(labels, name="species"))
    counts = pd.Series(y).value_counts().reindex(labels)
    per_sp_out = per_sp.round(4)
    per_sp_out.insert(0, "n_specimens", counts.values)
    per_sp_out.to_csv(out / "per_species_f1.csv")

    for name in args.models:
        cm = confusion_matrix(y, oof[("all", name)], labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            out / f"confusion_{name.lower().replace(' ', '_')}.csv")

    all_rows = summary[summary.feature_set == "all"]
    best = all_rows.iloc[0]["model"]
    worst = all_rows.iloc[-1]["model"]
    for name in dict.fromkeys([best, worst]):
        cm = confusion_matrix(y, oof[("all", name)], labels=labels)
        plot_confusion(cm, labels, f"{name}: out-of-fold confusion ({args.folds}-fold CV, all characters)",
                       figs / f"confusion_{name.lower().replace(' ', '_')}.png")
    plot_per_species(per_sp, figs / "per_species_f1.png")

    meta = {
        "data": str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data),
        "n_specimens": int(len(df)), "n_species": int(len(labels)),
        "n_numeric_features": len(num), "n_categorical_features": len(cat),
        "excluded_columns": ["specimen_id (assigned in species order; would leak the label)"],
        "cv": f"StratifiedKFold(n_splits={args.folds}, shuffle=True, random_state={args.seed})",
        "decision_tree_tuning": "GridSearchCV inside each outer training fold",
        "sample_per_species": args.sample or None,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "summary": summary.round(4).to_dict(orient="records"),
        "per_species_f1": per_sp.round(4).reset_index().to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
