"""Fast end-to-end check of the evaluation loop on a small sample."""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from prepare_data import ROOT, TARGET, feature_columns, load_processed  # noqa: E402
from train_evaluate import evaluate, sample_per_species  # noqa: E402

PROCESSED = ROOT / "data" / "processed" / "spiders.csv"


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data not built")
def test_evaluate_on_sample():
    df = sample_per_species(load_processed(), 4, seed=0)
    assert df[TARGET].value_counts().max() == 4
    num, cat = feature_columns(df)
    folds, oof, labels, y = evaluate(df, ["Logistic Regression", "Decision Tree"],
                                     {"all": (num, cat)}, folds=2, seed=0, n_jobs=1)
    assert len(labels) == 45
    assert set(folds["model"]) == {"Logistic Regression", "Decision Tree"}
    assert folds["macro_f1"].between(0, 1).all()
    assert len(oof[("all", "Logistic Regression")]) == len(df)
