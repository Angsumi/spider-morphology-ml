"""Shared model and preprocessing definitions.

Hyperparameters follow the original notebook
(notebooks/Spider_ML_complete_with_RF_interpretability.ipynb):
  - numeric: median imputation + standard scaling
  - categorical: most-frequent imputation + one-hot (unknown levels ignored)
  - Decision tree: grid over max_depth x min_samples_leaf (5-fold, accuracy),
    run inside each outer fold here so the outer score is not tuned on its own test data
  - KNN k=5; logistic regression max_iter=3000; random forest 300 trees; RBF SVM
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42

MODEL_NAMES = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers)


def make_classifier(name: str, seed: int = SEED, n_jobs: int = 1):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000, random_state=seed)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=n_jobs)
    if name == "SVM":
        return SVC(kernel="rbf", random_state=seed)
    raise ValueError(name)


def make_model(name: str, numeric: list[str], categorical: list[str],
               seed: int = SEED, n_jobs: int = 1, inner_folds: int = 5,
               inner_shuffle: bool = True):
    pipe = Pipeline([
        ("preprocessor", make_preprocessor(numeric, categorical)),
        ("classifier", make_classifier(name, seed, n_jobs)),
    ])
    if name == "Decision Tree":
        grid = {
            "classifier__max_depth": [None, 10, 15, 20, 25],
            "classifier__min_samples_leaf": [1, 2, 3, 5],
        }
        return GridSearchCV(
            pipe, grid, scoring="accuracy", n_jobs=n_jobs,
            cv=(StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
                if inner_shuffle else inner_folds),
        )
    return pipe
