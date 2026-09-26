"""Phase 5b -- LightGBM binary classifier: candidate pair -> match probability.

LightGBM is primary per the plan's reconciliation of external reviews (faster
histogram-based CPU training/lower memory than XGBoost at this scale); both
satisfy the MIT/Apache-2.0, <=8B-param constraint trivially since we are
training our own model from scratch, not redistributing a pretrained one.

`lightgbm` is imported lazily inside each function, not at module level --
see the matching note in blocking.py: under Windows `spawn`, worker processes
re-import this whole module just to reconstruct the __main__ import graph,
even though workers never call any function in this file. A module-level
`import lightgbm` means every worker pays that import for nothing. Safe here
because `from __future__ import annotations` makes the `lgb.Booster` type
hints lazy strings, never evaluated at runtime.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import config

if TYPE_CHECKING:
    import lightgbm as lgb

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "bagging_freq": 1,  # without it LightGBM ignores subsample
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": config.RANDOM_SEED,
    "verbosity": -1,
}


def train(
    X: pd.DataFrame,
    y: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 20,
) -> "lgb.Booster":
    import lightgbm as lgb

    merged_params = {**DEFAULT_PARAMS, **(params or {})}
    train_set = lgb.Dataset(X, label=y, feature_name=list(X.columns))
    valid_sets, valid_names = [], []  # no train-set logloss every round (time only)
    callbacks = [lgb.log_evaluation(period=0)]
    if X_val is not None and y_val is not None:
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=list(X.columns))
        valid_sets.append(val_set)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    booster = lgb.train(
        merged_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets or None,
        valid_names=valid_names or None,
        callbacks=callbacks,
    )
    return booster


def predict_proba(booster: "lgb.Booster", X: pd.DataFrame) -> np.ndarray:
    best_iter = booster.best_iteration if booster.best_iteration > 0 else None
    return booster.predict(X, num_iteration=best_iter)


def save_model(booster: "lgb.Booster", path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load_model(path: str | Path) -> "lgb.Booster":
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


# Seed-bagged ensemble (idea from a teammate's branch): members train on the same rows and differ only in
# random_state, i.e. in every bagging / feature-subsample draw; averaging their probabilities smooths
# split-point noise. n_members=1 is the plain single model.
DEFAULT_ENSEMBLE_SEEDS = (config.RANDOM_SEED, 2024, 7)


def train_ensemble(X, y, X_val=None, y_val=None, params: dict | None = None, num_boost_round: int = 5000,
                   early_stopping_rounds: int = 50, n_members: int = 3,
                   seeds: tuple = DEFAULT_ENSEMBLE_SEEDS) -> list:
    seeds = (list(seeds) * n_members)[:max(1, n_members)]
    return [train(X, y, X_val, y_val, {**(params or {}), "random_state": int(s)},
                  num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds) for s in seeds]


def predict_proba_ensemble(models: list, X: pd.DataFrame) -> np.ndarray:
    """Mean probability over members."""
    return np.mean([predict_proba(b, X) for b in models], axis=0)


def save_models(models: list, path_stem: str | Path) -> None:
    """One member: <stem>.txt (the single-model layout); several: <stem>_0.txt, <stem>_1.txt, ..."""
    if len(models) == 1:
        save_model(models[0], f"{path_stem}.txt")
    else:
        for i, b in enumerate(models):
            save_model(b, f"{path_stem}_{i}.txt")


def load_models(path_stem: str | Path, n_members: int = 1) -> list:
    if n_members <= 1:
        return [load_model(f"{path_stem}.txt")]
    return [load_model(f"{path_stem}_{i}.txt") for i in range(n_members)]


def feature_importance(booster: "lgb.Booster") -> pd.Series:
    imp = pd.Series(
        booster.feature_importance(importance_type="gain"), index=booster.feature_name()
    )
    return imp.sort_values(ascending=False)
