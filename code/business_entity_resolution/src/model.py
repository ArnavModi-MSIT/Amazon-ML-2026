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
    valid_sets = [train_set]
    valid_names = ["train"]
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
        valid_sets=valid_sets,
        valid_names=valid_names,
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


def feature_importance(booster: "lgb.Booster") -> pd.Series:
    imp = pd.Series(
        booster.feature_importance(importance_type="gain"), index=booster.feature_name()
    )
    return imp.sort_values(ascending=False)
