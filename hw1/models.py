"""Model definitions, CV evaluation, and ensemble utilities."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ── model factories ───────────────────────────────────────────────────────────

def make_lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=255,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def make_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.03,
        max_leaf_nodes=255,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
        verbose=0,
    )


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )


# ── evaluation ────────────────────────────────────────────────────────────────

def cv_auc(model, X: pd.DataFrame, y: np.ndarray, cv: int = 3) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def _rank_average(oof_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Blend OOF predictions via weighted rank averaging (scale-invariant)."""
    from scipy.stats import rankdata
    ranked = [rankdata(p) for p in oof_list]
    wsum = sum(weights)
    return np.sum([w * r for w, r in zip(weights, ranked)], axis=0) / wsum


def ensemble_oof_auc(
    model_factories: list, X: pd.DataFrame, y: np.ndarray, cv: int = 3,
) -> tuple[float, list[float]]:
    """Proper OOF evaluation with AUC-weighted rank averaging.

    Returns (ensemble AUC, [per-model AUCs]).
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    oof_per_model = [np.zeros(len(y), dtype=np.float64) for _ in model_factories]

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]
        for i, factory in enumerate(model_factories):
            m = factory()
            m.fit(X_tr, y_tr)
            oof_per_model[i][va_idx] = m.predict_proba(X_va)[:, 1]

    per_model_auc = [roc_auc_score(y, p) for p in oof_per_model]

    blended_rank = _rank_average(oof_per_model, weights=per_model_auc)
    ensemble_auc = float(roc_auc_score(y, blended_rank))

    return ensemble_auc, per_model_auc


# ── ensemble ─────────────────────────────────────────────────────────────────

def train_and_predict_ensemble(
    models: list,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Train all models, return rank-averaged probability for X_test."""
    if weights is None:
        weights = [1.0] * len(models)

    preds = []
    for m in models:
        m.fit(X_train, y_train)
        preds.append(m.predict_proba(X_test)[:, 1])

    return _rank_average(preds, weights)
