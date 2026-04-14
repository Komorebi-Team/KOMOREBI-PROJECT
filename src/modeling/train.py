import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from scipy.stats import uniform, randint

from src.utils import compute_class_sample_weight

logger = logging.getLogger(__name__)


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
    """Baseline: predecir siempre la clase mayoritaria."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> Pipeline:
    """Logistic regression con features escaladas y class_weight balanced."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
    ])
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> RandomForestClassifier:
    """Random forest con class_weight balanced."""
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> HistGradientBoostingClassifier:
    """Gradient boosting con sample_weight para compensar desbalanceo."""
    sample_weights = compute_class_sample_weight(y_train)

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def tune_gradient_boosting(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    groups: Optional[pd.Series] = None, 
    n_iter: int = 50,
    random_state: int = 42
) -> RandomizedSearchCV:
    """
    Tuning de hiperparametros de Gradient Boosting con RandomizedSearchCV.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Datos de entrenamiento.
    groups : pd.Series, optional
        Grupos para GroupKFold (advertiser_zrive_id).
    n_iter : int
        Numero de combinaciones a probar.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    RandomizedSearchCV
        Modelo tuneado.
    """
    sample_weights = compute_class_sample_weight(y_train)

    param_dist = {
        "max_iter": randint(100, 500),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.2),
        "min_samples_leaf": randint(10, 50),
        "max_leaf_nodes": randint(20, 60),
        "l2_regularization": uniform(0, 1),
    }

    cv = GroupKFold(n_splits=5) if groups is not None else 5

    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=random_state),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
    )

    fit_params = {"sample_weight": sample_weights}
    if groups is not None:
        search.fit(X_train, y_train, groups=groups, **fit_params)
    else:
        search.fit(X_train, y_train, **fit_params)

    logger.info(
        "Tuning completado: mejor ROC AUC (CV)=%.3f, params=%s",
        search.best_score_, search.best_params_,
    )

    return search

def train_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> Any:
    """XGBoost con scale_pos_weight para compensar desbalanceo."""
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        logger.error("XGBoost is not installed. Please install it using 'pip install xgboost'.")
        raise ImportError("XGBoost is not installed.") from e

    scale = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
