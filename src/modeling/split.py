import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

LEAK_PATTERNS = ["stable_price", "invoice_post_month", "n_months_post_onboarding"]
NON_FEATURE_COLS = ["churned_3m", "contract_duration", "advertiser_zrive_id"]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Devuelve las columnas que se usan como features, excluyendo
    target, leakage (datos post-onboarding) e identificadores.
    """
    exclude = set(NON_FEATURE_COLS)
    for col in df.columns:
        for pattern in LEAK_PATTERNS:
            if pattern in col:
                exclude.add(col)
    return [c for c in df.columns if c not in exclude]


def split_by_advertiser(
    df: pd.DataFrame,
    target: str = "churned_3m",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split train/test agrupado por advertiser.

    Todos los contratos de un mismo advertiser van al mismo set
    para evitar leakage por advertiser.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a nivel contrato con features y target.
    target : str
        Nombre de la columna target.
    test_size : float
        Proporcion del test set.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    feature_cols = get_feature_cols(df)

    y = df[target].astype(int)
    X = df[feature_cols].fillna(0)
    groups = df["advertiser_zrive_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_advs = set(groups.iloc[train_idx])
    test_advs = set(groups.iloc[test_idx])

    logger.info(
        "Split: train=%s, test=%s, features=%s, "
        "advertisers train=%s, test=%s, overlap=%s",
        len(X_train), len(X_test), len(feature_cols),
        len(train_advs), len(test_advs), len(train_advs & test_advs),
    )

    return X_train, X_test, y_train, y_test
