import logging
from typing import List, Tuple, Any, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

# Meta-columnas que nunca deben entrar al modelo
DEFAULT_NON_FEATURE_COLS = [
    "advertiser_zrive_id",
    "contract_id",
    "period_int",
    "contract_start_date",
    "contract_end_period",
    "contrato_churn_date",
    "contract_duration_months",
    "advertiser_group_id",
    "advertiser_province",
    "is_right_censored",
    "right_censoring_case",
    "province_id",
    "region",
]

# Variables con leakage (usan datos post-onboarding)
DEFAULT_LEAKAGE_COLS = [
    "n_months_post_onboarding",
    "stable_price_mean",
    "stable_price_median",
    "stable_price_std",
]


def get_feature_cols(
    df: pd.DataFrame,
    target: str,
    extra_non_features: List[str] | None = None,
) -> List[str]:
    """
    Identifica columnas de features excluyendo meta, target, leakage y flags.

    Parameters
    ----------
    extra_non_features : list[str], optional
        Columnas adicionales a excluir.
    """
    exclude = set(DEFAULT_NON_FEATURE_COLS + DEFAULT_LEAKAGE_COLS + [target])
    if extra_non_features:
        exclude.update(extra_non_features)

    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and not c.endswith("_was_missing")
        and not c.startswith("invoice_post_")
    ]

    return feature_cols


def split_by_advertiser(
    df: pd.DataFrame,
    target: str = "churned_5m",
    test_size: float = 0.2,
    random_state: int = 42,
    extra_non_features: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split train/test agrupado por advertiser.
    Todos los contratos del mismo advertiser van al mismo set.
    """
    feature_cols = get_feature_cols(
        df,
        target,
        extra_non_features=extra_non_features,
    )

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
