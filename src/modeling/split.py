import logging
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

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
    "province_id"
]

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
    leakage_cols: List[str] | None = None,
    include_missing_flags: bool = True,
) -> List[str]:
    """
    Identifica columnas de features excluyendo meta, target y leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrada a nivel contrato.
    target : str
        Nombre de la variable objetivo.
    extra_non_features : list[str], optional
        Columnas adicionales a excluir.
    leakage_cols : list[str], optional
        Columnas con leakage a excluir. Si no se proporcionan,
        se usan las definidas por defecto.
    include_missing_flags : bool, default=True
        Si es True, mantiene columnas terminadas en `_was_missing`.
        Si es False, las excluye del set de features.

    Returns
    -------
    list[str]
        Lista de columnas de entrada para el modelo.
    """
    exclude = set(DEFAULT_NON_FEATURE_COLS + [target])

    if leakage_cols is None:
        exclude.update(DEFAULT_LEAKAGE_COLS)
    else:
        exclude.update(leakage_cols)

    if extra_non_features:
        exclude.update(extra_non_features)

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("invoice_post_"):
            continue
        if not include_missing_flags and col.endswith("_was_missing"):
            continue

        feature_cols.append(col)

    return feature_cols

def split_by_advertiser(
    df: pd.DataFrame,
    target: str = "churned_5m",
    test_size: float = 0.2,
    random_state: int = 42,
    extra_non_features: List[str] | None = None,
    leakage_cols: List[str] | None = None,
    include_missing_flags: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split train/test agrupado por advertiser.
    Todos los contratos del mismo advertiser van al mismo set.
    """
    feature_cols = get_feature_cols(
        df,
        target=target,
        extra_non_features=extra_non_features,
        leakage_cols=leakage_cols,
        include_missing_flags=include_missing_flags,
    )

    y = df[target].astype(int)
    X = df[feature_cols].copy()
    groups = df["advertiser_zrive_id"]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_advs = set(groups.iloc[train_idx])
    test_advs = set(groups.iloc[test_idx])

    logger.info(
        "Split: train=%s, test=%s, features=%s, include_missing_flags=%s, "
        "advertisers train=%s, test=%s, overlap=%s",
        len(X_train),
        len(X_test),
        len(feature_cols),
        include_missing_flags,
        len(train_advs),
        len(test_advs),
        len(train_advs & test_advs),
    )

    return X_train, X_test, y_train, y_test