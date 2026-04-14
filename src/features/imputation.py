import logging
from typing import Any, List, Optional
import pandas as pd

from src.utils import validate_columns

logger = logging.getLogger(__name__)


def _impute_zero_features(df: pd.DataFrame, custom_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Imputa con 0 columnas seleccionadas.

    Parameters
    ----------
    df : pd.DataFrame
    custom_cols : list[str], optional
        Lista de columnas a imputar con 0. 
        Si es None, usa las listas predefinidas (backward compatibility).
    """
    df = df.copy()

    if custom_cols is not None:
        zero_impute_cols = [c for c in custom_cols if c in df.columns]
    else:
        # Listas predefinidas por defecto
        distinct_cols = [
            "monthly_distinct_ads",
            "monthly_distinct_ads_month1",
            "monthly_distinct_ads_month2",
            "monthly_distinct_ads_month3",
        ]
        invoice_cols = [
            "monthly_total_invoice_month1",
            "monthly_total_invoice_month2",
            "monthly_total_invoice_month3",
        ]
        ratio_cols = [
            "usage_ratio",
            "cost_per_lead",
            "conversion_rate",
            "premium_ratio",
        ]
        zero_impute_cols = [
            c for c in distinct_cols + invoice_cols + ratio_cols if c in df.columns
        ]

    n_missing_before = (
        int(df[zero_impute_cols].isna().sum().sum()) if zero_impute_cols else 0
    )

    if zero_impute_cols:
        df[zero_impute_cols] = df[zero_impute_cols].fillna(0)

    logger.info(
        "impute_zero_features: imputadas %s columnas con 0 (%s nulos totales antes de imputar).",
        len(zero_impute_cols),
        n_missing_before,
    )

    return df


def _add_avg_price_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea flags de missing para columnas de precio medio por anuncio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a nivel contrato con features ya construidas.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas `<col>_was_missing` añadidas para las columnas
        de precio medio presentes en el dataframe.
    """
    df = df.copy()

    avg_price_cols = [
        "monthly_avg_ad_price",
        "monthly_avg_ad_price_month1",
        "monthly_avg_ad_price_month2",
        "monthly_avg_ad_price_month3",
    ]
    avg_price_cols = [c for c in avg_price_cols if c in df.columns]

    missing_counts = {}

    for col in avg_price_cols:
        flag_col = f"{col}_was_missing"
        df[flag_col] = df[col].isna().astype(int)
        missing_counts[col] = int(df[flag_col].sum())

    logger.info(
        "add_avg_price_missing_flags: creados %s missing flags.",
        len(avg_price_cols),
    )
    logger.debug("Missing counts avg price: %s", missing_counts)

    return df


def prepare_model_features(
    df: pd.DataFrame, 
    impute_config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """
    Aplica transformaciones finales sobre features ya construidas.

    Parameters
    ----------
    df : pd.DataFrame
    impute_config : dict, optional
        Configuración de imputación. Keys:
        - 'zero_impute_cols': list[str] (columnas a rellenar con 0)
    """
    expected_cols = {
        "monthly_distinct_ads",
        "monthly_total_invoice_month1",
        "usage_ratio",
        "monthly_avg_ad_price",
    }
    validate_columns(df, expected_cols, "prepare_model_features")

    df = df.copy()
    
    impute_cfg = impute_config or {}
    df = _impute_zero_features(df, custom_cols=impute_cfg.get("zero_impute_cols"))
    df = _add_avg_price_missing_flags(df)

    logger.info(
        "prepare_model_features: dataset final preparado para modelado con %s filas y %s columnas.",
        df.shape[0],
        df.shape[1],
    )

    return df