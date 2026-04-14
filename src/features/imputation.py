import logging
from typing import Any, List, Optional
import pandas as pd

from src.utils import validate_columns

logger = logging.getLogger(__name__)


def _impute_zero_features(
    df: pd.DataFrame,
    custom_cols: list[str] | None = None,
    mode: str = "default",
) -> pd.DataFrame:
    """
    Imputa con 0 columnas seleccionadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    custom_cols : list[str], optional
        Lista de columnas personalizadas a imputar con 0.
    mode : {"default", "custom", "default_plus_custom"}, default="default"
        Estrategia para seleccionar columnas a imputar:
        - "default": usa solo las columnas por defecto
        - "custom": usa solo `custom_cols`
        - "default_plus_custom": combina columnas por defecto y `custom_cols`

    Returns
    -------
    pd.DataFrame
        DataFrame con la imputación aplicada.
    """
    df = df.copy()

    if mode not in {"default", "custom", "default_plus_custom"}:
        raise ValueError(
            "mode debe ser 'default', 'custom' o 'default_plus_custom'."
        )

    # Columnas por defecto parametrizadas por patrón
    default_distinct_cols = [
        c
        for c in df.columns
        if c == "monthly_distinct_ads" or c.startswith("monthly_distinct_ads_month")
    ]
    default_invoice_cols = [
        c
        for c in df.columns
        if c.startswith("monthly_total_invoice")
    ]
    default_ratio_cols = [
        c
        for c in [
            "usage_ratio",
            "cost_per_lead",
            "conversion_rate",
            "premium_ratio",
        ]
        if c in df.columns
    ]

    default_cols = default_distinct_cols + default_invoice_cols + default_ratio_cols
    custom_cols_valid = [c for c in (custom_cols or []) if c in df.columns]

    if mode == "default":
        zero_impute_cols = default_cols
        label = "Columnas imputadas por defecto"
    elif mode == "custom":
        zero_impute_cols = custom_cols_valid
        label = "Columnas imputadas personalizadas"
    else:  # default_plus_custom
        zero_impute_cols = list(dict.fromkeys(default_cols + custom_cols_valid))
        label = "Columnas imputadas por defecto + personalizadas"

    n_missing_before = (
        int(df[zero_impute_cols].isna().sum().sum()) if zero_impute_cols else 0
    )

    if zero_impute_cols:
        df[zero_impute_cols] = df[zero_impute_cols].fillna(0)

    logger.info(
        "impute_zero_features: imputadas %s columnas con 0 (%s nulos totales antes de imputar). %s: %s",
        len(zero_impute_cols),
        n_missing_before,
        label,
        zero_impute_cols,
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