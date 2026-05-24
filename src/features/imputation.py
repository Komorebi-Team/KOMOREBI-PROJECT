import logging
from typing import Any

import pandas as pd

from src.utils import validate_columns

logger = logging.getLogger(__name__)


def _add_missing_flag(
    df: pd.DataFrame,
    col: str,
    flag_suffix: str = "_was_missing",
) -> pd.DataFrame:
    """
    Añade una flag binaria indicando si una columna estaba originalmente en missing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    col : str
        Columna sobre la que crear la flag.
    flag_suffix : str, default="_was_missing"
        Sufijo de la flag.

    Returns
    -------
    pd.DataFrame
        DataFrame con la flag añadida si la columna existe.
    """
    df = df.copy()

    if col not in df.columns:
        logger.info(
            "_add_missing_flag: la columna `%s` no existe, no se crea flag.",
            col,
        )
        return df

    flag_col = f"{col}{flag_suffix}"

    if flag_col in df.columns:
        logger.info(
            "_add_missing_flag: la flag `%s` ya existe, no se sobreescribe.",
            flag_col,
        )
        return df

    df[flag_col] = df[col].isna().astype(int)

    logger.info(
        "_add_missing_flag: creada flag `%s` para `%s` (%s missings).",
        flag_col,
        col,
        int(df[flag_col].sum()),
    )

    return df


def _impute_monthly_distinct_ads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa con 0 las columnas relacionadas con `monthly_distinct_ads`.

    Se aplica sobre:
    - `monthly_distinct_ads`
    - `monthly_distinct_ads_month*` si existieran

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas imputadas.
    """
    df = df.copy()

    distinct_cols = [
        c
        for c in df.columns
        if c == "monthly_distinct_ads" or c.startswith("monthly_distinct_ads_month")
    ]

    if not distinct_cols:
        logger.info(
            "_impute_monthly_distinct_ads: no se encontraron columnas a imputar."
        )
        return df

    n_missing_before = int(df[distinct_cols].isna().sum().sum())
    df[distinct_cols] = df[distinct_cols].fillna(0)

    logger.info(
        "_impute_monthly_distinct_ads: imputadas %s columnas con 0 (%s nulos antes). Columnas: %s",
        len(distinct_cols),
        n_missing_before,
        distinct_cols,
    )

    return df


def _impute_monthly_total_reference_price(
    df: pd.DataFrame,
    group_col: str = "contract_id",
    add_missing_flag: bool = True,
) -> pd.DataFrame:
    """
    Imputa `monthly_total_reference_price` con lógica de negocio a nivel mensual.

    Reglas:
    1. Opcionalmente crea una flag de missing original.
    2. Si `monthly_total_reference_price` es NaN y no hay facturación
       (`monthly_total_invoice <= 0` o NaN), imputa 0.
    3. Si sigue en NaN, hay contrato activo y facturación positiva, intenta
       imputar con el último valor válido anterior dentro del mismo grupo.
    4. Si no puede imputarse, se mantiene NaN.

    Importante:
    El arrastre del valor previo solo usa como fuente observaciones con
    facturación positiva, para evitar propagar 0 imputados de meses sin billing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mensual.
    group_col : str, default="contract_id"
        Columna de agrupación para arrastrar el último valor válido anterior.
        Si no existe, se usará `advertiser_zrive_id`.
    add_missing_flag : bool, default=True
        Si es True, crea la flag `monthly_total_reference_price_was_missing`.

    Returns
    -------
    pd.DataFrame
        DataFrame con la imputación aplicada.
    """
    required_cols = {
        "advertiser_zrive_id",
        "period_int",
        "monthly_total_reference_price",
        "monthly_total_invoice",
        "has_active_contract",
    }
    validate_columns(
        df=df,
        required_cols=required_cols,
        func_name="_impute_monthly_total_reference_price",
    )

    df = df.copy()

    if group_col not in df.columns:
        logger.info(
            "_impute_monthly_total_reference_price: `%s` no existe, se usará `advertiser_zrive_id`.",
            group_col,
        )
        group_col = "advertiser_zrive_id"

    df = df.sort_values([group_col, "period_int"]).copy()

    n_missing_before = int(df["monthly_total_reference_price"].isna().sum())

    if add_missing_flag:
        df = _add_missing_flag(df, "monthly_total_reference_price")

    missing_mask_initial = df["monthly_total_reference_price"].isna()

    # Regla 1: si no hay facturación, imputar 0
    no_billing_mask = df["monthly_total_invoice"].fillna(0).le(0)
    fill_zero_mask = missing_mask_initial & no_billing_mask
    df.loc[fill_zero_mask, "monthly_total_reference_price"] = 0

    # Regla 2: si hay contrato activo y facturación positiva, arrastrar último valor válido anterior
    source_ref_price = df["monthly_total_reference_price"].where(
        df["monthly_total_invoice"].fillna(0).gt(0)
    )
    prev_valid_ref_price = source_ref_price.groupby(df[group_col]).ffill()

    missing_after_zero = df["monthly_total_reference_price"].isna()
    active_billed_mask = (
        df["has_active_contract"].fillna(False)
        & df["monthly_total_invoice"].fillna(0).gt(0)
    )
    fill_prev_mask = missing_after_zero & active_billed_mask & prev_valid_ref_price.notna()
    df.loc[fill_prev_mask, "monthly_total_reference_price"] = prev_valid_ref_price[fill_prev_mask]

    n_filled_zero = int(fill_zero_mask.sum())
    n_filled_prev = int(fill_prev_mask.sum())
    n_missing_after = int(df["monthly_total_reference_price"].isna().sum())

    logger.info(
        "_impute_monthly_total_reference_price: missing_before=%s | filled_zero=%s | "
        "filled_prev=%s | missing_after=%s | group_col=%s",
        n_missing_before,
        n_filled_zero,
        n_filled_prev,
        n_missing_after,
        group_col,
    )

    return df


def prepare_monthly_features_before_aggregation(
    df: pd.DataFrame,
    impute_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Prepara e imputa variables mensuales base antes de agregarlas a nivel contrato.

    Qué hace:
    - Imputa `monthly_distinct_ads` con 0
    - Imputa `monthly_total_reference_price` con lógica contextual
    - NO imputa `monthly_avg_ad_price`

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mensual a nivel contrato-periodo o advertiser-periodo.
    impute_config : dict, optional
        Configuración opcional. Keys:
        - 'group_col': str
        - 'add_reference_price_missing_flag': bool

    Returns
    -------
    pd.DataFrame
        DataFrame mensual preparado para agregación.
    """
    expected_cols = {
        "advertiser_zrive_id",
        "period_int",
        "monthly_distinct_ads",
        "monthly_total_invoice",
        "has_active_contract",
    }
    validate_columns(
        df=df,
        required_cols=expected_cols,
        func_name="prepare_monthly_features_before_aggregation",
    )

    df = df.copy()
    impute_cfg = impute_config or {}

    df = _impute_monthly_distinct_ads(df)

    if "monthly_total_reference_price" in df.columns:
        df = _impute_monthly_total_reference_price(
            df,
            group_col=impute_cfg.get("group_col", "contract_id"),
            add_missing_flag=impute_cfg.get("add_reference_price_missing_flag", True),
        )
    else:
        logger.info(
            "prepare_monthly_features_before_aggregation: no se encontró `monthly_total_reference_price`, se omite su imputación."
        )

    logger.info(
        "prepare_monthly_features_before_aggregation: dataset preparado con %s filas y %s columnas.",
        df.shape[0],
        df.shape[1],
    )

    return df


def _impute_zero_features(
    df: pd.DataFrame,
    custom_cols: list[str] | None = None,
    mode: str = "default",
) -> pd.DataFrame:
    """
    Imputa con 0 columnas seleccionadas del dataset ya agregado / listo para modelado.

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

        Las columnas por defecto son:
        - Todas las que sean `monthly_total_invoice*`
        - Todas las que sean `monthly_distinct_ads_month*`
        - `monthly_distinct_ads`
        - ratios base:
            `usage_ratio`, `cost_per_lead`, `conversion_rate`, `premium_ratio`
        - ratios output / invoice:
            `leads_per_invoice`, `visits_per_invoice`, `shows_per_invoice`,
            `published_ads_per_invoice`, `contracted_ads_per_invoice`,
            `unique_leads_per_invoice`, `calls_per_invoice`,
            `emails_per_invoice`, `phone_views_per_invoice`

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
            "unused_capacity_ratio",
            "distinct_ad_ratio",
            "cost_per_lead",
            "show_to_visit_rate",
            "visit_to_lead_rate",
            "show_to_lead_rate",
            "premium_ratio",
            "oro_ratio",
            "plata_ratio",
            "destacados_ratio",
            "shows_per_published_ad",
            "visits_per_published_ad",
            "leads_per_published_ad",
            "unique_leads_per_published_ad",
            "unique_lead_ratio",
            "phone_views_per_lead",
            "calls_per_lead",
            "emails_per_lead",
            "leads_per_invoice",
            "visits_per_invoice",
            "shows_per_invoice",
            "published_ads_per_invoice",
            "contracted_ads_per_invoice",
            "unique_leads_per_invoice",
            "calls_per_invoice",
            "emails_per_invoice",
            "phone_views_per_invoice",
            "premium_ads_per_invoice",
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
    else:
        zero_impute_cols = list(dict.fromkeys(default_cols + custom_cols_valid))
        label = "Columnas imputadas por defecto + personalizadas"

    n_missing_before = (
        int(df[zero_impute_cols].isna().sum().sum()) if zero_impute_cols else 0
    )

    if zero_impute_cols:
        df[zero_impute_cols] = df[zero_impute_cols].fillna(0)

    logger.info(
        "_impute_zero_features: imputadas %s columnas con 0 (%s nulos totales antes de imputar). %s: %s",
        len(zero_impute_cols),
        n_missing_before,
        label,
        zero_impute_cols,
    )

    return df


def _get_avg_price_cols(df: pd.DataFrame) -> list[str]:
    """
    Devuelve las columnas relacionadas con `monthly_avg_ad_price` presentes en el dataframe.
    """
    return [
        c
        for c in df.columns
        if c == "monthly_avg_ad_price" or c.startswith("monthly_avg_ad_price_month")
    ]


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
        DataFrame con columnas `<col>_was_missing` añadidas.
    """
    df = df.copy()

    avg_price_cols = _get_avg_price_cols(df)

    for col in avg_price_cols:
        df = _add_missing_flag(df, col)

    logger.info(
        "_add_avg_price_missing_flags: creados %s missing flags.",
        len(avg_price_cols),
    )

    return df


def _impute_avg_price_features(
    df: pd.DataFrame,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Imputa columnas de precio medio por anuncio tras crear sus missing flags.

    Esta imputación es técnica, pensada para dejar el dataset listo para modelado.
    Debe ejecutarse después de `_add_avg_price_missing_flags`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a nivel contrato con features ya construidas.
    fill_value : float, default=0.0
        Valor con el que rellenar los NaN.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas de precio medio imputadas.
    """
    df = df.copy()

    avg_price_cols = _get_avg_price_cols(df)

    if not avg_price_cols:
        logger.info(
            "_impute_avg_price_features: no se encontraron columnas de avg price para imputar."
        )
        return df

    n_missing_before = int(df[avg_price_cols].isna().sum().sum())
    df[avg_price_cols] = df[avg_price_cols].fillna(fill_value)

    logger.info(
        "_impute_avg_price_features: imputadas %s columnas con fill_value=%s (%s nulos antes). Columnas: %s",
        len(avg_price_cols),
        fill_value,
        n_missing_before,
        avg_price_cols,
    )

    return df


def prepare_model_features(
    df: pd.DataFrame,
    impute_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Aplica transformaciones finales sobre features ya agregadas y construidas.

    Qué hace:
    - Imputación técnica con 0 de columnas seleccionadas
    - Creación de missing flags para `monthly_avg_ad_price*`
    - Imputación técnica de `monthly_avg_ad_price*` tras crear los flags

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a nivel contrato con features ya construidas.
    impute_config : dict, optional
        Configuración opcional. Keys:
        - 'zero_impute_cols': list[str]
        - 'zero_impute_mode': {"default", "custom", "default_plus_custom"}
        - 'avg_price_fill_value': float

    Returns
    -------
    pd.DataFrame
        DataFrame final preparado para modelado.
    """
    df = df.copy()
    impute_cfg = impute_config or {}

    df = _impute_zero_features(
        df,
        custom_cols=impute_cfg.get("zero_impute_cols"),
        mode=impute_cfg.get("zero_impute_mode", "default"),
    )
    df = _add_avg_price_missing_flags(df)
    df = _impute_avg_price_features(
        df,
        fill_value=impute_cfg.get("avg_price_fill_value", 0.0),
    )

    logger.info(
        "prepare_model_features: dataset final preparado para modelado con %s filas y %s columnas.",
        df.shape[0],
        df.shape[1],
    )

    return df