import logging

import numpy as np
import pandas as pd

from src.utils import validate_columns

logger = logging.getLogger(__name__)


BEHAVIOR_COLS = [
    "monthly_contracted_ads",
    "monthly_published_ads",
    "monthly_distinct_ads",
    "monthly_oro_ads",
    "monthly_plata_ads",
    "monthly_destacados_ads",
    "monthly_shows",
    "monthly_visits",
    "monthly_leads",
    "monthly_total_phone_views",
    "monthly_total_calls",
    "monthly_total_emails",
    "monthly_total_invoice",
    "monthly_unique_leads",
    "monthly_avg_ad_price",
]

TREND_COLS = [
    "monthly_total_invoice",
    "monthly_leads",
    "monthly_visits",
    "monthly_shows",
]


def compute_contract_churn(
    df: pd.DataFrame,
    threshold: int = 3,
    include_right_censored: bool = True,
) -> pd.DataFrame:
    """
    Etiqueta contratos como churned si duran <= threshold meses.

    Opcionalmente, puede incluir en el output los contratos right-censored.
    Si no se incluyen, el output contiene únicamente contratos válidos para
    el análisis de churn en ese horizonte temporal.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mensual a nivel contrato-periodo con, al menos, las columnas:
        - `contract_id`
        - `period_int`
        - `is_right_censored`

    threshold : int, default=3
        Umbral máximo de duración para etiquetar el contrato como churned.

    include_right_censored : bool, default=False
        Si es True, mantiene en el output los contratos right-censored.
        Si es False, los excluye del output.

    Returns
    -------
    pd.DataFrame
        DataFrame mensual con las columnas añadidas:
        - `contract_duration_months`
        - `churned_{threshold}m`
    """
    validate_columns(
        df=df,
        required_cols={"contract_id", "period_int", "is_right_censored"},
        func_name="compute_contract_churn",
    )

    active = df.loc[df["contract_id"].notna()].copy()

    contract_summary = (
        active.groupby("contract_id", as_index=False)
        .agg(
            contract_duration_months=("period_int", "size"),
            is_right_censored=("is_right_censored", "first"),
        )
    )

    churn_col = f"churned_{threshold}m"

    contract_summary[churn_col] = (
        (contract_summary["contract_duration_months"] <= threshold)
        & (~contract_summary["is_right_censored"])
    )

    n_total_contracts = contract_summary["contract_id"].nunique()
    n_right_censored = int(contract_summary["is_right_censored"].sum())
    pct_right_censored = (
        n_right_censored / n_total_contracts * 100 if n_total_contracts > 0 else 0.0
    )

    if not include_right_censored:
        contract_summary = contract_summary.loc[
            ~contract_summary["is_right_censored"]
        ].copy()
        merge_how = "inner"
    else:
        merge_how = "left"

    n_retained_contracts = contract_summary["contract_id"].nunique()
    n_churned = int(contract_summary[churn_col].sum())
    pct_churned = (
        n_churned / n_retained_contracts * 100 if n_retained_contracts > 0 else 0.0
    )

    logger.info(
        "Contracts for churn_%sm | total=%s | right_censored=%s (%.2f%%) | include_right_censored=%s",
        threshold,
        n_total_contracts,
        n_right_censored,
        pct_right_censored,
        include_right_censored,
    )

    logger.info(
        "Contracts retained for churn_%sm analysis: %s/%s",
        threshold,
        n_retained_contracts,
        n_total_contracts,
    )

    logger.info(
        "Positive class %s: %s/%s (%.2f%%)",
        churn_col,
        n_churned,
        n_retained_contracts,
        pct_churned,
    )

    out = active.merge(
        contract_summary[
            ["contract_id", "contract_duration_months", churn_col]
        ],
        on="contract_id",
        how=merge_how,
    )

    return out


def get_first_n_months(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Devuelve los primeros `n` meses observados de cada contrato, manteniendo
    únicamente aquellos contratos con al menos `n` meses de historial.

    La función:
    1. ordena las observaciones por `contract_id` y `period_int`
    2. asigna un número de mes relativo dentro de cada contrato (`month_number`)
    3. filtra los contratos con longitud suficiente (`n_months >= n`)
    4. devuelve solo los meses `1..n` de esos contratos

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mensual a nivel contrato-periodo con, al menos:
        - `contract_id`
        - `period_int`

    n : int, default=3
        Número de meses iniciales a conservar por contrato.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado con los primeros `n` meses de cada contrato, incluyendo
        la columna `month_number`.
    """
    validate_columns(df, {"contract_id", "period_int"}, "get_first_n_months")

    active = df.loc[df["contract_id"].notna()].copy()
    active = active.sort_values(["contract_id", "period_int"])
    active["month_number"] = active.groupby("contract_id").cumcount() + 1

    contract_lengths = (
        active.groupby("contract_id", as_index=False)
        .size()
        .rename(columns={"size": "n_months"})
    )

    total_contracts = contract_lengths["contract_id"].nunique()
    valid_contracts = contract_lengths.loc[
        contract_lengths["n_months"] >= n, "contract_id"
    ]
    n_valid = valid_contracts.nunique()
    n_removed = total_contracts - n_valid
    pct_removed = n_removed / total_contracts * 100 if total_contracts > 0 else 0.0

    logger.info(
        "Contracts retained with at least %s months: %s/%s | removed: %s (%.2f%%)",
        n, n_valid, total_contracts, n_removed, pct_removed,
    )

    active = active.loc[active["contract_id"].isin(valid_contracts)].copy()
    return active.loc[active["month_number"] <= n].copy()


def _slope(x, y):
    """Pendiente lineal simple entre x e y."""
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(tmp) < 2 or tmp["y"].std() == 0:
        return 0.0

    return float(
        np.polyfit(
            tmp["x"].to_numpy(dtype=float),
            tmp["y"].to_numpy(dtype=float),
            1,
        )[0]
    )


def compute_behavior_features(df_first_months: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas de comportamiento de los primeros meses a nivel contrato.

    A partir de un dataframe mensual ya restringido a los primeros meses de cada
    contrato, construye un dataframe a nivel `contract_id` con:

    - medias por contrato
    - desviaciones estándar por contrato
    - valores mensuales individuales (`month1`, `month2`, ..., `monthN`)
    - ratios derivados
    - tendencias lineales para variables seleccionadas

    Criterio para `monthly_total_invoice`
    -------------------------------------
    Los valores iguales a `0` se tratan como no informativos y se convierten en
    `NaN` antes de calcular las features derivadas de facturación.

    Parameters
    ----------
    df_first_months : pd.DataFrame
        DataFrame mensual con los primeros meses de cada contrato. Debe incluir:
        - `contract_id`
        - `month_number`
        - columnas definidas en `BEHAVIOR_COLS`
        - columnas definidas en `TREND_COLS`

    Returns
    -------
    pd.DataFrame
        DataFrame a nivel contrato con una fila por `contract_id` y las features
        agregadas de comportamiento.
    """
    required_cols = {"contract_id", "month_number"} | set(BEHAVIOR_COLS) | set(TREND_COLS)
    validate_columns(df_first_months, required_cols, "compute_behavior_features")

    n_contracts_input = df_first_months["contract_id"].nunique()

    df_clean = df_first_months.copy()

    n_zero_invoice = (df_clean["monthly_total_invoice"] == 0).sum()
    df_clean.loc[df_clean["monthly_total_invoice"] == 0, "monthly_total_invoice"] = np.nan

    # 1. Media por contrato
    agg = df_clean.groupby("contract_id")[BEHAVIOR_COLS].mean()
    agg["monthly_total_invoice"] = agg["monthly_total_invoice"].fillna(0)

    # 2. Std por contrato
    behavior_std = df_clean.groupby("contract_id")[BEHAVIOR_COLS].std().fillna(0)
    behavior_std.columns = [f"{col}_std" for col in BEHAVIOR_COLS]
    agg = agg.join(behavior_std)

    # 3. Valor de cada mes individual
    pivoted = df_clean.pivot_table(
        index="contract_id",
        columns="month_number",
        values=BEHAVIOR_COLS,
        aggfunc="first",
    )
    pivoted.columns = [f"{col}_month{int(month)}" for col, month in pivoted.columns]
    agg = agg.join(pivoted)

    # 4. Ratios
    agg["usage_ratio"] = (
        agg["monthly_published_ads"] / agg["monthly_contracted_ads"].replace(0, np.nan)
    )

    agg["cost_per_lead"] = (
        agg["monthly_total_invoice"] / agg["monthly_leads"].replace(0, np.nan)
    )

    agg["conversion_rate"] = (
        agg["monthly_leads"] / agg["monthly_visits"].replace(0, np.nan)
    )

    agg["premium_ratio"] = (
        (
            agg["monthly_oro_ads"]
            + agg["monthly_plata_ads"]
            + agg["monthly_destacados_ads"]
        )
        / agg["monthly_published_ads"].replace(0, np.nan)
    )

    # 5. Tendencias
    for col in TREND_COLS:
        agg[f"{col}_trend"] = (
            df_clean.groupby("contract_id")[["month_number", col]]
            .apply(lambda g: _slope(g["month_number"], g[col]))
        )

    logger.info(
        "Behavior features built for %s contracts (input contracts: %s). "
        "Converted %s zero invoices to NaN.",
        agg.index.nunique(),
        n_contracts_input,
        int(n_zero_invoice),
    )

    logger.info(
        "NaNs in ratios | usage_ratio: %s | cost_per_lead: %s | conversion_rate: %s | premium_ratio: %s",
        int(agg["usage_ratio"].isna().sum()),
        int(agg["cost_per_lead"].isna().sum()),
        int(agg["conversion_rate"].isna().sum()),
        int(agg["premium_ratio"].isna().sum()),
    )

    return agg


def compute_stable_price(df: pd.DataFrame, n_months: int = 3) -> pd.DataFrame:
    """
    Calcula el precio estable de cada contrato a partir de los meses posteriores
    al onboarding.

    Para cada contrato, la función:
    1. ordena los periodos y asigna `month_number`
    2. descarta los primeros `n_months` meses
    3. calcula estadísticas de facturación post-onboarding
    4. añade los valores mensuales individuales de invoice tras onboarding

    Criterio para `monthly_total_invoice`
    -------------------------------------
    Los valores iguales a `0` se tratan como no informativos y se convierten en
    `NaN` antes de calcular las features derivadas de facturación post-onboarding.

    Nota
    ----
    Esta función utiliza información posterior al onboarding, por lo que no debe
    emplearse como feature del modelo 1 si el objetivo es predecir usando solo
    los primeros `n_months` meses.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mensual a nivel contrato-periodo con, al menos:
        - `contract_id`
        - `period_int`
        - `monthly_total_invoice`

    n_months : int, default=3
        Número de meses de onboarding a excluir del cálculo.

    Returns
    -------
    pd.DataFrame
        DataFrame a nivel contrato con:
        - `stable_price`
        - `stable_price_std`
        - `n_months_post_onboarding`
        - `invoice_post_month1`, `invoice_post_month2`, ...
    """

    validate_columns(df, {"contract_id", "period_int", "monthly_total_invoice"}, "compute_stable_price")

    active = df.loc[df["contract_id"].notna()].copy()
    active = active.sort_values(["contract_id", "period_int"])
    active["month_number"] = active.groupby("contract_id").cumcount() + 1

    total_contracts = active["contract_id"].nunique()

    post = active.loc[active["month_number"] > n_months].copy()
    contracts_with_post_period = post["contract_id"].nunique()

    n_zero_invoice = int((post["monthly_total_invoice"] == 0).sum())
    post.loc[post["monthly_total_invoice"] == 0, "monthly_total_invoice"] = np.nan

    post["post_month"] = post.groupby("contract_id").cumcount() + 1

    out = post.groupby("contract_id").agg(
        stable_price=("monthly_total_invoice", "mean"),
        stable_price_std=("monthly_total_invoice", "std"),
        n_months_post_onboarding=("monthly_total_invoice", "size"),
    )

    out["stable_price"] = out["stable_price"].fillna(0)
    out["stable_price_std"] = out["stable_price_std"].fillna(0)

    pivoted = post.pivot_table(
        index="contract_id",
        columns="post_month",
        values="monthly_total_invoice",
        aggfunc="first",
    )
    pivoted.columns = [f"invoice_post_month{int(month)}" for month in pivoted.columns]
    out = out.join(pivoted)

    logger.info(
        "Stable price computed for %s/%s contracts with post-onboarding history. "
        "Converted %s zero invoices to NaN.",
        contracts_with_post_period,
        total_contracts,
        n_zero_invoice,
    )

    return out

def add_contract_metadata(
    df_features: pd.DataFrame,
    df_contracts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Añade al dataframe de features los metadatos contractuales ya calculados
    previamente en preprocessing.

    La función no recalcula información temporal ni lógica contractual. Para cada
    `contract_id`, simplemente toma de `df_contracts` los metadatos disponibles y
    los une al dataframe de features.

    Metadatos candidatos
    --------------------
    Si están presentes en `df_contracts`, se incorporan las siguientes columnas:
    - `advertiser_zrive_id`
    - `advertiser_province`
    - `province_id`
    - `advertiser_group_id`
    - `contract_start_date`
    - `contrato_churn_date`
    - `contract_end_period`
    - `contract_duration_months`

    Además, si existe `advertiser_group_id`, se añade la columna derivada
    `has_group`.

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame de features a nivel contrato, indexado por `contract_id`.

    df_contracts : pd.DataFrame
        DataFrame mensual preprocesado con metadatos ya construidos a nivel
        contrato-periodo.

    Returns
    -------
    pd.DataFrame
        DataFrame de features enriquecido con metadatos contractuales y del
        advertiser.
    """
    validate_columns(
        df=df_contracts,
        required_cols={"contract_id"},
        func_name="add_contract_metadata",
    )

    active = df_contracts.loc[df_contracts["contract_id"].notna()].copy()

    candidate_cols = [
        "advertiser_zrive_id",
        "advertiser_province",
        "province_id",
        "advertiser_group_id",
        "contract_start_date",
        "contrato_churn_date",
        "contract_end_period",
        "contract_duration_months",
    ]

    metadata_cols = [col for col in candidate_cols if col in active.columns]

    contract_meta = (
        active.groupby("contract_id")[metadata_cols]
        .first()
    )

    if "advertiser_group_id" in contract_meta.columns:
        contract_meta["has_group"] = contract_meta["advertiser_group_id"].notna()

    n_feature_contracts = df_features.index.nunique()
    n_meta_contracts = contract_meta.index.nunique()

    out = df_features.join(contract_meta, how="left")

    n_unmatched = int(out["advertiser_zrive_id"].isna().sum()) if "advertiser_zrive_id" in out.columns else None

    logger.info(
        "Contract metadata joined: feature_contracts=%s, metadata_contracts=%s, unmatched_contracts=%s",
        n_feature_contracts,
        n_meta_contracts,
        n_unmatched,
    )

    if "has_group" in out.columns:
        logger.info(
            "Contracts with advertiser group: %s/%s",
            int(out["has_group"].fillna(False).sum()),
            len(out),
        )

    return out