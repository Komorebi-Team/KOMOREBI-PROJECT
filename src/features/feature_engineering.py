import logging

import numpy as np
import pandas as pd

from src.preprocessing.utils import validate_columns

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


def compute_contract_churn(df: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    """
    Etiqueta contratos como churned si duran <= threshold meses y no están
    right-censored.
    """
    active = df[df["contract_id"].notna()].copy()

    contract_summary = (
        active.groupby("contract_id", as_index=False)
        .agg(
            contract_duration=("period_int", "size"),
            is_right_censored=("is_right_censored", "first"),
        )
    )

    contract_summary[f"churned_{threshold}m"] = (
        (contract_summary["contract_duration"] <= threshold)
        & (~contract_summary["is_right_censored"])
    )

    out = active.merge(
        contract_summary[["contract_id", "contract_duration", f"churned_{threshold}m"]],
        on="contract_id",
        how="left",
    )

    return out


def get_first_n_months(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Devuelve los primeros n meses de cada contrato, manteniendo unicamente
    contratos con al menos n meses observados.
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
    Agrega métricas de comportamiento de los primeros meses a nivel contrato:
    medias, desviaciones estándar, valores por mes, ratios y tendencias.

    Criterio para monthly_total_invoice
    -------------------------------------
    Los valores iguales a 0 se tratan como no informativos y se convierten en
    NaN antes de calcular las features derivadas de facturación.
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
    Calcula el precio estable como la media de facturación a partir del mes n_months + 1
    para cada contrato.

    Criterio para monthly_total_invoice
    -------------------------------------
    Los valores iguales a 0 se tratan como no informativos y se convierten en
    NaN antes de calcular las features derivadas de facturación post-onboarding.

    Nota
    ----
    Esta función utiliza información posterior al onboarding, por lo que no debe
    emplearse como feature del modelo 1 si el objetivo es predecir usando solo
    los primeros n_months meses.
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


def add_contract_metadata(df_features, df_contracts, df_dim, last_period="2025-05"):
    """Anade provincia, grupo y flags de censoring."""
    last_p = pd.Period(last_period, freq="M")
    active = df_contracts[df_contracts["contract_id"].notna()]

    info = active.groupby("contract_id").agg(
        advertiser_zrive_id=("advertiser_zrive_id", "first"),
        contract_start=("period_int", "min"),
        contract_end=("period_int", "max"),
    )
    info["is_right_censored"] = info["contract_end"] == last_p

    out = df_features.join(info)

    dim_cols = df_dim[["advertiser_zrive_id", "province_id", "advertiser_group_id"]].copy()
    dim_cols["has_group"] = dim_cols["advertiser_group_id"].notna()
    out = out.merge(dim_cols, on="advertiser_zrive_id", how="left")

    return out
