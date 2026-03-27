import numpy as np
import pandas as pd
from pathlib import Path

from build_contracts import add_contract_id
from churn import compute_contract_churn_3m


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


def prepare_fct(df_fct):
    """Limpieza del snapshot mensual: conversion de period_int y filtrado basico."""
    df = df_fct.copy()
    df["period_int"] = pd.to_datetime(
        df["period_int"].astype(str), format="%Y%m"
    ).dt.to_period("M")
    # Hay 2 registros con facturacion negativa (devoluciones)
    df = df[df["monthly_total_invoice"] >= 0]
    return df


def filter_left_censored(df, df_dim, observation_start="2023-01"):
    """Quita contratos que ya existian antes de que empiecen los datos.

    Si el primer mes activo de un contrato es ene-2023 (primer mes del dataset)
    y en dim vemos que min_start_contrato_date < 2023, ese contrato ya llevaba
    tiempo activo y no podemos usar sus "primeros 3 meses" como onboarding real.
    """
    obs_start = pd.Period(observation_start, freq="M")

    active = df[df["contract_id"].notna()]
    first_month = (
        active.groupby("contract_id")
        .agg(first_period=("period_int", "min"),
             advertiser_zrive_id=("advertiser_zrive_id", "first"))
    )

    # Contratos que arrancan justo en el primer mes de datos
    suspects = first_month[first_month["first_period"] == obs_start]

    dim_dates = df_dim[["advertiser_zrive_id", "min_start_contrato_date"]].copy()
    dim_dates["started_before"] = (
        pd.to_datetime(dim_dates["min_start_contrato_date"])
        < pd.Timestamp(observation_start)
    )

    suspects = suspects.reset_index().merge(dim_dates, on="advertiser_zrive_id", how="left")
    to_remove = suspects.loc[suspects["started_before"] == True, "contract_id"].values

    df = df.copy()
    df.loc[df["contract_id"].isin(to_remove), "contract_id"] = None
    return df


def get_first_n_months(df, n=3):
    """Devuelve los primeros n meses de cada contrato (solo contratos con >=n meses)."""
    active = df[df["contract_id"].notna()].copy()
    active = active.sort_values(["contract_id", "period_int"])
    active["month_number"] = active.groupby("contract_id").cumcount() + 1

    contract_lengths = active.groupby("contract_id").size()
    valid = contract_lengths[contract_lengths >= n].index
    active = active[active["contract_id"].isin(valid)]

    return active[active["month_number"] <= n]


def _slope(x, y):
    """Pendiente lineal simple entre x e y."""
    if len(x) < 2 or y.std() == 0:
        return 0.0
    return np.polyfit(x.values.astype(float), y.values.astype(float), 1)[0]


def compute_behavior_features(df_first_months):
    """Agrega metricas de los primeros meses: medias, ratios y tendencias."""

    agg = df_first_months.groupby("contract_id")[BEHAVIOR_COLS].mean()

    # Para la media de facturacion, ignoramos meses con invoice=0 (6.4% de los activos)
    # porque distorsionan la media hacia abajo
    tmp = df_first_months.copy()
    tmp.loc[tmp["monthly_total_invoice"] == 0, "monthly_total_invoice"] = np.nan
    agg["monthly_total_invoice"] = (
        tmp.groupby("contract_id")["monthly_total_invoice"].mean().fillna(0)
    )

    # Ratios
    agg["usage_ratio"] = (
        agg["monthly_published_ads"] / agg["monthly_contracted_ads"].replace(0, np.nan)
    ).clip(upper=1.0)

    agg["cost_per_lead"] = (
        agg["monthly_total_invoice"] / agg["monthly_leads"].replace(0, np.nan)
    )

    agg["conversion_rate"] = (
        agg["monthly_leads"] / agg["monthly_visits"].replace(0, np.nan)
    )

    agg["premium_ratio"] = (
        (agg["monthly_oro_ads"] + agg["monthly_plata_ads"] + agg["monthly_destacados_ads"])
        / agg["monthly_published_ads"].replace(0, np.nan)
    )

    # Tendencias: pendiente de las metricas clave durante los primeros meses
    for col in TREND_COLS:
        agg[f"{col}_trend"] = (
            df_first_months.groupby("contract_id")
            .apply(lambda g: _slope(g["month_number"], g[col]), include_groups=False)
        )

    return agg


def compute_stable_price(df, n_months=3):
    """Precio estable = media de facturacion a partir del mes n+1."""
    active = df[df["contract_id"].notna()].copy()
    active = active.sort_values(["contract_id", "period_int"])
    active["month_number"] = active.groupby("contract_id").cumcount() + 1

    post = active[active["month_number"] > n_months]

    out = post.groupby("contract_id").agg(
        stable_price=("monthly_total_invoice", "mean"),
        stable_price_std=("monthly_total_invoice", "std"),
        n_months_post_onboarding=("monthly_total_invoice", "size"),
    )
    out["stable_price_std"] = out["stable_price_std"].fillna(0)
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
    # Si el contrato llega hasta el ultimo mes de datos, no sabemos si seguira o no
    info["is_right_censored"] = info["contract_end"] == last_p

    out = df_features.join(info)

    dim_cols = df_dim[["advertiser_zrive_id", "province_id", "advertiser_group_id"]].copy()
    dim_cols["has_group"] = dim_cols["advertiser_group_id"].notna().astype(int)
    out = out.merge(dim_cols, on="advertiser_zrive_id", how="left")

    return out


def build_modeling_dataset(data_path, n_months=3):
    """Pipeline completo de datos a dataset de modelado.

    Carga los parquets, identifica contratos, filtra left-censored,
    calcula features de los primeros n meses y los dos targets:
    churn binario y precio estable post-onboarding.
    """
    df_dim = pd.read_parquet(data_path / "zrive_dim_advertiser.parquet")
    df_fct = pd.read_parquet(
        data_path / "zrive_fct_monthly_snapshot_advertiser.parquet"
    )

    df_fct = prepare_fct(df_fct)
    df_fct = add_contract_id(df_fct)
    df_fct = compute_contract_churn_3m(df_fct)
    df_fct = filter_left_censored(df_fct, df_dim)

    df_first = get_first_n_months(df_fct, n=n_months)
    df_features = compute_behavior_features(df_first)

    # Targets
    df_stable = compute_stable_price(df_fct, n_months=n_months)
    df_features = df_features.join(df_stable, how="left")

    churn = (
        df_fct[df_fct["contract_id"].notna()]
        .groupby("contract_id")["churned_3m"].first()
    )
    df_features["churned_3m"] = churn

    df_final = add_contract_metadata(df_features, df_fct, df_dim)
    return df_final


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    df = build_modeling_dataset(data_path)

    print(f"Dataset: {df.shape[0]} contratos x {df.shape[1]} columnas")

    print(f"\nTargets:")
    print(f"  Churn <=3m: {df['churned_3m'].mean():.1%}")
    nc = df[df["churned_3m"] == 0]
    print(f"  Precio estable (no churned): media={nc['stable_price'].mean():.0f}, "
          f"mediana={nc['stable_price'].median():.0f}")

    print(f"\nCensoring:")
    print(f"  Right-censored: {df['is_right_censored'].sum()} ({df['is_right_censored'].mean():.1%})")

    print(f"\nSanity checks:")
    print(f"  NaN en stable_price: {df['stable_price'].isna().sum()}")
    print(f"  usage_ratio > 1: {(df['usage_ratio'] > 1).sum()}")
