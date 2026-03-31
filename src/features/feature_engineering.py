import numpy as np
import pandas as pd


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


def compute_contract_churn_3m(df):
    """Etiqueta contratos como churned si duran <= 3 meses."""
    out = df.copy()
    contract_duration = (
        out.groupby("contract_id")
        .size()
        .reset_index(name="duration")
    )
    short_contracts = contract_duration.loc[
        contract_duration["duration"] <= 3, "contract_id"
    ]
    out["churned_3m"] = out["contract_id"].isin(short_contracts).astype(int)
    return out


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

    tmp = df_first_months.copy()
    tmp.loc[tmp["monthly_total_invoice"] == 0, "monthly_total_invoice"] = np.nan
    agg["monthly_total_invoice"] = (
        tmp.groupby("contract_id")["monthly_total_invoice"].mean().fillna(0)
    )

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
    info["is_right_censored"] = info["contract_end"] == last_p

    out = df_features.join(info)

    dim_cols = df_dim[["advertiser_zrive_id", "province_id", "advertiser_group_id"]].copy()
    dim_cols["has_group"] = dim_cols["advertiser_group_id"].notna().astype(int)
    out = out.merge(dim_cols, on="advertiser_zrive_id", how="left")

    return out
