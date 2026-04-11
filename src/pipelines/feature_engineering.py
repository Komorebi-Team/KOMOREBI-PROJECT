import pandas as pd

from src.features.feature_engineering import (
    compute_contract_churn,
    get_first_n_months,
    compute_behavior_features,
    compute_stable_price,
)


def run_feature_engineering_pipeline(
    df_contracts: pd.DataFrame,
    n_months: int = 3,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline de feature engineering sobre los contratos preprocesados.

    Recibe el df_contracts (output del preprocessing pipeline) con filas mensuales
    y devuelve un dataset a nivel contrato listo para modelar.

    Parameters
    ----------
    df_contracts : pd.DataFrame
        DataFrame mensual con contract_id (output de run_preprocessing_pipeline).
    n_months : int
        Numero de meses de onboarding para calcular features y definir churn.

    Returns
    -------
    pd.DataFrame
        Dataset con una fila por contrato, features de comportamiento,
        precio estable y target de churn.
    """
    churn_col = f"churned_{n_months}m"

    # 1. Etiquetar churn (mismo df, columna nueva)
    df = compute_contract_churn(df_contracts, threshold=n_months)

    # 2. Features de los primeros n meses (agrega a nivel contrato)
    df_first = get_first_n_months(df, n=n_months)
    df_features = compute_behavior_features(df_first)

    # 3. Precio estable post-onboarding (vuelve al df original, no al agregado)
    df_stable = compute_stable_price(df, n_months=n_months)
    df_features = df_features.join(df_stable, how="left")

    # 4. Target: churn binario
    churn = (
        df[df["contract_id"].notna()]
        .groupby("contract_id")[churn_col]
        .first()
    )
    df_features[churn_col] = churn

    # 5. Metadata del contrato (right censoring, advertiser, etc.)
    contract_meta = (
        df[df["contract_id"].notna()]
        .groupby("contract_id")
        .agg(
            advertiser_zrive_id=("advertiser_zrive_id", "first"),
            contract_start=("period_int", "min"),
            contract_end=("period_int", "max"),
        )
    )

    if "is_right_censored" in df.columns:
        contract_meta["is_right_censored"] = (
            df[df["contract_id"].notna()]
            .groupby("contract_id")["is_right_censored"]
            .first()
        )

    if "province_id" in df.columns:
        province = (
            df[df["contract_id"].notna()]
            .groupby("contract_id")["province_id"]
            .first()
        )
        contract_meta["province_id"] = province

    if "advertiser_group_id" in df.columns:
        group = (
            df[df["contract_id"].notna()]
            .groupby("contract_id")["advertiser_group_id"]
            .first()
        )
        contract_meta["advertiser_group_id"] = group
        contract_meta["has_group"] = group.notna()

    df_features = df_features.join(contract_meta, how="left")

    return df_features
