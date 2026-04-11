import pandas as pd

from src.features.feature_engineering import (
    compute_contract_churn,
    get_first_n_months,
    compute_behavior_features,
    compute_stable_price,
    add_contract_metadata,
)

def run_feature_engineering_pipeline(
    df_contracts: pd.DataFrame,
    n_months: int = 3,
    include_right_censored: bool = True
) -> pd.DataFrame:
    """
    Ejecuta el pipeline de feature engineering sobre los contratos preprocesados.

    Parameters
    ----------
    df_contracts : pd.DataFrame
        DataFrame mensual con contract_id (output de run_preprocessing_pipeline).
    n_months : int
        Número de meses de onboarding para calcular features y definir churn.
    exclude_right_censored : bool, default=False
        Indica si los contratos right-censored deben excluirse antes de construir
        el dataset final para este horizonte temporal.

    Returns
    -------
    pd.DataFrame
        Dataset con una fila por contrato, features de comportamiento,
        precio estable y target de churn.
    """
    churn_col = f"churned_{n_months}m"

    # 1. Etiquetar churn (mismo df, columna nueva)
    df = compute_contract_churn(
        df_contracts,
        threshold=n_months,
        include_right_censored=include_right_censored,
    )

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

    # 5. Metadata del contrato
    df_features = add_contract_metadata(df_features, df)

    return df_features
