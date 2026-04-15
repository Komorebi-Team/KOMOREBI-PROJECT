import pandas as pd

from src.features import (
    compute_contract_churn,
    get_first_n_months,
    compute_behavior_features,
    compute_stable_price,
    add_contract_metadata,

)

from src.features import prepare_model_features

def run_feature_engineering_pipeline(
    df_contracts: pd.DataFrame,
    n_months: int = 2,
    churn_threshold: int = 5,
    include_right_censored: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline de feature engineering sobre los contratos preprocesados.

    A partir del dataframe mensual de salida del preprocessing, el pipeline:
    1. calcula la etiqueta de churn contractual para el horizonte `churn_threshold`
    2. selecciona los primeros `n_months` meses de cada contrato
    3. construye features agregadas de comportamiento a nivel contrato
    4. calcula el precio estable post-onboarding
    5. añade la etiqueta de churn y los metadatos contractuales al output final

    Parameters
    ----------
    df_contracts : pd.DataFrame
        DataFrame mensual con `contract_id`, output de `run_preprocessing_pipeline`.

    n_months : int, default=2
        Número de meses de onboarding utilizados para construir las features.

    churn_threshold : int, default=5
        Umbral máximo de duración (en meses) para etiquetar un contrato como
        churned. Un contrato con duración <= churn_threshold se considera churn.

    include_right_censored : bool, default=True
        Indica si los contratos right-censored se mantienen en el dataset al
        calcular la etiqueta de churn para este horizonte temporal.

    Returns
    -------
    pd.DataFrame
        Dataset a nivel contrato con:
        - features de comportamiento de los primeros `n_months` meses
        - features de precio estable post-onboarding
        - variable objetivo `churned_{churn_threshold}m`
        - metadatos contractuales añadidos al final del pipeline
    """
    churn_col = f"churned_{churn_threshold}m"

    # 1. Etiquetar churn (mismo df, columna nueva)
    df = compute_contract_churn(
        df_contracts,
        threshold=churn_threshold,
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

    # 6. Imputación de valores nulos
    df_features = prepare_model_features(df_features)

    return df_features