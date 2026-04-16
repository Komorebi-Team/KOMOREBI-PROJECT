import logging
from typing import Any, List, Optional, Dict

import pandas as pd

from src.pipelines.preprocessing import run_preprocessing_pipeline
from src.pipelines.feature_engineering import run_feature_engineering_pipeline
from src.pipelines.modeling import run_modeling_pipeline

logger = logging.getLogger(__name__)


def run_end2end_pipeline(
    df_fct: pd.DataFrame,
    df_dim: pd.DataFrame,
    *,
    model_class: Any,
    params: dict[str, Any] | None = None,
    n_months: int = 2,
    churn_threshold: int = 5,
    include_right_censored: bool = False,
    observation_end: pd.Period | None = None,
    test_size: float = 0.2,
    output_path: str | None = None,
    save_model_flag: bool = True,
    optimize_threshold_for: str = "f1",
    risk_config: dict[str, Any] | None = None,
    evaluation_config: dict[str, Any] | None = None,
    random_state: int = 42,
    feature_config: dict[str, Any] | None = None,
    impute_config: dict[str, Any] | None = None,
    cross_validate: bool = False,
    cv_n_splits: int = 5,
) -> dict[str, Any]:
    """
    Pipeline end-to-end con control total de parámetros y reproducibilidad.

    Orquesta en orden:
      1. Preprocessing       → contratos limpios y censoring
      2. Feature engineering → dataset a nivel contrato con features y target
      3. Modeling            → split, train, evaluate, (save)

    Parameters
    ----------
    df_fct : pd.DataFrame
        DataFrame mensual crudo (fact table).
    df_dim : pd.DataFrame
        DataFrame dimensional de advertiser.
    model_class : estimator class
        Clase del modelo a entrenar (p.ej. HistGradientBoostingClassifier).
    params : dict, optional
        Hiperparámetros del modelo.
    n_months : int
        Horizonte de churn en meses.
    include_right_censored : bool
        Si se incluyen contratos right-censored al calcular el target.
    observation_end : pd.Period, optional
        Último periodo de observación global para el censoring.
    test_size : float
        Proporción del conjunto de test.
    output_path : str, optional
        Directorio donde guardar el modelo.
    save_model_flag : bool
        Si True, guarda el modelo.
    optimize_threshold_for : str, default="f1"
        Métrica a maximizar al buscar el threshold.
    risk_config : dict, optional
        Configuración de perfiles de riesgo (keys: 'bins', 'labels').
    evaluation_config : dict, optional
        Configuración de métricas avanzadas (extra_metrics, top_k_list, revenue_col).
    random_state : int, default=42
        Semilla para reproducibilidad global.
    feature_config : dict, optional
        Configuración de features (prefixes, extra_non_features).
    impute_config : dict, optional
        Configuración de imputación (zero_impute_cols).
    cross_validate : bool, default=False
        Si True, ejecuta validación cruzada antes del entrenamiento final.
    cv_n_splits : int, default=5
        Número de folds para la validación cruzada.

    Returns
    -------
    dict con el output de todas las etapas.
    """
    logger.info("=== [1/3] Preprocessing ===")
    df_contracts, contract_summary = run_preprocessing_pipeline(
        df_fct,
        df_dim,
        observation_end=observation_end,
    )

    logger.info("=== [2/3] Feature engineering (n_months=%d, churn_threshold=%d) ===", n_months, churn_threshold)
    df_features = run_feature_engineering_pipeline(
        df_contracts,
        n_months=n_months,
        churn_threshold=churn_threshold,
        include_right_censored=include_right_censored,
        impute_config=impute_config,
    )

    logger.info("=== [3/3] Modeling ===")
    target = f"churned_{churn_threshold}m"
    risk_config = risk_config or {}
    modeling_results = run_modeling_pipeline(
        df_features,
        model_class=model_class,
        params=params,
        target=target,
        test_size=test_size,
        output_path=output_path,
        save_model_flag=save_model_flag,
        optimize_threshold_for=optimize_threshold_for,
        risk_bins=risk_config.get("bins"),
        risk_labels=risk_config.get("labels"),
        evaluation_config=evaluation_config,
        random_state=random_state,
        feature_config=feature_config,
        cross_validate=cross_validate,
        cv_n_splits=cv_n_splits,
    )

    logger.info("=== Pipeline end-to-end completado ===")

    return {
        "df_contracts": df_contracts,
        "contract_summary": contract_summary,
        "df_features": df_features,
        **modeling_results,
    }
