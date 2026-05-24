import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score

from src.modeling.split import split_by_advertiser
from src.modeling.metrics import (
    evaluate_model,
    find_best_threshold,
    build_risk_profiles,
    compute_top_k_metrics,
)
from src.modeling.plots import compute_shap_values
from src.utils import compute_class_sample_weight

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_class: Any,
    params: dict[str, Any] | None = None,
    sample_weight: Any = None,
) -> Any:
    """
    Instancia y entrena cualquier estimador scikit-learn compatible.
    """
    params = params or {}
    model = model_class(**params)

    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(X_train, y_train, **fit_kwargs)

    logger.info(
        "Modelo '%s' entrenado con params=%s",
        model.__class__.__name__,
        params,
    )
    return model


def save_model(model: Any, output_path: str) -> str:
    """
    Guarda el modelo entrenado en disco con nombre que incluye timestamp.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y_%m_%d-%H%M")
    model_name = f"{model.__class__.__name__}_{timestamp}.pkl"
    saved_path = output_dir / model_name

    joblib.dump(model, saved_path)
    logger.info("Modelo guardado en '%s'", saved_path)
    return str(saved_path)


def _cross_validate(
    model_class: Any,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    sample_weight: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """
    CV interna del pipeline: GroupKFold por advertiser con sample_weight.
    """
    cv_params = (params or {}).copy()
    if "random_state" in model_class().get_params():
        cv_params["random_state"] = random_state

    model_cv = model_class(**cv_params)
    cv = GroupKFold(n_splits=n_splits)

    scores = cross_val_score(
        model_cv,
        X_train,
        y_train,
        cv=cv,
        groups=groups,
        scoring="roc_auc",
        params={"sample_weight": sample_weight},
        n_jobs=-1,
    )

    logger.info(
        "CV GroupKFold (%d folds) — ROC AUC: %.3f ± %.3f",
        n_splits, scores.mean(), scores.std(),
    )
    return {"cv_mean_auc": scores.mean(), "cv_std_auc": scores.std()}


def run_modeling_pipeline(
    df: pd.DataFrame,
    *,
    model_class: Any,
    params: dict[str, Any] | None = None,
    target: str = "churned_3m",
    test_size: float = 0.2,
    output_path: str | None = None,
    save_model_flag: bool = True,
    cross_validate: bool = False,
    cv_n_splits: int = 5,
    optimize_threshold_for: str = "f1",
    risk_bins: list[float] | None = None,
    risk_labels: list[str] | None = None,
    evaluation_config: dict[str, Any] | None = None,
    random_state: int = 42,
    feature_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Ejecuta el pipeline completo de modelado:
      1. Split train/test agrupado por advertiser
      2. (Opcional) Cross-validation en train con GroupKFold
      3. Entrenamiento sobre todo el train set
      4. Evaluación en test
      5. Threshold óptimo (max F1, Recall o Precision)
      6. Cálculo de perfiles de riesgo
      7. Metricas de negocio (Top K% capturado y facturacion salvada)
      8. (Opcional) Guardado del modelo

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a nivel contrato.
    model_class : estimator class
        Clase del modelo a entrenar.
    params : dict, optional
        Hiperparámetros.
    target : str
        Nombre de la columna target.
    test_size : float
        Proporción del conjunto de test.
    output_path : str, optional
        Directorio donde guardar el modelo.
    save_model_flag : bool
        Si True, guarda el modelo.
    cross_validate : bool
        Si True, ejecuta CV.
    cv_n_splits : int
        Número de folds para la CV.
    optimize_threshold_for : str, default="f1"
        Métrica a maximizar al buscar el threshold.
    risk_bins : list, optional
        Cortes para los buckets de riesgo.
    risk_labels : list, optional
        Etiquetas para los buckets de riesgo.
    evaluation_config : dict, optional
        Configuración de evaluación avanzada.
    random_state : int, default=42
        Semilla para reproducibilidad global.
    feature_config : dict, optional
        Configuración de features (prefixes, extra_non_features).
    """
    feat_cfg = feature_config or {}
    X_train, X_test, y_train, y_test = split_by_advertiser(
        df,
        target=target,
        test_size=test_size,
        random_state=random_state,
        extra_non_features=feat_cfg.get("extra_non_features"),
    )
    groups_train = df.loc[X_train.index, "advertiser_zrive_id"]

    sample_weight = compute_class_sample_weight(y_train)

    # --- CV opcional ---
    cv_results = None
    if cross_validate:
        cv_results = _cross_validate(
            model_class=model_class,
            params=params or {},
            X_train=X_train,
            y_train=y_train,
            groups=groups_train,
            sample_weight=sample_weight,
            n_splits=cv_n_splits,
            random_state=random_state,
        )

    # Inyectamos el random_state en params si el modelo lo soporta
    model_params = (params or {}).copy()
    if "random_state" in model_class().get_params() and "random_state" not in model_params:
        model_params["random_state"] = random_state

    # --- Entrenamiento final sobre todo el train set ---
    model = train_model(
        X_train,
        y_train,
        model_class=model_class,
        params=model_params,
        sample_weight=sample_weight,
    )

    eval_cfg = evaluation_config or {}
    metrics = evaluate_model(
        model, 
        X_test, 
        y_test, 
        name=model.__class__.__name__,
        extra_metrics=eval_cfg.get("extra_metrics"),
    )
    threshold_info = find_best_threshold(
        y_test, metrics["y_proba"], optimize_for=optimize_threshold_for
    )

    # --- Perfiles de riesgo ---
    # Creamos un DF con todas las columnas originales para el conjunto de test
    # para asegurar que las metricas de negocio (facturacion, etc) estan presentes
    X_test_full = df.loc[X_test.index]

    risk_profiles = build_risk_profiles(
        model, X_test_full, y_test, y_proba=metrics["y_proba"], bins=risk_bins, labels=risk_labels
    )

    # --- Metricas de negocio (Top K%) ---
    business_metrics = compute_top_k_metrics(
        X_test_full,
        y_test,
        metrics["y_proba"],
        revenue_col=eval_cfg.get("revenue_col"),
        k_list=eval_cfg.get("top_k_list"),
    )

    # --- SHAP values (opcional) ---
    shap_results = None
    if eval_cfg.get("compute_shap", False):
        shap_results = compute_shap_values(model, X_train, X_test)

    model_path = None
    if save_model_flag:
        if output_path is None:
            raise ValueError("Si 'save_model_flag=True', debes proporcionar 'output_path'.")
        model_path = save_model(model, output_path)

    logger.info(
        "Pipeline de modelado completado — CV AUC=%.3f±%.3f | Test AUC=%.3f | threshold=%.3f",
        cv_results["cv_mean_auc"] if cv_results else float("nan"),
        cv_results["cv_std_auc"] if cv_results else float("nan"),
        metrics["roc_auc"],
        threshold_info["threshold"],
    )

    return {
        "model": model,
        "metrics": metrics,
        "threshold": threshold_info,
        "model_path": model_path,
        "X_test": X_test,
        "y_test": y_test,
        "groups_train": groups_train,
        "cv_results": cv_results,
        "risk_profiles": risk_profiles,
        "business_metrics": business_metrics,
        "shap_values": shap_results,
    }
