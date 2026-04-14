import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.modeling.split import split_by_advertiser
from src.modeling.evaluate import evaluate_model, find_best_threshold

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

    El diseño está inspirado en la lógica del CustomModel del proyecto anterior:
    recibe la clase del modelo y sus parámetros de forma parametrizada, de modo
    que el pipeline de modelado es agnóstico al algoritmo concreto.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    model_class : estimator class
        Clase del modelo a instanciar (p.ej. HistGradientBoostingClassifier,
        RandomForestClassifier, XGBClassifier…).
    params : dict, optional
        Hiperparámetros pasados al constructor del modelo. Si None, se usan
        los valores por defecto de la clase.
    sample_weight : array-like, optional
        Pesos de muestra. Útil para compensar desbalanceo de clases.

    Returns
    -------
    object
        Modelo ajustado.
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

    Parameters
    ----------
    model : estimator
        Modelo entrenado.
    output_path : str
        Directorio de destino.

    Returns
    -------
    str
        Ruta completa al fichero guardado.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y_%m_%d-%H%M")
    model_name = f"{model.__class__.__name__}_{timestamp}.pkl"
    saved_path = output_dir / model_name

    joblib.dump(model, saved_path)
    logger.info("Modelo guardado en '%s'", saved_path)
    return str(saved_path)


def run_modeling_pipeline(
    df: pd.DataFrame,
    *,
    model_class: Any,
    params: dict[str, Any] | None = None,
    target: str = "churned_3m",
    test_size: float = 0.2,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Ejecuta el pipeline completo de modelado:
      1. Split train/test agrupado por advertiser
      2. Entrenamiento del modelo con los parámetros recibidos
      3. Evaluación en test
      4. Threshold óptimo
      5. Guardado opcional del modelo

    El tuning de hiperparámetros no forma parte de este pipeline: es una fase
    de experimentación previa que debe ejecutarse con las funciones de
    `src.modeling.train` y cuyos resultados (best_params_) se pasan aquí como
    `params`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a nivel contrato, salida del pipeline de feature engineering.
    model_class : estimator class
        Clase del modelo a entrenar.
    params : dict, optional
        Hiperparámetros del modelo, idealmente obtenidos de un proceso de
        tuning previo. Si None se usan los defaults de la clase.
    target : str
        Nombre de la columna target.
    test_size : float
        Proporción del conjunto de test.
    output_path : str, optional
        Si se proporciona, guarda el modelo entrenado en ese directorio.

    Returns
    -------
    dict con claves:
        - 'model'       : modelo entrenado
        - 'metrics'     : dict de métricas en test (roc_auc, accuracy, …)
        - 'threshold'   : dict con threshold óptimo y métricas asociadas
        - 'model_path'  : ruta de guardado (o None si output_path no se pasó)
        - 'X_test'      : features de test
        - 'y_test'      : target de test
        - 'groups_train': grupos de advertisers del train (para CV externo)
    """
    X_train, X_test, y_train, y_test = split_by_advertiser(
        df, target=target, test_size=test_size
    )
    groups_train = df.loc[X_train.index, "advertiser_zrive_id"]

    # Compensar desbalanceo de clases con sample_weight
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = np.where(y_train == 1, scale, 1.0)

    model = train_model(
        X_train, y_train,
        model_class=model_class,
        params=params,
        sample_weight=sample_weight,
    )

    metrics = evaluate_model(model, X_test, y_test, name=model.__class__.__name__)
    threshold_info = find_best_threshold(y_test, metrics["y_proba"])

    model_path = None
    if output_path:
        model_path = save_model(model, output_path)

    logger.info(
        "Pipeline de modelado completado — ROC AUC=%.3f, threshold óptimo=%.3f",
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
    }