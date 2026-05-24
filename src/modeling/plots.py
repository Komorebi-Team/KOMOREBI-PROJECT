import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    roc_auc_score,
    average_precision_score,
)
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

def compute_shap_values(model, X_train, X_test):
    """
    Calcula valores SHAP para el conjunto de test.

    Parameters
    ----------
    model : estimator
    X_train : pd.DataFrame
        Usado como background para el explainer.
    X_test : pd.DataFrame
        Conjunto para el cual calcular explicaciones.

    Returns
    -------
    shap_values o None si falla.
    """
    try:
        import shap
    except ImportError:
        logger.warning("La libreria 'shap' no esta instalada. No se calcularan valores SHAP.")
        return None

    try:
        # Usar el subset de columnas que el modelo conoce para evitar errores
        if hasattr(model, "feature_names_in_"):
            X_train_sub = X_train[model.feature_names_in_]
            X_test_sub = X_test[model.feature_names_in_]
        else:
            X_train_sub = X_train
            X_test_sub = X_test

        logger.info("Calculando valores SHAP (esto puede tardar unos segundos)...")
        # Explainer automatico (selecciona TreeExplainer para modelos de arboles)
        explainer = shap.Explainer(model, X_train_sub)
        shap_values = explainer(X_test_sub, check_additivity=False)
        
        return shap_values
    except Exception as e:
        logger.error("Error al calcular SHAP: %s", e)
        return None

def plot_metrics(
    model_name: str,
    y_score: np.ndarray,
    y_test: pd.Series,
    figure: Optional[Tuple[plt.Figure, np.ndarray]] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, float]]:
    """
    Dibuja curvas Precision-Recall y ROC para un modelo binario.

    Parameters
    ----------
    model_name : str
        Nombre del modelo para mostrar en la leyenda.
    y_score : np.ndarray
        Probabilidades o scores de la clase positiva.
    y_test : pd.Series
        Target real binario.
    figure : tuple[plt.Figure, np.ndarray], optional
        Figura y ejes existentes para superponer curvas de varios modelos.

    Returns
    -------
    tuple
        (fig, ax, metrics_dict), donde metrics_dict contiene:
        - pr_auc
        - average_precision
        - roc_auc
        - prevalence
    """
    precision_, recall_, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall_, precision_)
    average_precision = average_precision_score(y_test, y_score)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    if figure is None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax = figure

    positive_prevalence = y_test.mean()

    # Precision-Recall
    prev_label = f"Prevalencia ({positive_prevalence:.3f})"
    existing_labels = [line.get_label() for line in ax[0].get_lines()]
    if prev_label not in existing_labels:
        ax[0].axhline(
            positive_prevalence,
            linestyle="--",
            color="gray",
            alpha=0.6,
            label=prev_label,
        )

    ax[0].plot(
        recall_,
        precision_,
        label=f"{model_name} (AP={average_precision:.3f}, PR AUC={pr_auc:.3f})",
    )
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-Recall Curve")
    ax[0].legend(loc="upper right", fontsize=9)

    # ROC
    baseline_label = "Baseline aleatorio"
    existing_labels_roc = [line.get_label() for line in ax[1].get_lines()]
    if baseline_label not in existing_labels_roc:
        ax[1].plot(
            [0, 1], [0, 1],
            linestyle="--",
            color="gray",
            alpha=0.6,
            label=baseline_label,
        )

    ax[1].plot(
        fpr,
        tpr,
        label=f"{model_name} (ROC AUC={roc_auc:.3f})",
    )
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("ROC Curve")
    ax[1].legend(loc="lower right", fontsize=9)

    metrics = {
        "pr_auc": pr_auc,
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "prevalence": positive_prevalence,
    }

    return fig, ax, metrics

def plot_feature_importance(
    model,
    X_test,
    y_test,
    feature_cols=None,
    top_n=15,
    use_shap=False,
    shap_sample=None,
    shap_summary=True,
):
    """
    Muestra importancia de variables con permutation importance o SHAP.

    Parameters
    ----------
    model : estimator
        Modelo entrenado.
    X_test : pd.DataFrame
        Conjunto de features sobre el que calcular importancias.
    y_test : pd.Series | np.ndarray
        Target real. Solo se usa para permutation importance.
    feature_cols : list[str], optional
        Nombres de columnas. Si X_test es DataFrame y no se pasa, se usan sus columnas.
    top_n : int, default=15
        Numero de variables a mostrar en el grafico de barras.
    use_shap : bool, default=False
        Si True, usa SHAP. Si False, usa permutation importance.
    shap_sample : int, optional
        Numero maximo de observaciones sobre las que calcular SHAP.
        Util para reducir coste computacional.
    shap_summary : bool, default=True
        Si use_shap=True, muestra tambien el summary plot ademas del bar plot.

    Returns
    -------
    pd.Series
        Serie con las importancias medias por variable.
    """
    if feature_cols is None:
        if isinstance(X_test, pd.DataFrame):
            feature_cols = X_test.columns.tolist()
        else:
            raise ValueError(
                "Si X_test no es un DataFrame, debes proporcionar feature_cols."
            )

    if not use_shap:
        perm_imp = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="roc_auc",
        )
        importance_series = pd.Series(
            perm_imp.importances_mean,
            index=feature_cols,
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        importance_series.tail(top_n).plot(kind="barh", ax=ax)
        ax.set_title(f"Top {top_n} features (permutation importance)")
        ax.set_xlabel("Importancia")
        plt.tight_layout()
        plt.show()

        return importance_series

    # ---- SHAP ----
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "Para usar SHAP debes instalarlo: pip install shap"
        ) from exc

    if not isinstance(X_test, pd.DataFrame):
        X_shap = pd.DataFrame(X_test, columns=feature_cols)
    else:
        X_shap = X_test.copy()

    if shap_sample is not None and len(X_shap) > shap_sample:
        X_shap = X_shap.sample(shap_sample, random_state=42)

    logger.info("Calculando SHAP values sobre %d observaciones...", len(X_shap))

    try:
        explainer = shap.Explainer(model, X_shap)
        shap_values = explainer(X_shap)
        values = shap_values.values

    except Exception:
        logger.warning(
            "No se pudo usar shap.Explainer de forma generica. "
            "Probando con KernelExplainer, que puede ser mas lento."
        )

        background = X_shap.sample(min(100, len(X_shap)), random_state=42)

        if hasattr(model, "predict_proba"):
            explainer = shap.KernelExplainer(model.predict_proba, background)
            values = explainer.shap_values(X_shap)
            if isinstance(values, list):
                values = values[1]
        else:
            explainer = shap.KernelExplainer(model.predict, background)
            values = explainer.shap_values(X_shap)

        shap_values = None

    if values.ndim == 3:
        values = values[:, :, 1]

    mean_abs_shap = np.abs(values).mean(axis=0)
    importance_series = pd.Series(
        mean_abs_shap,
        index=X_shap.columns,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance_series.tail(top_n).plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top_n} features (SHAP mean |value|)")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.show()

    if shap_summary:
        if shap_values is not None:
            shap.summary_plot(shap_values, X_shap, show=True)
        else:
            shap.summary_plot(values, X_shap, show=True)

    return importance_series

