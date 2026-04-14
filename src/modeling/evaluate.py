import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, GroupKFold

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, name="Modelo"):
    """Evalua un modelo y devuelve metricas basicas."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = model.score(X_test, y_test)
    auc = roc_auc_score(y_test, y_proba)

    logger.info("%s — Accuracy: %.3f, ROC AUC: %.3f", name, accuracy, auc)

    return {
        "name": name,
        "accuracy": accuracy,
        "roc_auc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def cross_validate_models(models, X_train, y_train, groups=None):
    """
    Cross-validation de varios modelos con GroupKFold.

    Parameters
    ----------
    models : dict
        {nombre: modelo} ya entrenados.
    X_train, y_train : arrays
    groups : array-like, optional
        Grupos para GroupKFold.

    Returns
    -------
    pd.DataFrame con mean y std de ROC AUC por modelo.
    """
    cv = GroupKFold(n_splits=5) if groups is not None else 5
    results = []

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            groups=groups,
            scoring="roc_auc",
            n_jobs=-1,
        )
        results.append({
            "Modelo": name,
            "Mean AUC": scores.mean(),
            "Std AUC": scores.std(),
        })
        logger.info("CV %s: mean=%.3f, std=%.3f", name, scores.mean(), scores.std())

    return pd.DataFrame(results)


def find_best_threshold(y_test, y_proba):
    """Encuentra el threshold que maximiza F1 para la clase positiva."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)

    # precision_recall_curve devuelve un threshold menos que precisions/recalls
    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0

    return {
        "threshold": threshold,
        "precision": precisions[best_idx],
        "recall": recalls[best_idx],
        "f1": f1_scores[best_idx],
    }


def plot_comparativa(results, y_test):
    """
    Genera graficos de comparativa: curvas ROC y confusion matrix del mejor modelo.

    Parameters
    ----------
    results : list[dict]
        Lista de resultados de evaluate_model.
    y_test : array
    """
    model_results = [r for r in results if r["roc_auc"] > 0.5]
    best = max(model_results, key=lambda r: r["roc_auc"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in model_results:
        RocCurveDisplay.from_predictions(
            y_test,
            r["y_proba"],
            name=r["name"],
            ax=axes[0],
        )
    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set_title("Curvas ROC")
    axes[0].legend()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best["y_pred"],
        display_labels=["No churn", "Churn"],
        ax=axes[1],
    )
    axes[1].set_title(f"Confusion Matrix - {best['name']}")

    plt.tight_layout()
    plt.show()


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


def build_risk_profiles(model, X, y, bins=None, labels=None):
    """
    Segmenta contratos en perfiles de riesgo y devuelve estadisticas.

    Parameters
    ----------
    model : estimator
        Modelo entrenado.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target real.
    bins : list, optional
        Limites de los buckets de probabilidad. Default: [0, 0.1, 0.3, 1.0]
    labels : list, optional
        Nombres de los buckets. Default: ["Bajo", "Medio", "Alto"]

    Returns
    -------
    pd.DataFrame con perfil de cada segmento.
    """
    if bins is None:
        bins = [0, 0.1, 0.3, 1.0]
    if labels is None:
        labels = ["Bajo", "Medio", "Alto"]

    df = X.copy()
    df["churn_proba"] = model.predict_proba(X)[:, 1]
    df["churned"] = y.values
    df["riesgo"] = pd.cut(df["churn_proba"], bins=bins, labels=labels)

    profiles = []
    for riesgo in labels:
        mask = df["riesgo"] == riesgo
        profiles.append({
            "Riesgo": riesgo,
            "N contratos": mask.sum(),
            "Churn real": df.loc[mask, "churned"].mean(),
            "Facturacion media": df.loc[mask, "monthly_total_invoice"].mean(),
            "Leads medio": df.loc[mask, "monthly_leads"].mean(),
            "Visitas media": df.loc[mask, "monthly_visits"].mean(),
            "Usage ratio": df.loc[mask, "usage_ratio"].mean(),
        })

    return pd.DataFrame(profiles)