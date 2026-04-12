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
            model, X_train, y_train,
            cv=cv, groups=groups,
            scoring="roc_auc", n_jobs=-1,
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

    return {
        "threshold": thresholds[best_idx],
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
    # Excluir baseline de las curvas ROC
    model_results = [r for r in results if r["roc_auc"] > 0.5]
    best = max(model_results, key=lambda r: r["roc_auc"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in model_results:
        RocCurveDisplay.from_predictions(
            y_test, r["y_proba"], name=r["name"], ax=axes[0],
        )
    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set_title("Curvas ROC")
    axes[0].legend()

    ConfusionMatrixDisplay.from_predictions(
        y_test, best["y_pred"],
        display_labels=["No churn", "Churn"],
        ax=axes[1],
    )
    axes[1].set_title(f"Confusion Matrix - {best['name']}")

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, X_test, y_test, feature_cols, top_n=15):
    """Feature importance con permutation importance."""
    perm_imp = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=42, scoring="roc_auc",
    )
    perm_series = pd.Series(
        perm_imp.importances_mean, index=feature_cols,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    perm_series.tail(top_n).plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top_n} features (permutation importance)")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    plt.show()


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
