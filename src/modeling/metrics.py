import logging
from typing import Any, List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, GroupKFold

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, threshold=0.5, name="Modelo"):
    """
    Evalúa un modelo binario usando probabilidades y un threshold configurable.

    Parameters
    ----------
    model : estimator
        Modelo entrenado con predict_proba.
    X_test : DataFrame
        Features de test.
    y_test : Series or array-like
        Target real.
    threshold : float, default=0.5
        Punto de corte para convertir probabilidades en clase predicha.
    name : str, default="Modelo"
        Nombre del modelo.

    Returns
    -------
    dict
        Métricas, predicciones y probabilidades.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    results = {
        "name": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "average_precision": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "log_loss": log_loss(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    print(f"--- Evaluación: {name} ---")
    print(f"Threshold: {threshold:.4f}")
    print(f"Average Precision / PR AUC: {results['average_precision']:.3f}")
    print(f"ROC AUC: {results['roc_auc']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1: {results['f1']:.3f}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Log Loss: {results['log_loss']:.3f}")
    print(f"Brier Score: {results['brier_score']:.3f}")

    print("\nMatriz de confusión:")
    print(results["confusion_matrix"])

    print("\nClassification report:")
    print(results["classification_report"])

    return results

def cross_validate_models(models, X_train, y_train, groups=None, scoring="average_precision", return_oof_preds=False):
    """
    Cross-validation de varios modelos con GroupKFold.

    Parameters
    ----------
    models : dict
        {nombre: modelo} ya entrenados.
    X_train, y_train : arrays
    groups : array-like, optional
        Grupos para GroupKFold.
    scoring : str, default="average_precision"
        Métrica de evaluación a usar en la validación cruzada.
    return_oof_preds : bool, default=False
        Si es True, devuelve una tupla (DataFrame, dict_probabilidades_oof).

    Returns
    -------
    pd.DataFrame (si return_oof_preds=False)
    Tuple[pd.DataFrame, dict] (si return_oof_preds=True)
    """
    cv = GroupKFold(n_splits=5) if groups is not None else 5
    results = []
    oof_preds = {}

    # Para que los nombres de las columnas en el dataframe queden bonitos
    score_name = scoring.replace("_", " ").title() if isinstance(scoring, str) else "Score"

    for name, model in models.items():
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            groups=groups,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True,
        )
        
        if return_oof_preds:
            # Calculamos las predicciones Out-Of-Fold (OOF) para encontrar el mejor threshold sin data leakage
            oof_proba = cross_val_predict(
                model,
                X_train,
                y_train,
                cv=cv,
                groups=groups,
                n_jobs=-1,
                method="predict_proba"
            )[:, 1]
            oof_preds[name] = oof_proba

        train_score_mean = cv_results["train_score"].mean()
        train_score_std = cv_results["train_score"].std()
        test_score_mean = cv_results["test_score"].mean()
        test_score_std = cv_results["test_score"].std()
        gap = train_score_mean - test_score_mean

        results.append({
            "Modelo": name,
            f"Train {score_name} Mean": train_score_mean,
            f"CV {score_name} Mean": test_score_mean,
            f"CV {score_name} Std": test_score_std,
            "Gap (Train - CV)": gap,
        })
        logger.info(
            "CV %s: Train=%.3f (±%.3f) | CV=%.3f (±%.3f) | Gap=%.3f",
            name, train_score_mean, train_score_std, test_score_mean, test_score_std, gap
        )

    df_results = pd.DataFrame(results)
    
    if return_oof_preds:
        return df_results, oof_preds
    return df_results

import numpy as np
from sklearn.metrics import precision_recall_curve

def find_best_threshold(
    y_true,
    y_proba,
    optimize_for="f1",
    min_precision=None,
    min_recall=None,
):
    """
    Encuentra el threshold óptimo según una métrica o restricción de negocio.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales.
    y_proba : array-like
        Probabilidades predichas para la clase positiva.
    optimize_for : str, default="f1"
        Métrica a maximizar: "f1", "precision", "recall".
    min_precision : float, optional
        Precision mínima exigida. Ejemplo: 0.4.
    min_recall : float, optional
        Recall mínimo exigido. Ejemplo: 0.7.

    Returns
    -------
    dict
        Threshold y métricas asociadas en ese punto.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # thresholds tiene un elemento menos que precision/recall
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    f1_scores = (
        2 * precisions * recalls 
        / (precisions + recalls + 1e-8)
    )

    valid_mask = np.ones_like(thresholds, dtype=bool)

    if min_precision is not None:
        valid_mask &= precisions >= min_precision

    if min_recall is not None:
        valid_mask &= recalls >= min_recall

    if not valid_mask.any():
        raise ValueError(
            "No hay ningún threshold que cumpla las restricciones indicadas."
        )

    if optimize_for == "f1":
        scores = np.where(valid_mask, f1_scores, -np.inf)
    elif optimize_for == "precision":
        scores = np.where(valid_mask, precisions, -np.inf)
    elif optimize_for == "recall":
        scores = np.where(valid_mask, recalls, -np.inf)
    else:
        raise ValueError(
            f"optimize_for debe ser 'f1', 'precision' o 'recall'. Recibido: {optimize_for}"
        )

    best_idx = np.argmax(scores)

    return {
        "threshold": thresholds[best_idx],
        "precision": precisions[best_idx],
        "recall": recalls[best_idx],
        "f1": f1_scores[best_idx],
        "optimized_for": optimize_for,
        "min_precision": min_precision,
        "min_recall": min_recall,
    }

def build_risk_profiles(model, X, y, y_proba=None, bins=None, labels=None):
    """
    Segmenta contratos en perfiles de riesgo y devuelve estadisticas.

    Parameters
    ----------
    model : estimator
        Modelo entrenado.
    X : pd.DataFrame
        Dataset (pueden ser features o el DF completo restringido a test).
    y : pd.Series
        Target real.
    y_proba : np.ndarray, optional
        Probabilidades ya calculadas. Si no se pasan, se calculan usando X.
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
    
    if y_proba is None:
        # Intentar predecir. Nota: sklearn falla si X tiene columnas extra no vistas en fit.
        # Si esto falla en el futuro, es mejor pasar y_proba desde fuera.
        if hasattr(model, "feature_names_in_"):
            y_proba = model.predict_proba(X[model.feature_names_in_])[:, 1]
        else:
            y_proba = model.predict_proba(X)[:, 1]
            
    df["churn_proba"] = y_proba
    df["churned"] = y.values
    df["riesgo"] = pd.cut(df["churn_proba"], bins=bins, labels=labels)

    # Columnas de negocio de interes si estan presentes
    business_cols = {
        "monthly_total_invoice": "Facturacion media",
        "monthly_leads": "Leads medio",
        "monthly_visits": "Visitas media",
        "usage_ratio": "Tasa de uso",
        "visit_to_lead_rate": "Conv. Visita-Lead",
        "leads_per_invoice": "ROI (Leads/€)",
    }

    profiles = []
    for riesgo in labels:
        mask = df["riesgo"] == riesgo
        profile = {
            "Riesgo": riesgo,
            "N contratos": int(mask.sum()),
            "Churn real": df.loc[mask, "churned"].mean(),
        }
        # Añadir metricas de negocio dinamicamente
        for col, col_label in business_cols.items():
            if col in df.columns:
                profile[col_label] = df.loc[mask, col].mean()

        profiles.append(profile)

    return pd.DataFrame(profiles)

def compute_top_k_metrics(X_test, y_test, y_proba, revenue_col=None, k_list=None):
    """
    Calcula precision y recall (y opcionalmente facturacion salvada) 
    para los top K% clientes con mas riesgo.

    Parameters
    ----------
    X_test : pd.DataFrame
    y_test : pd.Series
    y_proba : np.ndarray
        Probabilidades de churn.
    revenue_col : str, optional
        Nombre de la columna de facturacion (ej: 'monthly_total_invoice').
    k_list : list[int], optional
        Lista de percentiles top a evaluar. Default: [5, 10, 20].

    Returns
    -------
    pd.DataFrame con metricas de negocio por cada K%.
    """
    if k_list is None:
        k_list = [5, 10, 20]

    sorted_idx = np.argsort(-y_proba)
    n_total = len(y_test)
    total_churns = y_test.sum()

    results = []
    for k in k_list:
        n_int = int(n_total * k / 100)
        if n_int == 0:
            continue

        top_idx = sorted_idx[:n_int]
        tp = y_test.iloc[top_idx].sum()
        precision = tp / n_int
        recall = tp / total_churns if total_churns > 0 else 0

        res = {
            "Top %": k,
            "Intervenidos": n_int,
            "Churns capturados": tp,
            "Precision": precision,
            "Recall": recall,
        }

        if revenue_col and revenue_col in X_test.columns:
            invoices = X_test[revenue_col].values
            inv_saved = invoices[top_idx][y_test.iloc[top_idx] == 1].sum()
            res["Facturacion salvada"] = inv_saved

        results.append(res)

    return pd.DataFrame(results)

def print_classification_report(model_name: str, y_true, y_pred) -> Dict[str, Any]:
    """
    Imprime classification report y confusion matrix.
    Devuelve ambos en un diccionario por si se quieren reutilizar.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Classification Report (threshold=0.5)")
    print(f"{'='*60}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )
    print("Confusion Matrix:")
    print(cm)
    print()

    return {"report": report, "confusion_matrix": cm}

