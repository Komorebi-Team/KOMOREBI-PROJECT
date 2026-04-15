from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation, record_evaluation
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from src.modeling.evaluate import evaluate_model, find_best_threshold
from src.pipelines.feature_engineering import run_feature_engineering_pipeline
from src.pipelines.preprocessing import run_preprocessing_pipeline


DEFAULT_DATA_DIR = Path("data")
DEFAULT_FCT_PATH = DEFAULT_DATA_DIR / "zrive_fct_monthly_snapshot_advertiser.parquet"
DEFAULT_DIM_PATH = DEFAULT_DATA_DIR / "zrive_dim_advertiser.parquet"
DEFAULT_OUTPUT_DIR = Path("data/processed/models/lightgbm")
TARGET_COLUMN = "churned_3m"
PRICE_COLUMN = "stable_price_mean"

PROVINCE_TO_CCAA = {
    "A Coruña": "Galicia",
    "Álava": "Pais Vasco",
    "Albacete": "Castilla-La Mancha",
    "Alicante": "Comunidad Valenciana",
    "Almería": "Andalucia",
    "Asturias": "Asturias",
    "Ávila": "Castilla y Leon",
    "Badajoz": "Extremadura",
    "Barcelona": "Cataluna",
    "Burgos": "Castilla y Leon",
    "Cáceres": "Extremadura",
    "Cádiz": "Andalucia",
    "Cantabria": "Cantabria",
    "Castellón": "Comunidad Valenciana",
    "Ceuta": "Ceuta",
    "Ciudad Real": "Castilla-La Mancha",
    "Córdoba": "Andalucia",
    "Cuenca": "Castilla-La Mancha",
    "Girona": "Cataluna",
    "Granada": "Andalucia",
    "Guadalajara": "Castilla-La Mancha",
    "Guipúzcoa": "Pais Vasco",
    "Huelva": "Andalucia",
    "Huesca": "Aragon",
    "Islas Baleares": "Islas Baleares",
    "Jaén": "Andalucia",
    "La Coruña": "Galicia",
    "La Rioja": "La Rioja",
    "Las Palmas": "Canarias",
    "León": "Castilla y Leon",
    "Lleida": "Cataluna",
    "Lugo": "Galicia",
    "Madrid": "Comunidad de Madrid",
    "Málaga": "Andalucia",
    "Melilla": "Melilla",
    "Murcia": "Region de Murcia",
    "Navarra": "Navarra",
    "Orense": "Galicia",
    "Ourense": "Galicia",
    "Palencia": "Castilla y Leon",
    "Pontevedra": "Galicia",
    "Salamanca": "Castilla y Leon",
    "Santa Cruz de Tenerife": "Canarias",
    "Segovia": "Castilla y Leon",
    "Sevilla": "Andalucia",
    "Soria": "Castilla y Leon",
    "Tarragona": "Cataluna",
    "Tenerife": "Canarias",
    "Teruel": "Aragon",
    "Toledo": "Castilla-La Mancha",
    "Valencia": "Comunidad Valenciana",
    "Valladolid": "Castilla y Leon",
    "Vizcaya": "Pais Vasco",
    "Zamora": "Castilla y Leon",
    "Zaragoza": "Aragon",
}


def load_raw_inputs(
    fct_path: Path = DEFAULT_FCT_PATH,
    dim_path: Path = DEFAULT_DIM_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga las tablas base necesarias para construir el dataset contractual."""
    return pd.read_parquet(fct_path), pd.read_parquet(dim_path)


def build_contract_features(
    n_months: int = 3,
    include_right_censored: bool = False,
) -> pd.DataFrame:
    """Ejecuta preprocessing y feature engineering con la estructura actual del repo."""
    df_fct, df_dim = load_raw_inputs()
    df_contracts, _ = run_preprocessing_pipeline(df_fct, df_dim)
    return run_feature_engineering_pipeline(
        df_contracts,
        n_months=n_months,
        include_right_censored=include_right_censored,
    )


def map_province_to_ccaa(province: object) -> str:
    """Agrupa provincia en CCAA para reducir cardinalidad y hacer el one-hot posterior."""
    if pd.isna(province):
        return "Unknown"
    province_str = str(province).strip()
    return PROVINCE_TO_CCAA.get(province_str, "Unknown")


def relativize_monthly_features(
    df: pd.DataFrame,
    price_col: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Construye monthly_* relativas al precio salvo para columnas que deben
    mantenerse en nivel original.

    Se relativizan las columnas monthly_* excepto aquellas cuyo nombre incluye:
    - price
    - std
    - invoice
    - trend
    """
    out = df.copy()
    monthly_cols = [
        col
        for col in out.columns
        if col.startswith("monthly_") and not col.endswith("_was_missing")
    ]
    keep_absolute_cols = [
        col
        for col in monthly_cols
        if any(token in col for token in ["price", "std", "invoice", "trend"])
    ]
    monthly_cols_to_relativize = [
        col for col in monthly_cols if col not in keep_absolute_cols
    ]

    divisor = out[price_col].replace(0, np.nan)
    relative_cols: list[str] = []
    for col in monthly_cols_to_relativize:
        rel_col = f"{col}_rel_price"
        out[rel_col] = out[col] / divisor
        relative_cols.append(rel_col)

    return out, relative_cols, keep_absolute_cols


def prepare_lightgbm_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepara el dataframe final para LightGBM:
    - mensual relativo a stable_price_mean
    - provincia agrupada por CCAA con one-hot
    - seleccion explicita de variables finales
    """
    out = df.copy()
    if PRICE_COLUMN not in out.columns:
        raise KeyError(
            f"No se ha encontrado '{PRICE_COLUMN}' en el dataframe final de features."
        )

    out["ccaa"] = (
        out["advertiser_province"]
        .apply(map_province_to_ccaa)
        .str.replace(r"[^0-9A-Za-z]+", "_", regex=True)
        .str.strip("_")
    )
    out = pd.get_dummies(out, columns=["ccaa"], prefix="ccaa", dtype=int)
    out, relative_monthly_cols, absolute_monthly_cols = relativize_monthly_features(
        out, price_col=PRICE_COLUMN
    )

    ratio_cols = [c for c in out.columns if c.startswith(("usage_", "cost_", "conversion_", "premium_"))]
    price_cols = [PRICE_COLUMN]
    missing_flag_cols = [c for c in out.columns if c.endswith("_was_missing")]
    ccaa_cols = [c for c in out.columns if c.startswith("ccaa_")]
    optional_cols: list[str] = []

    feature_cols = (
        relative_monthly_cols
        + absolute_monthly_cols
        + ratio_cols
        + price_cols
        + missing_flag_cols
        + ccaa_cols
        + optional_cols
    )
    feature_cols = list(dict.fromkeys([c for c in feature_cols if c in out.columns]))
    return out, feature_cols


def split_train_valid_test(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str = TARGET_COLUMN,
    test_size: float = 0.2,
    valid_size_within_train: float = 0.2,
    random_state: int = 42,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Hace un split outer train/test y un split inner train/valid, ambos agrupados por advertiser."""
    groups = df["advertiser_zrive_id"]
    X = df[feature_cols].copy()
    y = df[target].astype(int).copy()

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(outer.split(X, y, groups=groups))

    X_train_val = X.iloc[train_val_idx].copy()
    y_train_val = y.iloc[train_val_idx].copy()
    groups_train_val = groups.iloc[train_val_idx].copy()

    inner = GroupShuffleSplit(n_splits=1, test_size=valid_size_within_train, random_state=random_state)
    inner_train_idx, valid_idx = next(inner.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train = X_train_val.iloc[inner_train_idx].copy()
    y_train = y_train_val.iloc[inner_train_idx].copy()
    X_valid = X_train_val.iloc[valid_idx].copy()
    y_valid = y_train_val.iloc[valid_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_train_val": X_train_val,
        "y_train_val": y_train_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }


def build_split_indices(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = 0.2,
    valid_size_within_train: float = 0.2,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Construye una única partición por advertiser para reutilizarla entre variantes."""
    groups = df["advertiser_zrive_id"]
    y = df[target].astype(int).copy()
    row_index = np.arange(len(df))

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(outer.split(row_index, y, groups=groups))

    groups_train_val = groups.iloc[train_val_idx].copy()
    y_train_val = y.iloc[train_val_idx].copy()

    inner = GroupShuffleSplit(n_splits=1, test_size=valid_size_within_train, random_state=random_state)
    inner_train_idx, valid_idx_relative = next(
        inner.split(train_val_idx, y_train_val, groups=groups_train_val)
    )

    train_idx = train_val_idx[inner_train_idx]
    valid_idx = train_val_idx[valid_idx_relative]

    return {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "train_val_idx": train_val_idx,
        "test_idx": test_idx,
    }


def materialize_split_from_indices(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_indices: dict[str, np.ndarray],
    target: str = TARGET_COLUMN,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Materializa X/y para una variante concreta manteniendo el mismo split."""
    X = df[feature_cols].copy()
    y = df[target].astype(int).copy()

    train_idx = split_indices["train_idx"]
    valid_idx = split_indices["valid_idx"]
    train_val_idx = split_indices["train_val_idx"]
    test_idx = split_indices["test_idx"]

    return {
        "X_train": X.iloc[train_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "X_valid": X.iloc[valid_idx].copy(),
        "y_valid": y.iloc[valid_idx].copy(),
        "X_train_val": X.iloc[train_val_idx].copy(),
        "y_train_val": y.iloc[train_val_idx].copy(),
        "X_test": X.iloc[test_idx].copy(),
        "y_test": y.iloc[test_idx].copy(),
        "feature_cols": feature_cols,
    }


def build_lightgbm_params(
    y_train: pd.Series,
    random_state: int = 42,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> dict[str, object]:
    """Configura LightGBM con un número alto de árboles para poder parar por early stopping."""
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / positives if positives else 1.0

    return {
        "objective": "binary",
        "n_estimators": 100,
        "learning_rate": 0.01,
        "num_leaves": 7,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "min_child_samples": 50,
        "min_split_gain": 0.01,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "force_col_wise": True,
        "n_jobs": -1,
    }


def train_lightgbm_with_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int = 42,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> tuple[LGBMClassifier, dict[str, dict[str, list[float]]]]:
    """Entrena LightGBM controlando aprendizaje con un conjunto de validación separado."""
    evals_result: dict[str, dict[str, list[float]]] = {}
    params = build_lightgbm_params(
        y_train,
        random_state=random_state,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
    )
    model = LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_names=["train", "valid"],
        eval_metric=["binary_logloss"],
        callbacks=[
            record_evaluation(evals_result),
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=0),
        ],
    )
    return model, evals_result


def refit_final_model(
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    best_iteration: int,
    random_state: int = 42,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> LGBMClassifier:
    """Reentrena el modelo final con train+valid usando el número óptimo de iteraciones."""
    params = build_lightgbm_params(
        y_train_val,
        random_state=random_state,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
    )
    params["n_estimators"] = int(best_iteration)
    model = LGBMClassifier(**params)
    model.fit(X_train_val, y_train_val)
    return model


def plot_learning_curve(
    evals_result: dict[str, dict[str, list[float]]],
    metric_name: str,
    output_path: Path,
) -> None:
    """Guarda la evolución de train/valid para una métrica de LightGBM."""
    plt.figure(figsize=(10, 6))
    for dataset_name, metrics in evals_result.items():
        if metric_name in metrics:
            plt.plot(metrics[metric_name], label=f"{dataset_name}_{metric_name}")
    plt.xlabel("Boosting iteration")
    plt.ylabel(metric_name)
    plt.title(f"LightGBM learning curve - {metric_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(model: LGBMClassifier, feature_cols: list[str], output_path: Path, top_n: int = 20) -> None:
    """Guarda las variables más importantes del modelo final."""
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    top = importance[importance > 0].tail(top_n)
    if top.empty:
        top = importance.tail(top_n)
    plt.figure(figsize=(10, 8))
    top.plot(kind="barh")
    plt.title("LightGBM feature importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_training_history(
    evals_result: dict[str, dict[str, list[float]]],
    output_path: Path,
) -> pd.DataFrame:
    """Guarda el histórico de métricas por iteración en formato largo."""
    records: list[dict[str, object]] = []
    for dataset_name, metrics in evals_result.items():
        for metric_name, values in metrics.items():
            for iteration, value in enumerate(values, start=1):
                records.append(
                    {
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "iteration": iteration,
                        "value": value,
                    }
                )
    history = pd.DataFrame(records)
    history.to_csv(output_path, index=False)
    return history


def save_feature_list(feature_cols: list[str], output_path: Path) -> pd.DataFrame:
    """Guarda la lista final de features del modelo."""
    feature_df = pd.DataFrame({"feature": feature_cols})
    feature_df.to_csv(output_path, index=False)
    return feature_df


def save_model_summary(
    metrics: dict[str, object],
    threshold_info: dict[str, float],
    best_iteration: int,
    split_info: dict[str, int | float],
    feature_count: int,
    output_path: Path,
) -> pd.DataFrame:
    """Guarda un resumen compacto del modelo final."""
    summary = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "best_iteration": int(best_iteration),
                "feature_count": int(feature_count),
                **split_info,
                "test_accuracy": float(metrics["accuracy"]),
                "test_roc_auc": float(metrics["roc_auc"]),
                "test_average_precision": float(metrics["average_precision_score"]),
                "test_log_loss": float(metrics["log_loss"]),
                "test_brier_score_loss": float(metrics["brier_score_loss"]),
                "test_matthews_corrcoef": float(metrics["matthews_corrcoef"]),
                "best_threshold": float(threshold_info["threshold"]),
                "best_threshold_precision": float(threshold_info["precision"]),
                "best_threshold_recall": float(threshold_info["recall"]),
                "best_threshold_f1": float(threshold_info["f1"]),
            }
        ]
    )
    summary.to_csv(output_path, index=False)
    return summary


def filter_feature_variant(feature_cols: list[str], variant: str) -> list[str]:
    """Devuelve la lista de features para una variante de ablación."""
    if variant == "baseline":
        filtered = feature_cols
    elif variant == "no_stable_price":
        filtered = [c for c in feature_cols if c != "stable_price_mean"]
    elif variant == "no_stable_price_no_invoice_price_trend":
        filtered = [
            c
            for c in feature_cols
            if c != "stable_price_mean"
            and not (
                c.startswith("monthly_")
                and any(token in c for token in ["invoice", "price", "trend"])
            )
        ]
    elif variant == "no_stable_price_no_price_trend":
        filtered = [
            c
            for c in feature_cols
            if c != "stable_price_mean"
            and not (
                c.startswith("monthly_")
                and any(token in c for token in ["price", "trend"])
            )
        ]
    elif variant == "no_stable_price_no_invoice_price_trend_std":
        filtered = [
            c
            for c in feature_cols
            if c != "stable_price_mean"
            and not (
                c.startswith("monthly_")
                and any(token in c for token in ["invoice", "price", "trend", "std"])
            )
        ]
    else:
        raise ValueError(f"Variante desconocida: {variant}")

    return list(dict.fromkeys(filtered))


def run_single_lightgbm_variant(
    df_model: pd.DataFrame,
    base_feature_cols: list[str],
    split_indices: dict[str, np.ndarray],
    variant: str,
    output_dir: Path,
    random_state: int = 42,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> dict[str, object]:
    """Entrena y evalúa una variante del set de features sobre el mismo split."""
    feature_cols = filter_feature_variant(base_feature_cols, variant)
    split = materialize_split_from_indices(df_model, feature_cols, split_indices)

    train_model, evals_result = train_lightgbm_with_validation(
        split["X_train"],
        split["y_train"],
        split["X_valid"],
        split["y_valid"],
        random_state=random_state,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
    )
    best_iteration = train_model.best_iteration_ or train_model.n_estimators
    final_model = refit_final_model(
        split["X_train_val"],
        split["y_train_val"],
        best_iteration=best_iteration,
        random_state=random_state,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
    )

    metrics = evaluate_model(
        final_model,
        split["X_test"],
        split["y_test"],
        name=f"LightGBM_{variant}",
        extra_metrics=["log_loss", "average_precision_score", "brier_score_loss", "matthews_corrcoef"],
    )
    threshold_info = find_best_threshold(split["y_test"], metrics["y_proba"], optimize_for="f1")

    output_dir.mkdir(parents=True, exist_ok=True)
    logloss_curve_path = output_dir / "learning_curve_logloss.png"
    importance_path = output_dir / "feature_importance.png"
    history_path = output_dir / "training_history.csv"
    feature_list_path = output_dir / "feature_list.csv"
    summary_path = output_dir / "model_summary.csv"
    model_path = output_dir / "lightgbm_model.pkl"

    plot_learning_curve(evals_result, "binary_logloss", logloss_curve_path)
    plot_feature_importance(final_model, split["feature_cols"], importance_path)
    history = save_training_history(evals_result, history_path)
    feature_list = save_feature_list(split["feature_cols"], feature_list_path)
    summary = save_model_summary(
        metrics,
        threshold_info,
        best_iteration,
        split_info={
            "train_rows": int(len(split["X_train"])),
            "valid_rows": int(len(split["X_valid"])),
            "test_rows": int(len(split["X_test"])),
        },
        feature_count=len(split["feature_cols"]),
        output_path=summary_path,
    )
    joblib.dump(final_model, model_path)

    return {
        "variant": variant,
        "metrics": metrics,
        "threshold": threshold_info,
        "summary": summary,
        "history": history,
        "feature_list": feature_list,
        "best_iteration": best_iteration,
        "outputs": {
            "logloss_curve": logloss_curve_path,
            "importance": importance_path,
            "history": history_path,
            "feature_list": feature_list_path,
            "summary": summary_path,
            "model": model_path,
        },
    }


def run_lightgbm_feature_ablation_study(random_state: int = 42) -> dict[str, object]:
    """Compara varias variantes del set de features sobre el mismo split."""
    df_features = build_contract_features(n_months=3, include_right_censored=False)
    df_model, base_feature_cols = prepare_lightgbm_frame(df_features)
    split_indices = build_split_indices(df_model, random_state=random_state)

    variants = [
        "baseline",
        "no_stable_price",
        "no_stable_price_no_invoice_price_trend",
        "no_stable_price_no_price_trend",
        "no_stable_price_no_invoice_price_trend_std",
    ]

    results: list[dict[str, object]] = []
    for variant in variants:
        variant_output_dir = DEFAULT_OUTPUT_DIR / "ablations" / variant
        variant_result = run_single_lightgbm_variant(
            df_model=df_model,
            base_feature_cols=base_feature_cols,
            split_indices=split_indices,
            variant=variant,
            output_dir=variant_output_dir,
            random_state=random_state,
        )
        results.append(
            {
                "variant": variant,
                "feature_count": len(filter_feature_variant(base_feature_cols, variant)),
                "best_iteration": int(variant_result["best_iteration"]),
                "test_roc_auc": float(variant_result["metrics"]["roc_auc"]),
                "test_average_precision": float(variant_result["metrics"]["average_precision_score"]),
                "test_log_loss": float(variant_result["metrics"]["log_loss"]),
                "best_threshold_precision": float(variant_result["threshold"]["precision"]),
                "best_threshold_recall": float(variant_result["threshold"]["recall"]),
                "best_threshold_f1": float(variant_result["threshold"]["f1"]),
            }
        )

    comparison = pd.DataFrame(results)
    comparison_path = DEFAULT_OUTPUT_DIR / "ablation_summary.csv"
    comparison.to_csv(comparison_path, index=False)

    return {
        "comparison": comparison,
        "comparison_path": comparison_path,
        "df_features": df_features,
        "df_model": df_model,
        "base_feature_cols": base_feature_cols,
    }


def main() -> None:
    results = run_lightgbm_feature_ablation_study()
    print("Comparativa de variantes LightGBM terminada.")
    print(results["comparison"].to_string(index=False))
    print(f"\nResumen guardado en: {results['comparison_path']}")


if __name__ == "__main__":
    main()
