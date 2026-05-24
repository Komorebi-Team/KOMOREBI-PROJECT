import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def _safe_div_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Realiza una división segura entre dos Series de pandas.

    Reemplaza los ceros en el denominador por NaN para evitar infinitos,
    y luego rellena los resultados NaN con 0.0.

    Parameters
    ----------
    numerator : pd.Series
        Serie con los valores del numerador.
    denominator : pd.Series
        Serie con los valores del denominador.

    Returns
    -------
    pd.Series
        Resultado de la división con valores nulos imputados a 0.0.
    """
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.0)

def _build_price_cols_to_scale(n: int) -> list[str]:
    """
    Construye la lista de variables absolutas monetarias que deben escalar.

    Parameters
    ----------
    n : int
        Número de meses iniciales del contrato sobre los que se calcularon métricas.

    Returns
    -------
    list of str
        Lista con los nombres exactos de las columnas relacionadas con la 
        facturación que deben multiplicarse por el factor de subida.
    """
    cols = [
        "monthly_total_invoice", 
        "monthly_total_invoice_trend",
        "monthly_total_invoice_std",
        "stable_price_mean",
        "stable_price_median",
        "stable_price_std"
    ]
    for i in range(1, n + 1):
        cols.append(f"monthly_total_invoice_month{i}")
        cols.append(f"invoice_post_month{i}")
    return cols

def _align_dtypes_to_reference(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Alinea los tipos de datos de un DataFrame simulado con los del modelo original.

    Evita que XGBoost falle debido a discrepancias de tipos de datos (por ejemplo, 
    int64 vs float64) generadas matemáticamente durante la simulación.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame simulado a corregir.
    reference_df : pd.DataFrame
        DataFrame original con el que se entrenó el modelo.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas ordenadas y casteadas exactamente 
        igual que reference_df.
    """
    df_aligned = df[reference_df.columns].copy()
    
    for col in df_aligned.columns:
        if df_aligned[col].dtype != reference_df[col].dtype:
            df_aligned[col] = df_aligned[col].astype(reference_df[col].dtype)
            
    return df_aligned

def _apply_price_elasticity_to_features(X_sim: pd.DataFrame, factor: float) -> pd.DataFrame:
    """
    Aplica la elasticidad del precio sobre los ratios de rentabilidad del cliente.

    Modifica matemáticamente las features normalizadas por factura sin depender 
    de las variables absolutas. Si la factura sube (factor > 1), la rentabilidad 
    percibida (ratios per_invoice) cae proporcionalmente.

    Parameters
    ----------
    X_sim : pd.DataFrame
        DataFrame con las features del cliente a simular.
    factor : float
        Multiplicador del precio (ej. 1.20 para una subida del 20%).

    Returns
    -------
    pd.DataFrame
        DataFrame con las variables "per_invoice" reescaladas a la baja.
    """
    X_sim = X_sim.copy()

    # Ratios donde el invoice está dividiendo
    ratio_cols = [c for c in X_sim.columns if "per_invoice" in c]
    for col in ratio_cols:
        X_sim[col] = X_sim[col] / factor

    return X_sim


def simulate_portfolio_invoice_scenarios(
    model,
    X_test: pd.DataFrame,
    X_train_reference: pd.DataFrame,
    n: int = 5,
    factors: list[float] = None,
    threshold: float = 0.5,
    scale_absolute_features: bool = False,
    elasticity_multiplier: float = 0.0,
    recalculate_ratios: bool = True,
) -> pd.DataFrame:
    """
    Simula el impacto de distintas subidas de precio a nivel portfolio.

    Multiplica las variables monetarias absolutas (opcional), empeora proporcionalmente 
    la eficiencia percibida por el cliente y genera predicciones de fuga y 
    estimaciones financieras usando el modelo proporcionado.

    Parameters
    ----------
    model : estimator
        Modelo de clasificación entrenado (con método predict_proba).
    X_test : pd.DataFrame
        Conjunto de datos (holdout) a someter a la simulación.
    X_train_reference : pd.DataFrame
        Datos de entrenamiento originales (usados para alinear tipos).
    n : int, default=5
        Número de meses evaluados en la lógica de onboarding.
    factors : list of float, optional
        Lista de multiplicadores de precio a simular. Por defecto simula 
        desde 1.0 (base) hasta 1.50 (+50%).
    threshold : float, default=0.5
        Punto de corte óptimo fijado por reglas de negocio para 
        clasificar una probabilidad como fuga confirmada.
    scale_absolute_features : bool, default=False
        Si es True, escala las variables absolutas de dinero (factura total).
        Si es False, solo se penalizan los ratios de rentabilidad, evitando 
        el sesgo de correlación "cliente grande = cliente seguro".
    elasticity_multiplier : float, default=0.0
        Penalizador manual para hibridar el modelo ML con una regla de negocio.
        Si es >0, aumenta la probabilidad de fuga base proporcionalmente a la subida de precio.
    recalculate_ratios : bool, default=True
        Si es True, recalcula las métricas derivadas (rentabilidad per_invoice) a la baja 
        cuando el precio sube. Si es False, congela estos ratios para hacer un experimento 
        puro de aislamiento de variables (Partial Dependence).

    Returns
    -------
    pd.DataFrame
        Tabla resumen con los resultados a nivel portfolio para cada 
        escenario, incluyendo tasas de fuga esperadas e impacto en facturación.
    """
    if factors is None:
        factors = [1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]

    price_cols_to_scale = _build_price_cols_to_scale(n)
    results = []

    for factor in factors:
        X_sim = X_test.copy()

        # 1. Multiplicar variables absolutas de dinero (Solo si está activado para evitar sesgos)
        if scale_absolute_features:
            # Petición específica: Solo escalar 'monthly_total_invoice' y no sus variables derivadas
            if "monthly_total_invoice" in X_sim.columns:
                X_sim["monthly_total_invoice"] = X_sim["monthly_total_invoice"] * factor

        # 2. Recalcular métricas derivadas (Elasticidad matemática pura)
        if recalculate_ratios:
            X_sim = _apply_price_elasticity_to_features(X_sim, factor=factor)
        
        # 3. Alinear dtypes para que XGBoost no falle
        X_sim = _align_dtypes_to_reference(X_sim, X_train_reference)

        # 4. Predicción de riesgo bajo el nuevo precio
        y_proba_base = model.predict_proba(X_sim)[:, 1]
        
        # 4.5. Aplicar multiplicador de elasticidad (Regla de negocio manual exponencial)
        incremento_pct = factor - 1.0
        y_proba = y_proba_base * np.exp(incremento_pct * elasticity_multiplier)
        y_proba = np.clip(y_proba, 0.0, 1.0)

        row = {
            "factor": factor,
            "incremento_precio_pct": (factor - 1) * 100,
            "mean_p_churn": float(np.mean(y_proba)),
            "n_obs": int(len(X_sim)),
        }

        # 5. Predicción en binario usando tu threshold de negocio
        y_pred = (y_proba >= threshold).astype(int)
        row["predicted_churn_rate"] = float(np.mean(y_pred))
        
        # 6. Cálculo de finanzas (Valor Esperado)
        # Se calcula usando la probabilidad (y_proba) para reflejar el riesgo continuo en lugar de decisiones binarias
        if "monthly_total_invoice" in X_test.columns:
            new_invoice = X_test["monthly_total_invoice"] * factor
            row["facturacion_retenida"] = float(np.sum((1 - y_proba) * new_invoice))
            row["facturacion_perdida"] = float(np.sum(y_proba * new_invoice))

        results.append(row)

    out = pd.DataFrame(results)
    
    # Añadir variaciones respecto al escenario Base (1.0)
    if (out["factor"] == 1.0).any():
        base_churn = out.loc[out["factor"] == 1.0, "predicted_churn_rate"].iloc[0]
        out["delta_churn_rate_pct"] = (out["predicted_churn_rate"] - base_churn) * 100
        
        if "facturacion_retenida" in out.columns:
            base_revenue = out.loc[out["factor"] == 1.0, "facturacion_retenida"].iloc[0]
            out["delta_revenue"] = out["facturacion_retenida"] - base_revenue

    return out
