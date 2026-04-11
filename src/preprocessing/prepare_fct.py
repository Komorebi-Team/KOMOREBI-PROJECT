import pandas as pd

from src.utils import validate_columns

def prepare_fct(
        df_fct: pd.DataFrame,
        df_dim: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Prepara el dataset mensual de snapshots (FCT) para su uso en el pipeline
    de análisis a nivel contrato.

    La función enriquece `df_fct` con información de contratos procedente de
    `df_dim`, necesaria para los pasos posteriores del pipeline (identificación
    de contratos, filtrado de left-censoring y cálculo de churn).

    Lógica aplicada:
    - Realiza un left join entre `df_fct` y `df_dim` por `advertiser_zrive_id`
      para incorporar variables de contrato (fechas de inicio, etc.).
    - Convierte `period_int` desde formato YYYYMM a `Period[M]` para asegurar
      consistencia temporal en análisis mensuales.
    - Elimina registros con facturación negativa, ya que corresponden a
      devoluciones o ajustes no representativos del comportamiento habitual.

    Parámetros
    ----------
    df_fct : pd.DataFrame
        DataFrame de snapshots mensuales a nivel advertiser. Debe contener:
        - advertiser_zrive_id
        - period_int
        - monthly_total_invoice

    df_dim : pd.DataFrame
        DataFrame de atributos del advertiser, que incluye información
        de contratos (por ejemplo, fechas de inicio).

    Returns
    -------
    pd.DataFrame
        DataFrame enriquecido con información de contrato, con `period_int`
        en formato mensual (`Period[M]`) y sin registros con facturación negativa.
    """
    validate_columns(
        df_fct,
        {"advertiser_zrive_id", "period_int", "monthly_total_invoice"},
        "prepare_fct",
    )
    validate_columns(
        df_dim,
        {"advertiser_zrive_id"},
        "prepare_fct",
    )

    df = df_fct.merge(
        df_dim,
        how="left",
        on="advertiser_zrive_id"
    )

    df["period_int"] = pd.to_datetime(           
        df["period_int"].astype(str),
        format="%Y%m",
        errors="raise"
    ).dt.to_period("M")

    df = df[df["monthly_total_invoice"] >= 0]

    return df