import numpy as np
import pandas as pd
import logging

from src.preprocessing.utils import validate_columns

logger = logging.getLogger(__name__)


def add_contract_id(df : pd.DataFrame) -> pd.DataFrame:
    """
    Asigna un identificador único de contrato (contract_id) a cada periodo continuo
    de actividad para cada advertiser.

    Un contrato se define como una secuencia de meses consecutivos para un mismo
    advertiser_zrive_id en los que `has_active_contract == True`.

    Un nuevo contrato comienza cuando:
    - La fila actual está activa y la fila anterior estaba inactiva, o
    - La fila actual está activa y existe un salto de más de 1 mes respecto a la anterior, o
    - La fila actual es el primer mes activo observado para ese advertiser
      (esto puede no representar el inicio real del contrato, ya que podría haber comenzado
      antes de la ventana de observación; este caso se analizará posteriormente en el
      tratamiento del left censoring).

    El contract_id se construye como:
        <advertiser_zrive_id>_<número_secuencial_de_contrato>

    Las filas donde `has_active_contract == False` tendrán contract_id = NaN.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada que debe contener al menos:
        - advertiser_zrive_id : identificador único del advertiser
        - period_int : periodo mensual (datetime o Period[M])
        - has_active_contract : indicador booleano de actividad del contrato

    Returns
    -------
    pd.DataFrame
        DataFrame original con una nueva columna:
        - contract_id : identificador único de cada contrato detectado
    """
    required_cols = {"advertiser_zrive_id", "period_int", "has_active_contract"}
    validate_columns(
        df,
        required_cols,
        "add_contract_id",
    )

    if df["has_active_contract"].isna().any():
        raise ValueError(
            "[add_contract_id] 'has_active_contract' contiene valores nulos."
        )
    df["has_active_contract"] = df["has_active_contract"].astype(bool)

    if not isinstance(df["period_int"].dtype, pd.PeriodDtype):
        df["period_int"] = pd.to_datetime(df["period_int"]).dt.to_period("M")   

    duplicated_mask = df.duplicated(subset=["advertiser_zrive_id", "period_int"], keep=False)
    if duplicated_mask.any():
        duplicated_rows = df.loc[duplicated_mask, ["advertiser_zrive_id", "period_int"]]
        raise ValueError(
            "[add_contract_id] existen filas duplicadas para el mismo advertiser_zrive_id y period_int. "
            f"Ejemplos:\n{duplicated_rows.head()}"
        )

    df = df.sort_values(["advertiser_zrive_id", "period_int"]).copy()

    df["prev_period_int"] = df.groupby("advertiser_zrive_id")["period_int"].shift(1)
    df["month_diff"] = np.where(
        df["prev_period_int"].notna(),
        df["period_int"].astype("int64") - df["prev_period_int"].astype("int64"),
        np.nan
    )

    df["prev_has_active_contract"] = (
        df.groupby("advertiser_zrive_id")["has_active_contract"].shift(1)
    )

    df["new_contract_start"] = (
        df["has_active_contract"] &
        (
            (df["prev_has_active_contract"].eq(False)) | 
            (df["prev_has_active_contract"].isna()) | 
            (df["month_diff"] > 1)
        )
    )

    contract_number = (
        df.groupby("advertiser_zrive_id")["new_contract_start"]
        .cumsum()
        .astype(str)
    )

    df["contract_id"] = np.where(
        df["has_active_contract"],
        df["advertiser_zrive_id"].astype(str) + "_" + contract_number.astype(str),
        np.nan,
    )

    n_advertisers = df["advertiser_zrive_id"].nunique()
    n_contracts = df["contract_id"].nunique(dropna=True)

    logger.info(
        "[add_contract_id] n_advertisers=%s, n_contracts=%s",
        n_advertisers,
        n_contracts,
    )

    return df
