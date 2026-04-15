import pandas as pd
import numpy as np
import logging
from typing import Optional

from src.utils import validate_columns

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(funcName)s - %(message)s"
)

def _build_contract_start_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un summary a nivel contrato con la información necesaria para
    validar el inicio del contrato.

    Para cada `contract_id`, calcula:
    - `first_period`: primer mes observado en FCT
    - `min_start_period`: mes asociado a `min_start_contrato_date`
    - `max_new_start_period`: mes asociado a `max_start_contrato_nuevo_date`

    Además, conserva las fechas originales de inicio disponibles en DIM para su
    uso posterior en la asignación de `start_source` y `contract_start_date`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mensual con, al menos, las columnas:
        - `contract_id`
        - `period_int`
        - `min_start_contrato_date`
        - `max_start_contrato_nuevo_date`

    Returns
    -------
    pd.DataFrame
        Summary a nivel contrato con:
        - `contract_id`
        - `first_period`
        - `min_start_contrato_date`
        - `max_start_contrato_nuevo_date`
        - `min_start_period`
        - `max_new_start_period`
    """

    contract_summary = (
        df.loc[df["contract_id"].notna()]
        .groupby("contract_id", as_index=False)
        .agg(
            first_period=("period_int", "min"),
            min_start_contrato_date=("min_start_contrato_date", "first"),
            max_start_contrato_nuevo_date=("max_start_contrato_nuevo_date", "first"),
        )
    )

    contract_summary["min_start_period"] = (
        pd.to_datetime(contract_summary["min_start_contrato_date"], errors="coerce")
        .dt.to_period("M")
    )

    contract_summary["max_new_start_period"] = (
        pd.to_datetime(contract_summary["max_start_contrato_nuevo_date"], errors="coerce")
        .dt.to_period("M")
    )

    return contract_summary

def _add_contract_start_metadata(contract_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Añade metadatos de validación del inicio del contrato a un summary a nivel
    contrato.

    Para cada `contract_id`, compara `first_period` con los periodos de inicio
    derivados de DIM y clasifica el origen del inicio (`start_source`) como:
    - `"min_start_contrato_date"`
    - `"max_start_contrato_nuevo_date"`
    - `"both"`
    - `"neither"`

    Además, construye `contract_start_date`:
    - si `start_source` ∈ {"min_start_contrato_date", "both"} →
      `min_start_contrato_date`
    - si `start_source` == "max_start_contrato_nuevo_date" →
      `max_start_contrato_nuevo_date`

    Parameters
    ----------
    contract_summary : pd.DataFrame
        Summary a nivel contrato con, al menos, las columnas:
        - `contract_id`
        - `first_period`
        - `min_start_contrato_date`
        - `max_start_contrato_nuevo_date`
        - `min_start_period`
        - `max_new_start_period`

    Returns
    -------
    pd.DataFrame
        Summary enriquecido con:
        - `matches_min_start`
        - `matches_max_new_start`
        - `start_source`
        - `contract_start_date`
    """
    df = contract_summary.copy()

    df["matches_min_start"] = df["first_period"] == df["min_start_period"]
    df["matches_max_new_start"] = df["first_period"] == df["max_new_start_period"]

    df["start_source"] = np.select(
        [
            df["matches_min_start"] & ~df["matches_max_new_start"],
            df["matches_max_new_start"] & ~df["matches_min_start"],
            df["matches_min_start"] & df["matches_max_new_start"],
        ],
        [
            "min_start_contrato_date",
            "max_start_contrato_nuevo_date",
            "both",
        ],
        default="neither",
    )

    df["contract_start_date"] = pd.NaT

    df.loc[
        df["start_source"].isin(["min_start_contrato_date", "both"]),
        "contract_start_date",
    ] = df["min_start_contrato_date"]

    df.loc[
        df["start_source"] == "max_start_contrato_nuevo_date",
        "contract_start_date",
    ] = df["max_start_contrato_nuevo_date"]

    df["contract_start_date"] = pd.to_datetime(
        df["contract_start_date"], errors="coerce"
    )

    return df


def filter_contracts_with_valid_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra contratos manteniendo únicamente aquellos cuyo primer mes observado
    en FCT (`first_period`) coincide, a nivel mensual, con alguna de las fechas
    de inicio disponibles en DIM (`min_start_contrato_date` o
    `max_start_contrato_nuevo_date`).

    Los contratos con `start_source = "neither"` se eliminan, ya que no es
    posible asignarles una fecha de inicio fiable. En concreto, un contrato se
    clasifica como `"neither"` cuando:

    - `first_period != min_start_period`
    - `first_period != max_new_start_period`

    Asumiendo DIM como fuente de verdad, este filtrado controla el left
    censoring, manteniendo únicamente contratos cuyo inicio puede validarse
    frente a las fechas de inicio disponibles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mensual con, al menos, las columnas:
        - `contract_id`
        - `advertiser_zrive_id`
        - `period_int`
        - `min_start_contrato_date`
        - `max_start_contrato_nuevo_date`

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado con contratos válidos, incluyendo:
        - `start_source`
        - `contract_start_date`
    """

    required_cols = {
        "contract_id",
        "advertiser_zrive_id",
        "period_int",
        "min_start_contrato_date",
        "max_start_contrato_nuevo_date",
    }

    validate_columns(
        df=df,
        required_cols=required_cols,
        func_name="filter_contracts_with_valid_start",
    )

    data = df.copy()

    contract_summary = (
        data
        .pipe(_build_contract_start_summary)
        .pipe(_add_contract_start_metadata)
    )

    valid_contracts = contract_summary.loc[
        contract_summary["start_source"] != "neither",
        "contract_id",
    ]

    df_contracts = data.loc[data["contract_id"].isin(valid_contracts)].copy()

    df_contracts = df_contracts.merge(
        contract_summary[["contract_id", "start_source", "contract_start_date"]],
        on="contract_id",
        how="left",
    )

    total = contract_summary["contract_id"].nunique()
    removed = (contract_summary["start_source"] == "neither").sum()

    logger.info(
        "n_total_contracts=%s, n_removed_neither=%s, n_remaining=%s",
        total,
        removed,
        total - removed,
    )

    return df_contracts

def add_right_censoring_flag(
    contract_summary: pd.DataFrame,
    observation_end: Optional[pd.Period] = None,
) -> pd.DataFrame:
    """
    Añade las columnas `is_right_censored` y `right_censoring_case` a un dataframe
    a nivel contrato.

    Definición
    ----------
    Un contrato se considera `right_censored` cuando alcanza el final de la
    ventana de observación y no se observa su churn dentro de dicha ventana.
    En estos casos, el evento podría ocurrir posteriormente, pero no es
    observable con la información disponible.

    Criterio aplicado
    -----------------
    - Si `contract_end_period < obs_end`, el contrato no se considera censurado.
    - Si `contract_end_period == obs_end`:
        * si `churn_period == obs_end`, el churn se considera observado y el
          contrato no está censurado
        * si `churn_period > obs_end`, el contrato se considera `right_censored`
        * si `contrato_churn_date` es nula, el contrato se considera `right_censored`
        * si `churn_period < obs_end`, el registro se clasifica como
          inconsistente y se excluye al final de la función

    Parameters
    ----------
    contract_summary : pd.DataFrame
        DataFrame a nivel contrato con, al menos, las columnas:
        - `contract_id`
        - `contract_end_period` (Period[M])
        - `contrato_churn_date` (datetime)

    observation_end : pd.Period, optional
        Último periodo global de observación. Si no se informa, se utiliza el
        máximo valor de `contract_end_period`.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas `is_right_censored` y `right_censoring_case`,
        excluyendo los contratos clasificados como `inconsistent_churn_before_window_end`.
    """
    validate_columns(
        df=contract_summary,
        required_cols={
            "contract_id",
            "contract_end_period",
            "contrato_churn_date",
        },
        func_name="add_right_censoring_flag",
    )

    df = contract_summary.copy()
    df["contrato_churn_date"] = pd.to_datetime(df["contrato_churn_date"], errors="coerce")

    if observation_end is None:
        obs_end = df["contract_end_period"].max()
    else:
        obs_end = observation_end

    churn_period = df["contrato_churn_date"].dt.to_period("M")
    reaches_window_end = df["contract_end_period"] == obs_end
    ends_before_window_end = df["contract_end_period"] < obs_end

    inconsistent_churn_before_window_end = (
        reaches_window_end
        & df["contrato_churn_date"].notna()
        & (churn_period < obs_end)
    )

    missing_churn_before_window_end = (
        ends_before_window_end
        & df["contrato_churn_date"].isna()
    )

    observed_churn_at_window_end = (
        reaches_window_end
        & df["contrato_churn_date"].notna()
        & (churn_period == obs_end)
    )

    right_censored = (
        reaches_window_end
        & (
            df["contrato_churn_date"].isna()
            | (churn_period > obs_end)
        )
    )

    df["right_censoring_case"] = "ended_before_window_end"

    df.loc[observed_churn_at_window_end, "right_censoring_case"] = (
        "observed_churn_at_window_end"
    )
    df.loc[right_censored, "right_censoring_case"] = "right_censored"
    df.loc[inconsistent_churn_before_window_end, "right_censoring_case"] = (
        "inconsistent_churn_before_window_end"
    )
    df.loc[missing_churn_before_window_end, "right_censoring_case"] = (
        "missing_churn_before_window_end"
    )

    df["is_right_censored"] = right_censored
    
    # Logging
    total_contracts_before_filter = len(df)
    n_right_censored = int(df["is_right_censored"].sum())
    n_inconsistent_boundary = int(inconsistent_churn_before_window_end.sum())
    n_missing_churn_before_end = int(missing_churn_before_window_end.sum())

    logger.info(
        "[add_right_censoring_flag] Right censored contracts before filtering: %s/%s (%.2f%%)",
        n_right_censored,
        total_contracts_before_filter,
        (
            n_right_censored / total_contracts_before_filter * 100
            if total_contracts_before_filter > 0 else 0.0
        ),
    )

    logger.info(
        "Contracts with inconsistent churn before window end: %s",
        n_inconsistent_boundary,
    )

    logger.info(
        "Contracts ending before window end with missing churn date: %s",
        n_missing_churn_before_end,
    )

    df = df.loc[df["right_censoring_case"] != "inconsistent_churn_before_window_end"].copy()

    total_contracts_after_filter = len(df)
    n_right_censored_after_filter = int(df["is_right_censored"].sum())

    logger.info(
        "Contracts retained after removing inconsistent cases: %s/%s",
        total_contracts_after_filter,
        total_contracts_before_filter,
    )

    logger.info(
        "Right censored contracts after filtering: %s/%s (%.2f%%)",
        n_right_censored_after_filter,
        total_contracts_after_filter,
        (
            n_right_censored_after_filter / total_contracts_after_filter * 100
            if total_contracts_after_filter > 0 else 0.0
        ),
    )

    return df
