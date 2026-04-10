import pandas as pd
import numpy as np
import logging

from src.preprocessing.utils import validate_columns

logger = logging.getLogger(__name__)

def filter_contracts_with_valid_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra contratos manteniendo únicamente aquellos cuyo primer mes observado en
    FCT (`first_period`) coincide, a nivel mensual, con alguna de las fechas de
    inicio disponibles en DIM (`min_start_contrato_date` o `max_start_contrato_nuevo_date`).

    Para cada `contract_id`, se compara `first_period` con ambas fechas y se clasifica
    el origen del inicio (`start_source`) como:
    - "min_start_contrato_date"
    - "max_start_contrato_nuevo_date"
    - "both"
    - "neither"

    Los contratos con `start_source = "neither"` se eliminan, ya que no es posible
    asignarles una fecha de inicio fiable (un contrato es "neither" cuando: 
    first_period != min_start_contrato_date AND first_period != max_start_contrato_nuevo_date)

    Se construye además `contract_start_date`:
    - si `start_source` ∈ {"min_start_contrato_date", "both"} → `min_start_contrato_date`
    - si `start_source` == "max_start_contrato_nuevo_date" → `max_start_contrato_nuevo_date`

    Asumiendo DIM como fuente de verdad, este filtrado controla el left censoring,
    manteniendo únicamente contratos cuyo inicio coincide con el inicio observado
    en FCT.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mensual con:
        - contract_id
        - advertiser_zrive_id
        - period_int
        - min_start_contrato_date
        - max_start_contrato_nuevo_date

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado con contratos válidos, incluyendo `contract_start_date`.
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
        data.loc[data["contract_id"].notna()]
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

    contract_summary["matches_min_start"] = (
        contract_summary["first_period"] == contract_summary["min_start_period"]
    )

    contract_summary["matches_max_new_start"] = (
        contract_summary["first_period"] == contract_summary["max_new_start_period"]
    )

    contract_summary["start_source"] = np.select(
        [
            contract_summary["matches_min_start"] & ~contract_summary["matches_max_new_start"],
            contract_summary["matches_max_new_start"] & ~contract_summary["matches_min_start"],
            contract_summary["matches_min_start"] & contract_summary["matches_max_new_start"],
        ],
        [
            "min_start_contrato_date",
            "max_start_contrato_nuevo_date",
            "both",
        ],
        default="neither",
    )

    # Assign start date
    contract_summary["contract_start_date"] = pd.NaT

    contract_summary.loc[
        contract_summary["start_source"].isin(["min_start_contrato_date", "both"]),
        "contract_start_date",
    ] = contract_summary["min_start_contrato_date"]

    contract_summary.loc[
        contract_summary["start_source"] == "max_start_contrato_nuevo_date",
        "contract_start_date",
    ] = contract_summary["max_start_contrato_nuevo_date"]

    contract_summary["contract_start_date"] = pd.to_datetime(
        contract_summary["contract_start_date"], errors="coerce"
    )

    # Filtrado
    valid_contracts = contract_summary.loc[
        contract_summary["start_source"] != "neither",
        "contract_id",
    ]

    df_filtered = data.loc[data["contract_id"].isin(valid_contracts)].copy()

    df_filtered = df_filtered.merge(
        contract_summary[["contract_id", "start_source", "contract_start_date"]],
        on="contract_id",
        how="left",
    )

    # Logging
    total = contract_summary["contract_id"].nunique()
    removed = (contract_summary["start_source"] == "neither").sum()

    logger.info(
        "[filter_contracts_with_valid_start] n_total_contracts=%s, n_removed_neither=%s, n_remaining=%s",
        total,
        removed,
        total - removed,
    )

    return df_filtered
