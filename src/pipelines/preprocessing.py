import pandas as pd

from src.preprocessing.prepare_fct import prepare_fct
from src.preprocessing.build_contracts import (
    add_contract_id,
    build_contract_summary,
    add_contract_end_period,
)
from src.preprocessing.filter_contracts import (
    filter_contracts_with_valid_start,
    add_right_censoring_flag,
)


def run_preprocessing_pipeline(
    df_fct: pd.DataFrame,
    df_dim: pd.DataFrame,
    observation_end: pd.Period | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el pipeline completo de preprocessing.

    Parameters
    ----------
    df_fct : pd.DataFrame
        DataFrame mensual base.
    df_dim : pd.DataFrame
        DataFrame dimensional con información de advertiser.
    observation_end : pd.Period, optional
        Último periodo global de observación. Si es None, se utiliza el máximo
        `contract_end_period` observado.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - `df_contracts`: dataframe mensual preprocesado
        - `contract_summary`: dataframe a nivel contrato con metadatos y censoring
    """
    df_contracts = (
        df_fct
        .pipe(prepare_fct, df_dim=df_dim)
        .pipe(add_contract_id)
        .pipe(filter_contracts_with_valid_start)
    )

    contract_summary = (
        df_contracts
        .pipe(build_contract_summary)
        .pipe(add_contract_end_period, df_contracts=df_contracts)
        .pipe(add_right_censoring_flag, observation_end=observation_end)
    )

    df_contracts = df_contracts.merge(
        contract_summary[
            [
                "contract_id",
                "contract_end_period",
                "is_right_censored",
                "right_censoring_case",
            ]
        ],
        on="contract_id",
        how="left",
    )

    return df_contracts, contract_summary