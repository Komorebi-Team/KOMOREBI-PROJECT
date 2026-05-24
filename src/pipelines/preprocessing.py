import pandas as pd

from src.preprocessing.prepare_fct import prepare_fct
from src.preprocessing.contracts import (
    add_contract_id,
    build_contract_summary,
)
from src.preprocessing.censoring import (
    filter_contracts_with_valid_start,
    add_right_censoring_flag,
)


def run_preprocessing_pipeline(
    df_fct: pd.DataFrame,
    df_dim: pd.DataFrame,
    observation_end: pd.Period | None = None,
) -> pd.DataFrame:
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
    pd.DataFrame
        DataFrame mensual preprocesado con flags de censura y end_period.
    """
    df_contracts = (
        df_fct
        .pipe(prepare_fct, df_dim=df_dim)
        .pipe(add_contract_id)
        .pipe(filter_contracts_with_valid_start)
        .pipe(add_right_censoring_flag, observation_end=observation_end)
    )

    return df_contracts