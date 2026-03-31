import pandas as pd

def filter_contracts_with_valid_start_dates(
    df: pd.DataFrame,
    observation_start: str = "2023-01"
) -> pd.DataFrame:
    """
    Filtra contratos manteniendo solo aquellos con evidencia de inicio dentro
    de la ventana de observación.

    Se conserva un contrato si:
    - min_start_contrato_date >= observation_start, ó
    - max_start_contrato_nuevo_date >= observation_start.

    Parámetros
    ----------
    df : pd.DataFrame
        Requiere las columnas 'contract_id', 'min_start_contrato_date' y
        'max_start_contrato_nuevo_date'.
    observation_start : str, opcional
        Inicio de la ventana ("YYYY-MM").

    Returns
    -------
    pd.DataFrame
    """
    obs_start = pd.Timestamp(observation_start)

    mask_valid = (
            (df["min_start_contrato_date"] >= obs_start) |
            (df["max_start_contrato_nuevo_date"] >= obs_start)
        )

    return df.loc[mask_valid].copy()

def filter_left_censored(df, observation_start="2023-01", keep_all=False):
    """
    Identifica y trata contratos left-censored en el dataset.

    Un contrato se considera left-censored cuando su primer mes observado en los datos
    coincide con el inicio de la ventana de observación, pero la información de fechas
    (min_start_contrato_date o max_start_contrato_nuevo_date) indica que el contrato
    ya estaba activo antes de dicho periodo. En estos casos, el inicio real del contrato
    no está completamente observado, lo que puede introducir sesgos en análisis de churn.

    Lógica aplicada:
    - Se identifican los contratos cuyo primer periodo observado es igual a observation_start.
    - Entre estos, se consideran left-censored aquellos cuyo inicio real (según DIM)
      es anterior a observation_start.
    - Estos contratos pueden eliminarse o anonimizarse según el parámetro keep_all.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada que debe contener al menos:
        - contract_id : identificador del contrato
        - period_int : periodo mensual (Period[M] o equivalente)
        - min_start_contrato_date : fecha de inicio original del contrato (datetime)
        - max_start_contrato_nuevo_date : fecha de última alta o renovación (datetime)

    observation_start : str, opcional (default="2023-01")
        Inicio de la ventana de observación en formato "YYYY-MM".

    keep_all : bool, opcional (default=False)
        - Si True: mantiene todas las filas pero elimina los contratos left-censored
          asignando contract_id = pd.NA.
        - Si False: elimina completamente las filas de los contratos left-censored.

    Returns
    -------
    pd.DataFrame
        DataFrame tratado:
        - Sin contratos left-censored si keep_all=False
        - Con contract_id nulo para dichos contratos si keep_all=True
    """

    obs_start = pd.Period(observation_start, freq="M")
    
    active = df[df["contract_id"].notna()]
    
    first_month = (
        active.groupby("contract_id")
        .agg(first_period=("period_int", "min"))
    )

    suspects_ids = first_month[first_month["first_period"] == obs_start].index

    suspects = df.loc[df["contract_id"].isin(suspects_ids)].copy()

    is_left_censored = (
        (suspects["min_start_contrato_date"].dt.to_period("M") != obs_start) &
        (suspects["max_start_contrato_nuevo_date"].dt.to_period("M") != obs_start)
    )

    to_remove_ids = suspects.loc[is_left_censored, "contract_id"].unique()

    if keep_all:
        suspects_result = df.copy()
        suspects_result.loc[suspects_result["contract_id"].isin(to_remove_ids), "contract_id"] = pd.NA
        return suspects_result

    return df.loc[~df["contract_id"].isin(to_remove_ids)].copy()



