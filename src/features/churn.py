import numpy as np
import pandas as pd

def compute_contract_churn_3m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a binary churn label at contract level based on contract duration.

    A contract is considered churned within 3 months (churned_3m = 1) if its
    duration is less than or equal to 3 months. Otherwise, it is considered
    non-churned (churned_3m = 0).

    The function assumes that:
    - Each row represents a monthly snapshot of an advertiser's contract activity.
    - A unique contract is identified by 'contract_id'.
    - The dataset contains at least 3 months of observable data per contract
      or is properly censored.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least the following columns:
        - 'contract_id': unique identifier of the contract
        - 'period_int': monthly period of observation (Period[M])
        - 'has_active_contract': boolean indicating contract activity

    Returns
    -------
    pd.DataFrame
        Original dataframe with an additional column:
        - 'churned_3m': int (1 if churned within 3 months, 0 otherwise)

    Notes
    -----
    - This function operates at contract level but returns results at row level.
    - It assumes no duplicate rows per ('contract_id', 'period_int').
    - Potential data leakage should be considered if future information is used
      to determine contract duration.
    """
    out = df.copy()

    contract_duration = out.groupby("contract_id").size()
    short_contracts = contract_duration.loc[lambda x: x <= 3].index

    out["churned_3m"] = out["contract_id"].isin(short_contracts).astype(int)

    return out
