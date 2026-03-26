import numpy as np
import pandas as pd

def add_contract_id(df : pd.DataFrame) -> pd.DataFrame:
    """
    This function receives a dataframe contaning monthly information about users activity on the platform, 
    including contract start and end dates, defines the contract and assigns an contract_id (it receives DF_FCT
    and returns the df with the new column "contract_id")

    A contract is defined as a consecutive monthly sequence for the same advertiser_zrive_id where 
    has_active_contract == True.

    A new contract starts when:
    - the current row is active and the previous row for that advertiser was inactive, or
    - the current row is active and there is a gap of more than 1 month, or
    - the current row is the first observed active row for that advertiser.

    Parameters:
        df : pd.DataFrame
            Input dataframe containing at least:
            - advertiser_zrive_id
            - period_int
            - has_active_contract
    Returns:
        df containing the contract_id
    """
    out = df.copy()

    if out["has_active_contract"].isna().any():
        raise ValueError("Column 'has_active_contract' contains null values.")
    out["has_active_contract"] = out["has_active_contract"].astype(bool)

    col = out["period_int"]
    if not (isinstance(col.dtype, pd.PeriodDtype) and col.dtype.freq == "ME"):
        out["period_int"] = pd.to_datetime(col).dt.to_period("M")

        # Validate uniqueness of advertiser-period
    duplicated_mask = out.duplicated(subset=["advertiser_zrive_id", "period_int"], keep=False)
    if duplicated_mask.any():
        duplicated_rows = out.loc[duplicated_mask, ["advertiser_zrive_id", "period_int"]]
        raise ValueError(
            "Found duplicated rows for the same advertiser_zrive_id and period_int. "
            f"Examples:\n{duplicated_rows.head()}"
        )

    out = out.sort_values(["advertiser_zrive_id", "period_int"]).copy()

    out["prev_period_int"] = out.groupby("advertiser_zrive_id")["period_int"].shift(1)
    out["month_diff"] = np.where(
        out["prev_period_int"].notna(),
        out["period_int"].astype("int64") - out["prev_period_int"].astype("int64"),
        np.nan
    )

    out["prev_has_active_contract"] = (
        out.groupby("advertiser_zrive_id")["has_active_contract"].shift(1)
    )

    out["new_contract_start"] = (
        out["has_active_contract"] &
        (
            (out["prev_has_active_contract"].eq(False)) | 
            (out["prev_has_active_contract"].isna()) | 
            (out["month_diff"] > 1)
        )
    )

    contract_number = (
        out.groupby("advertiser_zrive_id")["new_contract_start"]
        .cumsum()
        .astype(str)
    )

    out["contract_id"] = np.where(
        out["has_active_contract"],
        out["advertiser_zrive_id"].astype(str) + "_" + contract_number.astype(str),
        np.nan,
    )
    return out
