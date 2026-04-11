import logging
import pandas as pd
import logging

def validate_columns(df: pd.DataFrame, required_cols: set[str], func_name: str) -> None:
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[{func_name}] faltan columnas requeridas: {sorted(missing_cols)}"
        )
    
