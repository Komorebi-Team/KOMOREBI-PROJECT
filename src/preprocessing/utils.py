import logging
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def validate_columns(df: pd.DataFrame, required_cols: set, func_name: str):
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[{func_name}] faltan columnas requeridas: {sorted(missing_cols)}"
        )
    
