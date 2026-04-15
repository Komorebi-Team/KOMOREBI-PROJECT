import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_columns(df: pd.DataFrame, required_cols: set[str], func_name: str) -> None:
    """Lanza ValueError si al DataFrame le faltan columnas requeridas."""
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[{func_name}] faltan columnas requeridas: {sorted(missing_cols)}"
        )


def compute_class_sample_weight(y: pd.Series) -> np.ndarray:
    """
    Calcula pesos de muestra para compensar el desbalanceo de clases binario.

    Asigna peso `scale` a la clase positiva (1) y peso 1 a la negativa (0),
    donde `scale = n_negatives / n_positives`.

    Parameters
    ----------
    y : pd.Series
        Vector de etiquetas binarias (0/1).

    Returns
    -------
    np.ndarray
        Array de pesos, mismo orden que `y`.
    """
    scale = (y == 0).sum() / (y == 1).sum()
    return np.where(y == 1, scale, 1.0)
