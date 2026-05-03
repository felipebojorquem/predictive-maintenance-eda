import pandas as pd
import pandera.pandas as pa
from loguru import logger

from src.config import FAILURE_TYPES, TARGET_COL

SCHEMA = pa.DataFrameSchema(
    {
        "Air temperature [K]": pa.Column(float, pa.Check.in_range(290, 310)),
        "Process temperature [K]": pa.Column(float, pa.Check.in_range(305, 315)),
        "Rotational speed [rpm]": pa.Column(int, pa.Check.in_range(1000, 3000)),
        "Torque [Nm]": pa.Column(float, pa.Check.in_range(0, 80)),
        "Tool wear [min]": pa.Column(int, pa.Check.in_range(0, 300)),
        "Type": pa.Column(str, pa.Check.isin(["L", "M", "H"])),
        "Machine failure": pa.Column(int, pa.Check.isin([0, 1])),
    }
)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpieza estándar al dataset AI4I 2020.

    Parameters:
        df: pd.DataFrame Dataset crudo cargado desde CSV.

    Returns:
        pd.DataFrame: DataFrame limpio listo para feature engineering.
    """
    logger.info("Iniciando limpieza del dataset")
    original_shape = df.shape

    df = df.drop(columns=["UDI", "Product ID"], errors="ignore")

    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Nulos detectados:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("Sin valores nulos")

    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        df = df.drop_duplicates()
        logger.warning(f"Eliminados {n_duplicates} duplicados")

    for col in FAILURE_TYPES + [TARGET_COL]:
        df[col] = df[col].astype(int)
    df["Type"] = df["Type"].astype("category")

    logger.success(f"Limpieza completada: {original_shape} → {df.shape}")
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida el DataFrame contra el schema definido con pandera.

    Parameters:
        df: pd.DataFrame DataFrame a validar.

    Returns:
        pd.DataFrame: DataFrame validado.
    """
    logger.info("Validando schema del dataset")
    return SCHEMA.validate(df)