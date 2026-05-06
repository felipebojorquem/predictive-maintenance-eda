import pandas as pd
from loguru import logger


def assert_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Verifica que el DataFrame contiene las columnas requeridas.

    Parameters:
        df: DataFrame a verificar.
        required: Lista de columnas que deben existir.

    Raises:
        AssertionError: Si alguna columna requerida no existe.
    """
    missing = set(required) - set(df.columns)
    assert not missing, f"Columnas faltantes: {missing}"
    logger.debug(f"Validación de columnas OK — {len(required)} columnas verificadas")


def log_dataframe_info(df: pd.DataFrame, label: str = "") -> None:
    """
    Registra información básica del DataFrame en el log.

    Parameters:
        df: DataFrame a inspeccionar.
        label: Etiqueta opcional para identificar el log.
    """
    tag = f"[{label}] " if label else ""
    logger.info(f"{tag}Shape: {df.shape}")
    logger.info(f"{tag}Tipos:\n{df.dtypes}")
    logger.info(f"{tag}Nulos: {df.isnull().sum().sum()}")