from pathlib import Path
import pandas as pd
from loguru import logger

from src.config import PROCESSED_PATH, RAW_PATH

def load_csv(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Carga el dataset crudo desde CSV.

    Parameters:
        path: Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos sin procesar.
    """
    logger.info(f"Cargando dataset desde {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def save_parquet(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """
    Guarda el DataFrame procesado en formato Parquet.

    Parameters:
        df: DataFrame a guardar.
        path: Ruta de destino.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    logger.success(f"Dataset guardado en {path} ({path.stat().st_size / 1024:.1f} KB)")


def load_parquet(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """
    Carga el dataset procesado desde Parquet.

    Parameters:
        path: Ruta al archivo Parquet.

    Returns:
        pd.DataFrame: DataFrame procesado.
    """
    logger.info(f"Cargando dataset procesado desde {path}")
    return pd.read_parquet(path)