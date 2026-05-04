import numpy as np
import pandas as pd
from loguru import logger


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features con fundamento físico.

    Parameters:
        df: DataFrame limpio de la fase de cleaning.

    Returns:
        pd.DataFrame: DataFrame con 3 features adicionales.
    """
    logger.info("Construyendo features")
    df = df.copy()

    # Feature 1: Delta térmico
    # Para verificar si el proceso mantiene una diferencia estable entre
    # temperatura de proceso y temperatura ambiente.
    # Desviaciones indican problemas de disipación de calor (candidato a HDF).
    df["temp_delta"] = (df["Process temperature [K]"] - df["Air temperature [K]"])

    # Feature 2: Potencia mecánica estimada (vatios)
    # P = τ × ω, donde ω = rpm × 2π/60
    # Permite identificar la zona de operación en el espacio torque-velocidad.
    df["power_W"] = (df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60))

    # Feature 3: Tasa de desgaste normalizada
    # Relaciona el desgaste acumulado con la potencia operativa.
    # Valores altos → herramienta degradada operando en condiciones exigentes.
    df["wear_rate"] = df["Tool wear [min]"] / (df["power_W"] + 1e-6)

    logger.success("Features generadas: temp_delta, power_W, wear_rate")
    return df