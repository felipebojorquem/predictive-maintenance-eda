import pandas as pd
import pytest

from src.cleaning import clean
from src.features import build_features


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Dataset mínimo representativo para testing."""
    return pd.DataFrame(
        {
            "UDI": [1, 2, 3],
            "Product ID": ["M14860", "L47181", "L47182"],
            "Type": ["M", "L", "L"],
            "Air temperature [K]": [298.1, 298.2, 298.1],
            "Process temperature [K]": [308.6, 308.7, 308.6],
            "Rotational speed [rpm]": [1551, 1408, 1498],
            "Torque [Nm]": [42.8, 46.3, 49.4],
            "Tool wear [min]": [0, 3, 5],
            "Machine failure": [0, 0, 0],
            "TWF": [0, 0, 0],
            "HDF": [0, 0, 0],
            "PWF": [0, 0, 0],
            "OSF": [0, 0, 0],
            "RNF": [0, 0, 0],
        }
    )


# ─── Tests de clean() ────────────────────────────────────────────────────────

def test_clean_removes_id_columns(sample_df: pd.DataFrame) -> None:
    """UDI y Product ID deben eliminarse — no aportan al análisis."""
    result = clean(sample_df)
    assert "UDI" not in result.columns
    assert "Product ID" not in result.columns


def test_clean_no_nulls(sample_df: pd.DataFrame) -> None:
    """El dataset limpio no debe contener valores nulos."""
    result = clean(sample_df)
    assert result.isnull().sum().sum() == 0


def test_clean_type_column_is_category(sample_df: pd.DataFrame) -> None:
    """Type debe castearse a dtype 'category' para optimizar memoria."""
    result = clean(sample_df)
    assert result["Type"].dtype.name == "category"


def test_clean_failure_cols_are_int(sample_df: pd.DataFrame) -> None:
    """Las columnas de fallo deben ser enteros — son flags binarios."""
    result = clean(sample_df)
    for col in ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert pd.api.types.is_integer_dtype(result[col]), f"{col} no es entero"


# ─── Tests de build_features() ───────────────────────────────────────────────

def test_build_features_adds_columns(sample_df: pd.DataFrame) -> None:
    """Las tres features de ingeniería deben existir en el output."""
    df_clean = clean(sample_df)
    result = build_features(df_clean)
    assert "temp_delta" in result.columns
    assert "power_W" in result.columns
    assert "wear_rate" in result.columns


def test_temp_delta_physics(sample_df: pd.DataFrame) -> None:
    """temp_delta debe ser positivo: el proceso siempre es más caliente que el aire."""
    df_clean = clean(sample_df)
    result = build_features(df_clean)
    assert (result["temp_delta"] > 0).all()


def test_power_W_positive(sample_df: pd.DataFrame) -> None:
    """La potencia mecánica estimada debe ser estrictamente positiva."""
    df_clean = clean(sample_df)
    result = build_features(df_clean)
    assert (result["power_W"] > 0).all()