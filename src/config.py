# src/config.py
from pathlib import Path

# Rutas base — relativas a la raíz del proyecto
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "ai4i2020_clean.parquet"
REPORT_PATH = ROOT / "reports" / "profiling" / "eda_report.html"

# Constantes del dataset
TARGET_COL = "Machine failure"
FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_COLS = ["Type"]