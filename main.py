# main.py
"""
Pipeline reproducible: load → clean → validate → features → export → profile
Ejecutar desde la raíz del proyecto: uv run python main.py
"""
from pathlib import Path

from loguru import logger

from src.cleaning import clean, validate_schema
from src.config import REPORT_PATH
from src.features import build_features
from src.io import load_csv, save_parquet
from src.utils import log_dataframe_info

# Configurar logging a archivo
Path("logs").mkdir(exist_ok=True)
logger.add("logs/pipeline.log", rotation="1 MB", level="INFO")


def main() -> None:
    logger.info("=" * 50)
    logger.info("INICIANDO PIPELINE EDA — AI4I 2020")
    logger.info("=" * 50)

    # 1. Carga
    df = load_csv()
    log_dataframe_info(df, label="RAW")

    # 2. Limpieza
    df = clean(df)

    # 3. Validación de schema
    df = validate_schema(df)

    # 4. Feature engineering
    df = build_features(df)
    log_dataframe_info(df, label="PROCESSED")

    # 5. Exportar dataset procesado como Parquet
    save_parquet(df)

    # 6. Generar reporte de profiling
    logger.info("Generando reporte ydata-profiling")
    from ydata_profiling import ProfileReport

    profile = ProfileReport(df, title="AI4I 2020 EDA Report", explorative=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile.to_file(REPORT_PATH)
    logger.success(f"Reporte guardado en {REPORT_PATH}")

    logger.success("PIPELINE COMPLETADO")


if __name__ == "__main__":
    main()