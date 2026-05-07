from pathlib import Path
from loguru import logger
from src.cleaning import clean, validate_schema
from src.config import REPORT_PATH
from src.features import build_features
from src.io import load_csv, save_parquet
from src.utils import log_dataframe_info
from src.viz import (
    plot_failure_distribution,
    plot_temp_delta_by_failure,
    plot_torque_speed_scatter,
    plot_wear_vs_twf,
    plot_failure_rate_by_type,
    plot_correlation_heatmap,
)
import matplotlib.pyplot as plt

# Configurar logging a archivo
Path("logs").mkdir(exist_ok=True)
logger.add("logs/pipeline.log", rotation="1 MB", level="INFO")

FIGURES_PATH = Path("reports") / "figures"


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

    # 4. Crear Features
    df = build_features(df)
    log_dataframe_info(df, label="PROCESSED")

    # 5. Exportar dataset procesado como Parquet
    save_parquet(df)

    # 6. Generar reporte de profiling
    logger.info("Generando reporte data-profiling")
    from data_profiling import ProfileReport
    profile = ProfileReport(df, title="AI4I 2020 EDA Report", explorative=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile.to_file(REPORT_PATH)
    logger.success(f"Reporte guardado en {REPORT_PATH}")

    # 7. Exportar visualizaciones clave
    logger.info("Generando visualizaciones — reports/figures/")
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    # Figuras plotly → HTML (interactivas, no requieren kaleido)
    plot_failure_distribution(df).write_html(
        str(FIGURES_PATH / "fig1_failure_distribution.html")
    )
    plot_temp_delta_by_failure(df).write_html(
        str(FIGURES_PATH / "fig2_temp_delta_hdf.html")
    )
    plot_torque_speed_scatter(df).write_html(
        str(FIGURES_PATH / "fig3_torque_speed_scatter.html")
    )
    plot_failure_rate_by_type(df).write_html(
        str(FIGURES_PATH / "fig5_failure_rate_by_type.html")
    )

    # Figuras matplotlib → PNG
    fig4 = plot_wear_vs_twf(df)
    fig4.savefig(FIGURES_PATH / "fig4_wear_vs_twf.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    fig6 = plot_correlation_heatmap(df)
    fig6.savefig(FIGURES_PATH / "fig6_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)

    logger.success(f"6 figuras guardadas en {FIGURES_PATH}")
    logger.success("PIPELINE COMPLETADO")


if __name__ == "__main__":
    main()