import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure
from loguru import logger

from src.config import FAILURE_TYPES, NUMERIC_COLS


def plot_failure_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Gráfico 1: Distribución de tipos de fallo.

    Parameters:
        df: pd.DataFrame DataFrame procesado.

    Returns:
        go.Figure: Gráfico de barras con frecuencia de cada tipo de fallo.
    """
    counts = df[FAILURE_TYPES].sum().reset_index()
    counts.columns = ["Tipo de fallo", "Cantidad"]
    fig = px.bar(
        counts,
        x="Tipo de fallo",
        y="Cantidad",
        title="Distribución de Tipos de Fallo — AI4I 2020",
        color="Tipo de fallo",
        text="Cantidad",
        labels={"Cantidad": "Número de ocurrencias"},
    )
    fig.update_layout(showlegend=False)
    logger.info("Gráfico 1 generado: distribución de fallos")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Gráfico 2: Heatmap de correlación entre variables numéricas.

    Parameters:
        df: pd.DataFrame DataFrame procesado con features de ingeniería.

    Returns:
        plt.Figure: Heatmap de correlación incluyendo features.
    """
    cols = NUMERIC_COLS + ["temp_delta", "power_W", "wear_rate", "Machine failure"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=True,
    )
    ax.set_title("Correlación entre variables — incluye features de ingeniería")
    plt.tight_layout()
    logger.info("Gráfico 2 generado: heatmap de correlación")
    return fig


def plot_torque_speed_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Gráfico 3: Scatter torque x velocidad coloreado por fallo.

    Parameters:
        df: pd.DataFrame DataFrame procesado.

    Returns:
        go.Figure: Scatter plot identificando zona de peligro operativa.
    """
    fig = px.scatter(
        df,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        color="Machine failure",
        color_discrete_map={0: "#2196F3", 1: "#F44336"},
        opacity=0.6,
        title="Zona de Operación: Velocidad vs Torque",
        labels={"Machine failure": "Fallo"},
    )
    logger.info("Gráfico 3 generado: scatter torque vs velocidad")
    return fig


def plot_temp_delta_by_failure(df: pd.DataFrame) -> go.Figure:
    """
    Gráfico 4: Boxplot de delta térmico por fallo HDF.

    Parameters:
        df: pd.DataFrame DataFrame procesado con feature temp_delta.

    Returns:
        go.Figure: Boxplot que valida si temp_delta discrimina fallos térmicos.
    """
    fig = px.box(
        df,
        x="HDF",
        y="temp_delta",
        color="HDF",
        color_discrete_map={0: "#4CAF50", 1: "#FF5722"},
        title="Delta Térmico: Fallos por Disipación de Calor (HDF)",
        labels={
            "HDF": "Heat Dissipation Failure",
            "temp_delta": "ΔT Proceso-Aire [K]",
        },
    )
    logger.info("Gráfico 4 generado: boxplot delta térmico vs HDF")
    return fig


def plot_wear_by_product_type(df: pd.DataFrame) -> go.Figure:
    """
    Gráfico 5: Distribución de desgaste de herramienta por tipo de producto.

    Parameters:
        df: pd.DataFrame DataFrame procesado.

    Returns:
        go.Figure: Histograma de desgaste por tipo L/M/H.
    """
    fig = px.histogram(
        df,
        x="Tool wear [min]",
        color="Type",
        nbins=40,
        barmode="overlay",
        opacity=0.7,
        title="Desgaste de Herramienta por Tipo de Producto",
        labels={
            "Tool wear [min]": "Desgaste acumulado (min)",
            "Type": "Tipo",
        },
        color_discrete_map={"L": "#EF5350", "M": "#FFA726", "H": "#66BB6A"},
    )
    logger.info("Gráfico 5 generado: histograma desgaste por tipo")
    return fig


def plot_failure_rate_by_type(df: pd.DataFrame) -> go.Figure:
    """
    Gráfico 6: Tasa de fallo por tipo de producto.

    Parameters:
        df: pd.DataFrame DataFrame procesado.

    Returns:
        go.Figure: Barras con tasa de fallo (%) por tipo L/M/H.
    """
    rates = (
        df.groupby("Type")["Machine failure"]
        .agg(["sum", "count"])
        .reset_index()
    )
    rates["failure_rate"] = rates["sum"] / rates["count"] * 100
    fig = px.bar(
        rates,
        x="Type",
        y="failure_rate",
        color="Type",
        title="Tasa de Fallo por Tipo de Producto (%)",
        labels={
            "failure_rate": "Tasa de fallo (%)",
            "Type": "Tipo de producto",
        },
        text=rates["failure_rate"].round(2).astype(str) + "%",
    )
    logger.info("Gráfico 6 generado: tasa de fallo por tipo")
    return fig