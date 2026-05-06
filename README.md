# Predictive Maintenance EDA — AI4I 2020

Análisis exploratorio de datos del dataset AI4I 2020 para identificar patrones
operativos que preceden a fallos de maquinaria industrial. Este proyecto constituye
la base de un sistema de mantenimiento predictivo end-to-end, diseñado para escalar
hacia modelado ML, despliegue API y monitorización en producción.

---

## Dataset

| Atributo              | Valor                                                                            |
| --------------------- | -------------------------------------------------------------------------------- |
| **Fuente**            | UCI Machine Learning Repository                                                  |
| **URL**               | https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset |
| **Tamaño**            | 10.000 registros, 14 variables originales + 3 features de ingeniería             |
| **Licencia**          | CC BY 4.0                                                                        |
| **Formato procesado** | Parquet (snappy)                                                                 |

---

## Preguntas de investigación

1. ¿Cuál es la distribución de los 5 tipos de fallo y qué tipo domina?
2. ¿Bajo qué condiciones térmicas se concentran los fallos HDF?
3. ¿Existe una zona de peligro en el espacio torque × velocidad?
4. ¿El desgaste acumulado de herramienta predice el tipo de fallo TWF?
5. ¿El tipo de producto (L/M/H) tiene diferente tasa de fallos?

---

## Stack técnico

| Capa                 | Herramienta                           |
| -------------------- | ------------------------------------- |
| Gestor de entorno    | `uv` + `pyproject.toml` (PEP 517/621) |
| Python               | 3.11                                  |
| Procesamiento        | `pandas 2.x`, `numpy`                 |
| Almacenamiento       | `pyarrow` — Parquet (snappy)          |
| Validación de datos  | `pandera`                             |
| Visualización        | `plotly 5.x`, `seaborn 0.13+`         |
| EDA automático       | `data-profiling`                      |
| Logging              | `loguru`                              |
| Linting / Formatting | `ruff`                                |
| Tests                | `pytest`, `pytest-cov`                |
| Notebook             | JupyterLab 4.x                        |

---

## Estructura del proyecto

```
predictive_maintenance_eda/
├── pyproject.toml              # Dependencias y configuración
├── ruff.toml                   # Linter/formatter
├── README.md
├── main.py                     # Entrypoint reproducible del pipeline
│
├── data/
│   ├── raw/                    # ai4i2020.csv (.gitignore)
│   └── processed/              # ai4i2020_clean.parquet (.gitignore)
│
├── notebooks/
│   └── eda.ipynb               # Notebook principal
│
├── reports/
│   └── profiling/              # Reporte data-profiling (HTML)
│
├── src/
│   ├── config.py               # Rutas y constantes globales
│   ├── io.py                   # load_csv(), save_parquet(), load_parquet()
│   ├── cleaning.py             # clean(), validate_schema()
│   ├── features.py             # build_features()
│   ├── viz.py                  # Funciones de visualización
│   └── utils.py                # assert_columns(), log_dataframe_info()
│
└── tests/
    └── test_cleaning.py
```

---

## Pipeline reproducible

### 1. Instalar uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. Clonar el repositorio y configurar el entorno

```bash
git clone <repo-url>
cd predictive_maintenance_eda
uv python pin 3.11
uv venv
source .venv/bin/activate
uv sync
uv sync --group dev
```

### 3. Descargar el dataset

```bash
cd data/raw/
wget "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
unzip "ai4i+2020+predictive+maintenance+dataset.zip"
cd ../..
```

### 4. Ejecutar el pipeline completo

```bash
uv run python main.py
```

Esto genera:

- `data/processed/ai4i2020_clean.parquet` — dataset procesado
- `reports/profiling/eda_report.html` — reporte automático data-profiling
- `logs/pipeline.log` — log estructurado de la ejecución

### 5. Lanzar el notebook

```bash
uv run jupyter lab
```

Abrir `notebooks/eda.ipynb` con el kernel `Python (pdm-eda)`.

### 6. Ejecutar tests

```bash
uv run pytest tests/ -v --tb=short
```

---

## Hallazgos principales

### 1 — Distribución de fallos y desbalanceo de clase

El dataset presenta una tasa global de fallo del **3.39%** (339 de 10.000 registros).
El mecanismo de fallo más frecuente es **HDF** (Heat Dissipation Failure) con **115
casos** (33.9% de todos los fallos), seguido de OSF (98) y PWF (95). El fallo menos
frecuente es RNF (19 casos, 0.19%): al no estar asociado a ningún mecanismo de
degradación determinista, se descartará en la fase de modelado supervisado. El
desbalanceo 96.6/3.4 exigirá estrategias de compensación (SMOTE o `class_weight`)
en el clasificador.

> Referencia: `notebooks/eda.ipynb` — Sección 2 y Fig. 1

### 2 — Predictores principales por tipo de fallo

Cada tipo de fallo responde a un predictor dominante diferente:

- **HDF:** `temp_delta` discrimina con separación clara entre grupos. La mediana en
  operación normal es 9.8 K frente a 8.3 K en fallos HDF (reducción de **1.5 K**).
  La baja varianza del grupo HDF=1 (std=0.28 K) sugiere un umbral de activación
  próximo a 8.5 K.

- **PWF/OSF:** La zona de peligro se localiza en **1.200–1.400 rpm con torque >60 Nm**.
  Los fallos presentan una `power_W` media de **7.283 W** frente a **6.245 W** en
  operación normal (+16.6%). La correlación Torque↔RPM = **-0.88** confirma operación
  a potencia aproximadamente constante, validando `power_W` (P = τ × ω) como feature
  representativa del régimen operativo.

- **TWF:** `Tool wear [min]` predice TWF con separación casi perfecta. El **100%** de
  los fallos TWF ocurre con desgaste acumulado **>198 min** (mediana=214.5 min,
  RIC <20 min), mientras el 75% de la operación normal se mantiene por debajo de
  162 min. El patrón indica un umbral de activación determinista, no degradación
  gradual.

> Referencia: Fig. 2 — `plot_temp_delta_by_failure()` | Fig. 3 — `plot_torque_speed_scatter()` | Fig. 4 — `plot_wear_vs_twf()`

### 3 — Efecto del tipo de producto sobre la tasa de fallos

La tasa de fallo decrece de forma monotónica con la calidad del producto:
**L = 3.92%**, M = 2.77%, **H = 2.09%** — el tipo L falla **1.88x** más que el tipo H.
El perfil de desgaste acumulado es prácticamente idéntico entre los tres tipos
(distribución uniforme 0–210 min), descartando que la mayor tasa de L sea un
artefacto del tiempo de operación. La diferencia es intrínseca a las tolerancias de
fabricación y condiciones de proceso asignadas a cada calidad. `Type` tiene valor
predictivo para el clasificador y es independiente de `Tool wear [min]`.

> Referencia: Fig. 5a — `plot_failure_rate_by_type()` | Fig. 5b — `plot_wear_by_product_type()`

---

## Limitaciones del feature engineering

**`wear_rate`:** Esta feature mezcla una variable acumulada (`Tool wear [min]`) con
una variable instantánea (`power_W`). Físicamente, el tratamiento riguroso de la tasa
de desgaste requeriría la velocidad media histórica de operación de la herramienta,
obtenible mediante la ecuación de Taylor. En el contexto de este dataset sintético,
la feature actúa como proxy razonable, pero debe interpretarse con precaución en
aplicaciones industriales reales.

---

## Próximos pasos

- **Feature Engineering avanzado:** interacciones entre variables, codificación
  ordinal de `Type`, normalización para modelos lineales.
- **Modelado ML:** clasificación multiclase del tipo de fallo (Random Forest /
  XGBoost) y predicción de RUL (Remaining Useful Life).
- **Explicabilidad:** SHAP values para auditar las decisiones del modelo en contexto
  industrial.
- **Despliegue:** API FastAPI + Docker para consumir el modelo en tiempo real.
- **Monitorización:** MLflow + detección de drift en producción.

---

## Referencia del dataset

Matzka, S. (2020). _AI4I 2020 Predictive Maintenance Dataset_ [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
