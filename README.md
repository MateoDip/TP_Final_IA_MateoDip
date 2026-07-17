# 📊 Analizador de Datos con IA

Trabajo Práctico Final — Python para Inteligencia Artificial (UAI 2026)

Aplicación web que permite subir un dataset propio en CSV, explorarlo con herramientas de análisis de datos (Pandas) y conversar con él en lenguaje natural usando un modelo de lenguaje (LLM) alojado en Hugging Face. Combina las dos temáticas del curso: **Análisis de datos con Pandas** y **Uso de LLMs (prompting e inferencia)**, integradas en un mismo flujo: los resultados del análisis con Pandas (estadísticas, agregados, muestras de filas) se inyectan como contexto real en los prompts que recibe el modelo, para que sus respuestas estén basadas en los datos concretos del usuario y no en información genérica.

🔗 **Demo en vivo:** https://tp-final-mateodip.streamlit.app

## ¿Qué hace la aplicación?

- **Carga y limpieza automática** del CSV: normaliza nombres de columnas, elimina filas con datos faltantes, detecta y parsea columnas de fecha, y detecta y convierte columnas numéricas que vienen como texto con formato moneda (por ejemplo `"$1.200,50"`).
- **Resumen ejecutivo automático**: apenas se carga el archivo, la IA genera una interpretación de 4 a 6 líneas sobre los datos, sin que el usuario tenga que preguntar nada.
- **Filtros interactivos** en la barra lateral, generados dinámicamente según las columnas del dataset (categóricas, numéricas y de fecha).
- **Tablas descriptivas**: vista de los primeros registros y estadísticas descriptivas (promedio, desvío estándar, percentiles, etc.) traducidas y formateadas para que sean fáciles de leer.
- **Visualización de datos**: gráficos de barras, torta y línea temporal generados automáticamente con Plotly, según qué columnas detecte el dataset.
- **Chat con el dataset**: el usuario puede hacer preguntas en lenguaje natural sobre sus datos y recibir respuestas basadas en el contenido real (no solo en estadísticas generales).
- **Descarga de reporte en Excel** con los datos filtrados, las estadísticas y el ranking por categoría.

## Stack tecnológico

- **[Streamlit](https://streamlit.io/)** — interfaz web
- **[Pandas](https://pandas.pydata.org/)** — carga, limpieza, transformación y agregación de datos
- **[Plotly](https://plotly.com/python/)** — visualización de datos
- **[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index)** (router OpenAI-compatible) — inferencia sobre un LLM (`openai/gpt-oss-120b`) vía `requests`
- **[openpyxl](https://openpyxl.readthedocs.io/)** — generación del reporte Excel
- **python-dotenv** — manejo de variables de entorno en desarrollo local

## Estructura del repositorio

```
├── app.py                  # Aplicación principal (interfaz + lógica)
├── requirements.txt        # Dependencias del proyecto
├── ventas_pruebas.csv       # Dataset de ejemplo para probar la app
└── README.md
```

## Instalación local

1. Cloná el repositorio:
   ```bash
   git clone https://github.com/MateoDip/TP_Final_IA_MateoDip.git
   cd TP_Final_IA_MateoDip
   ```

2. Creá y activá un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate    # En Windows: venv\Scripts\activate
   ```

3. Instalá las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Configurá tu API key de Hugging Face. Creá un archivo `.env` en la raíz del proyecto (no se sube al repo, está en `.gitignore`) con:
   ```
   HF_API_KEY=tu_token_de_huggingface
   ```
   Podés generar un token en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), con el permiso **"Make calls to Inference Providers"** habilitado.

## Ejecución local

```bash
streamlit run app.py
```

La aplicación va a estar disponible en `http://localhost:8501`. Desde ahí podés subir el CSV de ejemplo (`ventas_pruebas.csv`) o cualquier dataset propio con estructura similar (una o más columnas numéricas, alguna columna categórica y opcionalmente una columna de fecha).

## Deploy

La aplicación está deployada en **Streamlit Community Cloud**: https://tp-final-mateodip.streamlit.app

Para deployar una copia propia, conectá el repositorio desde [share.streamlit.io](https://share.streamlit.io/) y configurá el secreto `HF_API_KEY` en **Settings → Secrets** con el mismo formato que el `.env` local:
```toml
HF_API_KEY = "tu_token_de_huggingface"
```

## Autor

Mateo Dip — Trabajo Práctico Final, Python (UAI 2026)
