import base64
import io
import time
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

# Cargamos las variables de entorno (el archivo .env)
load_dotenv()

# --- CLASE PRINCIPAL (POO) ---
class AnalizadorInteligente:
    def __init__(self, dataframe):
        self.df = dataframe
        try:
            self.api_key = st.secrets["HF_API_KEY"]
        except Exception:
            self.api_key = os.getenv("HF_API_KEY")

    def limpiar_y_preparar(self):
        """Aplica limpieza básica usando comprehensions y Pandas."""
        # List comprehension para normalizar los nombres de las columnas
        self.df.columns = [col.strip().lower().replace(" ", "_") for col in self.df.columns]
        # Eliminamos filas con valores nulos
        self.df = self.df.dropna()
        # Intentamos parsear automáticamente columnas que parecen fechas
        for col in self.df.columns:
            if "fecha" in col or "date" in col:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except (ValueError, TypeError):
                    pass
        return self.df

    def detectar_columnas(self):
        """Clasifica las columnas del dataset en fecha / numéricas / categóricas,
        para poder armar filtros, gráficos y contexto de forma genérica."""
        columnas_fecha = [c for c in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[c])]
        columnas_numericas = self.df.select_dtypes(include="number").columns.tolist()
        columnas_categoricas = [
            c for c in self.df.select_dtypes(include="object").columns
            if c not in columnas_fecha and self.df[c].nunique() < len(self.df)
        ]
        return columnas_fecha, columnas_numericas, columnas_categoricas

    def columna_monto_principal(self, columnas_numericas):
        """Elige la columna numérica más probable de representar dinero/ingresos."""
        palabras_clave = ["total", "monto", "ingreso", "venta", "importe", "precio"]
        candidatas = [c for c in columnas_numericas if any(p in c for p in palabras_clave)]
        return (candidatas or columnas_numericas or [None])[0]

    def obtener_contexto_estadistico(self):
        """Genera un resumen estadístico + agregados por categoría, para que el LLM
        pueda responder con datos concretos y no solo con estadísticas generales."""
        partes = [f"El dataset tiene {len(self.df)} filas y estas columnas: {', '.join(self.df.columns)}."]
        partes.append("Estadísticas generales:\n" + self.df.describe().to_string())

        columnas_fecha, columnas_numericas, columnas_categoricas = self.detectar_columnas()
        col_monto = self.columna_monto_principal(columnas_numericas)

        if columnas_categoricas and col_monto:
            col_categoria = columnas_categoricas[0]
            ranking = (
                self.df.groupby(col_categoria)[col_monto]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            partes.append(
                f"Ranking de '{col_categoria}' según la suma de '{col_monto}' (de mayor a menor):\n"
                + ranking.to_string()
            )

        partes.append("Muestra de filas del dataset:\n" + self.df.head(10).to_string())
        return "\n\n".join(partes)

    def _llamar_llm(self, contenido_usuario, max_tokens=250):
        """Hace la llamada real a la API de Hugging Face (router nuevo). Reutilizado
        tanto para el chat como para el resumen ejecutivo automático."""
        if not self.api_key:
            raise ValueError("No se encontró la API Key de Hugging Face.")

        # HF discontinuó api-inference.huggingface.co a fines de 2025.
        url_segura = "aHR0cHM6Ly9yb3V0ZXIuaHVnZ2luZ2ZhY2UuY28vdjEvY2hhdC9jb21wbGV0aW9ucw=="
        url = base64.b64decode(url_segura).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MiAppDeAnalisisDatos/1.0"
        }

        payload = {
            # mistralai/Mistral-7B-Instruct-v0.3 ya no lo sirve ningún Inference Provider.
            "model": "openai/gpt-oss-120b:cerebras",
            "messages": [{"role": "user", "content": contenido_usuario}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        respuesta = requests.post(url, headers=headers, json=payload, timeout=20)
        respuesta.raise_for_status()
        datos = respuesta.json()
        return datos["choices"][0]["message"]["content"]

    def consultar_llm(self, pregunta):
        """Responde una pregunta puntual del usuario sobre el dataset."""
        contexto = self.obtener_contexto_estadistico()
        prompt_completo = (
            "Sos un asistente que analiza datasets de ventas. A continuación tenés información real "
            "del dataset del usuario: estadísticas generales, un ranking agregado y una muestra de filas. "
            "Respondé la pregunta de forma breve, concreta y basada ÚNICAMENTE en estos datos. "
            "Si el ranking o la muestra ya contestan la pregunta, decilo directamente (por ejemplo, nombrando "
            "el producto o categoría exacto), sin quedarte solo en el número máximo/mínimo.\n\n"
            f"{contexto}\n\n"
            f"Pregunta: {pregunta}"
        )
        try:
            return self._llamar_llm(prompt_completo)
        except requests.exceptions.RequestException as e:
            return f"Error de conexión detallado: {e}"

    def generar_resumen_ejecutivo(self):
        """Genera automáticamente un resumen ejecutivo apenas se carga el dataset,
        sin que el usuario tenga que preguntar nada."""
        contexto = self.obtener_contexto_estadistico()
        prompt = (
            "Sos un analista de datos. A partir de la siguiente información de un dataset, "
            "escribí un resumen ejecutivo de 4 a 6 líneas en español, destacando los hallazgos "
            "más relevantes (qué producto/categoría se destaca, tendencias, algo llamativo). "
            "Interpretá los datos, no repitas los números crudos sin contexto.\n\n" + contexto
        )
        return self._llamar_llm(prompt, max_tokens=300)


# --- FUNCIÓN GENERADORA ---
def efecto_maquina_escribir(texto):
    """Generador para simular que el LLM está escribiendo en tiempo real."""
    for palabra in texto.split(" "):
        yield palabra + " "
        time.sleep(0.05)


def generar_reporte_excel(df_filtrado, stats, agrupado=None, nombre_categoria=None):
    """Arma un Excel en memoria con los datos filtrados, las estadísticas y el ranking."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_filtrado.to_excel(writer, sheet_name="Datos filtrados", index=False)
        stats.to_excel(writer, sheet_name="Estadisticas")
        if agrupado is not None and nombre_categoria:
            agrupado.to_excel(writer, sheet_name=f"Ranking {nombre_categoria}"[:31], index=False)
    return buffer.getvalue()


# --- INTERFAZ WEB (STREAMLIT) ---
st.set_page_config(page_title="Analizador IA", page_icon="📊", layout="wide")
st.title("📊 Analizador de Datos con IA")
st.markdown("Sube tu archivo CSV y chatea con tus datos utilizando Pandas y Hugging Face.")

# 1. Carga de Archivo (Input)
archivo_subido = st.file_uploader("Carga tu dataset (CSV)", type=["csv"])

if archivo_subido is not None:
    try:
        # Cargamos con Pandas
        df_original = pd.read_csv(archivo_subido)

        # Instanciamos nuestra clase
        analizador = AnalizadorInteligente(df_original)
        df_limpio = analizador.limpiar_y_preparar()
        columnas_fecha, columnas_numericas, columnas_categoricas = analizador.detectar_columnas()
        col_monto = analizador.columna_monto_principal(columnas_numericas)

        st.success("¡Archivo cargado y procesado correctamente!")

        # --- RESUMEN EJECUTIVO AUTOMÁTICO (se genera una sola vez por archivo) ---
        st.subheader("🤖 Resumen ejecutivo automático")
        clave_cache = f"resumen_{archivo_subido.name}_{archivo_subido.size}"
        if clave_cache not in st.session_state:
            with st.spinner("La IA está analizando tu dataset..."):
                try:
                    st.session_state[clave_cache] = analizador.generar_resumen_ejecutivo()
                except Exception as e:
                    st.session_state[clave_cache] = f"No se pudo generar el resumen automático: {e}"
        st.info(st.session_state[clave_cache])

        # --- FILTROS EN LA BARRA LATERAL ---
        st.sidebar.header("🔎 Filtros")
        df_filtrado = df_limpio.copy()

        for col in columnas_categoricas:
            valores = sorted(df_limpio[col].dropna().unique().tolist())
            seleccion = st.sidebar.multiselect(col.replace("_", " ").title(), valores, default=valores)
            if seleccion:
                df_filtrado = df_filtrado[df_filtrado[col].isin(seleccion)]

        for col in columnas_numericas:
            min_val, max_val = float(df_limpio[col].min()), float(df_limpio[col].max())
            if min_val < max_val:
                rango = st.sidebar.slider(col.replace("_", " ").title(), min_val, max_val, (min_val, max_val))
                df_filtrado = df_filtrado[(df_filtrado[col] >= rango[0]) & (df_filtrado[col] <= rango[1])]

        if columnas_fecha:
            col_fecha = columnas_fecha[0]
            fecha_min, fecha_max = df_limpio[col_fecha].min(), df_limpio[col_fecha].max()
            if fecha_min < fecha_max:
                rango_fecha = st.sidebar.date_input(
                    col_fecha.replace("_", " ").title(), (fecha_min, fecha_max)
                )
                if isinstance(rango_fecha, tuple) and len(rango_fecha) == 2:
                    df_filtrado = df_filtrado[
                        (df_filtrado[col_fecha] >= pd.Timestamp(rango_fecha[0])) &
                        (df_filtrado[col_fecha] <= pd.Timestamp(rango_fecha[1]))
                    ]

        st.sidebar.caption(f"Mostrando {len(df_filtrado)} de {len(df_limpio)} filas")

        if df_filtrado.empty:
            st.warning("Los filtros aplicados no dejan ninguna fila. Ajustá los filtros de la barra lateral.")
            st.stop()

        # A partir de acá, el chat y los cálculos usan el dataset ya filtrado
        analizador.df = df_filtrado

        # Mostramos los datos con Pandas, con nombres de columna más prolijos
        with st.expander("Ver primeros registros del Dataset"):
            df_vista = df_filtrado.head().copy()
            df_vista.columns = [col.replace("_", " ").title() for col in df_vista.columns]
            st.dataframe(df_vista)

        with st.expander("Ver estadísticas descriptivas"):
            stats = df_filtrado.describe().T.rename(columns={
                "count": "Cantidad de datos",
                "mean": "Promedio",
                "std": "Desvío estándar",
                "min": "Mínimo",
                "25%": "Percentil 25",
                "50%": "Mediana",
                "75%": "Percentil 75",
                "max": "Máximo",
            })
            stats.index = [i.replace("_", " ").title() for i in stats.index]
            st.dataframe(stats.style.format("{:.2f}"))

        # --- GRÁFICOS AUTOMÁTICOS ---
        st.subheader("📈 Visualización de datos")
        agrupado = None
        col_categoria_principal = columnas_categoricas[0] if columnas_categoricas else None

        if col_categoria_principal and col_monto:
            agrupado = (
                df_filtrado.groupby(col_categoria_principal)[col_monto]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            col1, col2 = st.columns(2)
            with col1:
                fig_barras = px.bar(
                    agrupado, x=col_categoria_principal, y=col_monto,
                    title=f"{col_monto.title()} por {col_categoria_principal.title()}",
                )
                st.plotly_chart(fig_barras, use_container_width=True)

            with col2:
                fig_torta = px.pie(
                    agrupado, names=col_categoria_principal, values=col_monto,
                    title=f"Distribución de {col_monto.title()} por {col_categoria_principal.title()}",
                )
                st.plotly_chart(fig_torta, use_container_width=True)

        if columnas_fecha and col_monto:
            col_fecha_principal = columnas_fecha[0]
            serie = df_filtrado.groupby(col_fecha_principal)[col_monto].sum().reset_index()
            fig_linea = px.line(
                serie, x=col_fecha_principal, y=col_monto, markers=True,
                title=f"{col_monto.title()} a lo largo del tiempo",
            )
            st.plotly_chart(fig_linea, use_container_width=True)

        if not (col_categoria_principal and col_monto) and not (columnas_fecha and col_monto):
            st.caption("No se detectaron columnas categóricas/de fecha combinables con una columna numérica para graficar.")

        # --- DESCARGA DE REPORTE ---
        st.subheader("⬇️ Descargar reporte")
        excel_bytes = generar_reporte_excel(df_filtrado, stats, agrupado, col_categoria_principal)
        st.download_button(
            label="Descargar reporte en Excel",
            data=excel_bytes,
            file_name="reporte_analisis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # --- INTERACCIÓN CON EL LLM (Chat) ---
        st.subheader("💬 Chatea con tu Dataset")
        pregunta_usuario = st.text_input("Hazle una pregunta a la IA sobre estos datos:")

        if st.button("Consultar IA"):
            if pregunta_usuario:
                with st.spinner("Analizando con el LLM..."):
                    try:
                        respuesta_ia = analizador.consultar_llm(pregunta_usuario)
                        st.write_stream(efecto_maquina_escribir(respuesta_ia))
                    except Exception as e:
                        st.error(f"Error al conectar con la IA: {e}")
            else:
                st.warning("Por favor, escribe una pregunta primero.")

    except Exception as e:
        # Manejo de excepciones si el archivo no es un CSV válido
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
