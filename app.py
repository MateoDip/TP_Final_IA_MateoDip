import base64
import streamlit as st
import pandas as pd
import requests
import os
import time
from dotenv import load_dotenv

# Cargamos las variables de entorno (el archivo .env)
load_dotenv()

# --- CLASE PRINCIPAL (POO) ---
class AnalizadorInteligente:
    def __init__(self, dataframe):
        self.df = dataframe
        try:
            self.api_key = st.secrets["HF_API_KEY"]
        except:
            self.api_key = os.getenv("HF_API_KEY")
        
    def limpiar_y_preparar(self):
        """Aplica limpieza básica usando comprehensions y Pandas."""
        # List comprehension para normalizar los nombres de las columnas
        self.df.columns = [col.strip().lower().replace(" ", "_") for col in self.df.columns]
        # Eliminamos filas con valores nulos
        self.df = self.df.dropna()
        return self.df

    def obtener_contexto_estadistico(self):
        """Genera un resumen estadístico + agregados por categoría, para que el LLM
        pueda responder con datos concretos y no solo con estadísticas generales."""
        partes = []
        partes.append(
            f"El dataset tiene {len(self.df)} filas y estas columnas: {', '.join(self.df.columns)}."
        )
        partes.append("Estadísticas generales:\n" + self.df.describe().to_string())

        columnas_numericas = self.df.select_dtypes(include="number").columns.tolist()
        columnas_texto = [
            col for col in self.df.select_dtypes(include="object").columns
            if "fecha" not in col and "date" not in col and self.df[col].nunique() < len(self.df)
        ]

        # Priorizamos columnas numéricas que parezcan de dinero/ingresos para el ranking
        palabras_clave = ["total", "monto", "ingreso", "venta", "importe", "precio"]
        columnas_relevantes = [c for c in columnas_numericas if any(p in c for p in palabras_clave)] or columnas_numericas

        if columnas_texto and columnas_relevantes:
            col_categoria = columnas_texto[0]
            col_monto = columnas_relevantes[0]
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

        # Muestra de filas reales para que la IA pueda citar casos concretos
        partes.append("Muestra de filas del dataset:\n" + self.df.head(10).to_string())

        return "\n\n".join(partes)

    def consultar_llm(self, pregunta):
        """Se comunica con la API de Hugging Face (router nuevo) usando requests de forma segura."""
        if not self.api_key:
            raise ValueError("No se encontró la API Key de Hugging Face.")

        # HF discontinuó api-inference.huggingface.co a fines de 2025.
        # El endpoint vigente es router.huggingface.co, con formato chat-completions (estilo OpenAI).
        url_segura = "aHR0cHM6Ly9yb3V0ZXIuaHVnZ2luZ2ZhY2UuY28vdjEvY2hhdC9jb21wbGV0aW9ucw=="
        url = base64.b64decode(url_segura).decode("utf-8")

        # Agregamos User-Agent y Content-Type para evitar bloqueos anti-bots
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MiAppDeAnalisisDatos/1.0"
        }

        # Inyectamos el contexto estadístico del dataset para que el LLM
        # pueda responder preguntas reales sobre los datos cargados.
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

        payload = {
            # mistralai/Mistral-7B-Instruct-v0.3 ya no lo sirve ningún Inference Provider.
            # Usamos un modelo que HF confirma activo en el router (formato modelo:proveedor).
            "model": "openai/gpt-oss-120b:cerebras",
            "messages": [
                {"role": "user", "content": prompt_completo}
            ],
            "max_tokens": 250,
            "temperature": 0.7
        }

        try:
            # Agregamos un timeout de 15 segundos para que no se quede colgado
            respuesta = requests.post(url, headers=headers, json=payload, timeout=15)
            respuesta.raise_for_status() # Esto nos avisará si la IA da error 400 o 500

            datos = respuesta.json()
            return datos["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            return f"Error de conexión detallado: {e}"

# --- FUNCIÓN GENERADORA ---
def efecto_maquina_escribir(texto):
    """Generador para simular que el LLM está escribiendo en tiempo real."""
    for palabra in texto.split(" "):
        yield palabra + " "
        time.sleep(0.05)

# --- INTERFAZ WEB (STREAMLIT) ---
st.set_page_config(page_title="Analizador IA", page_icon="📊")
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
        
        st.success("¡Archivo cargado y procesado correctamente!")
        
        # Mostramos los datos con Pandas, con nombres de columna más prolijos
        with st.expander("Ver primeros registros del Dataset"):
            df_vista = df_limpio.head().copy()
            df_vista.columns = [col.replace("_", " ").title() for col in df_vista.columns]
            st.dataframe(df_vista)

        with st.expander("Ver estadísticas descriptivas"):
            # Transponemos: cada variable es una fila, cada estadística una columna.
            # Así se lee como una ficha por variable en vez de una tabla críptica.
            stats = df_limpio.describe().T.rename(columns={
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

        # 2. Interacción con el LLM (Chat)
        st.subheader("💬 Chatea con tu Dataset")
        pregunta_usuario = st.text_input("Hazle una pregunta a la IA sobre estos datos:")
        
        if st.button("Consultar IA"):
            if pregunta_usuario:
                with st.spinner("Analizando con el LLM..."):
                    try:
                        respuesta_ia = analizador.consultar_llm(pregunta_usuario)
                        # Usamos nuestro generador para mostrar la respuesta
                        st.write_stream(efecto_maquina_escribir(respuesta_ia))
                    except Exception as e:
                        st.error(f"Error al conectar con la IA: {e}")
            else:
                st.warning("Por favor, escribe una pregunta primero.")

    except Exception as e:
        # Manejo de excepciones si el archivo no es un CSV válido
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
