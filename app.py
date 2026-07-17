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
        """Genera un resumen estadístico para inyectarlo al LLM."""
        return self.df.describe().to_string()

    def consultar_llm(self, pregunta):
        """Se comunica con la API de Hugging Face usando requests de forma segura."""
        if not self.api_key:
            raise ValueError("No se encontró la API Key de Hugging Face.")

        # Decodificamos la URL en tiempo real para destruir caracteres ocultos
        url_segura = "aHR0cHM6Ly9hcGktaW5mZXJlbmNlLmh1Z2dpbmdmYWNlLmNvL21vZGVscy9taXN0cmFsYWkvTWlzdHJhbC03Qi1JbnN0cnVjdC12MC4z"
        url = base64.b64decode(url_segura).decode("utf-8")
        
        # Agregamos User-Agent y Content-Type para evitar bloqueos anti-bots
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MiAppDeAnalisisDatos/1.0"
        }
        
        payload = {
            "inputs": pregunta,
            "parameters": {"max_new_tokens": 250, "temperature": 0.7}
        }
        
        try:
            # Agregamos un timeout de 15 segundos para que no se quede colgado
            respuesta = requests.post(url, headers=headers, json=payload, timeout=15)
            respuesta.raise_for_status() # Esto nos avisará si la IA da error 400 o 500
            
            datos = respuesta.json()
            return datos[0]["generated_text"]
            
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
        
        # Mostramos los datos con Pandas
        with st.expander("Ver primeros registros del Dataset"):
            st.dataframe(df_limpio.head())
            
        with st.expander("Ver estadísticas descriptivas"):
            st.dataframe(df_limpio.describe())

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
