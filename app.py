import os
import re
import unicodedata
from pathlib import Path
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

# =====================
# Cargar archivos
# =====================
load_dotenv()
content_dir = Path("content")
training_path =  "training set.txt"
vocabulario_path =  "vocabulario.txt"

with open(training_path, encoding="utf-8") as f:
    training_set = f.read()

with open(vocabulario_path, encoding="utf-8") as f:
    vocabulario_txt = f.read()

vocabulario = set(word.strip().lower() for word in vocabulario_txt.splitlines() if word.strip())

# =====================
# Construir mensajes
# =====================
def construir_mensajes(usuario_pregunta):
    system_prompt = (
        "Eres un asistente de chino mandarín para estudiantes hispanohablantes. "
        "Debes ayudar a traducir frases o explicar gramática usando únicamente el vocabulario y el set de entrenamiento proporcionado. "
        "Si una palabra no está en el vocabulario, no debes decir 'no puedo responder'. En cambio, intenta dar una alternativa cercana o aproximada que esté dentro del vocabulario. "
        "Si haces alguna adaptación, explícalo brevemente. "
        "La respuesta debe ser clara, educativa, y sin repetir el listado completo de palabras fuera del vocabulario. "
        "No incluyas disculpas. No repitas la pregunta. No uses frases como 'lo siento'. "
        "Responde de forma directa y útil. Solo menciona palabras fuera del vocabulario si son estrictamente necesarias para explicar la alternativa."
    )

    context_prompt = f"[SET DE ENTRENAMIENTO]\n{training_set}\n\n[VOCABULARIO]\n{vocabulario_txt}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context_prompt}\n\n[PREGUNTA DEL ESTUDIANTE]: {usuario_pregunta}"}
    ]

# =====================
# Inicializar clientes
# =====================
openai_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")

# =====================
# Función principal
# =====================
def responder_pregunta(pregunta, usar_deepseek=True):
    mensajes = construir_mensajes(pregunta)
    cliente = deepseek_client if usar_deepseek else openai_client
    modelo = "deepseek-chat" if usar_deepseek else "gpt-4.1-nano"

    respuesta = cliente.chat.completions.create(
        model=modelo,
        messages=mensajes
    )

    return respuesta.choices[0].message.content

# =====================
# Interfaz Streamlit
# =====================
st.set_page_config(page_title="Asistente de Mandarín", layout="centered")
st.title("🧠 Asistente de Mandarín")
st.markdown("Traduce frases o resuelve dudas gramaticales usando solo el vocabulario del curso.")

modelo = st.radio("Modelo:", ["DeepSeek", "OpenAI"])
usar_deepseek = modelo == "DeepSeek"

pregunta = st.text_input("Pregunta:", placeholder="¿Cómo se dice 'me gusta estudiar chino'?")

if st.button("Enviar") and pregunta:
    with st.spinner("Generando respuesta..."):
        respuesta = responder_pregunta(pregunta, usar_deepseek=usar_deepseek)
        st.success("Respuesta del asistente:")
        st.markdown(respuesta)


