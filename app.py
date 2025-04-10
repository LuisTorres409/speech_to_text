import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import json

# Token do Hugging Face via secrets
os.environ["HF_TOKEN"] = st.secrets["HUGGINGFACE_HUB_TOKEN"]

# Configuração da página
st.set_page_config(page_title="Transcritor de Áudio", layout="centered")
st.title("🎧 Transcritor de Áudio")
st.markdown("""
Faça upload de um arquivo `.mp3` e receba a transcrição completa.

> ⚠️ Áudios muito longos podem demorar para serem processados.
> Certifique-se de que o áudio tenha uma boa qualidade para melhor desempenho.
""")

# Seletor de modelo
model_size = st.selectbox(
    "Escolha o modelo Whisper:",
    options=["tiny", "base", "small", "medium"],
    index=1
)

# Upload do áudio
uploaded_file = st.file_uploader("Escolha um arquivo de áudio (.mp3)", type=["mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format='audio/mp3')
    st.info("Transcrevendo o áudio, por favor aguarde... Isso pode levar alguns minutos.")

    # Carrega modelo Whisper com configuração otimizada para CPU e áudios longos
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, _ = model.transcribe(
        tmp_path,
        vad_filter=True,
        beam_size=5,
        chunk_length=60,     # aumenta o tempo de cada chunk para processar áudios maiores
        no_speech_threshold=0.2,  # mais sensível a pausas curtas
        temperature=0.0
    )

    # Concatena transcrição final
    formatted_transcription = "\n\n".join(
        [seg.text.strip() for seg in segments if seg.text.strip()]
    )

    # Exibe e permite baixar
    st.subheader("📝 Transcrição")
    st.text_area("Texto transcrito:", formatted_transcription, height=300)

    st.subheader("⬇️ Baixar Transcrição")
    st.download_button("📄 Baixar como .txt", formatted_transcription.encode('utf-8'), "transcricao.txt")
    st.download_button("🗂️ Baixar como .json", json.dumps({"transcricao": formatted_transcription}, ensure_ascii=False, indent=2), "transcricao.json")

    os.remove(tmp_path)
else:
    st.info("Faça upload de um arquivo MP3 para começar a transcrição.")
