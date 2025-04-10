import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import json
from pathlib import Path

import os
os.environ["HF_TOKEN"] = st.secrets["HUGGINGFACE_HUB_TOKEN"]

# Título e descrição do app
st.set_page_config(page_title="Transcritor de Áudio", layout="centered")
st.title("🎧 Transcritor de Áudio")
st.markdown("Faça upload de um arquivo `.mp3` e receba a transcrição formatada.")

# Upload do arquivo de áudio
uploaded_file = st.file_uploader("Escolha um arquivo de áudio (.mp3)", type=["mp3"])

if uploaded_file is not None:
    # Salva temporariamente o arquivo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format='audio/mp3')
    st.info("Transcrevendo o áudio, por favor aguarde...")

    # Carrega o modelo Whisper otimizado (faster-whisper) com uso forçado da CPU
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(tmp_path)

    # Junta os textos com espaçamento entre frases
    formatted_transcription = "\n\n".join([seg.text.strip() for seg in segments if seg.text.strip()])

    # Exibe o texto formatado
    st.subheader("📝 Transcrição")
    st.text_area("Texto transcrito:", formatted_transcription, height=300)

    # Downloads
    st.subheader("⬇️ Baixar Transcrição")

    txt_bytes = formatted_transcription.encode('utf-8')
    st.download_button("📄 Baixar como .txt", data=txt_bytes, file_name="transcricao.txt")

    json_str = json.dumps({"transcricao": formatted_transcription}, ensure_ascii=False, indent=2)
    st.download_button("🗂️ Baixar como .json", data=json_str, file_name="transcricao.json")

    # Limpa o arquivo temporário
    os.remove(tmp_path)
else:
    st.info("Faça upload de um arquivo MP3 para começar a transcrição.")
