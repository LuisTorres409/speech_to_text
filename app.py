import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import json

# Token de autenticação do Hugging Face
os.environ["HF_TOKEN"] = st.secrets["HUGGINGFACE_HUB_TOKEN"]

# Configuração da página
st.set_page_config(page_title="Transcritor de Áudio", layout="centered")
st.title("🎧 Transcritor de Áudio")
st.markdown("Faça upload de um arquivo `.mp3` e receba a transcrição formatada.")

# Upload do áudio
uploaded_file = st.file_uploader("Escolha um arquivo de áudio (.mp3)", type=["mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format='audio/mp3')
    st.info("Transcrevendo o áudio, por favor aguarde...")

    # Carrega modelo Whisper com configuração otimizada para CPU e áudios longos
    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, _ = model.transcribe(
        tmp_path,
        vad_filter=True,         # Detecta voz para dividir o áudio
        beam_size=5,             # Aumenta a qualidade
        chunk_length=30          # Divide em pedaços de 30s (mais confiável)
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
