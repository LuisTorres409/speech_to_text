import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import json
from typing import List, Dict, Any

# Token do Hugging Face via secrets
os.environ["HF_TOKEN"] = st.secrets.get("HUGGINGFACE_HUB_TOKEN", "")

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
    options=["tiny", "base", "small", "medium", "large-v2"],
    index=2  # small como padrão para melhor equilíbrio qualidade/performance
)

# Upload do áudio
uploaded_file = st.file_uploader("Escolha um arquivo de áudio (.mp3, .wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format='audio/mp3')
    
    with st.spinner("Transcrevendo o áudio, por favor aguarde... Isso pode levar alguns minutos."):
        try:
            # Carrega modelo Whisper com configuração otimizada
            model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root="./models"  # Cache local para modelos
            )

            # Configurações melhoradas para transcrição musical
            segments, info = model.transcribe(
                tmp_path,
                vad_filter=False,  # Desativa VAD para músicas
                beam_size=5,
                chunk_length=30,  # Chunks menores para melhor processamento
                no_speech_threshold=0.6,  # Limiar mais alto para não descartar partes
                temperature=(0.0, 0.2, 0.4, 0.6),  # Variação de temperatura
                condition_on_previous_text=False,  # Melhor para músicas
                word_timestamps=False  # Desativa para melhor performance
            )

            # Processa todos os segmentos
            full_transcription = []
            segment_details = []
            
            for segment in segments:
                full_transcription.append(segment.text.strip())
                segment_details.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })

            # Concatena transcrição final
            formatted_transcription = "\n\n".join(full_transcription)

            # Exibe informações sobre o áudio
            st.success(f"Transcrição concluída! Detalhes do áudio:")
            st.json({
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "transcription_length": len(formatted_transcription)
            })

            # Exibe e permite baixar
            st.subheader("📝 Transcrição Completa")
            st.text_area("Texto transcrito:", formatted_transcription, height=300)

            st.subheader("⬇️ Baixar Transcrição")
            st.download_button(
                "📄 Baixar como .txt", 
                formatted_transcription.encode('utf-8'), 
                "transcricao.txt"
            )
            st.download_button(
                "🗂️ Baixar como .json", 
                json.dumps({
                    "metadata": {
                        "language": info.language,
                        "duration": info.duration
                    },
                    "segments": segment_details,
                    "full_text": formatted_transcription
                }, ensure_ascii=False, indent=2), 
                "transcricao.json"
            )

        except Exception as e:
            st.error(f"Erro durante a transcrição: {str(e)}")
        finally:
            os.remove(tmp_path)
else:
    st.info("Faça upload de um arquivo de áudio para começar a transcrição.")