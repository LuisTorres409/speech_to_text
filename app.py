import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import json
import time
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
    # Barra de progresso e status
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Iniciando processamento...")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format='audio/mp3')
    
    try:
        # Fase 1: Carregamento do modelo (10-30%)
        status_text.text("Carregando modelo Whisper...")
        progress_bar.progress(10)
        
        start_time = time.time()
        
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="./models"
        )
        
        progress_bar.progress(30)
        status_text.text("Modelo carregado. Iniciando transcrição...")

        # Configurações IDÊNTICAS ao código original
        segments, info = model.transcribe(
            tmp_path,
            vad_filter=False,  # Mantido igual
            beam_size=5,       # Mantido igual
            chunk_length=30,   # Mantido igual
            no_speech_threshold=0.6,  # Mantido igual
            temperature=(0.0, 0.2, 0.4, 0.6),  # Mantido igual
            condition_on_previous_text=False,  # Mantido igual
            word_timestamps=False  # Mantido igual
        )

        # Processamento IDÊNTICO ao código original
        full_transcription = []
        segment_details = []
        
        for segment in segments:
            # Atualização do progresso (30-90%)
            progress = min(30 + (segment.end / info.duration * 60), 90)
            progress_bar.progress(int(progress))
            
            status_text.text(f"Transcrevendo... {segment.end:.1f}/{info.duration:.1f}s")
            
            # Lógica original de processamento
            full_transcription.append(segment.text.strip())
            segment_details.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        # Concatenação IDÊNTICA
        formatted_transcription = "\n\n".join(full_transcription)
        processing_time = time.time() - start_time
        
        # Finalização (90-100%)
        progress_bar.progress(100)
        status_text.text("Finalizando...")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        # Exibição ORIGINAL + métricas
        st.success(f"Transcrição concluída em {processing_time:.1f} segundos!")
        
        # Métricas adicionais (sem alterar o layout original)
        st.subheader("📊 Estatísticas")
        col1, col2 = st.columns(2)
        col1.metric("Duração do Áudio", f"{info.duration:.1f}s")
        col2.metric("Tempo de Processamento", f"{processing_time:.1f}s")
        
        # Mantido IDÊNTICO ao original
        st.subheader("📝 Transcrição Completa")
        st.text_area("Texto transcrito:", formatted_transcription, height=300)

        # Mantido IDÊNTICO ao original
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
        progress_bar.progress(0)
        status_text.text("")
        st.error(f"Erro durante a transcrição: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
else:
    st.info("Faça upload de um arquivo de áudio para começar a transcrição.")