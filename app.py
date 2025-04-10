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
Faça upload de um arquivo de áudio e receba a transcrição completa.

> ⚠️ Áudios muito longos podem demorar para serem processados.
""")

# Inicialização do session state
if 'transcription_done' not in st.session_state:
    st.session_state.transcription_done = False
if 'transcription_data' not in st.session_state:
    st.session_state.transcription_data = None

# Seletor de modelo
model_size = st.selectbox(
    "Escolha o modelo Whisper:",
    options=["tiny", "base", "small"],
    index=2,
    key="model_select"  # Chave única para o widget
)

# Upload do áudio
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio (.mp3, .wav)", 
    type=["mp3", "wav"],
    key="file_uploader"
)

def reset_transcription():
    """Reseta o estado da transcrição"""
    st.session_state.transcription_done = False
    st.session_state.transcription_data = None

def run_transcription():
    """Executa o processo de transcrição"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        # Barra de progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Iniciando processamento...")
        
        # Fase 1: Carregamento do modelo
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

        # Configurações de transcrição
        segments, info = model.transcribe(
            tmp_path,
            vad_filter=False,
            beam_size=5,
            chunk_length=30,
            no_speech_threshold=0.6,
            temperature=(0.0, 0.2, 0.4, 0.6),
            condition_on_previous_text=False,
            word_timestamps=False
        )

        # Processamento dos segmentos
        full_transcription = []
        segment_details = []
        
        for segment in segments:
            progress = min(30 + (segment.end / info.duration * 60), 90)
            progress_bar.progress(int(progress))
            
            status_text.text(f"Transcrevendo... {segment.end:.1f}/{info.duration:.1f}s")
            
            full_transcription.append(segment.text.strip())
            segment_details.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        # Finalização
        formatted_transcription = "\n\n".join(full_transcription)
        processing_time = time.time() - start_time
        
        progress_bar.progress(100)
        status_text.text("Finalizando...")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        # Armazena os resultados no session state
        st.session_state.transcription_data = {
            "transcription": formatted_transcription,
            "details": segment_details,
            "info": info,
            "processing_time": processing_time
        }
        st.session_state.transcription_done = True
        
    except Exception as e:
        st.error(f"Erro durante a transcrição: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

# Botão de transcrição (só aparece se houver arquivo)
if uploaded_file is not None and not st.session_state.transcription_done:
    st.audio(uploaded_file, format='audio/mp3')
    
    # Botão para iniciar a transcrição
    if st.button("🔊 Iniciar Transcrição", type="primary"):
        run_transcription()
        st.rerun()

# Exibição dos resultados (se a transcrição foi concluída)
if st.session_state.transcription_done and st.session_state.transcription_data:
    data = st.session_state.transcription_data
    
    st.success(f"✅ Transcrição concluída em {data['processing_time']:.1f} segundos!")
    
    # Seção de estatísticas aprimoradas
    st.subheader("📊 Estatísticas Detalhadas")
    cols = st.columns(4)
    cols[0].metric("Duração", f"{data['info'].duration:.1f}s")
    cols[1].metric("Tempo Process.", f"{data['processing_time']:.1f}s")
    cols[2].metric("Idioma", data['info'].language)
    cols[3].metric("Caracteres", len(data['transcription']))
    
    # Detalhes adicionais
    with st.expander("🔍 Mais informações"):
        st.json({
            "Probabilidade Idioma": data['info'].language_probability,
            "Modelo Utilizado": model_size,
            "Nº Segmentos": len(data['details'])
        })
    
    # Transcrição
    st.subheader("📝 Transcrição Completa")
    st.text_area("Resultado:", data['transcription'], height=300, key="transcription_area")
    
    # Opções de download
    st.subheader("⬇️ Baixar Transcrição")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📄 Texto (.txt)", 
            data['transcription'].encode('utf-8'), 
            file_name="transcricao.txt"
        )
    
    with col2:
        st.download_button(
            "🗂️ JSON (.json)", 
            json.dumps({
                "metadata": {
                    "language": data['info'].language,
                    "duration": data['info'].duration,
                    "model": model_size
                },
                "segments": data['details'],
                "full_text": data['transcription']
            }, ensure_ascii=False, indent=2), 
            file_name="transcricao.json"
        )
    
    # Botão para nova transcrição
    if st.button("🔄 Realizar Nova Transcrição"):
        reset_transcription()
        st.rerun()

elif uploaded_file is None:
    st.info("ℹ️ Faça upload de um arquivo de áudio para começar.")

# Nota sobre mudanças de configuração
if uploaded_file is not None and not st.session_state.transcription_done:
    st.warning("⚠️ Altere as configurações antes de iniciar a transcrição. Mudanças após iniciar requerem nova transcrição.")