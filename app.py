import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import json
import time
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Transcritor de Áudio", layout="centered")
st.title("🎧 Transcritor de Áudio")
st.markdown("""
Faça upload de um arquivo de áudio ou grave diretamente e receba a transcrição completa.

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
    options=["tiny", "base", "small",'medium'],
    index=2,
    key="model_select"
)

# Opção de entrada de áudio
input_method = st.radio(
    "Selecione o método de entrada:",
    options=["Upload de arquivo", "Gravar áudio"],
    index=0,
    key="input_method"
)

audio_data = None

if input_method == "Upload de arquivo":
    # Upload do áudio
    uploaded_file = st.file_uploader(
        "Escolha um arquivo de áudio (.mp3, .wav)", 
        type=["mp3", "wav"],
        key="file_uploader"
    )
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')
        audio_data = uploaded_file.getvalue()
else:
    # Gravador de áudio nativo do Streamlit
    recorded_audio = st.audio_input("Clique para gravar um áudio:")
    if recorded_audio is not None:
        #st.audio(recorded_audio, format="audio/wav")
        audio_data = recorded_audio.getvalue()

def reset_transcription():
    """Reseta o estado da transcrição"""
    st.session_state.transcription_done = False
    st.session_state.transcription_data = None

def run_transcription(audio_bytes):
    """Executa o processo de transcrição"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        # Barra de progresso e status
        progress_bar = st.progress(0, text="0% concluído")
        status_text = st.empty()
        status_text.text("🔧 Preparando ambiente...")
        
        start_time = time.time()
        
        # Fase 1: Carregamento do modelo
        status_text.text("⏳ Carregando modelo...")
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="./models"
        )
        
        # Fase 2: Início da transcrição
        status_text.text("🎙️ Iniciando transcrição (0%)...")
        progress_bar.progress(0.0, text="0% concluído")
        
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

        # Fase 3: Processamento dos segmentos (0%-90%)
        full_transcription = []
        segment_details = []
        total_segments = sum(1 for _ in segments)
        segments = model.transcribe(
            tmp_path,
            vad_filter=False,
            beam_size=5,
            chunk_length=30,
            no_speech_threshold=0.6,
            temperature=(0.0, 0.2, 0.4, 0.6),
            condition_on_previous_text=False,
            word_timestamps=False
        )[0]
        
        processed_segments = 0
        
        for segment in segments:
            processed_segments += 1
            progress = (processed_segments / total_segments * 0.9)
            current_percent = min(progress, 0.9)
            
            progress_bar.progress(current_percent, text=f"{int(current_percent * 100)}% concluído")
            status_text.text(f"📝 Transcrevendo ({int(current_percent * 100)}%)...")
            
            full_transcription.append(segment.text.strip())
            segment_details.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        # Fase 4: Finalização
        status_text.text("✨ Finalizando (100%)...")
        progress_bar.progress(1.0, text="100% concluído")
        time.sleep(0.5)
        
        # Armazena resultados
        formatted_transcription = "\n\n".join(full_transcription)
        processing_time = time.time() - start_time
        
        st.session_state.transcription_data = {
            "transcription": formatted_transcription,
            "details": segment_details,
            "info": info,
            "processing_time": processing_time
        }
        st.session_state.transcription_done = True
        
    except Exception as e:
        st.error(f"❌ Erro durante a transcrição: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
        progress_bar.empty()
        status_text.empty()

# Interface principal
if audio_data is not None and not st.session_state.transcription_done:
    if st.button("🔊 Iniciar Transcrição", type="primary"):
        run_transcription(audio_data)
        st.rerun()

elif st.session_state.transcription_done and st.session_state.transcription_data:
    data = st.session_state.transcription_data
    
    st.success(f"✅ Transcrição concluída em {data['processing_time']:.1f} segundos!")
    
    # Estatísticas
    st.subheader("📊 Estatísticas Detalhadas")
    cols = st.columns(4)
    cols[0].metric("Duração", f"{data['info'].duration:.1f}s")
    cols[1].metric("Tempo Process.", f"{data['processing_time']:.1f}s")
    cols[2].metric("Idioma", data['info'].language)
    cols[3].metric("Caracteres", len(data['transcription']))
    
    with st.expander("🔍 Mais informações"):
        st.json({
            "Probabilidade Idioma": data['info'].language_probability,
            "Modelo Utilizado": model_size,
            "Nº Segmentos": len(data['details'])
        })
    
    # Transcrição completa
    st.subheader("📝 Transcrição Completa")
    st.text_area("Resultado:", data['transcription'], height=300, key="transcription_area")
    
    # Nova seção: Detalhes dos Segmentos
    st.subheader("⏱️ Detalhes dos Segmentos")
    # Converter os detalhes para um DataFrame para exibição
    segments_df = pd.DataFrame(data['details'])
    # Renomear colunas para melhor legibilidade
    segments_df.columns = ["Início (s)", "Fim (s)", "Texto"]
    # Formatar os tempos com 2 casas decimais
    segments_df["Início (s)"] = segments_df["Início (s)"].map("{:.2f}".format)
    segments_df["Fim (s)"] = segments_df["Fim (s)"].map("{:.2f}".format)
    # Exibir tabela interativa
    st.dataframe(
        segments_df,
        use_container_width=True,
        height=300  # Altura fixa para evitar expansão excessiva
    )
    
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
    
    if st.button("🔄 Realizar Nova Transcrição"):
        reset_transcription()
        st.rerun()

elif audio_data is None:
    st.info("ℹ️ Faça upload de um arquivo de áudio ou grave um áudio para começar.")

if audio_data is not None and not st.session_state.transcription_done:
    st.warning("⚠️ Altere as configurações antes de iniciar a transcrição. Mudanças após iniciar requerem nova transcrição.")