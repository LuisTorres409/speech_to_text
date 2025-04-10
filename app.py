import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import json
import time

# Configuração da página
st.set_page_config(page_title="Transcritor de Áudio Premium", layout="centered")
st.title("🎧 Transcritor de Áudio Profissional")
st.markdown("""
Faça upload de arquivos de áudio para obter transcrições de alta qualidade.

> 🔍 Para músicas ou áudios complexos, recomendo usar os modelos **medium** ou **large-v2**.
""")

# Seletor de modelo
model_size = st.selectbox(
    "Selecione o modelo de transcrição:",
    options=["tiny", "base", "small", "medium", "large-v2"],
    index=2,
    help="Modelos maiores = melhor qualidade, porém mais lentos"
)

# Upload do áudio
uploaded_file = st.file_uploader("Carregue seu arquivo de áudio", type=["mp3", "wav", "ogg"])

if uploaded_file is not None:
    # Informações do arquivo
    file_size = uploaded_file.size / (1024 * 1024)  # MB
    st.success(f"✅ Arquivo carregado: {uploaded_file.name} ({file_size:.2f} MB)")
    
    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Preparando ambiente...")
    
    # Cria arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    st.audio(uploaded_file, format='audio/mp3')
    
    try:
        # Fase 1: Carregamento do modelo
        status_text.text("Carregando modelo Whisper...")
        progress_bar.progress(15)
        
        start_time = time.time()
        
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="./models"
        )
        
        progress_bar.progress(30)
        status_text.text("Modelo pronto. Iniciando transcrição...")
        
        # Configurações ORIGINAIS que garantiam boa qualidade
        segments, info = model.transcribe(
            tmp_path,
            vad_filter=True,  # Mantido como True para melhor qualidade
            beam_size=5,
            chunk_length=30,  # Tamanho original
            no_speech_threshold=0.4,  # Valor original
            temperature=0.0,  # Configuração original
            condition_on_previous_text=True,  # Original
            word_timestamps=False
        )
        
        # Processamento com feedback
        full_transcription = []
        segment_details = []
        total_duration = info.duration
        
        for segment in segments:
            # Atualiza progresso sem interferir no processamento
            progress = min(30 + (segment.end / total_duration * 65), 95)
            progress_bar.progress(int(progress))
            
            status_text.text(f"Processando: {segment.end:.1f}s de {total_duration:.1f}s")
            
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
        
        st.balloons()
        st.success("Transcrição concluída com sucesso!")
        
        # Métricas (sem alterar o layout principal)
        st.subheader("📊 Estatísticas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Duração do Áudio", f"{total_duration:.1f} segundos")
            st.metric("Tempo de Processamento", f"{processing_time:.1f} segundos")
        
        with col2:
            st.metric("Idioma Detectado", info.language)
            st.metric("Caracteres Gerados", len(formatted_transcription))
        
        # Transcrição (layout original)
        st.subheader("📝 Transcrição Completa")
        st.text_area("Resultado:", formatted_transcription, height=300, label_visibility="collapsed")
        
        # Downloads (layout original)
        st.subheader("⬇️ Opções de Download")
        st.download_button(
            "Baixar como TXT", 
            formatted_transcription.encode('utf-8'), 
            file_name="transcricao.txt"
        )
        st.download_button(
            "Baixar como JSON", 
            json.dumps({
                "metadata": {
                    "model": model_size,
                    "language": info.language,
                    "duration": total_duration
                },
                "segments": segment_details,
                "full_text": formatted_transcription
            }, ensure_ascii=False, indent=2), 
            file_name="transcricao.json"
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
    st.info("Por favor, carregue um arquivo de áudio para iniciar a transcrição.")