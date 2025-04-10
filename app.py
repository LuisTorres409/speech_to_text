import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import json
from typing import List, Dict, Any
import time

# Configuração da página
st.set_page_config(page_title="Transcritor de Áudio", layout="centered")
st.title("🎧 Transcritor de Áudio")
st.markdown("""
Faça upload de um arquivo de áudio (.mp3, .wav) e receba a transcrição completa.

> ⚠️ Áudios muito longos podem demorar para serem processados.
> Para melhores resultados com músicas, use o modelo **medium** ou **large-v2**.
""")

# Seletor de modelo com descrições
model_size = st.selectbox(
    "Escolha o modelo Whisper:",
    options=["tiny", "base", "small", "medium", "large-v2"],
    index=2,  # small como padrão
    help="Modelos maiores oferecem melhor precisão mas são mais lentos"
)

# Upload do áudio com mais opções
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio", 
    type=["mp3", "wav", "ogg"],
    help="Formatos suportados: MP3, WAV, OGG"
)

if uploaded_file is not None:
    # Mostra informações do arquivo
    file_details = {
        "Nome": uploaded_file.name,
        "Tipo": uploaded_file.type,
        "Tamanho": f"{uploaded_file.size / (1024*1024):.2f} MB"
    }
    st.json(file_details)
    
    # Salva arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    st.audio(uploaded_file, format='audio/mp3')
    
    # Barra de progresso e status
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Preparando para transcrever...")
    
    try:
        # Inicia temporizador
        start_time = time.time()
        
        # Atualiza status
        status_text.text("Carregando modelo Whisper... (isso pode demorar na primeira execução)")
        progress_bar.progress(10)
        
        # Carrega modelo Whisper com configuração otimizada para CPU
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="./whisper_models",
            cpu_threads=4  # Otimização para Streamlit Cloud
        )
        
        progress_bar.progress(30)
        status_text.text("Modelo carregado. Iniciando transcrição...")
        
        # Configurações para música e áudios longos
        segments_generator, info = model.transcribe(
            tmp_path,
            vad_filter=False,  # Importante para músicas
            beam_size=5,
            chunk_length=20,  # Chunks menores para melhor feedback
            no_speech_threshold=0.6,
            temperature=(0.0, 0.2, 0.4, 0.6),
            condition_on_previous_text=False
        )
        
        # Processa segmentos com feedback de progresso
        full_transcription = []
        segment_details = []
        total_duration = info.duration if info.duration > 0 else 1
        
        status_text.text(f"Transcrevendo áudio de {total_duration:.1f} segundos...")
        
        for i, segment in enumerate(segments_generator):
            # Atualiza progresso baseado no tempo do segmento
            progress = min(70 + (segment.end / total_duration * 30), 95)
            progress_bar.progress(int(progress))
            
            # Atualiza status
            status_text.text(
                f"Processando: {segment.end:.1f}/{total_duration:.1f}s "
                f"(Velocidade: {segment.end/(time.time()-start_time):.1f}x)"
            )
            
            full_transcription.append(segment.text.strip())
            segment_details.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        # Concatena transcrição final
        formatted_transcription = "\n\n".join(full_transcription)
        
        # Finaliza progresso
        progress_bar.progress(100)
        status_text.text(f"Transcrição concluída em {time.time()-start_time:.1f} segundos!")
        st.success("✅ Transcrição finalizada com sucesso!")
        
        # Mostra estatísticas
        st.subheader("📊 Estatísticas")
        col1, col2, col3 = st.columns(3)
        col1.metric("Duração", f"{total_duration:.1f}s")
        col2.metric("Texto gerado", f"{len(formatted_transcription)} chars")
        col3.metric("Velocidade", f"{total_duration/(time.time()-start_time):.1f}x")
        
        # Exibe transcrição
        st.subheader("📝 Transcrição Completa")
        st.text_area("Texto transcrito:", formatted_transcription, height=300, label_visibility="collapsed")
        
        # Opções de download
        st.subheader("⬇️ Baixar Transcrição")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📄 Texto (.txt)", 
                formatted_transcription.encode('utf-8'), 
                file_name="transcricao.txt",
                help="Baixar transcrição em formato de texto simples"
            )
        
        with col2:
            st.download_button(
                "🗂️ JSON (.json)", 
                json.dumps({
                    "metadata": {
                        "model": model_size,
                        "language": info.language,
                        "duration": info.duration
                    },
                    "segments": segment_details,
                    "full_text": formatted_transcription
                }, ensure_ascii=False, indent=2), 
                file_name="transcricao.json",
                help="Baixar transcrição com metadados e segmentação temporal"
            )
    
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("")
        st.error(f"❌ Ocorreu um erro durante a transcrição: {str(e)}")
    
    finally:
        # Limpeza do arquivo temporário
        try:
            os.remove(tmp_path)
        except:
            pass
else:
    st.info("ℹ️ Faça upload de um arquivo de áudio para começar a transcrição.")
    st.markdown("""
    ### Dicas para melhores resultados:
    - Use áudios com boa qualidade e mínimo de ruído de fundo
    - Para músicas, prefira o modelo **medium** ou **large-v2**
    - Arquivos muito longos (>30 min) podem atingir limites de tempo no Streamlit Cloud
    """)