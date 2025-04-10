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
Faça upload de um arquivo de áudio (.mp3, .wav) e receba a transcrição completa.

> ⚠️ Áudios muito longos podem demorar para serem processados.
> Para melhores resultados com músicas, use o modelo **medium** ou **large-v2**.
""")

# Seletor de modelo
model_size = st.selectbox(
    "Escolha o modelo Whisper:",
    options=["tiny", "base", "small", "medium", "large-v2"],
    index=2,
    help="Modelos maiores oferecem melhor precisão mas são mais lentos"
)

# Upload do áudio
uploaded_file = st.file_uploader("Escolha um arquivo de áudio (.mp3, .wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    # Mostra informações básicas do arquivo
    file_size = uploaded_file.size / (1024 * 1024)  # Convertendo para MB
    st.info(f"📄 Arquivo carregado: {uploaded_file.name} ({file_size:.2f} MB)")
    
    # Barra de progresso e status
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Iniciando processamento...")
    
    # Cria arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    st.audio(uploaded_file, format='audio/mp3')
    
    with st.spinner("Preparando para transcrição..."):
        try:
            start_time = time.time()
            
            # Fase 1: Carregamento do modelo
            status_text.text("Carregando modelo Whisper...")
            progress_bar.progress(10)
            
            model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root="./models",
                cpu_threads=4
            )
            
            progress_bar.progress(30)
            status_text.text("Modelo carregado. Iniciando transcrição...")
            
            # Fase 2: Transcrição
            segments, info = model.transcribe(
                tmp_path,
                vad_filter=False,
                beam_size=5,
                chunk_length=20,
                no_speech_threshold=0.6,
                temperature=(0.0, 0.2, 0.4, 0.6),
                condition_on_previous_text=False
            )
            
            # Processa segmentos com atualização de progresso
            full_transcription = []
            segment_details = []
            total_duration = info.duration if info.duration > 0 else 1
            
            for segment in segments:
                # Atualiza progresso (30-95% durante a transcrição)
                progress = min(30 + (segment.end / total_duration * 65), 95)
                progress_bar.progress(int(progress))
                
                status_text.text(
                    f"Transcrevendo: {segment.end:.1f}/{total_duration:.1f}s "
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
            processing_time = time.time() - start_time
            
            # Fase 3: Finalização
            progress_bar.progress(100)
            status_text.text(f"Transcrição concluída em {processing_time:.1f} segundos!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Transcrição finalizada com sucesso!")
            
            # Seção de métricas aprimoradas
            st.subheader("📊 Métricas de Desempenho")
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Duração do Áudio", f"{total_duration:.1f} segundos")
            with cols[1]:
                st.metric("Tempo de Processamento", f"{processing_time:.1f} segundos")
            with cols[2]:
                speed_factor = total_duration / processing_time if processing_time > 0 else 0
                st.metric("Velocidade", f"{speed_factor:.1f}x tempo real")
            with cols[3]:
                st.metric("Texto Gerado", f"{len(formatted_transcription)} caracteres")
            
            # Métricas adicionais
            with st.expander("🔍 Estatísticas Detalhadas"):
                cols2 = st.columns(3)
                with cols2[0]:
                    st.metric("Taxa de Caracteres/seg", 
                             f"{(len(formatted_transcription)/total_duration):.1f}" if total_duration > 0 else "N/A")
                with cols2[1]:
                    st.metric("Segmentos Gerados", len(segment_details))
                with cols2[2]:
                    avg_seg_length = total_duration/len(segment_details) if segment_details else 0
                    st.metric("Duração Média/segmento", f"{avg_seg_length:.1f}s")
            
            # Exibe transcrição
            st.subheader("📝 Transcrição Completa")
            st.text_area("Resultado:", formatted_transcription, height=300, label_visibility="collapsed")
            
            # Opções de download
            st.subheader("⬇️ Baixar Transcrição")
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📄 Texto (.txt)", 
                    formatted_transcription.encode('utf-8'), 
                    file_name="transcricao.txt"
                )
            
            with col2:
                st.download_button(
                    "🗂️ JSON Completo (.json)", 
                    json.dumps({
                        "metadata": {
                            "model": model_size,
                            "language": info.language,
                            "duration": info.duration,
                            "processing_time": processing_time
                        },
                        "segments": segment_details,
                        "full_text": formatted_transcription
                    }, ensure_ascii=False, indent=2), 
                    file_name="transcricao.json"
                )
            
            # Adiciona gráfico de distribuição dos segmentos (opcional)
            if len(segment_details) > 1:
                with st.expander("📈 Distribuição Temporal dos Segmentos"):
                    import pandas as pd
                    import matplotlib.pyplot as plt
                    
                    df = pd.DataFrame(segment_details)
                    df['duration'] = df['end'] - df['start']
                    
                    fig, ax = plt.subplots()
                    ax.bar(df['start'], df['duration'], width=df['duration'], align='edge')
                    ax.set_xlabel('Tempo (s)')
                    ax.set_ylabel('Duração (s)')
                    ax.set_title('Distribuição dos Segmentos de Transcrição')
                    st.pyplot(fig)
        
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("")
            st.error(f"❌ Erro durante a transcrição: {str(e)}")
            st.exception(e)
        
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
else:
    st.info("ℹ️ Faça upload de um arquivo de áudio para começar a transcrição.")
    st.markdown("""
    ### Dicas para melhores resultados:
    - Use áudios com boa qualidade (evite ruídos de fundo)
    - Modelos maiores = melhor precisão, mas mais lentos
    - Arquivos longos (>30min) podem atingir limites de tempo
    """)