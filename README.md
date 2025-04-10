# 🎧 Transcritor de Áudio 

Este projeto é um aplicativo web simples feito com **Streamlit** que permite transcrever arquivos de áudio `.mp3` utilizando o modelo **Whisper**, via a versão otimizada **[faster-whisper](https://github.com/guillaumekln/faster-whisper)**.

---

## 🚀 Funcionalidades

- Upload de arquivos `.mp3`
- Transcrição automática de áudio com Whisper (via CPU)
- Exibição do texto formatado com quebras de linha
- Opções para download da transcrição nos formatos:
  - `.txt`
  - `.json`

---

## 🧠 Tecnologias utilizadas

- [Streamlit](https://streamlit.io/)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [PyTorch](https://pytorch.org/)
- [CTranslate2](https://opennmt.net/CTranslate2/)

---

## 🛠️ Como rodar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/transcricao-whisper-app.git
cd transcricao-whisper-app
