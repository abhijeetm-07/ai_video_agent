# 🎥 AI Video & Meeting Assistant

An end-to-end intelligent video and meeting processing agent that extracts audio from YouTube URLs or local media files, performs high-fidelity speech-to-text transcription (supporting both English and Hinglish), extracts structured meeting insights using Mistral AI, and indexes the discussion into a local vector database for interactive RAG (Retrieval-Augmented Generation) Q&A.

---

## 🚀 Features

- **Multi-Source Audio Ingestion**: Process YouTube video URLs directly using `yt-dlp` or local audio/video media files (`.mp4`, `.mp3`, `.wav`, etc.).
- **Dual-Engine Speech-to-Text**:
  - **English**: Local transcription via OpenAI Whisper (`small`, `base`, etc.).
  - **Hinglish**: Intelligent chunking (25-second windows) and transcription/translation to English using Sarvam AI (`saaras:v2.5`).
- **Audio Processing & Preprocessing**: Automatic downsampling to 16 kHz mono WAV via `pydub` and FFmpeg with intelligent chunking for long recordings.
- **LLM-Powered Meeting Intelligence (Mistral AI)**:
  - **Auto Title Generation**: Short, descriptive executive titles.
  - **Map-Reduce Summarization**: Hierarchical chunk summarization followed by executive synthesis with clean bullet points.
  - **Action Items Extraction**: Identifies task descriptions, assigned owners, and deadlines.
  - **Key Decisions Extraction**: Captures finalized decisions, policies, and rationales.
  - **Unresolved Questions Extraction**: Spots open issues, unanswered topics, and follow-ups.
- **Retrieval-Augmented Generation (RAG) Interactive Chat**:
  - Transcripts are embedded using HuggingFace (`all-MiniLM-L6-v2`) and stored in ChromaDB.
  - Interactive CLI chat mode to query meeting transcripts with grounded, strictly attributed answers.

---

## 🏗️ Architecture & Pipeline

```text
[ YouTube URL / Local Media File ]
               │
               ▼
   [ Audio Ingestion & Conversion ]
    (yt-dlp / pydub / FFmpeg -> 16kHz Mono WAV Chunks)
               │
               ▼
   [ Speech-to-Text Transcription ]
    ├── English  ──> OpenAI Whisper (Local)
    └── Hinglish ──> Sarvam AI (Translate to English)
               │
               ▼
   [ Full Meeting Transcript ]
         │                   │
         ▼                   ▼
[ LLM Analysis (Mistral) ]  [ Vector Indexing (ChromaDB) ]
  • Title Generation          • HuggingFace Embeddings
  • Structured Summary        • Semantic Chunking
  • Action Items              • Similarity Retriever
  • Key Decisions                     │
  • Open Questions                    ▼
                        [ Interactive RAG Q&A Chat ]
```

---

## 📁 Repository Structure

```text
ai_video_agent/
├── core/
│   ├── extractor.py        # Action items, key decisions, and questions extraction
│   ├── rag_engine.py       # LangChain LCEL RAG pipeline & strict Q&A prompt
│   ├── summarize.py        # Map-reduce meeting summarizer & title generator
│   ├── transcriber.py      # Whisper & Sarvam AI STT orchestrator
│   └── vector_store.py     # ChromaDB & HuggingFace embeddings setup
├── utils/
│   └── audio_processor.py  # YouTube downloader, WAV conversion, & audio chunking
├── downloads/              # Local cache directory for downloaded audio
├── vector_db/              # Persistent Chroma vector store directory (git-ignored)
├── main.py                 # Pipeline execution & interactive CLI
├── requirement.txt         # Project dependencies
└── .env                    # Environment variables and API keys
```

---

## 🛠️ Prerequisites & Setup

### 1. System Dependencies (FFmpeg)
Ensure FFmpeg is installed and accessible in your system `PATH`:

- **macOS** (via Homebrew):
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu/Debian**:
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```
- **Windows**:
  Download and install from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your System PATH.

---

### 2. Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/abhijeetm-07/ai_video_agent.git
cd ai_video_agent

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirement.txt
```

---

### 3. API Keys Configuration

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY="your-mistral-api-key"
SARVAM_API_KEY="your-sarvam-api-key"
WHISPER_MODEL="small"
SARVAM_STT_MODEL="saaras:v2.5"
```

- **`MISTRAL_API_KEY`**: Obtain from [Mistral AI Console](https://console.mistral.ai/).
- **`SARVAM_API_KEY`**: Obtain from [Sarvam AI Dashboard](https://www.sarvam.ai/) (required for Hinglish transcription).
- **`WHISPER_MODEL`**: Choose between `tiny`, `base`, `small`, `medium`, or `large` based on hardware availability.

---

## 💻 Usage

Run the main CLI application:

```bash
python main.py
```

### Example Interaction:

1. **Input Source**: Enter a YouTube URL (e.g. `https://www.youtube.com/watch?v=...`) or a path to a local media file (e.g. `sample_meeting.mp4`).
2. **Language Selection**: Enter `english` or `hinglish`.
3. **Pipeline Output**:
   - Audio is converted and transcribed.
   - Outputs meeting Title, Structured Executive Summary, Action Items with owners and deadlines, Key Decisions, and Open Questions.
4. **Chat with the Meeting**:
   - An interactive prompt lets you ask any question about the discussion.
   - Type `exit` or `quit` to end the session.

---

## 🔒 License

This project is licensed under the MIT License.
