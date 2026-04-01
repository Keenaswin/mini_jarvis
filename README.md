# 🤖 Mini Jarvis — Local-First Personal AI Assistant

> A fully offline, privacy-first personal AI assistant with semantic memory,
> natural-language querying, task scheduling, and local LLM inference.
> **No cloud. No telemetry. Your data never leaves your machine.**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 Semantic Memory | Store notes/tasks/conversations. Retrieve via cosine similarity (FAISS) |
| 🔍 RAG Pipeline | Retrieved memories are injected into LLM context before every response |
| 💬 Conversation History | Full multi-turn chat with SQLite-persisted history |
| ⏰ Task Reminders | Natural-language scheduling with a background reminder thread |
| 🔒 Fully Offline | Embeddings + LLM + storage all run locally |
| 🗂️ Modular Code | Clean separation: memory / retrieval / LLM / conversation / tasks |

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────┐
│                        main.py (CLI)                       │
└──────┬───────────┬──────────────┬──────────────┬──────────┘
       │           │              │              │
  memory.py   retrieval.py    llm.py     conversation.py
  (SQLite +   (FAISS search   (llama-    (history + context
   FAISS)      + re-ranking)   cpp-py)    window mgmt)
       │                                        │
   tasks.py                                utils.py
  (scheduler +
   reminders)
```

### Memory Pipeline
```
User Input → chunk_text() → SentenceTransformer.encode()
           → FAISS.add()  + SQLite INSERT
```

### Query Pipeline
```
User Query → embed_query() → FAISS.search() → fetch SQLite rows
           → build_context() → LocalLLM.chat() → Response
```

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB+ |
| CPU | 4-core | 8-core |
| Python | 3.10 | 3.11+ |
| Disk | 3 GB | 10 GB+ |
| GPU | None (CPU ok) | Apple Silicon / NVIDIA |

---

## 🚀 Installation

### 1. Clone / create the project directory

```bash
git clone https://github.com/you/mini-jarvis   # or unzip the files
cd mini-jarvis
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Troubleshooting llama-cpp-python on macOS:**
> ```bash
> # Apple Silicon (Metal acceleration — highly recommended)
> CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
> ```
>
> **NVIDIA GPU (CUDA):**
> ```bash
> CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
> ```

### 4. Download a model (optional but recommended)

Mini Jarvis works in **fallback mode** (retrieval only) without a model.
For full AI responses, download any GGUF model and place it in `./models/`.

**Recommended models (small & fast):**

| Model | Size | Download |
|---|---|---|
| Phi-3-mini-4k-instruct Q4_K_M | 2.3 GB | [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) |
| Llama-3.2-3B-Instruct Q4_K_M | 2.0 GB | [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-GGUF) |
| Mistral-7B-Instruct Q4_K_M | 4.4 GB | [TheBloke on HF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) |

```bash
mkdir models
# Download your chosen model into models/
# Example with huggingface-cli:
pip install huggingface_hub
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
    Phi-3-mini-4k-instruct-q4.gguf \
    --local-dir models/
```

### 5. Configure the model path

Either edit `MODEL_PATH` in `main.py`, or set an environment variable:

```bash
# macOS / Linux
export JARVIS_MODEL="models/Phi-3-mini-4k-instruct-q4.gguf"

# Windows PowerShell
$env:JARVIS_MODEL="models\Phi-3-mini-4k-instruct-q4.gguf"
```

---

## 💬 Usage

### Interactive chat (default)

```bash
python main.py
# or
python main.py chat
```

Inside the REPL, everything is a chat message **unless** it starts with `:`.

---

## 🗂️ REPL Commands

| Command | Description |
|---|---|
| `:note <text>` | Save a freeform note to memory |
| `:task <text>` | Create a task/reminder (auto-parses due time) |
| `:search <query>` | Semantic search over all memories |
| `:list [type]` | List memories — filter by `note`, `task`, or `conversation` |
| `:tasks` | Show all pending tasks |
| `:history` | View recent conversation turns |
| `:stats` | Show memory and session statistics |
| `:new` | Start a fresh conversation session |
| `:clear` | Clear current session history |
| `:delete <id>` | Delete a memory by its ID |
| `:help` | Show all commands |
| `:quit` | Exit Mini Jarvis |

---

## 📋 CLI Commands (non-interactive)

```bash
# Add a note
python main.py add "Python's GIL is being removed in 3.13+"

# Add a task with a due time (parsed automatically)
python main.py add "Submit project report tomorrow at 5 PM" --type task

# Semantic search
python main.py search "python concurrency"

# Search only notes
python main.py search "meeting notes" --type note

# List all notes
python main.py list --type note

# Task management
python main.py tasks list           # pending tasks
python main.py tasks all            # all tasks
python main.py tasks done 3         # mark task #3 complete
python main.py tasks delete 5       # delete task #5

# Statistics
python main.py stats
```

---

## 🧪 Example Session

```
You: What's the difference between multiprocessing and multithreading in Python?

Jarvis: Python's GIL means threads share memory but can't run CPU-bound code
        truly in parallel. Use threading for I/O-bound tasks and multiprocessing
        for CPU-bound tasks. Multiprocessing spawns separate interpreter
        processes — no GIL — but has higher memory overhead and inter-process
        communication cost.

You: :note The GIL is being removed in Python 3.13 via PEP 703 (nogil build)

✓  Note saved (id: a3f8c12b901d)

You: :task Finish reading PEP 703 tomorrow at 9am

✓  Task memory saved (id: 9b2e1d3c4f7a) and scheduled (task #1)

You: What did I note about Python's GIL?

Jarvis: Based on your notes, you recorded that the GIL is being removed
        in Python 3.13 via PEP 703 (the nogil build). You also have a task
        scheduled to finish reading PEP 703 tomorrow at 9 AM.

You: :stats

  ── Memory statistics ──

    total                4
    notes                2
    tasks                1
    conversations        1
    faiss_vectors        4
    sessions             1
    pending tasks        1
```

---

## 📁 Project Structure

```
mini-jarvis/
├── main.py           Entry point — CLI + REPL
├── memory.py         SQLite + FAISS memory store
├── retrieval.py      Semantic search + context builder
├── llm.py            llama-cpp-python wrapper + fallback mode
├── conversation.py   Chat history + context window management
├── tasks.py          Task CRUD + background scheduler
├── utils.py          Chunking, timestamps, datetime parsing, colors
├── requirements.txt  Python dependencies
├── README.md         This file
│
├── models/           ← place your .gguf model here
│   └── phi3-mini.Q4_K_M.gguf
│
├── minijarvis.db     ← auto-created on first run (SQLite)
├── minijarvis.faiss  ← auto-created on first run (FAISS index)
└── minijarvis_meta.pkl ← auto-created on first run (FAISS metadata)
```

---

## ⚙️ Configuration

All settings can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `JARVIS_MODEL` | `models/phi3-mini.Q4_K_M.gguf` | Path to .gguf model |
| `JARVIS_DB` | `minijarvis.db` | SQLite database file |
| `JARVIS_CTX` | `4096` | LLM context window (tokens) |
| `JARVIS_THREADS` | `4` | CPU threads for inference |
| `JARVIS_GPU` | `0` | GPU layers (0 = CPU only) |

---

## 🔒 Privacy Architecture

- **No network calls** after initial model/embedding download
- All data stored in local files (`minijarvis.db`, `*.faiss`, `*.pkl`)
- No telemetry, analytics, or logging to external services
- Delete everything: `rm minijarvis.db minijarvis.faiss minijarvis_meta.pkl`

---

## 🔮 Future Improvements

1. **Better chunking** — sentence-aware splitting using `nltk.sent_tokenize`
2. **Hybrid search** — combine BM25 (keyword) + FAISS (semantic) with RRF fusion
3. **Incremental FAISS** — use `IndexIVFFlat` for sub-linear search on large indexes
4. **Memory decay** — automatically reduce importance of old, low-access memories
5. **Tagging UI** — richer tag-based filtering and organisation
6. **Voice input** — integrate `whisper.cpp` for offline speech-to-text
7. **GUI** — Tkinter or a lightweight web UI (FastAPI + HTMX)
8. **Multimodal** — store and retrieve images with CLIP embeddings
9. **Export** — JSON/Markdown export of all memories
10. **Sync** — optional encrypted sync between devices via a local network share

---

## 🐛 Troubleshooting

**`ModuleNotFoundError: No module named 'llama_cpp'`**
→ Install with: `pip install llama-cpp-python`

**`FileNotFoundError: model not found`**
→ Download a .gguf file and set `JARVIS_MODEL` to its path.
   Mini Jarvis still runs in fallback mode without a model.

**Slow embedding on first run**
→ `sentence-transformers` downloads `all-MiniLM-L6-v2` (~80 MB) on first use.
   Subsequent runs use the cached model.

**llama.cpp is very slow**
→ Increase `JARVIS_THREADS` to match your CPU core count.
   On Apple Silicon: rebuild llama-cpp-python with Metal support (see Install).

---

## 📄 License

MIT — use freely, modify freely, keep your data yours.
