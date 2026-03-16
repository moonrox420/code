# CAN — PyQt “All Bells” Local RAG Coder

## Run
1) `python -m venv .venv && source .venv/bin/activate`
2) `pip install -r requirements.txt`
3) Put reference files in `data/raw/` (PDF/TXT supported) or drag/drop via Settings tab.
4) `python app.py`
5) Click **Rebuild Index** once, then ask (Ctrl/Cmd+Enter sends).

## Features
- Dark, tabbed PyQt5 UI with split-pane answer/context cards.
- Streaming tokens with cancel; temp and max-token controls; Top‑K slider.
- Drag/drop ingestion, PDF/TXT parsing, progress bar, one-click FAISS rebuild.
- Hot model swap from UI; GPU auto-detect with CPU fallback.
- Copy or export answer to Markdown; save/load chat history.
- Context viewer with scores + sources for transparency.

## TROUBLESHOOTING

### Using a local GGUF model (no Hugging Face downloads)

Set the `LOCAL_GGUF_MODEL` environment variable to the absolute path of your `.gguf` file
before starting the app.  When this variable is set the Hugging Face / Transformers LLM
initialisation is **fully skipped** — no model will be downloaded from the Hub.

**Requirements**

```bash
pip install llama-cpp-python   # CPU-only
# or, for CUDA GPU offload:
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall
```

**Example (Linux / macOS)**

```bash
export LOCAL_GGUF_MODEL="/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
export LLAMA_CPP_THREADS=8          # optional: number of CPU threads (0 = auto)
export LLAMA_CPP_N_GPU_LAYERS=35    # optional: layers to offload to GPU (0 = CPU only)
python app.py
```

**Example (Windows PowerShell)**

```powershell
$env:LOCAL_GGUF_MODEL = "C:\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
$env:LLAMA_CPP_THREADS = "8"
python app.py
```

If `LOCAL_GGUF_MODEL` points to a file that does not exist a `FileNotFoundError` is raised
immediately with a clear message.

### Using Hugging Face / Transformers (default)

Simply leave `LOCAL_GGUF_MODEL` **unset** (or set to an empty string).  The app will
download and cache the model specified by `MCFG_LLM` (default:
`mistralai/Mistral-7B-Instruct-v0.2`) via the normal Hugging Face Hub mechanism.

```bash
# Optionally override the HF model:
export MCFG_LLM="mistralai/Mistral-7B-Instruct-v0.2"
python app.py
```