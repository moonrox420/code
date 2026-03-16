# CAN — PyQt "All Bells" Local RAG Coder

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
- Optional llama.cpp GGUF backend (fully local, no HF downloads).

## Running fully local with GGUF (llama.cpp)

Use this mode to run inference entirely on-device and skip all Hugging Face downloads.

### Prerequisites

1. Install the llama.cpp Python binding:
   ```
   pip install llama-cpp-python
   ```
2. Have a local `.gguf` model file (e.g. a Q4_K_M or Q6_K quantization of any supported model).

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOCAL_GGUF_MODEL` | *(empty)* | **Required.** Absolute path to your `.gguf` file. When set, the HF/Transformers backend is bypassed entirely. |
| `LLAMA_CPP_THREADS` | `0` | CPU threads. `0` = auto (`os.cpu_count()`, fallback 4). Set to your physical core count for best performance. |
| `LLAMA_CPP_N_GPU_LAYERS` | `0` | Transformer layers to offload to GPU. `0` = CPU-only. If `llama-cpp-python` was built with GPU support, try `20`–`40` for a 14B quantized model and tune to fit your VRAM. |

`MCFG_LLM` (the HF model name) is ignored when `LOCAL_GGUF_MODEL` is set and can be left empty.

### Run example

**Linux/macOS:**
```bash
export LOCAL_GGUF_MODEL=/models/mistral-7b-instruct-v0.2.Q6_K.gguf
export LLAMA_CPP_THREADS=8          # set to your physical core count
export LLAMA_CPP_N_GPU_LAYERS=0     # set >0 only if GPU build (e.g. 20-40 for 14B)
python enhanced_rag_system.py
```

**Windows (PowerShell):**
```powershell
$env:LOCAL_GGUF_MODEL = "C:\models\mistral-7b-instruct-v0.2.Q6_K.gguf"
$env:LLAMA_CPP_THREADS = "8"
$env:LLAMA_CPP_N_GPU_LAYERS = "0"
python enhanced_rag_system.py
```

### Sanity check

After startup, look for this log line:
```
Using llama.cpp with threads=..., n_gpu_layers=..., model=...
```
Confirm there are **no** Hugging Face model download logs. To revert to the HF/Transformers backend, unset `LOCAL_GGUF_MODEL` (or set it to an empty string).
