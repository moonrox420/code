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
- Optional llama.cpp GGUF backend (fully local, no HF downloads).

## Running fully local with GGUF (llama.cpp)

Use this mode to avoid any Hugging Face model downloads.

**Prerequisites**
- Install the llama.cpp Python binding: `pip install llama-cpp-python`
- Have a local `.gguf` model file on disk.

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `LOCAL_GGUF_MODEL` | *(unset)* | Full path to your `.gguf` file. **Required** to enable GGUF mode. |
| `LLAMA_CPP_THREADS` | `0` | CPU threads. `0` = auto (`cpu_count`, fallback 4); otherwise set to your physical core count. |
| `LLAMA_CPP_N_GPU_LAYERS` | `0` | Layers offloaded to GPU. `0` = CPU-only. If built with GPU support, try `20`–`40` for a 14B quant and tune to fit VRAM. |
| `MCFG_LLM` | *(optional)* | HF model name/path. Can be left empty when `LOCAL_GGUF_MODEL` is set — the GGUF backend short-circuits any HF loading. |

**Run (all HF downloads skipped)**

Windows:
```powershell
$env:LOCAL_GGUF_MODEL = "C:\path\to\your_model.q6_k.gguf"
$env:LLAMA_CPP_THREADS = "0"
$env:LLAMA_CPP_N_GPU_LAYERS = "0"
$env:MCFG_LLM = ""
python enhanced_rag_system.py
```

Linux / macOS:
```bash
export LOCAL_GGUF_MODEL=/path/to/your_model.q6_k.gguf
export LLAMA_CPP_THREADS=0
export LLAMA_CPP_N_GPU_LAYERS=0
export MCFG_LLM=
python enhanced_rag_system.py
```

**Sanity check**
- Logs should contain: `Using llama.cpp with threads=..., n_gpu_layers=..., model=...`
- No Hugging Face model download messages should appear.
- Clear (unset) `LOCAL_GGUF_MODEL` to revert to the default HF/Transformers backend.

## TROUBLESHOOTING