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
- Local GGUF inference via llama.cpp (no HuggingFace downloads when `LOCAL_GGUF_MODEL` is set).

## Local GGUF Mode (llama-cpp-python)

Use a local `.gguf` file instead of downloading from HuggingFace:

```bash
pip install llama-cpp-python          # CPU build (default)
# For CUDA: CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
```

### Sanity-check: confirm llama.cpp mode with no HF download

```bash
# Linux / macOS
LOCAL_GGUF_MODEL=/path/to/your/model.gguf MCFG_LLM= python enhanced_rag_system.py

# Windows PowerShell
$env:LOCAL_GGUF_MODEL = "C:\path\to\your\model.gguf"
$env:MCFG_LLM = ""
python enhanced_rag_system.py
```

Look for log lines like:
```
Loading local GGUF via llama.cpp: /path/to/model.gguf (n_gpu_layers=0)
llama.cpp threads: auto ...
```
If you see those, llama.cpp is active and HF downloads are fully skipped.

### Performance tuning env vars

| Variable | Default | Notes |
|---|---|---|
| `LOCAL_GGUF_MODEL` | *(unset)* | Absolute path to `.gguf`; enables llama.cpp mode when set. |
| `LLAMA_CPP_THREADS` | `0` (auto) | Set to your **physical** core count for best CPU throughput, e.g. `8`. |
| `LLAMA_CPP_N_GPU_LAYERS` | `0` (CPU-only) | Layers to offload to GPU. Start with `20`–`40` for a 14B quant if llama-cpp-python was built with CUDA/Metal. |
| `MCFG_LLM` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model name; ignored when `LOCAL_GGUF_MODEL` is set. |

**Recommended starting points:**

- **CPU-only (e.g. 8-core machine):** `LLAMA_CPP_THREADS=8 LLAMA_CPP_N_GPU_LAYERS=0`
- **GPU offload (14B Q6\_K, 24 GB VRAM):** `LLAMA_CPP_N_GPU_LAYERS=40`
- **Hybrid (some layers on GPU, rest on CPU):** `LLAMA_CPP_N_GPU_LAYERS=20`

`LLAMA_CPP_THREADS=0` lets llama.cpp auto-detect; `LLAMA_CPP_N_GPU_LAYERS=0` keeps everything on CPU (safe default for CPU-only builds).

## Troubleshooting

**Still seeing HuggingFace downloads?**
- Make sure `LOCAL_GGUF_MODEL` is exported in the *same* shell session before running.
- Unset or clear `MCFG_LLM` so the HF fallback name is not used.
- Verify the path exists: `python -c "import os; print(os.path.isfile('<your path>'))"`

**`llama-cpp-python` ImportError:**
Install with: `pip install llama-cpp-python` (CPU) or rebuild with `CMAKE_ARGS="-DLLAMA_CUDA=on"` for GPU support.

**`torch_dtype` deprecation warning:**
Harmless — the code uses `torch_dtype=` in the HF path which some versions warn about. It does not affect functionality.
