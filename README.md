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

Use this mode to run entirely offline and avoid any Hugging Face model downloads.

**Prerequisites**
- Install the llama.cpp Python binding: `pip install llama-cpp-python`
- Have a local `.gguf` model file ready.

**Environment variables**

| Variable | Required | Default | Description |
|---|---|---|---|
| `LOCAL_GGUF_MODEL` | Yes (to enable GGUF mode) | _(empty)_ | Full path to your `.gguf` model file. Setting this enables GGUF mode. |
| `LLAMA_CPP_THREADS` | No | `0` (auto) | Number of CPU threads. Set to your physical core count for best performance; `0` = auto-detect via `cpu_count()`, fallback 4. |
| `LLAMA_CPP_N_GPU_LAYERS` | No | `0` (CPU-only) | Transformer layers to offload to GPU. Use `0` for CPU-only. If `llama-cpp-python` was built with GPU support, try `20`–`40` for a 14B quant and tune to fit your VRAM. |
| `MCFG_LLM` | No | Mistral-7B | Override the HF model name. Unused when `LOCAL_GGUF_MODEL` is set — GGUF mode bypasses all HF downloads regardless of this value. |

**Run (skips all HF downloads)**

Windows (PowerShell):
```powershell
$env:LOCAL_GGUF_MODEL = "C:\path\to\model.gguf"
$env:LLAMA_CPP_THREADS = "0"          # 0 = auto
$env:LLAMA_CPP_N_GPU_LAYERS = "0"     # 0 = CPU-only
$env:MCFG_LLM = ""                    # optional; ignored in GGUF mode
python enhanced_rag_system.py
```

Windows (cmd):
```cmd
set LOCAL_GGUF_MODEL=C:\path\to\model.gguf
set LLAMA_CPP_THREADS=0
set LLAMA_CPP_N_GPU_LAYERS=0
set MCFG_LLM=
python enhanced_rag_system.py
```

Linux / macOS:
```bash
export LOCAL_GGUF_MODEL=/path/to/model.gguf
export LLAMA_CPP_THREADS=0
export LLAMA_CPP_N_GPU_LAYERS=0
python enhanced_rag_system.py
```

**Sanity check**
- Logs should show: `Using llama.cpp with threads=..., n_gpu_layers=..., model=...`
- Confirm there are **no** Hugging Face model download messages.
- Clear (unset) `LOCAL_GGUF_MODEL` to revert to the default HF/Transformers backend.

## TROUBLESHOOTING

### GGUF mode not activating
- Ensure `LOCAL_GGUF_MODEL` is set to the **full path** of an existing `.gguf` file.
- Verify `llama-cpp-python` is installed: `pip show llama-cpp-python`
- Check for `FileNotFoundError` in logs if the path is wrong.

### GPU offload not working
- Confirm your `llama-cpp-python` build includes GPU support (CUDA or Metal).
- Increase `LLAMA_CPP_N_GPU_LAYERS` incrementally (e.g., `10`, `20`, `35`) and monitor VRAM usage.