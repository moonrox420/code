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

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOCAL_GGUF_MODEL` | _(empty)_ | Absolute path to a `.gguf` model file. When set, enables the llama.cpp backend and skips all Hugging Face downloads. |
| `LLAMA_CPP_THREADS` | `0` | CPU threads for llama.cpp inference. `0` = auto-detect (`os.cpu_count()`). Set to your physical core count for best throughput. |
| `LLAMA_CPP_N_GPU_LAYERS` | `0` | Transformer layers to offload to GPU. `0` = CPU-only. Requires `llama-cpp-python` built with GPU support; try `20`–`40` for a 14B quant and tune to fit your VRAM. |
| `MCFG_LLM` | `mistralai/Mistral-7B-Instruct-v0.2` | Hugging Face model ID used when **not** running in GGUF mode. Ignored when `LOCAL_GGUF_MODEL` is set. |

## TROUBLESHOOTING

### Running fully local with GGUF (llama.cpp)

To run without any Hugging Face model downloads:

1. Install the llama.cpp Python binding:
   ```
   pip install llama-cpp-python
   ```
2. Set the path to your GGUF model file:
   ```
   set LOCAL_GGUF_MODEL=C:\path\to\your_model.q6_k.gguf   # Windows
   export LOCAL_GGUF_MODEL=/path/to/your_model.q6_k.gguf  # Linux/macOS
   ```
3. (Optional) Tune performance:
   - `LLAMA_CPP_THREADS=<physical_cores>` — `0` (default) auto-detects via `os.cpu_count()`.
   - `LLAMA_CPP_N_GPU_LAYERS=0` for CPU-only; if built with GPU support, try `20`–`40` for a 14B quant and tune to fit your VRAM.
4. (Optional) Clear the HF model name to prevent any model resolution attempt:
   ```
   set MCFG_LLM=     # Windows
   export MCFG_LLM=  # Linux/macOS
   ```
5. Launch (HF downloads are skipped automatically when `LOCAL_GGUF_MODEL` is set):
   ```
   python enhanced_rag_system.py
   ```

**Sanity check:** After startup, look for the log line:
```
Using llama.cpp with threads=..., n_gpu_layers=..., model=...
```
and confirm there are **no** Hugging Face model download logs. Unset `LOCAL_GGUF_MODEL` (or set it to an empty string) to revert to the default HF/Transformers backend.