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

### Running fully local with GGUF (llama.cpp — no HF downloads)

1. **Install the backend:**
   ```
   pip install llama-cpp-python
   ```

2. **Set your model path** (the `.gguf` file on disk):
   ```
   # Linux/macOS
   export LOCAL_GGUF_MODEL=/path/to/model.q6_k.gguf

   # Windows
   set LOCAL_GGUF_MODEL=C:\path\to\model.q6_k.gguf
   ```

3. **Tune performance** (optional):

   | Variable | Default | Notes |
   |---|---|---|
   | `LLAMA_CPP_THREADS` | `0` (auto = physical cores) | Set to your physical core count for best throughput |
   | `LLAMA_CPP_N_GPU_LAYERS` | `0` (CPU-only) | If llama-cpp-python was built with GPU support, try `20`–`40` for a 14B quant and tune to fit VRAM |

   Recommended starting values:
   - CPU-only: `LLAMA_CPP_THREADS=<physical_cores>`, `LLAMA_CPP_N_GPU_LAYERS=0`
   - With GPU: `LLAMA_CPP_N_GPU_LAYERS=20` and increase until you run out of VRAM.

4. **Launch** (skips all HF downloads):
   ```
   python enhanced_rag_system.py
   # or
   python app.py
   ```

**Sanity check:** Look for the log line:
```
Using llama.cpp with threads=<N>, n_gpu_layers=<N>, model=<path>
```
and confirm the absence of any Hugging Face model-download progress bars.

**Reverting to HF/Transformers:** Unset `LOCAL_GGUF_MODEL` (and optionally set `MCFG_LLM` to your preferred HF model ID). The HF path is used by default when `LOCAL_GGUF_MODEL` is not set.
