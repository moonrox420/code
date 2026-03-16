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
3. (Optional) Tune performance with these environment variables:
   - `LLAMA_CPP_THREADS` — number of CPU threads to use. Set to your physical core count for best performance; `0` (default) = auto-detect.
   - `LLAMA_CPP_N_GPU_LAYERS` — number of transformer layers to offload to GPU. Use `0` (default) for CPU-only inference. If `llama-cpp-python` was built with GPU support, try `20`–`40` for a 14B quantized model and tune to fit your VRAM.
4. Launch (HF downloads will be skipped entirely):
   ```
   python enhanced_rag_system.py
   ```

**Sanity check:** After startup, look for the log line:
```
Using llama.cpp with threads=..., n_gpu_layers=..., model=...
```
and confirm there are **no** Hugging Face model download logs. Unset `LOCAL_GGUF_MODEL` (or set it to an empty string) to revert to the default HF/Transformers backend.

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
