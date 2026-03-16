# code
coder

---

## Enhanced RAG System

`enhanced_rag_system.py` is a Retrieval-Augmented Generation (RAG) pipeline that supports two LLM backends:

| Backend | When active | Notes |
|---|---|---|
| **HuggingFace / Transformers** | `LOCAL_GGUF_MODEL` **unset** (default) | Downloads model on first run (e.g. ~14 GB for Mistral-7B) |
| **llama.cpp (local GGUF)** | `LOCAL_GGUF_MODEL` **set** | No HF download; uses your local `.gguf` file |

### Installation

```bash
# Core retrieval stack (always required)
pip install faiss-cpu sentence-transformers numpy

# HF backend (default)
pip install transformers accelerate bitsandbytes torch

# GGUF / llama.cpp backend (opt-in)
pip install llama-cpp-python
# GPU build (CUDA):
# CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

---

## TROUBLESHOOTING

### Running with a local GGUF model (avoids HF downloads)

Set the `LOCAL_GGUF_MODEL` environment variable to the **absolute path** of your `.gguf` file before running:

**Linux / macOS**
```bash
export LOCAL_GGUF_MODEL="/path/to/your/model.gguf"
python enhanced_rag_system.py
```

**Windows (cmd)**
```bat
set LOCAL_GGUF_MODEL=C:\path\to\your\model.gguf
python enhanced_rag_system.py
```

**Windows (PowerShell)**
```powershell
$env:LOCAL_GGUF_MODEL = "C:\path\to\your\model.gguf"
python enhanced_rag_system.py
```

When `LOCAL_GGUF_MODEL` is set:
- The HuggingFace model is **completely skipped** (no download, no GPU initialisation).
- `llama-cpp-python` must be installed (`pip install llama-cpp-python`).

#### Optional tuning variables (GGUF backend only)

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CPP_THREADS` | `0` (auto) | CPU threads to use for inference |
| `LLAMA_CPP_N_GPU_LAYERS` | `0` (CPU-only) | Number of model layers to offload to GPU |

Example with GPU offloading:
```bash
export LOCAL_GGUF_MODEL="/models/q6_k.gguf"
export LLAMA_CPP_N_GPU_LAYERS=32
export LLAMA_CPP_THREADS=8
python enhanced_rag_system.py
```

---

### Reverting to the HuggingFace default

Simply **unset** `LOCAL_GGUF_MODEL`:

```bash
unset LOCAL_GGUF_MODEL       # Linux / macOS
# or
set LOCAL_GGUF_MODEL=        # Windows cmd
# or
Remove-Item Env:LOCAL_GGUF_MODEL   # Windows PowerShell
```

The HF model is controlled by the `MCFG_LLM` env var:
```bash
export MCFG_LLM="mistralai/Mistral-7B-Instruct-v0.2"   # default
python enhanced_rag_system.py
```

---

### Common errors

| Error | Cause | Fix |
|---|---|---|
| `LOCAL_GGUF_MODEL is set but llama-cpp-python is not installed` | Missing package | `pip install llama-cpp-python` |
| `LOCAL_GGUF_MODEL path not found: …` | Wrong path | Check the path and re-export the variable |
| Out-of-memory during HF load | Large model on limited VRAM | Switch to GGUF backend or reduce `MCFG_LLM` to a smaller model |
| `torch_dtype` deprecation warning | Old Transformers call | Already fixed in this codebase; uses `dtype=` |
