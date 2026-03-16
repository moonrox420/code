# CAN — Precision-Optimized Local RAG Coder

## Quick Start
1. `python -m venv .venv && source .venv/bin/activate`  *(Windows: `.venv\Scripts\activate`)*
2. `pip install -r requirements.txt`
3. Put reference files in `data/raw/` (PDF/TXT/DOCX/MD/HTML supported) or drag-drop via **Settings** tab.
4. `python enhanced_rag_system.py`
5. Click **Rebuild Index** once, then ask in the **Precision Chat** tab (Ctrl+Enter sends).

## Features
- Dark, tabbed PyQt5 UI with split-pane answer/context cards.
- Streaming tokens with cancel; temperature, max-tokens, and Top-K controls.
- Drag-and-drop ingestion — PDF, TXT, DOCX, Markdown, HTML, source code files.
- Query expansion, semantic reranking, HyDE (Hypothetical Document Embeddings).
- Analytics dashboard with per-file chunk statistics (auto-refreshes on tab switch).
- Query history dropdown for quick recall of past questions.
- Hot model swap from UI; GPU auto-detect with CPU fallback.
- Copy or export answer to Markdown; save/load chat history as JSON.
- Context viewer with relevance scores + sources for full transparency.
- Optional llama.cpp GGUF backend (fully local, no HF downloads needed).

---

## TROUBLESHOOTING

### Startup crash: `ImportError` at line ~29

The script uses **defensive import guards** for every optional dependency.
If you see an import error at startup, install the missing package:

```
pip install -r requirements.txt
```

If the error persists, install the specific package manually (see the sections below).

---

### PyMuPDF (`import fitz` / `ModuleNotFoundError: No module named 'fitz'`)

PyMuPDF is the **preferred** (faster, richer metadata) PDF backend.
It is **optional** — the script falls back to PyPDF2 automatically when absent.

To install PyMuPDF:
```
pip install PyMuPDF
```

> **Do NOT** install the old package named `fitz` from PyPI — that is a completely
> different, abandoned package and will cause an `ImportError`.
> The correct package name is **`PyMuPDF`**.

---

### DOCX parsing (`ImportError: No module named 'docx'` or wrong package)

DOCX support requires **`python-docx`**:
```
pip install python-docx
```

> **Warning:** Do NOT install the package called `docx` from PyPI.
> `docx` is a broken, unmaintained stub and will import as `docx` but provide none of
> the required functionality (`Document`, paragraph parsing, etc.).
> Always use `python-docx`.

After installing, verify with:
```python
from docx import Document   # should succeed silently
```

---

### Bad PyPI package names to avoid

| Correct package (pip name) | Wrong package | Notes |
|----------------------------|---------------|-------|
| `python-docx` | `docx` | Broken stub — no real DOCX parsing |
| `PyMuPDF` | `fitz` | Unrelated abandoned project |
| `faiss-cpu` or `faiss-gpu` | `faiss` | Unofficial build — may not work |

---

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
3. (Optional) Tune performance with environment variables:
   - `LLAMA_CPP_THREADS` — CPU threads (0 = auto-detect via `os.cpu_count()`).
   - `LLAMA_CPP_N_GPU_LAYERS` — layers to offload to GPU (0 = CPU-only).
4. Launch — HF downloads will be skipped entirely:
   ```
   python enhanced_rag_system.py
   ```

**Sanity check:** After startup, look for:
```
Using llama.cpp with threads=..., n_gpu_layers=..., model=...
```
Unset `LOCAL_GGUF_MODEL` (or set it to an empty string) to revert to the HF/Transformers backend.

---

### Token counting (`tiktoken` not installed)

`tiktoken` is optional. When absent the system automatically falls back to a
word-count approximation (~1.3 tokens/word). To enable exact token counting:
```
pip install tiktoken
```
