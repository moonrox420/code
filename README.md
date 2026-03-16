# code

An enhanced Retrieval-Augmented Generation (RAG) system with pluggable generation
backends and a streaming token interface for UIs.

---

## Quick start

```bash
pip install sentence-transformers faiss-cpu torch transformers
python enhanced_rag_system.py "What is this system?"
```

---

## Backends

### Default – HuggingFace Transformers

When `LOCAL_GGUF_MODEL` is **not** set, the system loads the model specified by
`HF_MODEL_NAME` (default: `mistralai/Mistral-7B-Instruct-v0.2`) via the
HuggingFace Transformers library.

> **Note:** The `torch_dtype` deprecation warning from older Transformers versions
> is addressed by passing `dtype=torch.float16` instead.  No functional change.

### Optional – Local GGUF (llama.cpp)

Set `LOCAL_GGUF_MODEL` to the **absolute path** of a GGUF file to use
`llama-cpp-python` as the generation backend.  **No HuggingFace / Transformers
downloads occur** when this variable is set.

```bash
pip install llama-cpp-python
# GPU build (optional):
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

export LOCAL_GGUF_MODEL=/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
python enhanced_rag_system.py "Summarise the documents."
```

If the file does not exist or `llama-cpp-python` is not installed the system
exits immediately with a descriptive error message.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOCAL_GGUF_MODEL` | *(unset)* | Path to a local GGUF model file.  Activates the GGUF backend. |
| `LLAMA_CPP_THREADS` | `0` | CPU thread count for llama.cpp (`0` = auto-detect). |
| `LLAMA_CPP_N_GPU_LAYERS` | `0` | Layers to offload to GPU (requires GPU-enabled build). |
| `HF_MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model used by the Transformers backend. |
| `EMBED_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for retrieval. |

---

## Programmatic usage

```python
from enhanced_rag_system import EnhancedRAGSystem

# GGUF backend (no HF downloads)
rag = EnhancedRAGSystem(local_gguf_model="/models/my-model.Q4_K_M.gguf")

# HF backend (default)
# rag = EnhancedRAGSystem()

rag.add_documents(["Paris is the capital of France.", "The Eiffel Tower is in Paris."])

for token in rag.query("Where is the Eiffel Tower?"):
    print(token, end="", flush=True)
print()
```

---

## Troubleshooting

### `FileNotFoundError: LOCAL_GGUF_MODEL is set but the file was not found`

The path in `LOCAL_GGUF_MODEL` does not exist.  Check the path and try again.

### `ImportError: llama-cpp-python is required when LOCAL_GGUF_MODEL is set`

Install the package:

```bash
pip install llama-cpp-python
```

For GPU offloading (CUDA):

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### HuggingFace `torch_dtype` deprecation warning

Older guides pass `torch_dtype=torch.float16` to `from_pretrained`.  This
codebase uses the non-deprecated `dtype=torch.float16` keyword instead, so no
warning is emitted.

### Out-of-memory errors with the HF backend

Use the GGUF backend with a quantised model (e.g. Q4_K_M) to significantly
reduce memory requirements.
