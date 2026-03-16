# code

Enhanced RAG System – a Retrieval-Augmented Generation pipeline that supports both
**HF/Transformers** (default) and a **local GGUF model via llama.cpp**.

## Quick start

```bash
pip install torch transformers sentence-transformers faiss-cpu bitsandbytes
python enhanced_rag_system.py
```

## Using a local GGUF model (avoids HF downloads)

When `LOCAL_GGUF_MODEL` is set, the system loads the specified `.gguf` file with
**llama.cpp** and **completely skips** downloading or initialising any HF model
(saving the ~14.5 GB Mistral-7B download, for example).

### Installation

```bash
pip install llama-cpp-python          # CPU build
# OR – for CUDA GPU offload:
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Running with a local GGUF file

**Linux / macOS**

```bash
export LOCAL_GGUF_MODEL=/path/to/your/model.gguf
# Optional tuning:
export LLAMA_CPP_THREADS=8           # 0 = auto-detect
export LLAMA_CPP_N_GPU_LAYERS=35     # >0 only if llama-cpp-python was built with GPU support
export LLAMA_CPP_N_CTX=4096          # context window size

python enhanced_rag_system.py
```

**Windows (PowerShell)**

```powershell
$env:LOCAL_GGUF_MODEL = "C:\Users\droxa\.cache\huggingface\hub\models--BlossomsAI--Qwen2.5-Coder-14B-Instruct-Uncensored-GGUF\snapshots\b15f5f5bf2c2ccaa66f82b58a1a410a5b74715d1\q6_k.gguf"
$env:LLAMA_CPP_THREADS = "8"
$env:LLAMA_CPP_N_GPU_LAYERS = "0"    # set >0 if GPU build available

python enhanced_rag_system.py
```

**Windows (cmd.exe)**

```cmd
set LOCAL_GGUF_MODEL=C:\path\to\model\q6_k.gguf
python enhanced_rag_system.py
```

### Reverting to the HF/Transformers default

Simply clear the environment variable – the system will use the HF model specified
by `MCFG_LLM` (default: `mistralai/Mistral-7B-Instruct-v0.2`).

```bash
unset LOCAL_GGUF_MODEL          # Linux / macOS
python enhanced_rag_system.py
```

```powershell
Remove-Item Env:LOCAL_GGUF_MODEL    # Windows PowerShell
python enhanced_rag_system.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOCAL_GGUF_MODEL` | *(unset)* | Absolute path to a `.gguf` file. When set, llama.cpp is used and HF is skipped entirely. |
| `MCFG_LLM` | `mistralai/Mistral-7B-Instruct-v0.2` | HF model ID (only used when `LOCAL_GGUF_MODEL` is unset). |
| `LLAMA_CPP_THREADS` | `0` | CPU threads for llama.cpp (`0` = auto-detect). |
| `LLAMA_CPP_N_GPU_LAYERS` | `0` | Layers to offload to GPU (`0` = CPU only). |
| `LLAMA_CPP_N_CTX` | `4096` | Context window size for llama.cpp. |

## TROUBLESHOOTING

**`LOCAL_GGUF_MODEL is set but llama-cpp-python is not installed`**
→ Run `pip install llama-cpp-python`.

**`LOCAL_GGUF_MODEL path not found`**
→ Check the path is absolute and the file exists.

**HF model keeps downloading even though I set `LOCAL_GGUF_MODEL`**
→ Make sure the variable is exported in the same shell session (`export`, not just
  `set` in bash).  Verify with `echo $LOCAL_GGUF_MODEL` (Linux) or
  `echo %LOCAL_GGUF_MODEL%` (Windows cmd).

**CUDA out-of-memory with HF backend**
→ Enable 4-bit quantisation (on by default when `load_in_4bit=True` in
  `ModelConfig`) or switch to a GGUF model and set `LLAMA_CPP_N_GPU_LAYERS` to
  offload only as many layers as VRAM allows.
