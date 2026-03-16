"""
Enhanced RAG (Retrieval-Augmented Generation) system.

Supports two generation backends:
  1. Local GGUF via llama-cpp-python  – activated by setting LOCAL_GGUF_MODEL
  2. HuggingFace Transformers          – used when LOCAL_GGUF_MODEL is unset (default)

Environment variables
---------------------
LOCAL_GGUF_MODEL        Path to a local GGUF model file.  When set, llama-cpp-python
                        is used and *no* HuggingFace / Transformers downloads occur.
LLAMA_CPP_THREADS       Number of CPU threads for llama.cpp (default: 0 = auto-detect).
LLAMA_CPP_N_GPU_LAYERS  Number of layers to offload to GPU (default: 0, requires a
                        GPU-enabled llama-cpp-python build).
HF_MODEL_NAME           HuggingFace model name/path used by the Transformers backend
                        (default: "mistralai/Mistral-7B-Instruct-v0.2").
EMBED_MODEL_NAME        Sentence-transformers model used for retrieval embeddings
                        (default: "sentence-transformers/all-MiniLM-L6-v2").
"""

from __future__ import annotations

import os
import sys
import threading
from queue import Empty, Queue
from typing import Generator, List, Optional

# ---------------------------------------------------------------------------
# Lazy imports – only pulled in when the relevant backend is active
# ---------------------------------------------------------------------------
_torch = None
_transformers = None
_llama_cpp = None
_sentence_transformers = None
_faiss = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch  # noqa: PLC0415
        _torch = torch
    return _torch


def _import_transformers():
    global _transformers
    if _transformers is None:
        import transformers  # noqa: PLC0415
        _transformers = transformers
    return _transformers


def _import_llama_cpp():
    global _llama_cpp
    if _llama_cpp is None:
        try:
            from llama_cpp import Llama  # noqa: PLC0415
            _llama_cpp = Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required when LOCAL_GGUF_MODEL is set.  "
                "Install it with:  pip install llama-cpp-python"
            ) from exc
    return _llama_cpp


def _import_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        _sentence_transformers = SentenceTransformer
    return _sentence_transformers


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LOCAL_GGUF_MODEL: Optional[str] = os.environ.get("LOCAL_GGUF_MODEL")
_LLAMA_CPP_THREADS: int = int(os.environ.get("LLAMA_CPP_THREADS", "0"))
_LLAMA_CPP_N_GPU_LAYERS: int = int(os.environ.get("LLAMA_CPP_N_GPU_LAYERS", "0"))
_HF_MODEL_NAME: str = os.environ.get(
    "HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"
)
_EMBED_MODEL_NAME: str = os.environ.get(
    "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------------------------
# Streaming helper (shared by both backends)
# ---------------------------------------------------------------------------

class TextStreamer:
    """Thread-safe streamer that yields tokens from a queue.

    Both the HF Transformers TextIteratorStreamer and the llama.cpp streaming
    generator are wrapped so that the UI / caller always consumes a single
    ``Generator[str, None, None]``.
    """

    def __init__(self) -> None:
        self._queue: Queue[Optional[str]] = Queue()
        self._done = threading.Event()

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    def put(self, token: str) -> None:
        """Push a token into the stream."""
        self._queue.put(token)

    def end(self) -> None:
        """Signal end-of-stream."""
        self._queue.put(None)

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------

    def __iter__(self) -> Generator[str, None, None]:
        while True:
            try:
                token = self._queue.get(timeout=60)
            except Empty:
                break
            if token is None:
                break
            yield token


# ---------------------------------------------------------------------------
# GGUF backend
# ---------------------------------------------------------------------------

class GGUFBackend:
    """Generation backend backed by a local GGUF file via llama-cpp-python."""

    def __init__(self, model_path: str, n_threads: int = 0, n_gpu_layers: int = 0) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"LOCAL_GGUF_MODEL is set but the file was not found: {model_path!r}.  "
                "Please provide a valid path to a GGUF model file."
            )

        Llama = _import_llama_cpp()

        kwargs: dict = {
            "model_path": model_path,
            "verbose": False,
        }
        if n_threads > 0:
            kwargs["n_threads"] = n_threads
        if n_gpu_layers > 0:
            kwargs["n_gpu_layers"] = n_gpu_layers

        print(f"[EnhancedRAG] Loading GGUF model from {model_path!r} …", file=sys.stderr)
        self._llm = Llama(**kwargs)
        print("[EnhancedRAG] GGUF model loaded.", file=sys.stderr)

    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> TextStreamer:
        """Return a :class:`TextStreamer` that yields tokens asynchronously."""
        streamer = TextStreamer()

        def _worker():
            try:
                for chunk in self._llm(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                ):
                    token: str = chunk["choices"][0]["text"]
                    streamer.put(token)
            finally:
                streamer.end()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return streamer


# ---------------------------------------------------------------------------
# HuggingFace Transformers backend
# ---------------------------------------------------------------------------

class HFBackend:
    """Generation backend backed by HuggingFace Transformers."""

    def __init__(self, model_name: str) -> None:
        transformers = _import_transformers()
        torch = _import_torch()

        print(
            f"[EnhancedRAG] Loading HF model {model_name!r} …  "
            "(set LOCAL_GGUF_MODEL to skip this download)",
            file=sys.stderr,
        )

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        # Use `dtype` (not the deprecated `torch_dtype`) when loading the model.
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        self._model.eval()
        print("[EnhancedRAG] HF model loaded.", file=sys.stderr)

    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> TextStreamer:
        """Return a :class:`TextStreamer` that yields tokens asynchronously."""
        transformers = _import_transformers()
        torch = _import_torch()

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)

        hf_streamer = transformers.TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "streamer": hf_streamer,
        }

        streamer = TextStreamer()

        # model.generate() feeds hf_streamer during generation – run it in its
        # own thread so we can concurrently drain hf_streamer into TextStreamer.
        def _gen():
            with torch.no_grad():
                self._model.generate(**gen_kwargs)

        gen_thread = threading.Thread(target=_gen, daemon=True)
        gen_thread.start()

        # Drain the HF streamer into our unified TextStreamer.
        def _drain():
            try:
                for token in hf_streamer:
                    streamer.put(token)
            finally:
                streamer.end()

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()

        return streamer


# ---------------------------------------------------------------------------
# Retrieval component
# ---------------------------------------------------------------------------

class DocumentStore:
    """Simple in-memory FAISS-backed document store for retrieval."""

    def __init__(self, embed_model_name: str = _EMBED_MODEL_NAME) -> None:
        SentenceTransformer = _import_sentence_transformers()
        self._embed_model = SentenceTransformer(embed_model_name)
        self._documents: List[str] = []
        self._index = None  # FAISS index, built lazily

    # ------------------------------------------------------------------

    def add_documents(self, documents: List[str]) -> None:
        """Embed and index a list of document strings."""
        import numpy as np  # noqa: PLC0415
        try:
            import faiss  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu (or faiss-gpu) is required for the document store.  "
                "Install it with:  pip install faiss-cpu"
            ) from exc

        embeddings = self._embed_model.encode(documents, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        if self._index is None:
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatL2(dim)

        self._index.add(embeddings)
        self._documents.extend(documents)

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        """Return the *top_k* most relevant documents for *query*."""
        if self._index is None or len(self._documents) == 0:
            return []

        import numpy as np  # noqa: PLC0415

        query_vec = self._embed_model.encode([query], convert_to_numpy=True).astype(
            np.float32
        )
        _, indices = self._index.search(query_vec, min(top_k, len(self._documents)))
        return [self._documents[i] for i in indices[0] if i >= 0]


# ---------------------------------------------------------------------------
# Main RAG system
# ---------------------------------------------------------------------------

class EnhancedRAGSystem:
    """Retrieval-Augmented Generation system with pluggable generation backends.

    Parameters
    ----------
    embed_model_name:
        Sentence-transformers model for encoding queries and documents.
    hf_model_name:
        HuggingFace model name/path used when the HF backend is active.
    local_gguf_model:
        Path to a local GGUF file.  When supplied (or when the
        ``LOCAL_GGUF_MODEL`` env var is set) the GGUF backend is used and *no*
        HF/Transformers downloads occur.
    llama_cpp_threads:
        CPU thread count for llama.cpp (0 = auto).
    llama_cpp_n_gpu_layers:
        Number of layers to offload to GPU (0 = CPU-only).
    """

    def __init__(
        self,
        *,
        embed_model_name: str = _EMBED_MODEL_NAME,
        hf_model_name: str = _HF_MODEL_NAME,
        local_gguf_model: Optional[str] = _LOCAL_GGUF_MODEL,
        llama_cpp_threads: int = _LLAMA_CPP_THREADS,
        llama_cpp_n_gpu_layers: int = _LLAMA_CPP_N_GPU_LAYERS,
    ) -> None:
        self._doc_store = DocumentStore(embed_model_name=embed_model_name)

        if local_gguf_model:
            self._backend: GGUFBackend | HFBackend = GGUFBackend(
                model_path=local_gguf_model,
                n_threads=llama_cpp_threads,
                n_gpu_layers=llama_cpp_n_gpu_layers,
            )
            self._backend_name = "gguf"
        else:
            self._backend = HFBackend(model_name=hf_model_name)
            self._backend_name = "hf"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Return ``'gguf'`` or ``'hf'`` depending on the active backend."""
        return self._backend_name

    def add_documents(self, documents: List[str]) -> None:
        """Index documents for retrieval."""
        self._doc_store.add_documents(documents)

    def query(
        self,
        question: str,
        *,
        top_k: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> TextStreamer:
        """Retrieve relevant context and stream a generated answer.

        Returns
        -------
        TextStreamer
            Iterate over it to consume generated tokens as they are produced.
            Example::

                for token in rag.query("What is RAG?"):
                    print(token, end="", flush=True)
        """
        context_docs = self._doc_store.retrieve(question, top_k=top_k)
        context = "\n\n".join(context_docs) if context_docs else "No context available."

        prompt = _build_prompt(question, context)

        return self._backend.generate_stream(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str, context: str) -> str:
    return (
        f"You are a helpful assistant. Use the following context to answer the "
        f"question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced RAG System – quick test")
    parser.add_argument("question", nargs="?", default="What is this system?")
    args = parser.parse_args()

    rag = EnhancedRAGSystem()
    rag.add_documents(
        [
            "The Enhanced RAG System combines dense retrieval with a generative model.",
            "It supports a local GGUF backend (llama-cpp-python) or HuggingFace Transformers.",
            "Set LOCAL_GGUF_MODEL to the path of a GGUF file to use the local backend.",
        ]
    )

    print(f"\n[backend={rag.backend_name}] {args.question}\n")
    for tok in rag.query(args.question):
        print(tok, end="", flush=True)
    print()
