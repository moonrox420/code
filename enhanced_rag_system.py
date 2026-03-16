"""
Enhanced RAG (Retrieval-Augmented Generation) System
=====================================================
Supports two LLM backends, selected via environment variables:

  1. HuggingFace/Transformers (default)
     Uses AutoModelForCausalLM + AutoTokenizer.
     Model is chosen by the MCFG_LLM env var (default: mistralai/Mistral-7B-Instruct-v0.2).

  2. Local GGUF via llama.cpp  (opt-in)
     Set LOCAL_GGUF_MODEL to the absolute path of a .gguf file.
     The HF model is then COMPLETELY SKIPPED – no download, no GPU memory.
     Requires:  pip install llama-cpp-python

Optional tuning env vars (GGUF path only):
  LLAMA_CPP_THREADS       – number of CPU threads  (0 = let llama.cpp choose)
  LLAMA_CPP_N_GPU_LAYERS  – layers to offload to GPU  (0 = CPU-only)

Retrieval backend: sentence-transformers + FAISS (in-memory).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variables – read once at module import time
# ---------------------------------------------------------------------------

LOCAL_GGUF_MODEL: str = os.getenv("LOCAL_GGUF_MODEL", "").strip()
LLAMA_CPP_THREADS: int = int(os.getenv("LLAMA_CPP_THREADS", "0"))
LLAMA_CPP_N_GPU_LAYERS: int = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", "0"))

# ---------------------------------------------------------------------------
# Lazy / conditional imports
# ---------------------------------------------------------------------------

# llama-cpp-python – only needed when LOCAL_GGUF_MODEL is set.
Llama = None  # will be replaced below if the GGUF path is active
if LOCAL_GGUF_MODEL:
    try:
        from llama_cpp import Llama  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError(
            "LOCAL_GGUF_MODEL is set but llama-cpp-python is not installed.\n"
            "Fix: pip install llama-cpp-python\n"
            "For GPU support see: https://github.com/abetlen/llama-cpp-python"
        ) from exc

# HF / Transformers – imported lazily so that setting LOCAL_GGUF_MODEL
# never triggers a torch/transformers import (and therefore no GPU init).
if not LOCAL_GGUF_MODEL:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextIteratorStreamer,
    )

# Retrieval stack – always needed.
try:
    import faiss  # type: ignore
    import numpy as np
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as exc:
    raise ImportError(
        "Missing retrieval dependencies.\n"
        "Fix: pip install faiss-cpu sentence-transformers numpy"
    ) from exc

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """LLM and generation hyper-parameters."""

    # HF model – ignored when LOCAL_GGUF_MODEL is set.
    llm_name: str = field(
        default_factory=lambda: os.getenv(
            "MCFG_LLM", "mistralai/Mistral-7B-Instruct-v0.2"
        )
    )

    # Whether to use the local GGUF backend (auto-detected from env).
    use_local_gguf: bool = field(default_factory=lambda: bool(LOCAL_GGUF_MODEL))

    # Generation parameters (shared by both backends).
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # HF-only settings.
    load_in_4bit: bool = True         # use 4-bit quantisation when available
    device_map: str = "auto"


@dataclass
class RetrieverConfig:
    """Document retrieval hyper-parameters."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    k_retrieve: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64


# ---------------------------------------------------------------------------
# Document indexer
# ---------------------------------------------------------------------------


class DocumentIndexer:
    """Embeds and indexes text chunks; retrieves the top-k most relevant ones."""

    def __init__(self, cfg: RetrieverConfig) -> None:
        self.cfg = cfg
        logger.info("Loading embedding model: %s", cfg.embedding_model)
        self.embedder = SentenceTransformer(cfg.embedding_model)
        self._chunks: List[str] = []
        self._index: Optional[faiss.IndexFlatIP] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """Split *text* into overlapping fixed-size chunks."""
        size = self.cfg.chunk_size
        overlap = self.cfg.chunk_overlap
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += size - overlap
        return chunks

    def add_texts(self, texts: Iterable[str]) -> None:
        """Chunk, embed, and add *texts* to the FAISS index."""
        new_chunks: List[str] = []
        for doc in texts:
            new_chunks.extend(self._chunk_text(doc))

        if not new_chunks:
            return

        logger.info("Indexing %d new chunks …", len(new_chunks))
        embeddings = self._embed(new_chunks)  # (N, D) float32

        if self._index is None:
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(embeddings)
        self._chunks.extend(new_chunks)
        logger.info("Index now contains %d chunks.", len(self._chunks))

    def add_files(self, paths: Iterable[str | Path]) -> None:
        """Read text files and add them to the index."""
        texts: List[str] = []
        for p in paths:
            p = Path(p)
            if not p.is_file():
                logger.warning("Skipping non-existent path: %s", p)
                continue
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
        self.add_texts(texts)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: Optional[int] = None) -> List[str]:
        """Return the top-*k* most relevant chunks for *query*."""
        if self._index is None or not self._chunks:
            return []
        k = k or self.cfg.k_retrieve
        k = min(k, len(self._chunks))
        q_emb = self._embed([query])
        _, indices = self._index.search(q_emb, k)
        return [self._chunks[i] for i in indices[0] if i < len(self._chunks)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> "np.ndarray":
        """Return L2-normalised embeddings as float32 numpy array."""
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# RAG generator
# ---------------------------------------------------------------------------


class EnhancedRagGenerator:
    """
    Retrieval-Augmented Generator.

    Selects backend at construction time:
      • LOCAL_GGUF_MODEL set  →  llama.cpp (local file, no HF download)
      • LOCAL_GGUF_MODEL unset →  HuggingFace Transformers
    """

    def __init__(
        self,
        mcfg: Optional[ModelConfig] = None,
        rcfg: Optional[RetrieverConfig] = None,
    ) -> None:
        self.mcfg = mcfg or ModelConfig()
        self.rcfg = rcfg or RetrieverConfig()
        self.indexer = DocumentIndexer(self.rcfg)

        # These are set by _load_models().
        self.llama_cpp_model: Optional["Llama"] = None  # type: ignore[type-arg]
        self.model = None
        self.tokenizer = None

        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        if self.mcfg.use_local_gguf:
            self._load_gguf()
        else:
            self._load_hf()

    def _load_gguf(self) -> None:
        """Load a local GGUF file via llama.cpp – HF is never touched."""
        path = LOCAL_GGUF_MODEL
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"LOCAL_GGUF_MODEL path not found: {path}\n"
                "Set LOCAL_GGUF_MODEL to the absolute path of a valid .gguf file."
            )

        n_threads = LLAMA_CPP_THREADS if LLAMA_CPP_THREADS > 0 else None
        logger.info("Loading local GGUF via llama.cpp: %s", path)
        logger.info(
            "  n_gpu_layers=%d  n_threads=%s",
            LLAMA_CPP_N_GPU_LAYERS,
            n_threads if n_threads is not None else "auto",
        )

        self.llama_cpp_model = Llama(  # type: ignore[call-arg]
            model_path=path,
            n_ctx=4096,
            n_threads=n_threads,
            n_gpu_layers=LLAMA_CPP_N_GPU_LAYERS,
            logits_all=False,
            use_mlock=False,
            verbose=False,
        )
        # Explicitly ensure HF objects are absent.
        self.model = None
        self.tokenizer = None
        logger.info("GGUF model loaded successfully.")

    def _load_hf(self) -> None:
        """Load an HF/Transformers model (default path)."""
        model_name = self.mcfg.llm_name
        logger.info("Loading HF tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )

        logger.info("Loading HF model: %s", model_name)
        load_kwargs: dict = {
            "device_map": self.mcfg.device_map,
        }

        if self.mcfg.load_in_4bit and _bitsandbytes_available():
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            load_kwargs["quantization_config"] = bnb_cfg
            logger.info("  Using 4-bit quantisation (bitsandbytes).")
        else:
            # Use `dtype` (not the deprecated `torch_dtype`) to set precision.
            load_kwargs["dtype"] = torch.float16
            logger.info("  Using fp16 (no 4-bit quant).")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        logger.info("HF model loaded successfully.")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_precision_prompt(
        self, query: str, context_chunks: List[str]
    ) -> str:
        """Build an instruction-tuned prompt with retrieved context."""
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"
        return (
            "[INST] You are a precise and helpful assistant. "
            "Use the context below to answer the question accurately. "
            "If the answer is not in the context, say so.\n\n"
            f"### Context:\n{context}\n\n"
            f"### Question:\n{query} [/INST]"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, query: str, **kwargs) -> Tuple[str, List[str]]:
        """
        Non-streaming generation.

        Returns (answer_text, retrieved_chunks).
        """
        chunks = self.indexer.retrieve(query, k=self.rcfg.k_retrieve)
        prompt = self.build_precision_prompt(query, chunks)

        if self.llama_cpp_model is not None:
            response = self._generate_gguf(prompt, **kwargs)
        else:
            response = self._generate_hf(prompt, **kwargs)

        return response, chunks

    def generate_stream(
        self, query: str, **kwargs
    ) -> Tuple[Iterable[str], List[str]]:
        """
        Streaming generation.

        Returns (token_iterable, retrieved_chunks).
        Iterate over the first element to receive tokens as they are produced.
        """
        chunks = self.indexer.retrieve(query, k=self.rcfg.k_retrieve)
        prompt = self.build_precision_prompt(query, chunks)

        if self.llama_cpp_model is not None:
            token_iter = self._stream_gguf(prompt, **kwargs)
        else:
            token_iter = self._stream_hf(prompt, **kwargs)

        return token_iter, chunks

    # ------------------------------------------------------------------
    # Backend: llama.cpp
    # ------------------------------------------------------------------

    def _gguf_kwargs(self, **overrides) -> dict:
        return {
            "max_tokens": overrides.pop("max_new_tokens", self.mcfg.max_new_tokens),
            "temperature": overrides.pop("temperature", self.mcfg.temperature),
            "top_p": overrides.pop("top_p", self.mcfg.top_p),
            "top_k": overrides.pop("top_k", self.mcfg.top_k),
            "repeat_penalty": overrides.pop(
                "repetition_penalty", self.mcfg.repetition_penalty
            ),
            **overrides,
        }

    def _generate_gguf(self, prompt: str, **kwargs) -> str:
        output = self.llama_cpp_model(prompt, **self._gguf_kwargs(**kwargs))
        return output["choices"][0]["text"]

    def _stream_gguf(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        for chunk in self.llama_cpp_model(
            prompt, stream=True, **self._gguf_kwargs(**kwargs)
        ):
            yield chunk["choices"][0]["text"]

    # ------------------------------------------------------------------
    # Backend: HuggingFace Transformers
    # ------------------------------------------------------------------

    def _hf_inputs(self, prompt: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    def _generation_kwargs(self, **overrides) -> dict:
        return {
            "max_new_tokens": overrides.pop("max_new_tokens", self.mcfg.max_new_tokens),
            "temperature": overrides.pop("temperature", self.mcfg.temperature),
            "top_p": overrides.pop("top_p", self.mcfg.top_p),
            "top_k": overrides.pop("top_k", self.mcfg.top_k),
            "repetition_penalty": overrides.pop(
                "repetition_penalty", self.mcfg.repetition_penalty
            ),
            "do_sample": True,
            **overrides,
        }

    def _generate_hf(self, prompt: str, **kwargs) -> str:
        inputs = self._hf_inputs(prompt)
        gen_kwargs = self._generation_kwargs(**kwargs)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        # Strip the prompt tokens from the output.
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][prompt_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _stream_hf(self, prompt: str, **kwargs) -> "TextIteratorStreamer":
        inputs = self._hf_inputs(prompt)
        gen_kwargs = self._generation_kwargs(**kwargs)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer

        def _run():
            with torch.no_grad():
                self.model.generate(**inputs, **gen_kwargs)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return streamer

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def add_documents(self, texts: Iterable[str]) -> None:
        """Index raw text strings for retrieval."""
        self.indexer.add_texts(texts)

    def add_files(self, paths: Iterable[str | Path]) -> None:
        """Index the contents of text files for retrieval."""
        self.indexer.add_files(paths)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bitsandbytes_available() -> bool:
    """Return True if bitsandbytes is importable."""
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# CLI entry-point (basic demo)
# ---------------------------------------------------------------------------


def _cli_demo() -> None:
    """Interactive demo: index a file (if given) and answer questions."""
    import sys

    print("=== Enhanced RAG System demo ===")
    if LOCAL_GGUF_MODEL:
        print(f"Backend : llama.cpp  ({LOCAL_GGUF_MODEL})")
    else:
        mcfg_llm = os.getenv("MCFG_LLM", "mistralai/Mistral-7B-Instruct-v0.2")
        print(f"Backend : HuggingFace  ({mcfg_llm})")
    print()

    rag = EnhancedRagGenerator()

    # Optionally index files passed as CLI arguments.
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        print(f"Indexing {len(files)} file(s) …")
        rag.add_files(files)
        print("Done.\n")

    try:
        while True:
            query = input("Question (Ctrl-C to quit): ").strip()
            if not query:
                continue
            print("Generating …\n")
            token_iter, context = rag.generate_stream(query)
            for token in token_iter:
                print(token, end="", flush=True)
            print("\n")
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    _cli_demo()
