"""
enhanced_rag_system.py
======================
Retrieval-Augmented Generation (RAG) system with two interchangeable LLM backends:

  * **HF/Transformers** (default) – requires a CUDA-capable GPU and ~15 GB VRAM for
    Mistral-7B-Instruct-v0.2.  The model is downloaded automatically the first time.
  * **llama.cpp** (local GGUF) – set the ``LOCAL_GGUF_MODEL`` environment variable to
    the absolute path of a ``.gguf`` file.  When this variable is set, **no HF model is
    downloaded or initialised**.

Environment variables
---------------------
``LOCAL_GGUF_MODEL``
    Absolute path to a ``.gguf`` model file.  When set, the llama.cpp backend is used
    exclusively and the Transformers backend is never touched.

``MCFG_LLM``
    HF model ID used when ``LOCAL_GGUF_MODEL`` is *not* set.
    Default: ``mistralai/Mistral-7B-Instruct-v0.2``

``LLAMA_CPP_THREADS``
    Number of CPU threads for llama.cpp.  ``0`` means *auto-detect* (default).

``LLAMA_CPP_N_GPU_LAYERS``
    Number of model layers to offload to GPU with llama.cpp.  ``0`` = CPU only
    (default).  Requires a llama-cpp-python build with GPU support.

``LLAMA_CPP_N_CTX``
    Context-window size for llama.cpp.  Default: ``4096``.
"""

from __future__ import annotations

import logging
import os
import textwrap
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
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
# Environment toggles (resolved at import time)
# ---------------------------------------------------------------------------
LOCAL_GGUF_MODEL: str = os.getenv("LOCAL_GGUF_MODEL", "").strip()
LLAMA_CPP_THREADS: int = int(os.getenv("LLAMA_CPP_THREADS", "0"))   # 0 = auto
LLAMA_CPP_N_GPU_LAYERS: int = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", "0"))
LLAMA_CPP_N_CTX: int = int(os.getenv("LLAMA_CPP_N_CTX", "4096"))

# ---------------------------------------------------------------------------
# Lazy / optional heavy imports
# ---------------------------------------------------------------------------
if LOCAL_GGUF_MODEL:
    try:
        from llama_cpp import Llama as _LlamaCpp  # type: ignore
    except ImportError as _exc:
        raise ImportError(
            "LOCAL_GGUF_MODEL is set but llama-cpp-python is not installed.\n"
            "Install it with:  pip install llama-cpp-python\n"
            "(For GPU support see https://github.com/abetlen/llama-cpp-python)"
        ) from _exc
    logger.info("llama.cpp backend selected (LOCAL_GGUF_MODEL=%s)", LOCAL_GGUF_MODEL)
else:
    _LlamaCpp = None  # type: ignore
    logger.info(
        "HF/Transformers backend selected (set LOCAL_GGUF_MODEL to use llama.cpp)"
    )

# HF imports are deferred to _load_models() so the module can always be
# imported regardless of whether torch / transformers are installed.

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the LLM component."""

    # HF model ID; ignored when LOCAL_GGUF_MODEL is set.
    llm_name: str = field(
        default_factory=lambda: os.getenv(
            "MCFG_LLM", "mistralai/Mistral-7B-Instruct-v0.2"
        )
    )
    # Whether to use the local llama.cpp backend.
    use_local_gguf: bool = field(default_factory=lambda: bool(LOCAL_GGUF_MODEL))

    # HF generation defaults
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # HF loading options
    load_in_4bit: bool = True   # BitsAndBytes 4-bit quantisation (saves ~75 % VRAM)
    use_flash_attention_2: bool = False  # requires flash-attn package


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval component."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    k_retrieve: int = 5
    # Minimum cosine-similarity threshold (0 = accept everything)
    min_similarity: float = 0.0


# ---------------------------------------------------------------------------
# Document indexer
# ---------------------------------------------------------------------------

class DocumentIndexer:
    """Lightweight FAISS-backed vector store for document chunks.

    Falls back gracefully to a brute-force cosine search when ``faiss`` is not
    installed (slower but dependency-free).
    """

    def __init__(self, cfg: RetrievalConfig) -> None:
        self.cfg = cfg
        self._chunks: List[str] = []
        self._embeddings: Optional["np.ndarray"] = None  # (N, D) float32
        self._index = None  # faiss index or None

        # Deferred imports
        import numpy as np  # type: ignore
        self._np = np

        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading embedding model: %s", cfg.embedding_model)
        self._encoder = SentenceTransformer(cfg.embedding_model)

        try:
            import faiss  # type: ignore
            self._faiss = faiss
            logger.info("FAISS available – using IndexFlatIP for retrieval")
        except ImportError:
            self._faiss = None
            logger.warning(
                "faiss not installed – falling back to brute-force cosine search. "
                "Install with: pip install faiss-cpu"
            )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """Split *text* into overlapping chunks of fixed character length."""
        size = self.cfg.chunk_size
        overlap = self.cfg.chunk_overlap
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end].strip())
            start += size - overlap
        return [c for c in chunks if c]

    def add_documents(self, documents: Iterable[str]) -> int:
        """Chunk, embed and index *documents*.  Returns total chunk count."""
        new_chunks: List[str] = []
        for doc in documents:
            new_chunks.extend(self._chunk_text(doc))
        if not new_chunks:
            return 0

        logger.info("Encoding %d new chunks…", len(new_chunks))
        new_embs = self._encoder.encode(
            new_chunks,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # enables cosine similarity via dot-product
            convert_to_numpy=True,
        ).astype(self._np.float32)

        self._chunks.extend(new_chunks)
        if self._embeddings is None:
            self._embeddings = new_embs
        else:
            self._embeddings = self._np.vstack([self._embeddings, new_embs])

        self._rebuild_index()
        logger.info("Index now contains %d chunks.", len(self._chunks))
        return len(new_chunks)

    def _rebuild_index(self) -> None:
        if self._faiss is None or self._embeddings is None:
            return
        dim = self._embeddings.shape[1]
        index = self._faiss.IndexFlatIP(dim)
        index.add(self._embeddings)
        self._index = index

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return up to *k* ``(chunk, score)`` pairs most relevant to *query*."""
        if not self._chunks:
            return []

        q_emb = self._encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(self._np.float32)

        k = min(k, len(self._chunks))

        if self._index is not None:
            scores, indices = self._index.search(q_emb, k)
            results = [
                (self._chunks[i], float(scores[0][j]))
                for j, i in enumerate(indices[0])
                if i >= 0
            ]
        else:
            # Brute-force fallback
            assert self._embeddings is not None
            sims = (self._embeddings @ q_emb.T).squeeze()
            top_idx = self._np.argsort(sims)[::-1][:k]
            results = [(self._chunks[i], float(sims[i])) for i in top_idx]

        # Apply similarity threshold
        results = [
            (chunk, score)
            for chunk, score in results
            if score >= self.cfg.min_similarity
        ]
        return results


# ---------------------------------------------------------------------------
# Generator (LLM)
# ---------------------------------------------------------------------------

class EnhancedRagGenerator:
    """Orchestrates retrieval + generation for a query.

    Supports two backends, selected at construction time:

    * **llama.cpp** when ``mcfg.use_local_gguf`` is ``True`` (and
      ``LOCAL_GGUF_MODEL`` points to a ``.gguf`` file).
    * **HF/Transformers** otherwise.
    """

    def __init__(
        self,
        mcfg: Optional[ModelConfig] = None,
        rcfg: Optional[RetrievalConfig] = None,
    ) -> None:
        self.mcfg = mcfg or ModelConfig()
        self.rcfg = rcfg or RetrievalConfig()

        self.indexer = DocumentIndexer(self.rcfg)

        # These are set by _load_models()
        self.model = None
        self.tokenizer = None
        self.llama_cpp = None  # llama_cpp.Llama instance

        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load whichever LLM backend is configured."""

        # ── llama.cpp path ────────────────────────────────────────────────
        if self.mcfg.use_local_gguf and LOCAL_GGUF_MODEL:
            if not os.path.isfile(LOCAL_GGUF_MODEL):
                raise FileNotFoundError(
                    f"LOCAL_GGUF_MODEL path not found: {LOCAL_GGUF_MODEL}"
                )
            logger.info("Loading local GGUF via llama.cpp: %s", LOCAL_GGUF_MODEL)
            self.llama_cpp = _LlamaCpp(
                model_path=LOCAL_GGUF_MODEL,
                n_ctx=LLAMA_CPP_N_CTX,
                n_threads=LLAMA_CPP_THREADS if LLAMA_CPP_THREADS > 0 else None,
                n_gpu_layers=LLAMA_CPP_N_GPU_LAYERS,
                logits_all=False,
                use_mlock=False,
                seed=42,
                verbose=False,
            )
            self.tokenizer = None
            self.model = None
            logger.info("llama.cpp model loaded successfully.")
            return  # ← HF path is completely skipped

        # ── HF / Transformers path ────────────────────────────────────────
        # These heavy imports happen only when HF backend is actually needed.
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                TextIteratorStreamer,
            )
        except ImportError as exc:
            raise ImportError(
                "HF/Transformers backend requires torch and transformers.\n"
                "Install with: pip install torch transformers\n"
                "Or use LOCAL_GGUF_MODEL to switch to the llama.cpp backend."
            ) from exc

        model_id = self.mcfg.llm_name
        logger.info("Loading HF model: %s", model_id)

        # Build device / dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_dtype = torch.float16 if device == "cuda" else torch.float32

        # 4-bit quantisation (BitsAndBytes) – dramatically reduces VRAM usage.
        bnb_cfg = None
        if self.mcfg.load_in_4bit and device == "cuda":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("4-bit BitsAndBytes quantisation enabled.")

        extra_kwargs: dict = {}
        if self.mcfg.use_flash_attention_2:
            extra_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            # Use `dtype` (not the deprecated `torch_dtype`) to silence warning
            dtype=compute_dtype if bnb_cfg is None else None,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            **extra_kwargs,
        )
        self.model.eval()
        logger.info(
            "HF model loaded on %s (dtype=%s, 4-bit=%s).",
            device,
            compute_dtype,
            self.mcfg.load_in_4bit,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_precision_prompt(
        self,
        query: str,
        retrieved: List[Tuple[str, float]],
    ) -> str:
        """Assemble a RAG prompt from *query* and *retrieved* context chunks."""
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved, 1):
            context_parts.append(f"[{i}] (relevance={score:.3f})\n{chunk}")
        context = "\n\n".join(context_parts) if context_parts else "No context found."

        return textwrap.dedent(f"""\
            You are a precise and helpful assistant.

            Use the following retrieved context to answer the question accurately.
            If the context does not contain sufficient information, say so clearly.

            ### Context
            {context}

            ### Question
            {query}

            ### Answer
        """)

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        query: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[Iterable[str], List[Tuple[str, float]]]:
        """Return a ``(token_stream, retrieved_chunks)`` tuple.

        The *token_stream* is an iterable that yields text tokens one-by-one
        as they are generated.
        """
        max_tokens = max_new_tokens or self.mcfg.max_new_tokens
        temp = temperature if temperature is not None else self.mcfg.temperature

        final_results = self.indexer.retrieve(query, k=self.rcfg.k_retrieve)
        prompt = self.build_precision_prompt(query, final_results)

        # ── llama.cpp streaming ───────────────────────────────────────────
        if self.llama_cpp is not None:
            def _llama_stream() -> Generator[str, None, None]:
                for out in self.llama_cpp(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=self.mcfg.top_p,
                    top_k=self.mcfg.top_k,
                    repeat_penalty=self.mcfg.repetition_penalty,
                    stream=True,
                ):
                    yield out["choices"][0]["text"]

            return _llama_stream(), final_results

        # ── HF / Transformers streaming ───────────────────────────────────
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "No LLM backend is loaded. "
                "Either set LOCAL_GGUF_MODEL or ensure Transformers is installed."
            )

        import torch  # type: ignore
        from transformers import TextIteratorStreamer  # type: ignore

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=self.mcfg.top_p,
            top_k=self.mcfg.top_k,
            repetition_penalty=self.mcfg.repetition_penalty,
            do_sample=temp > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Run generation in a background thread so the main thread can
        # iterate over the streamer without blocking.
        gen_thread = threading.Thread(
            target=self.model.generate, kwargs=gen_kwargs, daemon=True
        )
        gen_thread.start()

        return streamer, final_results

    # ------------------------------------------------------------------
    # Convenience: non-streaming query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Generate a full answer (non-streaming).

        Returns ``(answer_text, retrieved_chunks)``.
        """
        stream, chunks = self.generate_stream(
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        answer = "".join(stream)
        return answer.strip(), chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_documents(self, documents: Iterable[str]) -> int:
        """Chunk, embed, and index *documents*.  Returns total chunk count."""
        return self.indexer.add_documents(documents)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _cli_demo() -> None:
    """Simple interactive demo that exercises both backends."""
    import sys

    print("=" * 60)
    print("Enhanced RAG System – interactive demo")
    print("=" * 60)
    if LOCAL_GGUF_MODEL:
        print(f"Backend : llama.cpp  ({LOCAL_GGUF_MODEL})")
    else:
        print(f"Backend : HF/Transformers ({os.getenv('MCFG_LLM', 'mistralai/Mistral-7B-Instruct-v0.2')})")
    print()

    rag = EnhancedRagGenerator()

    # Seed with a tiny knowledge base so the demo is self-contained.
    sample_docs = [
        "The Eiffel Tower is located in Paris, France. It was designed by Gustave Eiffel "
        "and completed in 1889. It stands 330 metres tall.",
        "Python is a high-level, general-purpose programming language. It was created by "
        "Guido van Rossum and first released in 1991. Python emphasises code readability.",
        "The speed of light in a vacuum is approximately 299,792,458 metres per second.",
    ]
    rag.add_documents(sample_docs)

    questions = [
        "How tall is the Eiffel Tower?",
        "Who created Python?",
    ]

    for q in questions:
        print(f"Q: {q}")
        print("A: ", end="", flush=True)
        stream, _ = rag.generate_stream(q)
        for token in stream:
            print(token, end="", flush=True)
        print("\n")

    if sys.stdin.isatty():
        print("Type your question (Ctrl-C to quit):")
        try:
            while True:
                q = input("> ").strip()
                if not q:
                    continue
                print("A: ", end="", flush=True)
                stream, chunks = rag.generate_stream(q)
                for token in stream:
                    print(token, end="", flush=True)
                print()
                print(f"  [retrieved {len(chunks)} chunk(s)]")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")


if __name__ == "__main__":
    _cli_demo()
