"""
CAN — Precision-Optimized RAG System
Enhanced for maximum knowledge accuracy and retrieval precision
"""

import os
import sys
import json
import glob
import shutil
import hashlib
import logging
import threading
import traceback
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Iterable, Set
from datetime import datetime
from pathlib import Path
from enum import Enum
import warnings

import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sentence_transformers import SentenceTransformer, CrossEncoder
import qdarkstyle

# Env toggles and runtime settings
LOCAL_GGUF_MODEL = os.getenv("LOCAL_GGUF_MODEL", "").strip()
LLAMA_CPP_THREADS = int(os.getenv("LLAMA_CPP_THREADS", "0"))  # 0 = auto
LLAMA_CPP_N_GPU_LAYERS = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", "0"))  # >0 only if built with GPU
HF_LLM_NAME_DEFAULT = "mistralai/Mistral-7B-Instruct-v0.2"  # public default


def _resolve_threads() -> int:
    """Return LLAMA_CPP_THREADS if >0, else os.cpu_count() (fallback 4)."""
    if LLAMA_CPP_THREADS > 0:
        return LLAMA_CPP_THREADS
    return os.cpu_count() or 4


HF_LLM_NAME = os.getenv("MCFG_LLM", HF_LLM_NAME_DEFAULT)

# Optional llama.cpp import
Llama = None
if LOCAL_GGUF_MODEL:
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as e:
        raise ImportError(
            "LOCAL_GGUF_MODEL is set but llama-cpp-python is not installed. "
            "Install with: pip install llama-cpp-python"
        ) from e

# PyMuPDF — optional, preferred PDF backend (pip install PyMuPDF)
try:
    import fitz  # type: ignore  # noqa: F401
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None  # type: ignore

# python-docx — DOCX parsing (pip install python-docx). NOT the broken 'docx' package.
try:
    from docx import Document as DocxDocument  # type: ignore
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    DocxDocument = None  # type: ignore

# Markdown/HTML helpers — optional but recommended
try:
    import markdown  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False
    markdown = None  # type: ignore
    BeautifulSoup = None  # type: ignore

try:
    import html2text  # type: ignore
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False
    html2text = None  # type: ignore

# Token counting — optional; falls back to word-count when absent
try:
    import tiktoken  # type: ignore
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    tiktoken = None  # type: ignore

from PyQt5 import QtCore, QtGui, QtWidgets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("can_rag.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------- Enhanced Configuration ---------------- #
class ChunkStrategy(Enum):
    SEMANTIC = "semantic"
    FIXED = "fixed"
    SLIDING = "sliding"
    HYBRID = "hybrid"


@dataclass
class ModelConfig:
    llm_name: str = HF_LLM_NAME
    embed_name: str = "sentence-transformers/all-mpnet-base-v2"
    embed_fallback: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    use_local_gguf: bool = bool(LOCAL_GGUF_MODEL)


@dataclass
class RagConfig:
    raw_dir: str = "data/raw"
    chunks_dir: str = "data/chunks"
    index_path: str = "data/index.faiss"
    meta_path: str = "data/meta.json"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 32
    max_chunk_size: int = 768
    k_retrieve: int = 10
    k_final: int = 5
    chunk_strategy: ChunkStrategy = ChunkStrategy.HYBRID
    semantic_threshold: float = 0.85
    deduplicate: bool = True
    min_similarity: float = 0.95
    enable_reranking: bool = True
    query_expansion: bool = True
    enable_hyde: bool = True
    cache_embeddings: bool = True
    version: str = "1.0.0"


@dataclass
class DocumentMetadata:
    source_path: str
    file_type: str
    file_size: int
    created_date: datetime
    modified_date: datetime
    md5_hash: str
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: str = "en"
    title: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    chunk_id: str
    document_id: str
    start_position: int
    end_position: int
    token_count: int
    sentence_count: int
    paragraph_id: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    embedding_model: str = ""


# ---------------- Enhanced Text Processing ---------------- #
class TextProcessor:
    def __init__(self, cfg: RagConfig):
        self.cfg = cfg
        # Gracefully handle tiktoken initialization failures (network issues, etc.)
        self._tiktoken_enc = None
        if HAS_TIKTOKEN:
            try:
                self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                # Fallback to word-based counting if tiktoken fails to initialize
                pass

    def count_tokens(self, text: str) -> int:
        if self._tiktoken_enc is not None:
            return len(self._tiktoken_enc.encode(text))
        # Word-count approximation: ~1.3 tokens per word on average
        return int(len(text.split()) * 1.3)

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def split_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def semantic_chunking(self, text: str, metadata: DocumentMetadata) -> List[Tuple[str, ChunkMetadata]]:
        chunks = []
        paragraphs = self.split_paragraphs(text)
        current_chunk, current_tokens, chunk_start = [], 0, 0

        for para_idx, paragraph in enumerate(paragraphs):
            para_tokens = self.count_tokens(paragraph)

            if para_tokens > self.cfg.max_chunk_size:
                sentences = self.split_sentences(paragraph)
                for sent in sentences:
                    sent_tokens = self.count_tokens(sent)
                    self._add_to_chunks(sent, sent_tokens, chunks, metadata, para_idx, chunk_start)
                    chunk_start += len(sent)
                continue

            if current_tokens + para_tokens <= self.cfg.chunk_size:
                current_chunk.append(paragraph)
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunk_tokens = self.count_tokens(chunk_text)
                    if chunk_tokens >= self.cfg.min_chunk_size:
                        chunk_meta = self._create_chunk_metadata(
                            chunk_text, metadata, chunk_start, chunk_start + len(chunk_text), para_idx, chunk_tokens
                        )
                        chunks.append((chunk_text, chunk_meta))
                current_chunk = [paragraph]
                current_tokens = para_tokens
                chunk_start += sum(len(p) for p in current_chunk[:-1])

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_tokens = self.count_tokens(chunk_text)
            if chunk_tokens >= self.cfg.min_chunk_size:
                chunk_meta = self._create_chunk_metadata(
                    chunk_text, metadata, chunk_start, chunk_start + len(chunk_text), len(paragraphs) - 1, chunk_tokens
                )
                chunks.append((chunk_text, chunk_meta))

        return chunks

    def _add_to_chunks(self, text: str, token_count: int, chunks: List, metadata: DocumentMetadata, position: int, start_pos: int):
        if token_count >= self.cfg.min_chunk_size:
            chunk_meta = self._create_chunk_metadata(text, metadata, start_pos, start_pos + len(text), position, token_count)
            chunks.append((text, chunk_meta))

    def _create_chunk_metadata(self, text: str, doc_meta: DocumentMetadata, start_pos: int, end_pos: int, para_idx: int, token_count: Optional[int] = None) -> ChunkMetadata:
        chunk_id = hashlib.md5(f"{doc_meta.md5_hash}:{start_pos}:{end_pos}".encode()).hexdigest()
        # Reuse token_count if already computed to avoid redundant tokenization
        if token_count is None:
            token_count = self.count_tokens(text)
        return ChunkMetadata(
            chunk_id=chunk_id,
            document_id=doc_meta.md5_hash,
            start_position=start_pos,
            end_position=end_pos,
            token_count=token_count,
            sentence_count=len(self.split_sentences(text)),
            paragraph_id=para_idx,
            page_number=None,
            section_title=self._extract_section_title(text),
            keywords=self._extract_keywords(text),
            embedding_model=self.cfg.version,
        )

    def _extract_section_title(self, text: str) -> Optional[str]:
        lines = text.split("\n")
        for line in lines[:3]:
            if len(line) < 100 and line.strip().endswith(":"):
                return line.strip()
            if line.isupper() and len(line) < 80:
                return line.strip()
        return None

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
        freq: Dict[str, int] = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        stopwords = {"this", "that", "with", "from", "have", "were", "they", "what", "when", "which", "would", "could", "should"}
        keywords = [w for w in sorted(freq.items(), key=lambda x: x[1], reverse=True) if w[0] not in stopwords][:max_keywords]
        return [k[0] for k in keywords]


# ---------------- Enhanced Document Indexer ---------------- #
class EnhancedDocumentIndexer:
    def __init__(self, embed_model: SentenceTransformer, cfg: RagConfig):
        self.embed = embed_model
        # Extract embedding model name once at initialization
        _mcd = getattr(embed_model, "_model_card_data", None)
        self._embed_model_name: str = (
            getattr(_mcd, "model_name", None) or cfg.version
        )
        self.cfg = cfg
        self.text_processor = TextProcessor(cfg)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        # In-memory index cache — invalidated when ingest() rebuilds the index
        self._index_cache: Optional[faiss.Index] = None
        self._chunks_cache: Optional[List[str]] = None
        self._meta_cache: Optional[Dict] = None
        os.makedirs(cfg.raw_dir, exist_ok=True)
        os.makedirs(cfg.chunks_dir, exist_ok=True)

    def _read_file(self, path: str) -> Tuple[str, DocumentMetadata]:
        path_obj = Path(path)
        file_type = path_obj.suffix.lower()
        stat = path_obj.stat()
        created_date = datetime.fromtimestamp(stat.st_ctime)
        modified_date = datetime.fromtimestamp(stat.st_mtime)

        with open(path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        text = ""
        page_count = None
        word_count = None
        title = None
        author = None

        try:
            if file_type == ".pdf":
                text, page_count, title, author = self._read_pdf(path)
            elif file_type == ".docx":
                text, word_count = self._read_docx(path)
            elif file_type == ".md":
                text = self._read_markdown(path)
            elif file_type in [".html", ".htm"]:
                text = self._read_html(path)
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            text = self.text_processor.clean_text(text)
            word_count = word_count or len(text.split())

        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise

        metadata = DocumentMetadata(
            source_path=path,
            file_type=file_type,
            file_size=stat.st_size,
            created_date=created_date,
            modified_date=modified_date,
            md5_hash=file_hash,
            page_count=page_count,
            word_count=word_count,
            title=title or path_obj.stem,
            author=author,
            tags=self._extract_tags(path, text),
        )
        return text, metadata

    def _read_pdf(self, path: str) -> Tuple[str, int, Optional[str], Optional[str]]:
        text_parts, title, author = [], None, None
        page_count = 0

        if HAS_PYMUPDF:
            try:
                doc = fitz.open(path)  # type: ignore[union-attr]
                page_count = len(doc)
                pdf_meta = doc.metadata
                title = pdf_meta.get("title") or None
                author = pdf_meta.get("author") or None
                for page_num in range(page_count):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                doc.close()
                return "\n\n".join(text_parts), page_count, title, author
            except Exception as e:
                logger.warning(f"PyMuPDF failed for {path}, falling back to PyPDF2: {e}")
                text_parts.clear()

        # Fallback: PyPDF2
        from PyPDF2 import PdfReader
        with open(path, "rb") as f:
            reader = PdfReader(f)
            page_count = len(reader.pages)
            pdf_meta = reader.metadata
            title = pdf_meta.get("/Title") if pdf_meta else None
            author = pdf_meta.get("/Author") if pdf_meta else None
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
        return "\n\n".join(text_parts), page_count, title, author

    def _read_docx(self, path: str) -> Tuple[str, int]:
        if not HAS_DOCX or DocxDocument is None:
            raise ImportError(
                "python-docx is required to read .docx files. "
                "Install with: pip install python-docx  (NOT the broken 'docx' package)"
            )
        doc = DocxDocument(path)
        text_parts = []
        word_count = 0
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
                word_count += len(para.text.split())
        return "\n".join(text_parts), word_count

    def _read_markdown(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            md_text = f.read()
        if HAS_MARKDOWN and markdown is not None and BeautifulSoup is not None:
            html = markdown.markdown(md_text)
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text()
        # Fallback: strip markdown syntax with regex
        text = re.sub(r"#{1,6}\s*", "", md_text)
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        return text.strip()

    def _read_html(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        if HAS_HTML2TEXT and html2text is not None:
            converter = html2text.HTML2Text()
            converter.ignore_links = False
            converter.ignore_images = True
            return converter.handle(html)
        if HAS_MARKDOWN and BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text()
        # Last resort: strip tags with regex
        return re.sub(r"<[^>]+>", " ", html).strip()

    def _extract_tags(self, path: str, text: str) -> List[str]:
        tags = []
        parent = Path(path).parent.name
        if parent and parent != ".":
            tags.append(parent.lower())
        for line in text.split("\n")[:10]:
            if line.lower().startswith("tags:"):
                tag_part = line[5:].strip()
                tags.extend([t.strip().lower() for t in tag_part.split(",")])
                break
        return list(set(tags))

    def _deduplicate_chunks(self, chunks: List[Tuple[str, Any]], embeddings: np.ndarray) -> Tuple[List, np.ndarray]:
        if not self.cfg.deduplicate or len(chunks) < 2:
            return chunks, embeddings

        # Pass 1: exact-text hash dedup (O(n)) — cheap and catches copy-paste duplicates
        seen_hashes: set = set()
        pass1_chunks, pass1_embs = [], []
        for (chunk_text, meta), emb in zip(chunks, embeddings):
            h = hashlib.md5(chunk_text.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                pass1_chunks.append((chunk_text, meta))
                pass1_embs.append(emb)

        if len(pass1_chunks) < 2:
            return pass1_chunks, np.array(pass1_embs)

        # Pass 2: near-duplicate embedding similarity using vectorized operations
        embs_arr = np.array(pass1_embs, dtype=np.float32)
        # NOTE: Embeddings are typically encoded with normalize_embeddings=True upstream,
        # so this L2-normalization is usually a no-op. We still normalize here as a
        # defensive check to ensure cosine similarity is well-defined even if callers
        # change the embedding configuration in the future.
        norms = np.linalg.norm(embs_arr, axis=1, keepdims=True)
        embs_arr = embs_arr / np.maximum(norms, 1e-8)

        # Compute similarity matrix using batched matrix multiplication (much faster than loop)
        sim_matrix = embs_arr @ embs_arr.T
        np.fill_diagonal(sim_matrix, 0.0)

        # Vectorized deduplication: mark duplicates based on similarity threshold
        unique_mask = np.ones(len(pass1_chunks), dtype=bool)
        for i in range(len(pass1_chunks)):
            if not unique_mask[i]:
                continue
            # Find all duplicates of i in one operation
            duplicates = np.where((unique_mask) & (sim_matrix[i] > self.cfg.min_similarity))[0]
            # Mark later duplicates as non-unique (keep first occurrence)
            unique_mask[duplicates[duplicates > i]] = False

        unique_chunks = [c for c, keep in zip(pass1_chunks, unique_mask) if keep]
        unique_embs = embs_arr[unique_mask]
        return unique_chunks, unique_embs

    def ingest(self, progress_cb=None) -> Dict[str, Any]:
        files = glob.glob(os.path.join(self.cfg.raw_dir, "**", "*.*"), recursive=True)
        files = [f for f in files if Path(f).suffix.lower() in [".pdf", ".txt", ".md", ".html", ".htm", ".docx", ".py", ".js", ".java", ".cpp"]]
        all_chunks, all_meta, processed_files, failed_files = [], [], [], []
        total_files = len(files)

        for file_index, file_path in enumerate(files):
            try:
                text, doc_metadata = self._read_file(file_path)
                chunks_with_meta = self.text_processor.semantic_chunking(text, doc_metadata)
                for chunk_text, chunk_meta in chunks_with_meta:
                    all_chunks.append(chunk_text)
                    all_meta.append({"chunk_meta": chunk_meta.__dict__, "doc_meta": doc_metadata.__dict__})
                processed_files.append({"path": file_path, "chunks": len(chunks_with_meta), "metadata": doc_metadata.__dict__})
            except Exception as e:
                failed_files.append({"path": file_path, "error": str(e)})
            if progress_cb:
                progress_cb(int((file_index + 1) / max(total_files, 1) * 50))

        if not all_chunks:
            if progress_cb:
                progress_cb(0)
            return {"total_chunks": 0, "processed": processed_files, "failed": failed_files}

        embeddings = self.embed.encode(all_chunks, batch_size=16, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        if progress_cb:
            progress_cb(75)

        if self.cfg.deduplicate:
            dedup_chunks, dedup_emb = self._deduplicate_chunks(list(zip(all_chunks, all_meta)), embeddings)
            all_chunks, all_meta, embeddings = zip(*[(c, m, e) for (c, m), e in zip(dedup_chunks, dedup_emb)])
            all_chunks, all_meta, embeddings = list(all_chunks), list(all_meta), np.array(embeddings)

        if self.cfg.cache_embeddings:
            for chunk_text, embedding in zip(all_chunks, embeddings):
                cache_key = hashlib.md5(chunk_text.encode()).hexdigest()
                self.embedding_cache[cache_key] = embedding

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, self.cfg.index_path)

        metadata = {
            "version": self.cfg.version,
            "created": datetime.now().isoformat(),
            "embedding_model": self._embed_model_name,
            "total_chunks": len(all_chunks),
            "total_documents": len(processed_files),
            "dimension": dimension,
            "chunks": all_meta,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "config": self.cfg.__dict__,
        }

        with open(self.cfg.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        chunks_file = os.path.join(self.cfg.chunks_dir, "chunks.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)

        if progress_cb:
            progress_cb(100)

        # Invalidate in-memory cache so the next retrieve loads the new index
        self.invalidate_cache()

        return {
            "total_chunks": len(all_chunks),
            "processed_files": len(processed_files),
            "failed_files": len(failed_files),
            "metadata": metadata,
        }

    def load(self) -> Tuple[faiss.Index, List[str], Dict]:
        # Return cached values if available (avoids disk I/O on every retrieve call)
        if self._index_cache is not None and self._chunks_cache is not None and self._meta_cache is not None:
            return self._index_cache, self._chunks_cache, self._meta_cache

        if not os.path.exists(self.cfg.index_path):
            raise FileNotFoundError(f"Index not found at {self.cfg.index_path}")
        if not os.path.exists(self.cfg.meta_path):
            raise FileNotFoundError(f"Metadata not found at {self.cfg.meta_path}")
        index = faiss.read_index(self.cfg.index_path)
        with open(self.cfg.meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        chunks_file = os.path.join(self.cfg.chunks_dir, "chunks.json")
        chunks: List[str] = []
        if os.path.exists(chunks_file):
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

        # Populate cache
        self._index_cache = index
        self._chunks_cache = chunks
        self._meta_cache = metadata
        return index, chunks, metadata

    def invalidate_cache(self) -> None:
        """Call after a new index is written to disk so the next retrieve reloads it."""
        self._index_cache = None
        self._chunks_cache = None
        self._meta_cache = None

    def _dedup_results_by_text(self, results: List[Tuple[str, Any, float]]) -> List[Tuple[str, Any, float]]:
        """Deduplicate retrieval results by chunk text, preserving order.

        Each element of *results* is a ``(chunk_text, metadata, score)`` tuple.
        Uses Python's native string hashing — faster than computing md5 per result.
        """
        seen_chunk_texts: Set[str] = set()
        unique_results: List[Tuple[str, Any, float]] = []
        for result in results:
            if result[0] not in seen_chunk_texts:
                seen_chunk_texts.add(result[0])
                unique_results.append(result)
        return unique_results

    def retrieve(self, query: str, k: int = None, query_expansion: bool = None) -> List[Tuple[str, Dict, float]]:
        if k is None:
            k = self.cfg.k_retrieve
        if query_expansion is None:
            query_expansion = self.cfg.query_expansion

        index, chunks, metadata = self.load()
        expanded_queries = [query]
        if query_expansion:
            expanded_queries.extend(self._expand_query(query))
        all_results = []
        seen_chunks = set()

        for expanded_query in expanded_queries:
            query_embedding = self.embed.encode([expanded_query], convert_to_numpy=True, normalize_embeddings=True)
            scores, indices = index.search(query_embedding, min(k * 2, len(chunks)))
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(chunks) and idx not in seen_chunks:
                    chunk_text = chunks[idx]
                    chunk_meta = metadata["chunks"][idx] if idx < len(metadata["chunks"]) else {}
                    all_results.append((chunk_text, chunk_meta, float(score)))
                    seen_chunks.add(idx)

        all_results.sort(key=lambda x: x[2], reverse=True)
        return self._dedup_results_by_text(all_results)[:k]

    def _expand_query(self, query: str) -> List[str]:
        expansions = []
        if not query.lower().startswith(("how", "what", "why", "when", "where", "which")):
            expansions.append(f"how to {query}")
        expansions.append(f"explain {query}")
        expansions.append(f"examples of {query}")
        tech_terms = {
            "code": ["implementation", "program", "algorithm"],
            "error": ["bug", "issue", "problem", "exception"],
            "function": ["method", "procedure", "routine"],
            "class": ["type", "object", "structure"],
            "api": ["interface", "endpoint", "service"],
        }
        for term, alternatives in tech_terms.items():
            if term in query.lower():
                for alt in alternatives:
                    expansions.append(query.lower().replace(term, alt))
        return list(set(expansions))


# ---------------- Enhanced RAG Generator ---------------- #
class EnhancedRagGenerator:
    def __init__(self, mcfg: ModelConfig, rcfg: RagConfig):
        self.mcfg = mcfg
        self.rcfg = rcfg
        self.llama_cpp = None
        self.tokenizer = None
        self.model = None
        self._load_models()
        self.indexer = EnhancedDocumentIndexer(self.embed_model, rcfg)
        self.reranker = None
        if rcfg.enable_reranking:
            try:
                self.reranker = CrossEncoder(mcfg.reranker_name, device=mcfg.device)
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
                self.rcfg.enable_reranking = False

    def _load_models(self):
        logger.info("Loading models...")
        # Embeddings
        try:
            self.embed_model = SentenceTransformer(self.mcfg.embed_name, device=self.mcfg.device)
            logger.info(f"Loaded embedding model: {self.mcfg.embed_name}")
        except Exception as e:
            logger.warning(f"Primary embedding load failed: {e}, fallback to {self.mcfg.embed_fallback}")
            self.embed_model = SentenceTransformer(self.mcfg.embed_fallback, device=self.mcfg.device)

        # Local GGUF path
        if self.mcfg.use_local_gguf and LOCAL_GGUF_MODEL:
            if not Llama:
                raise ImportError("llama-cpp-python not available while LOCAL_GGUF_MODEL is set.")
            if not os.path.isfile(LOCAL_GGUF_MODEL):
                raise FileNotFoundError(f"LOCAL_GGUF_MODEL path not found: {LOCAL_GGUF_MODEL}")
            threads = _resolve_threads()
            logger.info(
                f"Using llama.cpp with threads={threads}, n_gpu_layers={LLAMA_CPP_N_GPU_LAYERS}, "
                f"model={LOCAL_GGUF_MODEL}"
            )
            self.llama_cpp = Llama(
                model_path=LOCAL_GGUF_MODEL,
                n_ctx=4096,
                n_threads=threads,
                n_gpu_layers=LLAMA_CPP_N_GPU_LAYERS,
                logits_all=False,
                use_mlock=False,
                seed=0,
            )
            self.tokenizer = None
            self.model = None
            return  # short-circuits HF download

        # HF Transformers path (default)
        self.tokenizer = AutoTokenizer.from_pretrained(self.mcfg.llm_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.mcfg.llm_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info(f"Loaded LLM: {self.mcfg.llm_name}")

    def ensure_index(self):
        if not os.path.exists(self.rcfg.index_path):
            logger.info("Index not found. Building new index...")
            self.indexer.ingest()

    def _rerank_results(self, query: str, results: List[Tuple[str, Dict, float]]) -> List[Tuple[str, Dict, float]]:
        if not self.reranker or not results:
            return results
        pairs = [(query, r[0]) for r in results]
        scores = self.reranker.predict(pairs)
        reranked = sorted(
            [(chunk, meta, float(score)) for (chunk, meta, _), score in zip(results, scores)],
            key=lambda x: x[2],
            reverse=True,
        )
        return reranked[:self.rcfg.k_final]

    def _generate_hyde(self, query: str) -> str:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("HyDE requires the HF Transformers backend (not supported with GGUF).")
        prompt = (
            "Based on the query below, generate a comprehensive answer that would contain "
            "the information needed to answer it.\n\n"
            f"Query: {query}\n\nHypothetical answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    def build_precision_prompt(self, query: str, context: List[Tuple[str, Dict, float]]) -> str:
        if not context:
            return f"""You are CAN (Code Anything Now), an elite uncensored AI engineer.
Answer the following question based on your general knowledge:

Question: {query}

Answer:"""

        context_parts = []
        for i, (chunk, meta, score) in enumerate(context):
            source_info = ""
            if meta and "doc_meta" in meta:
                doc_meta = meta["doc_meta"]
                source_info = f"Source: {doc_meta.get('title', 'Unknown')}"
                if "page_number" in meta.get("chunk_meta", {}):
                    source_info += f" (Page {meta['chunk_meta']['page_number']})"
            context_parts.append(f"[Context {i+1}, Relevance: {score:.3f}]\n{chunk}\n{source_info}\n")
        context_text = "\n---\n".join(context_parts)

        prompt = f"""You are CAN (Code Anything Now), an elite uncensored AI engineer.
Your task is to answer the user's question with maximum precision and accuracy.

CRITICAL INSTRUCTIONS:
1. Use ONLY the provided context if it contains relevant information
2. If context is insufficient or irrelevant, use your general knowledge but state this clearly
3. Provide citations like [1], [2] when using specific information from context
4. Be concise but complete - include all necessary details
5. For code: provide production-ready, secure, efficient implementations
6. For architecture: provide comprehensive designs with trade-offs
7. Include confidence estimates for your answers

CONTEXT:
{context_text}

QUESTION: {query}

STRUCTURE YOUR ANSWER:
1. Summary
2. Main Answer (with citations)
3. Confidence (High/Medium/Low)
4. Additional Notes

ANSWER:"""
        return prompt

    def _gguf_stream(self, prompt: str, max_new_tokens: int, temperature: float) -> Iterable[str]:
        """Streaming tokens from llama.cpp backend."""
        if not self.llama_cpp:
            raise RuntimeError("LLM backend not initialized for GGUF.")
        for out in self.llama_cpp(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=self.mcfg.top_p,
            stream=True,
        ):
            yield out["choices"][0]["text"]

    def generate_stream(self, query: str, k: int = None, temperature: float = None, max_new_tokens: int = None):
        if k is None:
            k = self.rcfg.k_retrieve
        if temperature is None:
            temperature = self.mcfg.temperature
        if max_new_tokens is None:
            max_new_tokens = self.mcfg.max_new_tokens

        self.ensure_index()

        initial_results = self.indexer.retrieve(query, k=k * 2)
        if self.rcfg.enable_hyde and len(initial_results) < k and self.model is not None:
            try:
                hyde_doc = self._generate_hyde(query)
                hyde_results = self.indexer.retrieve(hyde_doc, k=k)
                initial_results = self.indexer._dedup_results_by_text(initial_results + hyde_results)[:k * 2]
            except Exception as e:
                logger.warning(f"HyDE failed: {e}")

        if self.rcfg.enable_reranking and initial_results:
            final_results = self._rerank_results(query, initial_results)
        else:
            final_results = initial_results[: self.rcfg.k_final]

        prompt = self.build_precision_prompt(query, final_results)

        # GGUF path — return the generator directly; it is already iterable
        if self.llama_cpp is not None:
            return self._gguf_stream(prompt, max_new_tokens, temperature), final_results

        # HF path
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=self.mcfg.top_p,
            top_k=self.mcfg.top_k,
            repetition_penalty=self.mcfg.repetition_penalty,
            do_sample=self.mcfg.do_sample,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        return streamer, final_results


# ---------------- Workers ---------------- #
class AskWorker(QtCore.QThread):
    tokenSignal = QtCore.pyqtSignal(str)
    ctxSignal = QtCore.pyqtSignal(list)
    errorSignal = QtCore.pyqtSignal(str)

    def __init__(self, rag: EnhancedRagGenerator, query: str, k: int, temperature: float, max_tokens: int):
        super().__init__()
        self.rag = rag
        self.query = query
        self.k = k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._stop = False

    def run(self):
        try:
            streamer, ctx = self.rag.generate_stream(self.query, self.k, self.temperature, self.max_tokens)
            self.ctxSignal.emit(ctx)
            for token in streamer:
                if self._stop:
                    break
                self.tokenSignal.emit(token)
        except Exception as e:
            self.errorSignal.emit(f"{e}\n{traceback.format_exc()}")

    def stop(self):
        self._stop = True


class IngestWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    done = QtCore.pyqtSignal(int)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, indexer: EnhancedDocumentIndexer):
        super().__init__()
        self.indexer = indexer

    def run(self):
        try:
            result = self.indexer.ingest(progress_cb=self.progress.emit)
            self.done.emit(result.get("total_chunks", 0))
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


# ---------------- UI ---------------- #
class DropArea(QtWidgets.QLabel):
    filesDropped = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__("Drag & drop files/folders here to ingest")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet(
            "border: 2px dashed #4a90e2; padding: 20px; border-radius: 12px; "
            "color: #b8c7e0; background: #0f1624;"
        )
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        self.filesDropped.emit(paths)


class EnhancedMainWindow(QtWidgets.QMainWindow):
    def __init__(self, rag: EnhancedRagGenerator):
        super().__init__()
        self.rag = rag
        self.worker: Optional[AskWorker] = None
        self.setWindowTitle("CAN — Precision RAG Coder")
        self.setMinimumSize(1400, 900)
        self.precision_settings = {
            "query_expansion": rag.rcfg.query_expansion,
            "enable_reranking": rag.rcfg.enable_reranking,
            "enable_hyde": rag.rcfg.enable_hyde,
            "show_confidence": True,
            "show_citations": True,
        }
        self._build_ui()

    def _build_ui(self):
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._build_chat_tab(), "Precision Chat")
        self._tabs.addTab(self._build_settings_tab(), "Advanced Settings")
        self._analytics_tab_idx = self._tabs.addTab(self._build_analytics_tab(), "Analytics")
        self._tabs.addTab(self._build_about_tab(), "About")
        # Auto-refresh analytics when the tab is selected
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self._tabs)
        self.status = QtWidgets.QStatusBar()
        self.status.showMessage("Ready — Precision Mode Active")
        self.setStatusBar(self.status)

    def _on_tab_changed(self, index: int) -> None:
        if index == self._analytics_tab_idx:
            self._refresh_analytics()

    def _build_chat_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        precision_panel = QtWidgets.QHBoxLayout()
        self.expansion_cb = QtWidgets.QCheckBox("Query Expansion")
        self.expansion_cb.setChecked(self.precision_settings["query_expansion"])
        self.expansion_cb.stateChanged.connect(self._toggle_query_expansion)
        self.rerank_cb = QtWidgets.QCheckBox("Reranking")
        self.rerank_cb.setChecked(self.precision_settings["enable_reranking"])
        self.rerank_cb.stateChanged.connect(self._toggle_reranking)
        self.hyde_cb = QtWidgets.QCheckBox("HyDE")
        self.hyde_cb.setChecked(self.precision_settings["enable_hyde"])
        self.hyde_cb.stateChanged.connect(self._toggle_hyde)
        self.confidence_cb = QtWidgets.QCheckBox("Show Confidence")
        self.confidence_cb.setChecked(self.precision_settings["show_confidence"])
        self.citations_cb = QtWidgets.QCheckBox("Show Citations")
        self.citations_cb.setChecked(self.precision_settings["show_citations"])
        for widget in [self.expansion_cb, self.rerank_cb, self.hyde_cb, self.confidence_cb, self.citations_cb]:
            precision_panel.addWidget(widget)
        precision_panel.addStretch()
        v.addLayout(precision_panel)

        top = QtWidgets.QHBoxLayout()
        self.k_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.k_slider.setRange(1, 20)
        self.k_slider.setValue(self.rag.rcfg.k_final)
        self.k_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.top_k_label = QtWidgets.QLabel(f"Top‑K: {self.k_slider.value()}")
        self.k_slider.valueChanged.connect(lambda val: self.top_k_label.setText(f"Top‑K: {val}"))

        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 1.5)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(self.rag.mcfg.temperature)
        self.tokens_spin = QtWidgets.QSpinBox()
        self.tokens_spin.setRange(64, 2048)
        self.tokens_spin.setValue(self.rag.mcfg.max_new_tokens)

        ingest_btn = QtWidgets.QPushButton("Rebuild Index")
        ingest_btn.clicked.connect(self.handle_ingest)
        cancel_btn = QtWidgets.QPushButton("Cancel Stream")
        cancel_btn.clicked.connect(self.cancel_stream)
        clear_btn = QtWidgets.QPushButton("Clear Chat")
        clear_btn.clicked.connect(self.clear_chat)
        copy_btn = QtWidgets.QPushButton("Copy Answer")
        copy_btn.clicked.connect(self.copy_answer)
        export_btn = QtWidgets.QPushButton("Export Answer.md")
        export_btn.clicked.connect(self.export_answer)

        top.addWidget(self.top_k_label)
        top.addWidget(self.k_slider)
        top.addWidget(QtWidgets.QLabel("Temp"))
        top.addWidget(self.temp_spin)
        top.addWidget(QtWidgets.QLabel("Max tokens"))
        top.addWidget(self.tokens_spin)
        top.addStretch(1)
        top.addWidget(copy_btn)
        top.addWidget(export_btn)
        top.addWidget(cancel_btn)
        top.addWidget(clear_btn)
        top.addWidget(ingest_btn)
        v.addLayout(top)

        # Query history bar
        history_row = QtWidgets.QHBoxLayout()
        history_row.addWidget(QtWidgets.QLabel("History:"))
        self.history_combo = QtWidgets.QComboBox()
        self.history_combo.setEditable(False)
        self.history_combo.setMinimumWidth(300)
        self.history_combo.currentTextChanged.connect(
            lambda text: self.prompt_edit.setPlainText(text) if text else self.prompt_edit.clear()
        )
        history_row.addWidget(self.history_combo, stretch=1)
        v.addLayout(history_row)

        self.prompt_edit = QtWidgets.QTextEdit()
        self.prompt_edit.setPlaceholderText("Ask for code, refactors, architecture plans… (Ctrl/Cmd+Enter to send)")
        self.prompt_edit.setMinimumHeight(80)
        self.prompt_edit.keyPressEvent = self._wrap_enter(self.prompt_edit.keyPressEvent)

        ask_btn = QtWidgets.QPushButton("Ask")
        ask_btn.setStyleSheet("font-weight: 700; padding: 10px 16px;")
        ask_btn.clicked.connect(self.handle_ask)

        self.answer_view = QtWidgets.QTextBrowser()
        self.answer_view.setOpenExternalLinks(True)
        self.answer_view.setStyleSheet("font-size: 14px;")
        self.ctx_view = QtWidgets.QTextBrowser()
        self.ctx_view.setStyleSheet("background:#0f1624;")
        self.ctx_view.setMinimumWidth(420)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.answer_view)
        splitter.addWidget(self.ctx_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        v.addWidget(self.prompt_edit)
        v.addWidget(ask_btn)
        v.addWidget(splitter)
        return w

    def _build_settings_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.drop = DropArea()
        self.drop.filesDropped.connect(self.handle_drop)
        v.addWidget(self.drop)

        grid = QtWidgets.QFormLayout()
        self.model_edit = QtWidgets.QLineEdit(self.rag.mcfg.llm_name)
        self.embed_edit = QtWidgets.QLineEdit(self.rag.mcfg.embed_name)
        self.raw_dir_edit = QtWidgets.QLineEdit(self.rag.rcfg.raw_dir)
        grid.addRow("LLM model", self.model_edit)
        grid.addRow("Embed model", self.embed_edit)
        grid.addRow("Raw data folder", self.raw_dir_edit)

        # Font size control
        font_row = QtWidgets.QHBoxLayout()
        font_row.addWidget(QtWidgets.QLabel("Answer font size:"))
        self.font_spin = QtWidgets.QSpinBox()
        self.font_spin.setRange(8, 24)
        self.font_spin.setValue(14)
        self.font_spin.valueChanged.connect(
            lambda pt: self.answer_view.setFont(QtGui.QFont("Monospace", pt))
        )
        font_row.addWidget(self.font_spin)
        font_row.addStretch()
        grid.addRow("", font_row)
        v.addLayout(grid)

        apply_btn = QtWidgets.QPushButton("Apply & Reload Models")
        apply_btn.clicked.connect(self.handle_reload_models)
        v.addWidget(apply_btn)

        folder_btn = QtWidgets.QPushButton("Open raw folder in OS")
        folder_btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(self.raw_dir_edit.text() or self.rag.rcfg.raw_dir)
        ))
        v.addWidget(folder_btn)

        save_chat_btn = QtWidgets.QPushButton("Save Chat History")
        save_chat_btn.clicked.connect(self.save_chat)
        load_chat_btn = QtWidgets.QPushButton("Load Chat History")
        load_chat_btn.clicked.connect(self.load_chat)
        v.addWidget(save_chat_btn)
        v.addWidget(load_chat_btn)

        v.addStretch(1)
        return w

    def _build_analytics_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.metrics_text = QtWidgets.QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("font-family: monospace;")
        refresh_btn = QtWidgets.QPushButton("Refresh Analytics")
        refresh_btn.clicked.connect(self._refresh_analytics)
        v.addWidget(QtWidgets.QLabel("System Analytics & Precision Metrics"))
        v.addWidget(self.metrics_text)
        v.addWidget(refresh_btn)
        return w

    def _build_about_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        lbl = QtWidgets.QLabel(
            "CAN — Code Anything Now\n"
            "• Local, uncensored RAG (FAISS + SentenceTransformers)\n"
            "• Streaming tokens, dark theme, drag/drop ingestion, PDF/TXT/DOCX/HTML support\n"
            "• Hot model swap, adjustable Top‑K, temperature & tokens\n"
            "• Context viewer with scores, chat save/load, export to Markdown\n"
            "• Query expansion, reranking, HyDE, analytics dashboard\n"
            "• GPU‑aware with CPU fallback\n"
            "• Optional llama.cpp GGUF backend via LOCAL_GGUF_MODEL"
        )
        lbl.setAlignment(QtCore.Qt.AlignTop)
        v.addWidget(lbl)
        v.addStretch(1)
        return w

    # ---- Helpers & Handlers ----
    def _wrap_enter(self, orig_keypress):
        def handler(event):
            if (event.modifiers() & QtCore.Qt.ControlModifier) and event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self.handle_ask()
            else:
                orig_keypress(event)
        return handler

    def handle_drop(self, paths: List[str]):
        raw_dir = self.rag.rcfg.raw_dir
        os.makedirs(raw_dir, exist_ok=True)
        for p in paths:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in files:
                        shutil.copy(os.path.join(root, f), os.path.join(raw_dir, f))
            else:
                shutil.copy(p, os.path.join(raw_dir, os.path.basename(p)))
        self.status.showMessage("Files copied. Rebuild index to embed.", 6000)

    def handle_reload_models(self):
        try:
            new_llm = self.model_edit.text().strip()
            new_embed = self.embed_edit.text().strip()
            if new_llm:
                self.rag.mcfg.llm_name = new_llm
            if new_embed:
                self.rag.mcfg.embed_name = new_embed
            new_raw = self.raw_dir_edit.text().strip()
            if new_raw:
                self.rag.rcfg.raw_dir = new_raw
            self.rag.__init__(self.rag.mcfg, self.rag.rcfg)
            self.status.showMessage("Models reloaded.", 6000)
        except Exception as e:
            self.status.showMessage(f"Reload failed: {e}", 8000)

    def handle_ingest(self):
        self.progress = QtWidgets.QProgressDialog("Indexing...", "Abort", 0, 100, self)
        self.progress.setWindowModality(QtCore.Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        # Store reference on self so the worker is not garbage-collected mid-run
        self._ingest_worker = IngestWorker(self.rag.indexer)
        self._ingest_worker.progress.connect(self.progress.setValue)
        self._ingest_worker.done.connect(self._ingest_done)
        self._ingest_worker.failed.connect(lambda msg: self.status.showMessage(msg, 8000))
        self._ingest_worker.start()

    def _ingest_done(self, count: int):
        self.progress.setValue(100)
        self.progress.close()
        self.status.showMessage(f"Indexed {count} chunks.", 6000)

    def clear_chat(self):
        self.answer_view.clear()
        self.ctx_view.clear()

    def copy_answer(self):
        QtWidgets.QApplication.clipboard().setText(self.answer_view.toPlainText())
        self.status.showMessage("Answer copied.", 3000)

    def export_answer(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Answer", "answer.md", "Markdown (*.md)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.answer_view.toPlainText())
            self.status.showMessage("Answer exported.", 4000)

    def handle_ask(self):
        query = self.prompt_edit.toPlainText().strip()
        if not query:
            return
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        # Add to history (avoid duplicates at top)
        existing = [self.history_combo.itemText(i) for i in range(self.history_combo.count())]
        if query not in existing:
            self.history_combo.insertItem(0, query)
            if self.history_combo.count() > 50:
                self.history_combo.removeItem(self.history_combo.count() - 1)
        self.answer_view.clear()
        self.ctx_view.clear()
        self.status.showMessage("Retrieving context & generating answer…")
        self.worker = AskWorker(
            self.rag,
            query,
            self.k_slider.value(),
            self.temp_spin.value(),
            self.tokens_spin.value(),
        )
        self.worker.tokenSignal.connect(self._on_token)
        self.worker.ctxSignal.connect(self._on_ctx)
        self.worker.errorSignal.connect(self._on_error)
        self.worker.finished.connect(lambda: self.status.showMessage("Done.", 5000))
        self.worker.start()

    def cancel_stream(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status.showMessage("Stream cancelled.", 4000)

    def _on_token(self, tok: str):
        cursor = self.answer_view.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(tok)
        self.answer_view.ensureCursorVisible()

    def _on_ctx(self, ctx):
        cards = []
        for i, c in enumerate(ctx):
            cards.append(
                f"[{i+1}] score={c[2]:.3f}\n{c[0]}\nsource: {c[1].get('doc_meta', {}).get('title','unknown')}\n"
                "----------------------------------------"
            )
        self.ctx_view.setPlainText("\n".join(cards))

    def _on_error(self, msg: str):
        self.status.showMessage("Error occurred", 6000)
        self.answer_view.setPlainText(msg)

    def save_chat(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Chat", "chat.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"answer": self.answer_view.toPlainText(), "context": self.ctx_view.toPlainText()}, f, indent=2)
            self.status.showMessage("Chat saved.", 4000)

    def load_chat(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Chat", "", "JSON (*.json)")
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.answer_view.setPlainText(data.get("answer", ""))
            self.ctx_view.setPlainText(data.get("context", ""))
            self.status.showMessage("Chat loaded.", 4000)

    def _refresh_analytics(self):
        try:
            with open(self.rag.rcfg.meta_path, "r") as f:
                metadata = json.load(f)
            metrics = [
                "=== PRECISION RAG ANALYTICS ===",
                f"Version: {metadata.get('version', 'N/A')}",
                f"Created: {metadata.get('created', 'N/A')}",
                f"Embedding Model: {metadata.get('embedding_model', 'N/A')}",
                f"Total Documents: {metadata.get('total_documents', 0)}",
                f"Total Chunks: {metadata.get('total_chunks', 0)}",
                f"Index Dimension: {metadata.get('dimension', 0)}",
                "",
                "=== PRECISION SETTINGS ===",
                f"Query Expansion: {self.precision_settings['query_expansion']}",
                f"Reranking: {self.precision_settings['enable_reranking']}",
                f"HyDE: {self.precision_settings['enable_hyde']}",
                f"Chunk Strategy: {self.rag.rcfg.chunk_strategy.value}",
                f"Deduplication: {self.rag.rcfg.deduplicate}",
                "",
                "=== FILE STATISTICS ===",
            ]
            for file_info in metadata.get("processed_files", [])[:10]:
                metrics.append(f"• {Path(file_info['path']).name}: {file_info.get('chunks', 0)} chunks")
            if len(metadata.get("processed_files", [])) > 10:
                metrics.append(f"... and {len(metadata['processed_files']) - 10} more files")
            self.metrics_text.setPlainText("\n".join(metrics))
        except Exception as e:
            self.metrics_text.setPlainText(f"Error loading analytics: {e}")

    def _toggle_precision_setting(self, setting_key: str, state: int) -> None:
        """Update a precision setting in the local cache and on the RAG config.

        Expected *setting_key* values: ``"query_expansion"``, ``"enable_reranking"``,
        ``"enable_hyde"`` — all are attributes of :class:`RagConfig`.
        """
        enabled = state == QtCore.Qt.Checked
        self.precision_settings[setting_key] = enabled
        setattr(self.rag.rcfg, setting_key, enabled)

    def _toggle_query_expansion(self, state: int) -> None:
        self._toggle_precision_setting("query_expansion", state)

    def _toggle_reranking(self, state: int) -> None:
        self._toggle_precision_setting("enable_reranking", state)

    def _toggle_hyde(self, state: int) -> None:
        self._toggle_precision_setting("enable_hyde", state)

    def closeEvent(self, event):
        """Gracefully stop background workers before closing."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        if hasattr(self, "_ingest_worker") and self._ingest_worker.isRunning():
            self._ingest_worker.wait(3000)
        event.accept()


# ---------------- Entrypoint ---------------- #
def main():
    mcfg = ModelConfig(
        llm_name=HF_LLM_NAME,
        embed_name="sentence-transformers/all-mpnet-base-v2",
        temperature=0.3,
        max_new_tokens=1024,
    )
    rcfg = RagConfig(
        chunk_strategy=ChunkStrategy.HYBRID,
        k_retrieve=12,
        k_final=6,
        enable_reranking=True,
        query_expansion=True,
        enable_hyde=True,
        deduplicate=True,
        semantic_threshold=0.85,
    )
    rag = EnhancedRagGenerator(mcfg, rcfg)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    font = QtGui.QFont("Segoe UI", 10)
    app.setFont(font)

    win = EnhancedMainWindow(rag)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()