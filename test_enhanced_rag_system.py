"""Unit tests for enhanced_rag_system pure-Python logic.

These tests cover components that don't require GPU, large models, or a running
PyQt5 application, so they can be executed in any CI environment.
"""

import hashlib
import unittest
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so the module-level optional imports don't crash the test run.
# ---------------------------------------------------------------------------
import sys
import types

for _mod in (
    "faiss",
    "torch",
    "transformers",
    "sentence_transformers",
    "qdarkstyle",
    "PyQt5",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
    "PyPDF2",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Stub torch.cuda.is_available so ModelConfig default device resolves cleanly.
_torch_stub = sys.modules["torch"]
if not hasattr(_torch_stub, "cuda"):
    _torch_stub.cuda = MagicMock()  # type: ignore[attr-defined]
_torch_stub.cuda.is_available = MagicMock(return_value=False)  # type: ignore[attr-defined]
_torch_stub.float16 = MagicMock()  # type: ignore[attr-defined]
_torch_stub.float32 = MagicMock()  # type: ignore[attr-defined]
_torch_stub.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))  # type: ignore[attr-defined]

# Stub transformers attributes used at import time.
_tf_stub = sys.modules["transformers"]
_tf_stub.AutoModelForCausalLM = MagicMock()  # type: ignore[attr-defined]
_tf_stub.AutoTokenizer = MagicMock()  # type: ignore[attr-defined]
_tf_stub.TextIteratorStreamer = MagicMock()  # type: ignore[attr-defined]

# Stub sentence_transformers attributes.
_st_stub = sys.modules["sentence_transformers"]
_st_stub.CrossEncoder = MagicMock()  # type: ignore[attr-defined]
_st_stub.SentenceTransformer = MagicMock()  # type: ignore[attr-defined]

# Stub faiss attributes.
_faiss_stub = sys.modules["faiss"]
_faiss_stub.IndexFlatIP = MagicMock()  # type: ignore[attr-defined]
_faiss_stub.Index = MagicMock()  # type: ignore[attr-defined]
_faiss_stub.normalize_L2 = MagicMock()  # type: ignore[attr-defined]
_faiss_stub.read_index = MagicMock()  # type: ignore[attr-defined]
_faiss_stub.write_index = MagicMock()  # type: ignore[attr-defined]

# Stub PyQt5 sub-modules used at module level.
for _qt_sub in ("PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets"):
    _m = sys.modules[_qt_sub]
    _m.QThread = MagicMock()  # type: ignore[attr-defined]
    _m.pyqtSignal = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    _m.Qt = MagicMock()  # type: ignore[attr-defined]
    _m.QMainWindow = MagicMock()  # type: ignore[attr-defined]
    _m.QLabel = MagicMock()  # type: ignore[attr-defined]
    _m.QTextCursor = MagicMock()  # type: ignore[attr-defined]
    _m.QFont = MagicMock()  # type: ignore[attr-defined]

import enhanced_rag_system as ers  # noqa: E402  (import after stubs)


class TestModelConfigDefaults(unittest.TestCase):
    def test_default_device_is_cpu_when_no_gpu(self):
        cfg = ers.ModelConfig()
        self.assertEqual(cfg.device, "cpu")

    def test_default_max_new_tokens(self):
        cfg = ers.ModelConfig()
        self.assertEqual(cfg.max_new_tokens, 1024)

    def test_use_local_gguf_false_by_default(self):
        cfg = ers.ModelConfig()
        self.assertFalse(cfg.use_local_gguf)


class TestRagConfigDefaults(unittest.TestCase):
    def test_deduplicate_is_true_by_default(self):
        cfg = ers.RagConfig()
        self.assertTrue(cfg.deduplicate)

    def test_min_similarity_threshold(self):
        cfg = ers.RagConfig()
        self.assertAlmostEqual(cfg.min_similarity, 0.95)

    def test_chunk_strategy_is_hybrid(self):
        cfg = ers.RagConfig()
        self.assertEqual(cfg.chunk_strategy, ers.ChunkStrategy.HYBRID)


class TestChunkStrategy(unittest.TestCase):
    def test_all_strategies_defined(self):
        strategies = {s.value for s in ers.ChunkStrategy}
        self.assertIn("semantic", strategies)
        self.assertIn("fixed", strategies)
        self.assertIn("sliding", strategies)
        self.assertIn("hybrid", strategies)


class TestTextProcessorTokenCount(unittest.TestCase):
    def setUp(self):
        self.cfg = ers.RagConfig()
        self.processor = ers.TextProcessor(self.cfg)

    def test_count_tokens_returns_positive_int(self):
        result = self.processor.count_tokens("hello world this is a test")
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_empty_string_returns_zero(self):
        result = self.processor.count_tokens("")
        self.assertEqual(result, 0)

    def test_longer_text_gives_higher_count(self):
        short = self.processor.count_tokens("hi")
        long = self.processor.count_tokens("hi " * 100)
        self.assertGreater(long, short)


class TestTextProcessorCleanText(unittest.TestCase):
    def setUp(self):
        self.processor = ers.TextProcessor(ers.RagConfig())

    def test_collapses_whitespace(self):
        result = self.processor.clean_text("hello   world\n\n  test")
        self.assertNotIn("   ", result)

    def test_strips_control_characters(self):
        result = self.processor.clean_text("hello\x00world\x1f!")
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x1f", result)

    def test_normalises_crlf(self):
        result = self.processor.clean_text("line1\r\nline2\rline3")
        self.assertNotIn("\r", result)

    def test_strips_leading_trailing_whitespace(self):
        result = self.processor.clean_text("  padded  ")
        self.assertEqual(result, "padded")


class TestTextProcessorSplitSentences(unittest.TestCase):
    def setUp(self):
        self.processor = ers.TextProcessor(ers.RagConfig())

    def test_basic_sentence_split(self):
        sentences = self.processor.split_sentences(
            "Hello world. How are you? I am fine!"
        )
        self.assertEqual(len(sentences), 3)

    def test_empty_string(self):
        sentences = self.processor.split_sentences("")
        self.assertEqual(sentences, [])


class TestTextProcessorSplitParagraphs(unittest.TestCase):
    def setUp(self):
        self.processor = ers.TextProcessor(ers.RagConfig())

    def test_splits_on_blank_lines(self):
        text = "para one\n\npara two\n\npara three"
        paragraphs = self.processor.split_paragraphs(text)
        self.assertEqual(len(paragraphs), 3)

    def test_ignores_empty_paragraphs(self):
        text = "para one\n\n\n\npara two"
        paragraphs = self.processor.split_paragraphs(text)
        self.assertEqual(len(paragraphs), 2)


class TestTextProcessorExtractKeywords(unittest.TestCase):
    def setUp(self):
        self.processor = ers.TextProcessor(ers.RagConfig())

    def test_returns_at_most_max_keywords(self):
        text = "machine learning algorithms neural network deep training"
        keywords = self.processor._extract_keywords(text, max_keywords=3)
        self.assertLessEqual(len(keywords), 3)

    def test_excludes_stopwords(self):
        text = "this that with from have were they what when which would could"
        keywords = self.processor._extract_keywords(text)
        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "were",
            "they",
            "what",
            "when",
            "which",
            "would",
            "could",
        }
        for kw in keywords:
            self.assertNotIn(kw, stopwords)


class TestResolveThreads(unittest.TestCase):
    def test_returns_positive_int(self):
        result = ers._resolve_threads()
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    @patch.object(ers, "LLAMA_CPP_THREADS", 8)
    def test_honours_env_override(self):
        result = ers._resolve_threads()
        self.assertEqual(result, 8)


class TestDeduplicateChunks(unittest.TestCase):
    """Test EnhancedDocumentIndexer._deduplicate_chunks without loading real models."""

    def _make_indexer(self, deduplicate=True, min_similarity=0.95):
        indexer = ers.EnhancedDocumentIndexer.__new__(ers.EnhancedDocumentIndexer)
        indexer.cfg = ers.RagConfig(
            deduplicate=deduplicate, min_similarity=min_similarity
        )
        return indexer

    def test_exact_duplicates_removed(self):
        import numpy as np

        indexer = self._make_indexer()
        chunks = [("hello world", {}), ("hello world", {}), ("different text", {})]
        # Embeddings: first two identical, third different
        embs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        result_chunks, result_embs = indexer._deduplicate_chunks(chunks, embs)
        texts = [c[0] for c in result_chunks]
        self.assertEqual(len(texts), 2)
        self.assertIn("hello world", texts)
        self.assertIn("different text", texts)

    def test_near_duplicates_removed(self):
        import numpy as np

        indexer = self._make_indexer()
        chunks = [("text A", {}), ("text B", {}), ("text C", {})]
        # text A and text B are near-duplicates (cosine similarity > 0.95)
        v_ab = np.array([0.999, 0.045])
        v_ab /= np.linalg.norm(v_ab)
        v_c = np.array([0.0, 1.0])
        embs = np.array([v_ab, v_ab + 1e-4, v_c])
        # Re-normalise
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        result_chunks, result_embs = indexer._deduplicate_chunks(chunks, embs)
        self.assertEqual(len(result_chunks), 2)

    def test_no_dedup_when_disabled(self):
        import numpy as np

        indexer = self._make_indexer(deduplicate=False)
        chunks = [("same", {}), ("same", {})]
        embs = np.array([[1.0, 0.0], [1.0, 0.0]])
        result_chunks, result_embs = indexer._deduplicate_chunks(chunks, embs)
        self.assertEqual(len(result_chunks), 2)


class TestExpandQuery(unittest.TestCase):
    """Test EnhancedDocumentIndexer._expand_query without loading real models."""

    def _make_indexer(self):
        indexer = ers.EnhancedDocumentIndexer.__new__(ers.EnhancedDocumentIndexer)
        indexer.cfg = ers.RagConfig()
        return indexer

    def test_returns_list(self):
        indexer = self._make_indexer()
        result = indexer._expand_query("how to sort a list")
        self.assertIsInstance(result, list)

    def test_prepends_how_to_for_non_question(self):
        indexer = self._make_indexer()
        result = indexer._expand_query("sort a list")
        self.assertTrue(any("how to" in r for r in result))

    def test_expands_known_term(self):
        indexer = self._make_indexer()
        result = indexer._expand_query("write a function")
        combined = " ".join(result)
        self.assertTrue(
            any(alt in combined for alt in ["method", "procedure", "routine"])
        )

    def test_no_duplicates(self):
        indexer = self._make_indexer()
        result = indexer._expand_query("sort code")
        self.assertEqual(len(result), len(set(result)))


if __name__ == "__main__":
    unittest.main()
