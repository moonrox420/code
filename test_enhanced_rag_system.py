"""
Unit tests for the Enhanced RAG System
"""

import os
import tempfile
import shutil
import json
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime

from enhanced_rag_system import (
    ModelConfig,
    RagConfig,
    DocumentMetadata,
    ChunkMetadata,
    TextProcessor,
    ChunkStrategy,
)


class TestTextProcessor:
    """Test TextProcessor class"""

    @pytest.fixture
    def processor(self):
        cfg = RagConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
        return TextProcessor(cfg)

    def test_clean_text(self, processor):
        """Test text cleaning functionality"""
        dirty_text = "This  has    multiple   spaces\r\nand\x00control\x01chars"
        clean = processor.clean_text(dirty_text)
        assert "  " not in clean
        assert "\x00" not in clean
        assert "\x01" not in clean

    def test_split_sentences(self, processor):
        """Test sentence splitting"""
        text = "First sentence. Second sentence! Third sentence? Fourth."
        sentences = processor.split_sentences(text)
        assert len(sentences) == 4
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"

    def test_split_paragraphs(self, processor):
        """Test paragraph splitting"""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        paragraphs = processor.split_paragraphs(text)
        assert len(paragraphs) == 3
        assert "Paragraph one." in paragraphs[0]

    def test_count_tokens_fallback(self, processor):
        """Test token counting with word-based approximation"""
        text = "This is a test sentence with several words."
        count = processor.count_tokens(text)
        # Should be approximately 1.3 tokens per word
        word_count = len(text.split())
        assert count > word_count  # More tokens than words
        assert count < word_count * 2  # But not too many more

    def test_extract_keywords(self, processor):
        """Test keyword extraction"""
        text = "Python programming is great. Python is powerful. Programming with Python is fun."
        keywords = processor._extract_keywords(text, max_keywords=3)
        assert len(keywords) <= 3
        assert "python" in keywords or "programming" in keywords

    def test_extract_section_title(self, processor):
        """Test section title extraction"""
        text_with_title = "INTRODUCTION:\nThis is the introduction text."
        title = processor._extract_section_title(text_with_title)
        assert title is not None
        assert "INTRODUCTION" in title


class TestChunkStrategy:
    """Test chunk strategy enum"""

    def test_chunk_strategy_values(self):
        """Test that chunk strategies have correct values"""
        assert ChunkStrategy.SEMANTIC.value == "semantic"
        assert ChunkStrategy.FIXED.value == "fixed"
        assert ChunkStrategy.SLIDING.value == "sliding"
        assert ChunkStrategy.HYBRID.value == "hybrid"


class TestModelConfig:
    """Test ModelConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        cfg = ModelConfig()
        assert cfg.temperature > 0
        assert cfg.temperature < 2
        assert cfg.max_new_tokens > 0
        assert cfg.device in ["cuda", "cpu"]

    def test_custom_values(self):
        """Test custom configuration values"""
        cfg = ModelConfig(
            llm_name="custom-model",
            temperature=0.5,
            max_new_tokens=512
        )
        assert cfg.llm_name == "custom-model"
        assert cfg.temperature == 0.5
        assert cfg.max_new_tokens == 512


class TestRagConfig:
    """Test RagConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        cfg = RagConfig()
        assert cfg.chunk_size > 0
        assert cfg.chunk_overlap >= 0
        assert cfg.k_retrieve > 0
        assert cfg.k_final > 0
        assert cfg.enable_reranking is True
        assert cfg.deduplicate is True

    def test_custom_values(self):
        """Test custom configuration values"""
        cfg = RagConfig(
            chunk_size=256,
            k_retrieve=5,
            enable_reranking=False
        )
        assert cfg.chunk_size == 256
        assert cfg.k_retrieve == 5
        assert cfg.enable_reranking is False


class TestDocumentMetadata:
    """Test DocumentMetadata dataclass"""

    def test_creation(self):
        """Test metadata creation"""
        meta = DocumentMetadata(
            source_path="/path/to/doc.pdf",
            file_type=".pdf",
            file_size=1024,
            created_date=datetime.now(),
            modified_date=datetime.now(),
            md5_hash="abc123",
            page_count=10,
            word_count=500
        )
        assert meta.source_path == "/path/to/doc.pdf"
        assert meta.file_type == ".pdf"
        assert meta.page_count == 10
        assert meta.word_count == 500


class TestChunkMetadata:
    """Test ChunkMetadata dataclass"""

    def test_creation(self):
        """Test chunk metadata creation"""
        meta = ChunkMetadata(
            chunk_id="chunk123",
            document_id="doc123",
            start_position=0,
            end_position=100,
            token_count=50,
            sentence_count=3,
            paragraph_id=1
        )
        assert meta.chunk_id == "chunk123"
        assert meta.document_id == "doc123"
        assert meta.token_count == 50
        assert meta.sentence_count == 3


class TestSemanticChunking:
    """Test semantic chunking functionality"""

    @pytest.fixture
    def processor(self):
        cfg = RagConfig(
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_size=10,
            max_chunk_size=100
        )
        return TextProcessor(cfg)

    @pytest.fixture
    def sample_metadata(self):
        return DocumentMetadata(
            source_path="/test/doc.txt",
            file_type=".txt",
            file_size=1000,
            created_date=datetime.now(),
            modified_date=datetime.now(),
            md5_hash="test123"
        )

    def test_semantic_chunking(self, processor, sample_metadata):
        """Test semantic chunking with sample text"""
        text = """
        First paragraph with some content.

        Second paragraph with more content.

        Third paragraph with even more content.
        """
        chunks = processor.semantic_chunking(text, sample_metadata)
        assert len(chunks) > 0
        for chunk_text, chunk_meta in chunks:
            assert isinstance(chunk_text, str)
            assert isinstance(chunk_meta, ChunkMetadata)
            assert len(chunk_text.strip()) > 0


class TestIntegration:
    """Integration tests"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_config_serialization(self):
        """Test that configs can be serialized to JSON"""
        cfg = RagConfig()
        cfg_dict = cfg.__dict__
        json_str = json.dumps(cfg_dict, default=str)
        assert len(json_str) > 0
        loaded = json.loads(json_str)
        assert loaded["chunk_size"] == cfg.chunk_size

    def test_text_processor_pipeline(self, temp_dir):
        """Test complete text processing pipeline"""
        cfg = RagConfig(chunk_size=100, min_chunk_size=10)
        processor = TextProcessor(cfg)

        # Create sample document
        text = "This is a test document. " * 50  # Create longer text

        # Clean and process
        clean_text = processor.clean_text(text)
        assert len(clean_text) > 0

        # Create metadata
        metadata = DocumentMetadata(
            source_path=os.path.join(temp_dir, "test.txt"),
            file_type=".txt",
            file_size=len(text),
            created_date=datetime.now(),
            modified_date=datetime.now(),
            md5_hash="test"
        )

        # Chunk the text
        chunks = processor.semantic_chunking(clean_text, metadata)
        assert len(chunks) > 0

        # Verify all chunks meet minimum size
        for chunk_text, chunk_meta in chunks:
            assert chunk_meta.token_count >= cfg.min_chunk_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
