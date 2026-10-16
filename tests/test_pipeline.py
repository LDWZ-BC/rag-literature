"""Basic tests for RAG literature pipeline."""

import os
import sys
from pathlib import Path

import pytest


class TestConfig:
    """Verify configuration module works with defaults."""

    def test_data_root_default(self):
        from rag_literature.config import DATA_ROOT
        assert DATA_ROOT is not None

    def test_llama_server_bin_default(self):
        from rag_literature.config import LLAMA_SERVER_BIN
        assert LLAMA_SERVER_BIN is not None

    def test_server_port_default(self):
        from rag_literature.config import SERVER_PORT
        assert 1024 <= SERVER_PORT <= 65535

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RAG_DATA_ROOT", "/tmp/test_rag")
        monkeypatch.setenv("RAG_SERVER_PORT", "9999")
        # Force re-import
        import importlib
        import rag_literature.config
        importlib.reload(rag_literature.config)
        from rag_literature.config import DATA_ROOT, SERVER_PORT
        assert str(DATA_ROOT) == "/tmp/test_rag"
        assert SERVER_PORT == 9999

    def test_embedding_model_default(self):
        from rag_literature.config import EMBEDDING_MODEL
        assert "MiniLM" in EMBEDDING_MODEL or "mini" in EMBEDDING_MODEL.lower()


class TestImports:
    """Verify all core modules import without errors."""

    def test_config_imports(self):
        from rag_literature.config import (
            DATA_ROOT, LLAMA_SERVER_BIN, SERVER_PORT, MAX_RETRIES,
            RAG_JSON_DIR, CHROMA_DATA_DIR, MINERU_CONDA_ENV,
        )
        assert True

    def test_utils_imports(self):
        from rag_literature.utils.files import read_json, write_json
        from rag_literature.utils.naming import sanitize_name
        assert True

    def test_models_imports(self):
        from rag_literature.models.manager import LLamaServerManager, server_manager
        from rag_literature.models.translator import translate_en2zh, translate_batch
        assert True

    def test_pipeline_imports(self):
        from rag_literature.pipeline.step1_language import detect_language
        from rag_literature.pipeline.step2_mineru import run_mineru
        from rag_literature.pipeline.step3_metadata import rename_and_move
        from rag_literature.pipeline.step4_clean_translate import clean_v2_json
        from rag_literature.pipeline.step6_rag_update import incremental_update
        from rag_literature.pipeline.orchestrator import process_pdf, process_all
        assert True


class TestServerManager:
    """Verify server manager logic."""

    def test_singleton(self):
        from rag_literature.models.manager import LLamaServerManager
        a = LLamaServerManager()
        b = LLamaServerManager()
        assert a is b

    def test_mark_activity_noop(self):
        from rag_literature.models.manager import LLamaServerManager
        mgr = LLamaServerManager()
        # Should not raise
        mgr.mark_activity()
        assert True

    def test_stop_without_server(self):
        from rag_literature.models.manager import LLamaServerManager
        mgr = LLamaServerManager()
        # Should not raise
        mgr.stop()
        assert True


class TestSectionHeadingFilter:
    """Verify section heading blacklist works."""

    def test_known_section_headings(self):
        from rag_literature.pipeline.step3_metadata import _is_likely_section_heading
        assert _is_likely_section_heading("Annotation")
        assert _is_likely_section_heading("Comment")
        assert _is_likely_section_heading("abstract")
        assert _is_likely_section_heading("Introduction.")
        assert _is_likely_section_heading("research report")

    def test_real_titles_pass(self):
        from rag_literature.pipeline.step3_metadata import _is_likely_section_heading
        assert not _is_likely_section_heading("Vortices in high-temperature superconductors")
        assert not _is_likely_section_heading("Creep motion of a magnetic wall: Avalanche size divergence")
        assert not _is_likely_section_heading("Optical trapping and manipulation of single cells")


class TestStep4CleanTranslate:
    """Verify cleaning logic works on sample data."""

    def test_split_sentences_basic(self):
        from rag_literature.pipeline.step4_clean_translate import split_into_sentences
        sents = split_into_sentences("Hello world. This is a test. Another sentence.")
        assert len(sents) >= 2

    def test_journal_abbreviations(self):
        from rag_literature.pipeline.step4_clean_translate import split_into_sentences
        # "Phys. Rev. Lett." should NOT be split at "Phys." or "Rev."
        sents = split_into_sentences("Published in Phys. Rev. Lett. 66, 1125 (1994). Next sentence.")
        # Check that journal abbreviations are preserved (not split)
        assert "phys. rev. lett." in sents[0].lower()


class TestLanguageDetection:
    """Verify language detection."""

    def test_detect_language_import(self):
        from rag_literature.pipeline.step1_language import detect_language
        assert callable(detect_language)
