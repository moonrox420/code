import os
import sys
import json
import glob
import shutil
import threading
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional

import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sentence_transformers import SentenceTransformer
from PyQt5 import QtCore, QtGui, QtWidgets
import qdarkstyle

# PyMuPDF — optional, preferred PDF backend (pip install PyMuPDF)
try:
    import fitz  # type: ignore
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False
    fitz = None  # type: ignore

# PyPDF2 — required fallback
from PyPDF2 import PdfReader


# ---------------- Configuration ---------------- #
@dataclass
class ModelConfig:
    llm_name: str = "meta-llama/Llama-3-8b-Instruct"
    embed_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 640
    temperature: float = 0.42
    top_p: float = 0.92


@dataclass
class RagConfig:
    raw_dir: str = "data/raw"
    chunks_dir: str = "data/chunks"
    index_path: str = "data/index.faiss"
    meta_path: str = "data/meta.json"
    chunk_size: int = 480
    chunk_overlap: int = 80
    k: int = 6


# ---------------- RAG Engine ---------------- #
class DocumentIndexer:
    def __init__(self, embed_model: SentenceTransformer, cfg: RagConfig):
        self.embed = embed_model
        self.cfg = cfg
        os.makedirs(cfg.raw_dir, exist_ok=True)
        os.makedirs(cfg.chunks_dir, exist_ok=True)

    def _read_file(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            if _HAS_PYMUPDF:
                try:
                    doc = fitz.open(path)  # type: ignore[union-attr]
                    text = "\n".join(doc[i].get_text() for i in range(len(doc)))
                    doc.close()
                    return text
                except Exception:
                    pass  # fall through to PyPDF2
            reader = PdfReader(path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def _split(self, text: str) -> List[str]:
        words = text.split()
        step = self.cfg.chunk_size - self.cfg.chunk_overlap
        return [" ".join(words[i:i + self.cfg.chunk_size]) for i in range(0, len(words), step)]

    def ingest(self, progress_cb=None) -> int:
        files = glob.glob(os.path.join(self.cfg.raw_dir, "**", "*.*"), recursive=True)
        chunks, meta = [], []
        total = len(files)
        for file_index, fp in enumerate(files):
            text = self._read_file(fp)
            for chunk_index, chunk_text in enumerate(self._split(text)):
                meta.append({"source": fp, "chunk_id": chunk_index})
                chunks.append(chunk_text)
            if progress_cb:
                progress_cb(int((file_index + 1) / max(total, 1) * 90))
        if not chunks:
            if progress_cb:
                progress_cb(0)
            return 0
        with open(self.cfg.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.cfg.chunks_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        emb = self.embed.encode(chunks, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, self.cfg.index_path)
        if progress_cb:
            progress_cb(100)
        return len(chunks)

    def load(self):
        if not (os.path.exists(self.cfg.index_path) and os.path.exists(self.cfg.meta_path)):
            raise FileNotFoundError("Index missing. Rebuild index first.")
        index = faiss.read_index(self.cfg.index_path)
        with open(os.path.join(self.cfg.chunks_dir, "chunks.json"), encoding="utf-8") as f:
            chunks = json.load(f)
        with open(self.cfg.meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return index, chunks, meta

    def retrieve(self, query: str, k: int) -> List[Tuple[str, dict, float]]:
        index, chunks, meta = self.load()
        q_emb = self.embed.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = index.search(q_emb, k)
        return [(chunks[i], meta[i], float(s)) for i, s in zip(idxs[0], scores[0])]


class RagGenerator:
    def __init__(self, mcfg: ModelConfig, rcfg: RagConfig):
        self.mcfg, self.rcfg = mcfg, rcfg
        self.tokenizer = AutoTokenizer.from_pretrained(mcfg.llm_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            mcfg.llm_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.embed = SentenceTransformer(mcfg.embed_name, device=mcfg.device)
        self.indexer = DocumentIndexer(self.embed, rcfg)

    def ensure_index(self):
        if not os.path.exists(self.rcfg.index_path):
            self.indexer.ingest()

    def build_prompt(self, q: str, ctx) -> str:
        ctx_txt = "\n\n".join(
            [f"[{i+1}] {c[0]}\n(source: {c[1]['source']}, score={c[2]:.3f})" for i, c in enumerate(ctx)]
        )
        return (
            "You are CAN, an elite uncensored engineer. Use ONLY the provided context if helpful.\n"
            f"Context:\n{ctx_txt}\n\nQuestion: {q}\n\n"
            "Answer concisely with production-grade code and reasoning."
        )

    def generate_stream(self, query: str, k: int, temperature: float, max_new_tokens: int):
        self.mcfg.temperature = temperature
        self.mcfg.max_new_tokens = max_new_tokens
        self.ensure_index()
        ctx = self.indexer.retrieve(query, k)
        prompt = self.build_prompt(query, ctx)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs = dict(
            **inputs,
            max_new_tokens=self.mcfg.max_new_tokens,
            temperature=self.mcfg.temperature,
            top_p=self.mcfg.top_p,
            do_sample=True,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        t = threading.Thread(target=self.model.generate, kwargs=kwargs)
        t.start()
        return streamer, ctx


# ---------------- UI Components ---------------- #
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


class IngestWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    done = QtCore.pyqtSignal(int)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, indexer: DocumentIndexer):
        super().__init__()
        self.indexer = indexer

    def run(self):
        try:
            count = self.indexer.ingest(progress_cb=self.progress.emit)
            self.done.emit(count)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class AskWorker(QtCore.QThread):
    tokenSignal = QtCore.pyqtSignal(str)
    ctxSignal = QtCore.pyqtSignal(list)
    errorSignal = QtCore.pyqtSignal(str)

    def __init__(self, rag: RagGenerator, query: str, k: int, temperature: float, max_tokens: int):
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, rag: RagGenerator):
        super().__init__()
        self.rag = rag
        self.worker: Optional[AskWorker] = None
        self.setWindowTitle("CAN — Local RAG Coder (PyQt)")
        self.setMinimumSize(1320, 860)
        self._build_ui()

    def _build_ui(self):
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_chat_tab(), "Chat")
        tabs.addTab(self._build_settings_tab(), "Settings")
        tabs.addTab(self._build_about_tab(), "About")
        self.setCentralWidget(tabs)
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

    # ---- Chat Tab ----
    def _build_chat_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        self.top_k_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.top_k_slider.setRange(1, 20)
        self.top_k_slider.setValue(self.rag.rcfg.k)
        self.top_k_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.top_k_label = QtWidgets.QLabel(f"Top‑K: {self.top_k_slider.value()}")
        self.top_k_slider.valueChanged.connect(lambda val: self.top_k_label.setText(f"Top‑K: {val}"))
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
        top.addWidget(self.top_k_slider)
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

        self.prompt_edit = QtWidgets.QTextEdit()
        self.prompt_edit.setPlaceholderText("Ask for code, refactors, architecture plans… (Ctrl/Cmd+Enter to send)")
        self.prompt_edit.keyPressEvent = self._wrap_enter(self.prompt_edit.keyPressEvent)

        ask_btn = QtWidgets.QPushButton("Ask")
        ask_btn.setStyleSheet("font-weight: 700; padding: 10px 16px;")
        ask_btn.clicked.connect(self.handle_ask)

        self.answer_view = QtWidgets.QTextBrowser()
        self.answer_view.setOpenExternalLinks(True)
        self.answer_view.setStyleSheet("font-size: 14px;")
        self.ctx_view = QtWidgets.QTextBrowser()
        self.ctx_view.setStyleSheet("background:#0f1624;")
        self.ctx_view.setMinimumWidth(400)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.answer_view)
        splitter.addWidget(self.ctx_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        v.addLayout(top)
        v.addWidget(self.prompt_edit)
        v.addWidget(ask_btn)
        v.addWidget(splitter)
        return w

    # ---- Settings Tab ----
    def _build_settings_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.drop = DropArea()
        self.drop.filesDropped.connect(self.handle_drop)
        v.addWidget(self.drop)

        grid = QtWidgets.QFormLayout()
        self.model_edit = QtWidgets.QLineEdit(self.rag.mcfg.llm_name)
        self.embed_edit = QtWidgets.QLineEdit(self.rag.mcfg.embed_name)
        grid.addRow("LLM model", self.model_edit)
        grid.addRow("Embed model", self.embed_edit)
        v.addLayout(grid)

        apply_btn = QtWidgets.QPushButton("Apply & Reload Models")
        apply_btn.clicked.connect(self.handle_reload_models)
        v.addWidget(apply_btn)

        folder_btn = QtWidgets.QPushButton("Open raw folder")
        folder_btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.rag.rcfg.raw_dir)))
        v.addWidget(folder_btn)

        save_chat_btn = QtWidgets.QPushButton("Save Chat History")
        save_chat_btn.clicked.connect(self.save_chat)
        load_chat_btn = QtWidgets.QPushButton("Load Chat History")
        load_chat_btn.clicked.connect(self.load_chat)
        v.addWidget(save_chat_btn)
        v.addWidget(load_chat_btn)

        v.addStretch(1)
        return w

    # ---- About Tab ----
    def _build_about_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        lbl = QtWidgets.QLabel(
            "CAN — Code Anything Now\n"
            "• Local, uncensored RAG (FAISS + SentenceTransformers)\n"
            "• Streaming tokens, dark theme, drag/drop ingestion, PDF/TXT support\n"
            "• Hot model swap, adjustable Top‑K, temperature & tokens\n"
            "• Context viewer with scores, chat save/load, export to Markdown\n"
            "• GPU‑aware with CPU fallback"
        )
        lbl.setAlignment(QtCore.Qt.AlignTop)
        v.addWidget(lbl)
        v.addStretch(1)
        return w

    # ---- Helpers ----
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
            self.rag.mcfg.llm_name = self.model_edit.text().strip()
            self.rag.mcfg.embed_name = self.embed_edit.text().strip()
            self.rag.__init__(self.rag.mcfg, self.rag.rcfg)
            self.status.showMessage("Models reloaded.", 6000)
        except Exception as e:
            self.status.showMessage(str(e), 8000)

    def handle_ingest(self):
        self.progress = QtWidgets.QProgressDialog("Indexing...", "Abort", 0, 100, self)
        self.progress.setWindowModality(QtCore.Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        worker = IngestWorker(self.rag.indexer)
        worker.progress.connect(self.progress.setValue)
        worker.done.connect(self._ingest_done)
        worker.failed.connect(lambda msg: self.status.showMessage(msg, 8000))
        worker.start()

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
        self.answer_view.clear()
        self.ctx_view.clear()
        self.status.showMessage("Thinking…")
        self.worker = AskWorker(
            self.rag,
            query,
            self.top_k_slider.value(),
            self.temp_spin.value(),
            self.tokens_spin.value(),
        )
        self.worker.tokenSignal.connect(self._on_token)
        self.worker.ctxSignal.connect(self._on_ctx)
        self.worker.errorSignal.connect(self._on_error)
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
                f"[{i+1}] score={c[2]:.3f}\n{c[0]}\nsource: {c[1]['source']}\n"
                "----------------------------------------"
            )
        self.ctx_view.setPlainText("\n".join(cards))

    def _on_error(self, msg: str):
        self.status.showMessage("Error occurred", 6000)
        self.answer_view.setPlainText(msg)

    # chat history
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


# ---------------- Entrypoint ---------------- #
def main():
    cfg = ModelConfig()
    rcfg = RagConfig()
    rag = RagGenerator(cfg, rcfg)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win = MainWindow(rag)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()