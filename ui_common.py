"""Common UI components shared between app.py and enhanced_rag_system.py"""
import traceback
from typing import Optional
from PyQt5 import QtCore, QtWidgets


class DropArea(QtWidgets.QLabel):
    """Drag & drop area for files and folders"""
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
    """Background worker for document ingestion"""
    progress = QtCore.pyqtSignal(int)
    done = QtCore.pyqtSignal(int)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, indexer):
        super().__init__()
        self.indexer = indexer

    def run(self):
        try:
            result = self.indexer.ingest(progress_cb=self.progress.emit)
            # Handle both dict and int return types
            if isinstance(result, dict):
                count = result.get("total_chunks", 0)
            else:
                count = result
            self.done.emit(count)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class AskWorker(QtCore.QThread):
    """Background worker for query processing"""
    tokenSignal = QtCore.pyqtSignal(str)
    ctxSignal = QtCore.pyqtSignal(list)
    errorSignal = QtCore.pyqtSignal(str)

    def __init__(self, rag, query: str, k: int, temperature: float, max_tokens: int):
        super().__init__()
        self.rag = rag
        self.query = query
        self.k = k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._stop = False

    def run(self):
        try:
            streamer, ctx = self.rag.generate_stream(
                self.query, self.k, self.temperature, self.max_tokens
            )
            self.ctxSignal.emit(ctx)
            for token in streamer:
                if self._stop:
                    break
                self.tokenSignal.emit(token)
        except Exception as e:
            self.errorSignal.emit(f"{e}\n{traceback.format_exc()}")

    def stop(self):
        self._stop = True
