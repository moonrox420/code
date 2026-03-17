"""Shared UI components used by both app.py and enhanced_rag_system.py."""

from PyQt5 import QtCore, QtWidgets


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
