from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class TaskWorker(QObject):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, task: Callable[[], object]):
        super().__init__()
        self.task = task

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.succeeded.emit(self.task())
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

