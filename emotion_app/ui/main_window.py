from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from emotion_app.domain import EMOTIONS, EMOTION_LABELS_ZH, RecognitionResult
from emotion_app.recognizers.base import FileRecognizerProtocol, TextRecognizerProtocol
from emotion_app.workers import TaskWorker
from emotion_app.ui.image_tab import ImageRecognitionTab


APP_STYLE = """
QMainWindow, QWidget { background: #f4f7fb; color: #172033; font-family: "Microsoft YaHei"; }
QGroupBox { background: white; border: 1px solid #dce3ef; border-radius: 10px;
            margin-top: 12px; padding: 14px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QTextEdit, QLineEdit { background: white; border: 1px solid #cfd8e6; border-radius: 7px;
                       padding: 8px; selection-background-color: #3d6df2; }
QPushButton { background: #315fdb; color: white; border: 0; border-radius: 7px;
              padding: 9px 16px; font-weight: 600; }
QPushButton:hover { background: #254dbb; }
QPushButton:disabled { background: #aeb9ce; }
QProgressBar { border: 1px solid #d5ddeb; border-radius: 5px; background: #eef2f8;
               text-align: center; min-height: 20px; }
QProgressBar::chunk { background: #5a80ed; border-radius: 4px; }
QTabWidget::pane { border: 1px solid #dce3ef; background: white; border-radius: 8px; }
QTabBar::tab { background: #e9eef8; padding: 10px 28px; margin-right: 3px; }
QTabBar::tab:selected { background: #315fdb; color: white; }
"""


class MainWindow(QMainWindow):
    def __init__(
        self,
        text_recognizer: TextRecognizerProtocol,
        speech_recognizer: FileRecognizerProtocol,
        image_recognizer: FileRecognizerProtocol,
    ):
        super().__init__()
        self.text_recognizer = text_recognizer
        self.speech_recognizer = speech_recognizer
        self.image_recognizer = image_recognizer
        self._active_thread: QThread | None = None
        self._active_worker: TaskWorker | None = None
        self._probability_bars: dict[str, QProgressBar] = {}
        self._speech_probability_bars: dict[str, QProgressBar] = {}
        self._selected_audio = ""
        self._selected_image = ""

        prepare_runtime = getattr(self.text_recognizer, "prepare_runtime", None)
        if self.text_recognizer.available and callable(prepare_runtime):
            try:
                prepare_runtime()
            except Exception:
                # The status panel and predict call will expose the exact
                # dependency error; the desktop shell must still open.
                pass

        self.setWindowTitle("多模态情绪智能识别系统")
        self.resize(1040, 760)
        self.setMinimumSize(900, 680)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 18, 24, 20)
        root.setSpacing(12)

        title = QLabel("多模态情绪智能识别系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 25px; font-weight: 700; color: #183a8a;")
        root.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_text_tab(), "文本识别")
        self.tabs.addTab(self._build_file_tab("speech"), "语音识别")
        self.tabs.addTab(self._build_file_tab("image"), "图像识别")
        self.tabs.currentChanged.connect(self._tab_changed)
        root.addWidget(self.tabs, 1)

        self.setCentralWidget(central)

    def _build_text_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        left = QVBoxLayout()
        input_group = QGroupBox("单条文本识别")
        input_layout = QVBoxLayout(input_group)
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("请输入中文或英文文本……")
        self.text_input.setMinimumHeight(170)
        input_layout.addWidget(self.text_input)
        actions = QHBoxLayout()
        example_cn = QPushButton("中文示例")
        example_en = QPushButton("English Example")
        clear = QPushButton("清空")
        self.predict_button = QPushButton("开始识别")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        example_cn.clicked.connect(
            lambda: self.text_input.setPlainText("今天终于完成了这个项目，我真的非常开心！")
        )
        example_en.clicked.connect(
            lambda: self.text_input.setPlainText("I am surprised and delighted by the good news!")
        )
        clear.clicked.connect(self._clear_text_result)
        self.predict_button.clicked.connect(self._predict_text)
        self.cancel_button.clicked.connect(self._cancel_task)
        for button in (example_cn, example_en, clear, self.predict_button, self.cancel_button):
            actions.addWidget(button)
        input_layout.addLayout(actions)
        left.addWidget(input_group)

        batch_group = QGroupBox("Excel 批量识别")
        batch_layout = QFormLayout(batch_group)
        self.batch_input = QLineEdit()
        self.batch_output = QLineEdit()
        input_row = QHBoxLayout()
        input_row.addWidget(self.batch_input)
        pick_input = QPushButton("选择文件")
        pick_input.clicked.connect(self._choose_batch_input)
        input_row.addWidget(pick_input)
        output_row = QHBoxLayout()
        output_row.addWidget(self.batch_output)
        pick_output = QPushButton("保存到")
        pick_output.clicked.connect(self._choose_batch_output)
        output_row.addWidget(pick_output)
        self.batch_button = QPushButton("开始批量识别")
        self.batch_button.clicked.connect(self._predict_batch)
        batch_layout.addRow("输入 Excel：", input_row)
        batch_layout.addRow("输出 Excel：", output_row)
        batch_layout.addRow("", self.batch_button)
        left.addWidget(batch_group)
        layout.addLayout(left, 3)

        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout(result_group)
        self.result_label = QLabel("等待识别")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 26px; font-weight: 700; color: #315fdb; padding: 12px;"
        )
        self.confidence_label = QLabel("置信度：--")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.confidence_label)
        for emotion in EMOTIONS:
            row = QHBoxLayout()
            label = QLabel(EMOTION_LABELS_ZH[emotion])
            label.setFixedWidth(45)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setFormat("0.0%")
            self._probability_bars[emotion] = bar
            row.addWidget(label)
            row.addWidget(bar)
            result_layout.addLayout(row)
        result_layout.addStretch(1)
        layout.addWidget(result_group, 2)
        return tab

    def _build_file_tab(self, modality: str) -> QWidget:
        is_image = modality == "image"
        if is_image:
            self.image_tab = ImageRecognitionTab(self.image_recognizer, self._run_task, self)
            return self.image_tab
        recognizer = self.image_recognizer if is_image else self.speech_recognizer
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(12)
        heading = QLabel("图像情绪识别" if is_image else "语音情感识别")
        heading.setStyleSheet("font-size: 22px; font-weight: 700; color: #183a8a;")
        path_edit = QLineEdit()
        path_edit.setReadOnly(True)
        choose = QPushButton("选择图像" if is_image else "选择 WAV / MP3 音频")
        predict = QPushButton("开始识别")
        predict.setEnabled(recognizer.available)
        preview = QLabel("尚未选择文件")
        preview.setAlignment(Qt.AlignCenter)
        preview.setMinimumHeight(160 if is_image else 56)
        preview.setStyleSheet("border: 1px dashed #b7c3d8; border-radius: 8px; color: #7a8498;")

        def choose_file() -> None:
            file_filter = "图像 (*.png *.jpg *.jpeg)" if is_image else "全部音频 (*.wav *.mp3);;WAV 音频 (*.wav);;MP3 音频 (*.mp3)"
            selected, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
            if not selected:
                return
            path_edit.setText(selected)
            if is_image:
                self._selected_image = selected
                pixmap = QPixmap(selected)
                preview.setPixmap(pixmap.scaled(620, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self._selected_audio = selected
                preview.setText(f"已选择音频：{Path(selected).name}")

        choose.clicked.connect(choose_file)
        predict.clicked.connect(
            lambda: self._predict_file(
                recognizer,
                self._selected_image if is_image else self._selected_audio,
                None if is_image else predict,
            )
        )
        buttons = QHBoxLayout()
        buttons.addWidget(choose)
        buttons.addWidget(predict)
        layout.addWidget(heading)
        layout.addWidget(path_edit)
        layout.addLayout(buttons)
        layout.addWidget(preview)

        if not is_image:
            result_box = QGroupBox("语音识别结果")
            result_layout = QVBoxLayout(result_box)
            self.speech_result_label = QLabel("等待识别")
            self.speech_result_label.setAlignment(Qt.AlignCenter)
            self.speech_result_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #315fdb;")
            self.speech_confidence_label = QLabel("置信度：--")
            self.speech_confidence_label.setAlignment(Qt.AlignCenter)
            result_layout.addWidget(self.speech_result_label)
            result_layout.addWidget(self.speech_confidence_label)
            for emotion in EMOTIONS:
                row = QHBoxLayout()
                name = QLabel(EMOTION_LABELS_ZH[emotion])
                name.setFixedWidth(45)
                bar = QProgressBar()
                bar.setRange(0, 1000)
                bar.setFormat("0.0%")
                self._speech_probability_bars[emotion] = bar
                row.addWidget(name)
                row.addWidget(bar)
                result_layout.addLayout(row)
            layout.addWidget(result_box, 1)
        else:
            layout.addStretch(1)
        if not is_image:
            tab.setMinimumHeight(610)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(tab)
            return scroll
        return tab

    def _tab_changed(self, index: int) -> None:
        if index != 2 and hasattr(self, "image_tab"):
            self.image_tab.stop_camera()

    def _predict_text(self) -> None:
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "输入提示", "文本内容不能为空。")
            return
        self._run_task(lambda: self.text_recognizer.predict(text), self._show_result)

    def _predict_file(
        self, recognizer: FileRecognizerProtocol, path: str, button: QPushButton | None = None
    ) -> None:
        if not path:
            QMessageBox.warning(self, "文件错误", "请先选择需要识别的文件。")
            return
        if button is not None:
            button.setEnabled(False)

        def completed(result: RecognitionResult) -> None:
            if button is not None:
                button.setEnabled(True)
            if not result.ok:
                QMessageBox.warning(self, "识别失败", result.error or recognizer.status)
                return
            self.speech_result_label.setText(EMOTION_LABELS_ZH[result.emotion or "neutral"])
            self.speech_confidence_label.setText(f"置信度：{result.confidence:.2%}")
            for emotion, bar in self._speech_probability_bars.items():
                value = float(result.probabilities.get(emotion, 0.0))
                bar.setValue(round(value * 1000))
                bar.setFormat(f"{value:.1%}")

        self._run_task(lambda: recognizer.predict(path), completed)
    def _show_result(self, result: RecognitionResult) -> None:
        if not result.ok:
            self.result_label.setText("识别失败")
            self.confidence_label.setText("置信度：--")
            QMessageBox.warning(self, "识别失败", result.error or "未知错误")
            return
        self.result_label.setText(EMOTION_LABELS_ZH[result.emotion or "neutral"])
        self.confidence_label.setText(f"置信度：{result.confidence:.2%}")
        for emotion, bar in self._probability_bars.items():
            value = float(result.probabilities.get(emotion, 0.0))
            bar.setValue(round(value * 1000))
            bar.setFormat(f"{value:.1%}")

    def _clear_text_result(self) -> None:
        self.text_input.clear()
        self.result_label.setText("等待识别")
        self.confidence_label.setText("置信度：--")
        for bar in self._probability_bars.values():
            bar.setValue(0)
            bar.setFormat("0.0%")

    def _choose_batch_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Excel", "", "Excel (*.xlsx *.xls)")
        if path:
            self.batch_input.setText(path)
            if not self.batch_output.text():
                source = Path(path)
                self.batch_output.setText(str(source.with_name(f"{source.stem}_情绪识别结果.xlsx")))

    def _choose_batch_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "情绪识别结果.xlsx", "Excel (*.xlsx)")
        if path:
            self.batch_output.setText(path if path.lower().endswith(".xlsx") else f"{path}.xlsx")

    def _predict_batch(self) -> None:
        input_path = Path(self.batch_input.text().strip())
        output_path = Path(self.batch_output.text().strip())
        if not input_path.is_file():
            QMessageBox.warning(self, "文件错误", "请选择有效的 Excel 输入文件。")
            return
        if not output_path.name:
            QMessageBox.warning(self, "文件错误", "请选择输出文件。")
            return

        def task() -> tuple[int, Path]:
            frame = pd.read_excel(input_path)
            text_column = next((name for name in ("text", "content", "文本", "内容") if name in frame.columns), None)
            if text_column is None:
                raise ValueError("Excel 必须包含 text、content、文本或内容列。")
            emotions: list[str] = []
            confidences: list[float | None] = []
            errors: list[str] = []
            for raw_text in frame[text_column].fillna("").astype(str):
                if QThread.currentThread().isInterruptionRequested():
                    raise RuntimeError("批量识别已取消，未写出不完整结果。")
                result = self.text_recognizer.predict(raw_text)
                emotions.append(EMOTION_LABELS_ZH.get(result.emotion or "", ""))
                confidences.append(result.confidence if result.ok else None)
                errors.append(result.error or "")
            frame["预测情绪"] = emotions
            frame["置信度"] = confidences
            frame["错误信息"] = errors
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_excel(output_path, index=False)
            return len(frame), output_path

        self._run_task(
            task,
            lambda payload: QMessageBox.information(
                self, "批量识别完成", f"已处理 {payload[0]} 条文本。\n结果：{payload[1]}"
            ),
        )

    def _run_task(self, task, on_success) -> None:
        if self._active_thread is not None:
            QMessageBox.information(self, "任务进行中", "请等待当前任务完成或先取消。")
            return
        thread = QThread(self)
        worker = TaskWorker(task)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(lambda payload: self._task_succeeded(payload, on_success))
        worker.failed.connect(lambda message: QMessageBox.warning(self, "任务失败", message))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._task_finished)
        self._active_thread = thread
        self._active_worker = worker
        self.predict_button.setEnabled(False)
        self.batch_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        thread.start()

    def _task_succeeded(self, payload, on_success) -> None:
        if self._active_thread is not None and self._active_thread.isInterruptionRequested():
            QMessageBox.information(self, "任务已取消", "任务结果已丢弃。")
            return
        on_success(payload)

    def _cancel_task(self) -> None:
        if self._active_thread is not None:
            self._active_thread.requestInterruption()
            self.cancel_button.setEnabled(False)

    def _task_finished(self) -> None:
        self._active_thread = None
        self._active_worker = None
        self.predict_button.setEnabled(True)
        self.batch_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def closeEvent(self, event) -> None:
        if hasattr(self, "image_tab"):
            self.image_tab.stop_camera()
        if self._active_thread is not None and self._active_thread.isRunning():
            self._active_thread.requestInterruption()
            self._active_thread.quit()
            self._active_thread.wait(3000)
        super().closeEvent(event)
