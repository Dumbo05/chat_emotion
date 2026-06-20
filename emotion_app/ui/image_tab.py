from __future__ import annotations

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from emotion_app.domain import EMOTIONS, EMOTION_LABELS_ZH, RecognitionResult


class ImageRecognitionTab(QWidget):
    def __init__(self, recognizer, run_task, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.run_task = run_task
        self.selected_image = ""
        self.camera = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_camera)
        self.probability_bars = {}
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)
        left_group = QGroupBox("图像 / 摄像头")
        left = QVBoxLayout(left_group)
        status = QLabel(self.recognizer.status)
        status.setWordWrap(True)
        status.setStyleSheet("background: #eaf6ee; padding: 10px; border-radius: 7px;" if self.recognizer.available else "background: #fff6dc; padding: 10px; border-radius: 7px;")
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        actions = QHBoxLayout()
        choose = QPushButton("导入图像")
        self.predict_button = QPushButton("识别图像")
        self.camera_button = QPushButton("启动摄像头")
        self.predict_button.setEnabled(self.recognizer.available)
        self.camera_button.setEnabled(self.recognizer.available)
        choose.clicked.connect(self._choose_image)
        self.predict_button.clicked.connect(self._predict_image)
        self.camera_button.clicked.connect(self._toggle_camera)
        for button in (choose, self.predict_button, self.camera_button):
            actions.addWidget(button)
        self.preview = QLabel("导入一张人脸图像，或启动摄像头进行实时识别")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(500, 380)
        self.preview.setStyleSheet("background: #172033; border: 1px solid #b7c3d8; border-radius: 8px; color: white;")
        left.addWidget(status)
        left.addWidget(self.path_edit)
        left.addLayout(actions)
        left.addWidget(self.preview, 1)
        layout.addWidget(left_group, 3)
        result_group = QGroupBox("图像识别结果")
        result = QVBoxLayout(result_group)
        self.result_label = QLabel("等待识别")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #315fdb;")
        self.confidence_label = QLabel("置信度：--")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        result.addWidget(self.result_label)
        result.addWidget(self.confidence_label)
        for emotion in EMOTIONS:
            row = QHBoxLayout()
            name = QLabel(EMOTION_LABELS_ZH[emotion])
            name.setFixedWidth(45)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setFormat("0.0%")
            self.probability_bars[emotion] = bar
            row.addWidget(name)
            row.addWidget(bar)
            result.addLayout(row)
        result.addStretch(1)
        layout.addWidget(result_group, 2)

    def _choose_image(self):
        selected, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像 (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not selected:
            return
        self.stop_camera()
        self.selected_image = selected
        self.path_edit.setText(selected)
        pixmap = QPixmap(selected)
        self.preview.setPixmap(pixmap.scaled(620, 430, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _predict_image(self):
        if not self.selected_image:
            QMessageBox.warning(self, "文件错误", "请先导入需要识别的图像。")
            return
        self.predict_button.setEnabled(False)
        def completed(result):
            self.predict_button.setEnabled(True)
            self.show_result(result)
            if not result.ok:
                QMessageBox.warning(self, "识别失败", result.error or self.recognizer.status)
        self.run_task(lambda: self.recognizer.predict(self.selected_image), completed)

    def show_result(self, result: RecognitionResult):
        if not result.ok:
            self.result_label.setText("未识别到人脸")
            self.confidence_label.setText("置信度：--")
            for bar in self.probability_bars.values():
                bar.setValue(0)
                bar.setFormat("0.0%")
            return
        self.result_label.setText(EMOTION_LABELS_ZH[result.emotion or "neutral"])
        self.confidence_label.setText(f"置信度：{result.confidence:.2%}")
        for emotion, bar in self.probability_bars.items():
            value = float(result.probabilities.get(emotion, 0.0))
            bar.setValue(round(value * 1000))
            bar.setFormat(f"{value:.1%}")

    def _toggle_camera(self):
        if self.camera is not None:
            self.stop_camera()
            return
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        camera = cv2.VideoCapture(0, backend)
        if not camera.isOpened():
            camera.release()
            QMessageBox.warning(self, "摄像头错误", "无法打开默认摄像头，请检查权限或是否被其他程序占用。")
            return
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera = camera
        self.camera_button.setText("停止摄像头")
        self.path_edit.setText("实时摄像头 0")
        self.timer.start(100)

    def stop_camera(self):
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.camera_button.setText("启动摄像头")

    def _update_camera(self):
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.stop_camera()
            QMessageBox.warning(self, "摄像头错误", "摄像头画面读取失败。")
            return
        try:
            faces = self.recognizer.predict_frame(frame)
            for face in faces:
                x, y, width, height = face.box
                cv2.rectangle(frame, (x, y), (x + width, y + height), (49, 95, 219), 2)
                text = f"{face.result.emotion} {face.result.confidence:.0%}"
                cv2.putText(frame, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (49, 95, 219), 2, cv2.LINE_AA)
            if faces:
                self.show_result(max(faces, key=lambda item: item.box[2] * item.box[3]).result)
        except Exception as exc:
            self.stop_camera()
            QMessageBox.warning(self, "实时识别失败", str(exc))
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)
