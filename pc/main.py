# -*- coding: utf-8 -*-
"""语音输入精简版（仅 SenseVoice）

只保留一个 ASR 模型：SenseVoice Small（sherpa-onnx 离线识别，自带标点 + ITN）。
文本后处理为纯规则流水线（proofreading 阶段 0-8），不依赖任何 LLM / 远程服务。

用法：
    1. 下载模型到 voice-models/（见 scripts/download_models.py）
    2. python main.py

操作：
    - 点击「开始录音」按钮，或按住右 Shift 键 ≥0.3 秒开始录音，松开停止
    - 识别完成后自动粘贴到当前活动窗口（可在设置中关闭）
"""

import sys
import os
import time
import logging
import queue
import gc
import warnings

# 抑制 PyQt5 SIP 的无害 DeprecationWarning（sipPyTypeDict）
warnings.filterwarnings("ignore", message=".*sipPyTypeDict.*", category=DeprecationWarning)


def _preload_msvc_runtime() -> None:
    """Windows：预加载系统版 MSVC 运行时，避免 PyQt5 自带的旧版 DLL 冲突。

    PyQt5 wheel 内置 MSVCP140.dll 14.26（2020），而 onnxruntime / sherpa-onnx
    等用新版运行时构建。两者同进程加载会导致内存布局不一致 → 偶发
    0xc0000005 访问冲突（崩溃位置随机：导入时/录音时/退出时）。

    解决：在导入 PyQt5 之前，先按 DLL 名加载系统版（或 Python 自带）的运行时。
    Windows 对同名 DLL 复用已加载实例，Qt 加载依赖时就不会再拉起 PyQt5
    自带的旧版。没有安装 VC++ Redistributable 的机器会正常回退。
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        for name in (
            "vcruntime140_1.dll",
            "vcruntime140.dll",
            "msvcp140_2.dll",
            "msvcp140_1.dll",
            "msvcp140.dll",
        ):
            try:
                ctypes.windll.LoadLibrary(name)
            except Exception:
                pass
    except Exception:
        pass


_preload_msvc_runtime()

# 导入跨平台工具模块并初始化 CUDA 环境（按平台判断）
from platform_utils import (
    setup_cuda_environment,
    get_model_path,
    get_project_root,
)
setup_cuda_environment()

# 配置 jieba 缓存目录，避免写入系统临时目录
jieba_cache_dir = None
try:
    jieba_cache_dir = os.path.join(get_project_root(), "jieba_cache")
    os.makedirs(jieba_cache_dir, exist_ok=True)
    os.environ["JIEBA_CACHE_DIR"] = jieba_cache_dir
except Exception as e:
    print(f"设置缓存目录失败: {e}")

import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QMessageBox,
    QLabel,
    QComboBox,
    QHBoxLayout,
    QCheckBox,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
    QGroupBox,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.Qt import Qt
import sherpa_onnx

# 模块路径（保证从任意目录启动都能 import 到同目录模块）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from post_processing import (
    record_learned_rule,
    _de_swap_conflicts_rule_only,
)
from proofreading import proofread
from config import config
from input_injector import get_text_injector, reset_text_injector
from hotkey_listener import start_global_hotkey
from platform_utils import get_voice_models_dir
from pynput import keyboard

# 模型下载（首次运行自动下载，HuggingFace 免认证）
from model_downloader import download_model as _download_model, MODEL_DIR_NAME as _MODEL_DIR_NAME

# ---------------------------------------------------------------------------
# SenseVoice 模型目录解析
# 默认：voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/
# （标准版 2024-07-17 int8，约 163MB，自带标点 + ITN；不要用 2025-09-09
#  粤语变体——无标点且普通话识别丢字。）
# 可在 config.json 的 asr.sensevoice_model_dir 中改为已有目录名（相对
# voice-models/）或绝对路径，无需改代码。
# ---------------------------------------------------------------------------
def _resolve_sensevoice_model_dir() -> str:
    """返回 SenseVoice 模型目录的绝对路径（支持相对名或绝对路径）。"""
    model_dir = config.get_sensevoice_model_dir()
    if os.path.isabs(model_dir):
        return model_dir
    return get_model_path(model_dir)


def _timing_enabled() -> bool:
    """耗时打点开关（debug.timing_enabled，默认关）。"""
    try:
        return bool(config.get("debug.timing_enabled", False))
    except Exception:
        return False


def _timing(name: str, t0: float) -> None:
    """打点输出：_timing('ASR 识别(sensevoice)', t0)。"""
    if not _timing_enabled():
        return
    try:
        logging.info(f"[timing] {name}: {(time.perf_counter() - t0) * 1000:.1f} ms")
    except Exception:
        pass


def _apply_rules_only(text: str) -> str:
    """纯规则后处理（阶段 0-8），LLM 润色阶段（阶段 9）显式跳过。

    直接调用 proofreading 引擎并传 llm_caller=None，保证无任何网络请求；
    与完整版流水线（proofread 规则层）行为逐字节一致。
    """
    if not text:
        return text
    try:
        result = proofread(text, mode="basic", llm_caller=None)
        return result.text
    except Exception as e:
        logging.warning(f"校对引擎调用失败，已降级为原文: {e}")
        return text


class AudioRecorder(QThread):
    """录音线程：PortAudio 回调采集 16kHz 单声道 PCM。"""

    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()

    def __init__(self, device=None):
        super().__init__()
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.device = device  # None = 系统默认输入设备

    def run(self):
        self.is_recording = True
        self.recording_started.emit()

        # 使用 callback API 采集音频（兼容所有音频驱动）
        callback_buffer = []

        def _cb(indata, frames, time_info, status):
            if self.is_recording:
                callback_buffer.append(indata.copy())

        stream = None
        fallback_attempted = False
        try:
            try:
                stream = sd.InputStream(
                    device=self.device,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    callback=_cb,
                )
                stream.start()
            except Exception as e:
                logging.error(f"音频设备 '{self.device}' 打开失败: {e}")
                if self.device is not None:
                    logging.info("尝试用系统默认设备回退...")
                    if stream:
                        try:
                            stream.stop()
                            stream.close()
                        except Exception:
                            pass
                        stream = None
                    callback_buffer.clear()
                    stream = sd.InputStream(
                        device=None,
                        channels=1,
                        samplerate=self.sample_rate,
                        blocksize=self.chunk_size,
                        callback=_cb,
                    )
                    stream.start()
                    self.device = None  # 已回退到默认
                    fallback_attempted = True
                    logging.info("已成功回退到系统默认音频设备")
                else:
                    self.audio_data = []
                    raise

            # 保持线程存活，音频数据由回调函数异步收集
            while self.is_recording:
                sd.sleep(50)

            if stream:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                stream = None
            self.audio_data = callback_buffer

        except Exception:
            pass
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                stream = None
            if fallback_attempted:
                self.audio_data = callback_buffer if callback_buffer else []

        self.recording_stopped.emit()

    def stop_recording(self):
        self.is_recording = False

    def set_device(self, device):
        """运行时切换录音设备（None = 系统默认），下次录音生效。"""
        self.device = device


class TranscriptionWorker(QThread):
    """音频转录工作线程：SenseVoice 推理 + 规则后处理。"""

    finished = pyqtSignal(str, str)  # (raw_text, corrected_text)

    def __init__(self, audio_data, sensevoice_model, post_processor):
        super().__init__()
        self.audio_data = audio_data
        self.sensevoice_model = sensevoice_model
        self.post_processor = post_processor

    def run(self):
        stream = None
        try:
            logging.info("工作线程启动，使用 SenseVoice 模型处理音频...")
            _asr_t0 = time.perf_counter()

            text = ""
            stream = self.sensevoice_model.create_stream()
            stream.accept_waveform(
                sample_rate=16000, waveform=self.audio_data.flatten()
            )
            self.sensevoice_model.decode_stream(stream)
            result = stream.result
            text = result.text if hasattr(result, "text") else str(result)
            logging.info(f"SenseVoice转录结果: '{text}'")
            _timing("ASR 识别(sensevoice)", _asr_t0)

            # 纯规则后处理（无 LLM）
            _pp_t0 = time.perf_counter()
            try:
                processed_text = self.post_processor(text)
                logging.info(f"后处理后文本: '{processed_text}'")
                self.finished.emit(text, processed_text)
            except Exception as e:
                logging.error(f"后处理失败，使用原始文本: {e}")
                self.finished.emit(text, text)
            _timing("文本后处理(全部规则)", _pp_t0)

        except Exception as e:
            logging.error(f"工作线程致命错误: {e}")
            self.finished.emit(f"处理错误: {e}", f"处理错误: {e}")
        finally:
            try:
                if stream is not None:
                    del stream
                self.audio_data = None
            except Exception:
                pass
            gc.collect()


class ModelDownloadWorker(QThread):
    """后台下载 SenseVoice 模型（首次运行时自动下载，HuggingFace 免认证）。"""

    progress = pyqtSignal(int, int, int, int)  # done, total, file_index, file_count
    finished_dl = pyqtSignal(bool, str)         # ok, message

    def __init__(self, target_root: str, parent=None):
        super().__init__(parent)
        self.target_root = target_root

    def run(self):
        try:
            ok, msg = _download_model(self.target_root, progress_cb=self._on_progress)
        except Exception as e:
            ok, msg = False, f"下载异常: {type(e).__name__}: {e}"
        self.finished_dl.emit(ok, msg)

    def _on_progress(self, done, total, idx, n):
        self.progress.emit(done, total, idx, n)


class MainWindow(QMainWindow):
    # 跨线程信号（热键监听线程 → 主线程）
    start_recording_signal = pyqtSignal()
    stop_recording_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音输入（SenseVoice 精简版）")
        self.setGeometry(100, 100, 640, 600)
        self.setMinimumSize(560, 540)

        self.sensevoice_model = None

        # 创建 UI
        self.init_ui()

        # 初始化录音器
        saved_device = config.get_audio_input_device()
        self.recorder = AudioRecorder(device=saved_device if saved_device else None)
        self.recorder.recording_started.connect(self.on_recording_started)
        self.recorder.recording_stopped.connect(self.on_recording_stopped)

        # 连接跨线程信号
        self.start_recording_signal.connect(self.start_recording)
        self.stop_recording_signal.connect(self.stop_recording)

        # 后处理器（纯规则；TextPostProcessor 仅用于兼容接口，实际走 _apply_rules_only）
        self.post_processor = _apply_rules_only

        # 用于采纳/拒绝修正的原始/修正文本
        self._last_raw_text = ""
        self._last_corrected_text = ""

        # 首次运行自动下载（模型缺失时懒启动）
        self._download_worker = None

        # 加载模型
        self.load_models()

        # 刷新麦克风列表
        self._refresh_mic_list()

        # 全局快捷键监听：右 Shift 长按开始录音、松开停止
        self.shift_pressed_time = None
        self.shift_active = False
        self.keyboard_listener = start_global_hotkey(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )

        # 闲置自动卸载 ASR 模型：闲置 N 分钟后释放内存/显存，下次录音 lazy 重载
        self._last_asr_activity_ts = time.time()
        self._models_unloaded_by_idle = False
        self._idle_check_timer = QTimer(self)
        self._idle_check_timer.setInterval(60 * 1000)
        self._idle_check_timer.timeout.connect(self._check_idle_and_unload)
        if config.get_idle_unload_minutes() > 0:
            self._idle_check_timer.start()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # 标题
        title_label = QLabel("语音输入系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label)

        # 控制行：引擎 + 设置
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("识别引擎:"))
        self.asr_type_label = QLabel()
        self.asr_type_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        self.asr_type_label.setText("SenseVoice Small")
        control_layout.addWidget(self.asr_type_label)
        control_layout.addStretch()
        self.settings_button = QPushButton("设置")
        self.settings_button.clicked.connect(self.show_settings)
        control_layout.addWidget(self.settings_button)
        layout.addLayout(control_layout)

        # 麦克风行
        mic_layout = QHBoxLayout()
        mic_layout.addWidget(QLabel("🎤 麦克风:"))
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(280)
        self.mic_combo.setToolTip(
            "选择录音使用的麦克风设备。『默认设备』= 系统默认输入设备"
        )
        self.mic_combo.currentIndexChanged.connect(self._on_mic_changed)
        mic_layout.addWidget(self.mic_combo)
        self.mic_refresh_btn = QPushButton("🔄 刷新")
        self.mic_refresh_btn.setToolTip("重新扫描音频设备")
        self.mic_refresh_btn.clicked.connect(self._refresh_mic_list)
        self.mic_refresh_btn.setMaximumWidth(60)
        mic_layout.addWidget(self.mic_refresh_btn)
        mic_layout.addStretch()
        layout.addLayout(mic_layout)

        # 录音按钮
        self.record_button = QPushButton("开始录音")
        self.record_button.setStyleSheet(
            "font-size: 16px; padding: 10px; background-color: #4CAF50; color: white;"
        )
        self.record_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_button)

        # 状态标签
        self.status_label = QLabel("点击按钮开始录音，或按住右 Shift 键")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # 识别结果
        layout.addWidget(QLabel("识别结果:"))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(120)
        layout.addWidget(self.result_text)

        # 修正按钮行：采纳 / 拒绝 / 重新粘贴
        action_layout = QHBoxLayout()
        self.accept_correction_btn = QPushButton("✓ 采纳修正")
        self.accept_correction_btn.clicked.connect(self.on_accept_correction)
        self.accept_correction_btn.setEnabled(False)
        action_layout.addWidget(self.accept_correction_btn)
        self.reject_correction_btn = QPushButton("✗ 拒绝修正")
        self.reject_correction_btn.clicked.connect(self.on_reject_correction)
        self.reject_correction_btn.setEnabled(False)
        action_layout.addWidget(self.reject_correction_btn)
        self.repaste_btn = QPushButton("↻ 重新粘贴")
        self.repaste_btn.clicked.connect(self.on_repaste)
        self.repaste_btn.setEnabled(False)
        action_layout.addWidget(self.repaste_btn)
        layout.addLayout(action_layout)

        # 手动修正
        layout.addWidget(QLabel("手动修正（保存后自动学习规则）:"))
        self.manual_edit = QTextEdit()
        self.manual_edit.setMinimumHeight(80)
        layout.addWidget(self.manual_edit)
        self.save_manual_btn = QPushButton("保存手动修正")
        self.save_manual_btn.clicked.connect(self.on_save_manual_correction)
        self.save_manual_btn.setEnabled(False)
        layout.addWidget(self.save_manual_btn)

        # 关键：把布局挂到 central widget 上，否则控件不会在窗口中显示
        central_widget.setLayout(layout)

    # ------------------------------------------------------------------
    # 麦克风
    # ------------------------------------------------------------------
    def _refresh_mic_list(self):
        current = config.get_audio_input_device()
        self.mic_combo.blockSignals(True)
        self.mic_combo.clear()
        self.mic_combo.addItem("默认设备", None)
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    name = dev["name"]
                    ch = dev["max_input_channels"]
                    self.mic_combo.addItem(f"{name} ({ch}ch)", str(i))
            idx = self.mic_combo.findData(current)
            if idx >= 0:
                self.mic_combo.setCurrentIndex(idx)
        except Exception as e:
            logging.warning(f"枚举音频设备失败: {e}")
        self.mic_combo.blockSignals(False)

    def _on_mic_changed(self, index):
        dev = self.mic_combo.itemData(index)
        config.set_audio_input_device(dev if dev is not None else "")
        self.recorder.set_device(int(dev) if dev is not None else None)

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load_models(self):
        self.sensevoice_model = None
        gc.collect()
        if self.load_sensevoice_model():
            self.status_label.setText("SenseVoice 模型加载成功，可以开始录音")
        else:
            self.status_label.setText("SenseVoice 模型加载失败")
            self._offer_model_download()

    # ------------------------------------------------------------------
    # 首次运行：模型自动下载（HuggingFace 免认证）
    # ------------------------------------------------------------------
    def _offer_model_download(self):
        """模型缺失时弹窗询问是否下载；是则后台下载，完成后自动重载。"""
        if self._download_worker is not None and self._download_worker.isRunning():
            return
        reply = QMessageBox.question(
            self,
            "模型未找到",
            "未找到 SenseVoice 模型（约 163MB）。\n\n"
            "是否现在从 HuggingFace 自动下载？\n"
            "（下载一次即可，之后本地运行无需网络）",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            self.status_label.setText(
                "模型未加载。可运行 scripts/download_models.py 或修改 config.json 的 asr.sensevoice_model_dir"
            )
            return

        self.status_label.setText("正在下载模型（约 163MB）...")
        self._download_worker = ModelDownloadWorker(get_voice_models_dir(), self)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished_dl.connect(self._on_download_finished)
        self._download_worker.start()

    def _on_download_progress(self, done, total, idx, n):
        if total > 0:
            pct = done * 100 // total
            self.status_label.setText(
                f"正在下载模型 [{idx + 1}/{n}] {done // 1024 // 1024}/{total // 1024 // 1024} MB ({pct}%)"
            )

    def _on_download_finished(self, ok, msg):
        if ok:
            self.status_label.setText("模型下载完成，正在加载...")
            if self.load_sensevoice_model():
                self.status_label.setText("SenseVoice 模型加载成功，可以开始录音")
            else:
                self.status_label.setText(f"模型已下载但加载失败: {msg}")
        else:
            self.status_label.setText(f"模型下载失败: {msg}")
            QMessageBox.warning(self, "下载失败", msg)

    def load_sensevoice_model(self):
        """加载 SenseVoice Small（sherpa-onnx，自带标点 + ITN）。"""
        try:
            model_dir = _resolve_sensevoice_model_dir()
            model_path = os.path.join(model_dir, "model.int8.onnx")
            tokens_path = os.path.join(model_dir, "tokens.txt")
            if not os.path.exists(model_path):
                logging.warning(f"SenseVoice 模型文件不存在: {model_path}")
                return False

            self.sensevoice_model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model_path,
                tokens=tokens_path,
                decoding_method="greedy_search",
                num_threads=config.get_sensevoice_num_threads(),
                provider=self._get_safe_provider(),
                use_itn=config.is_sensevoice_itn_enabled(),
            )
            logging.info("SenseVoice 模型加载成功")
            return True
        except Exception as e:
            logging.warning(f"SenseVoice 模型加载失败: {str(e)}")
            return False

    _cuda_fallback_used = False

    def _get_safe_provider(self) -> str:
        """安全获取 provider：CUDA 失败过或未启用则返回 cpu。"""
        if self._cuda_fallback_used or not config.is_gpu_enabled():
            return "cpu"
        return "cuda"

    # ------------------------------------------------------------------
    # 闲置自动卸载
    # ------------------------------------------------------------------
    def _check_idle_and_unload(self) -> None:
        idle_min = config.get_idle_unload_minutes()
        if idle_min <= 0:
            return
        elapsed = time.time() - self._last_asr_activity_ts
        if elapsed < idle_min * 60:
            return
        if self._models_unloaded_by_idle:
            return
        logging.info(f"ASR 模型闲置 {elapsed/60:.1f} 分钟，自动卸载释放资源...")
        self.sensevoice_model = None
        self._models_unloaded_by_idle = True
        self.status_label.setText(f"⏸ ASR 模型已卸载（闲置 {idle_min}min），按录音自动重载")
        gc.collect()

    def _ensure_asr_model_loaded(self) -> None:
        if not self._models_unloaded_by_idle:
            return
        logging.info("检测到 ASR 模型已被闲置卸载，执行 lazy 重载...")
        self.status_label.setText("加载模型中...")
        QApplication.processEvents()
        self.load_models()
        self._models_unloaded_by_idle = False
        self._last_asr_activity_ts = time.time()

    # ------------------------------------------------------------------
    # 录音控制
    # ------------------------------------------------------------------
    def toggle_recording(self):
        if not self.recorder.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self._ensure_asr_model_loaded()
        self._last_asr_activity_ts = time.time()
        if config.get_idle_unload_minutes() > 0 and not self._idle_check_timer.isActive():
            self._idle_check_timer.start()
        self.recorder.start()
        self.shift_active = True
        self.status_label.setText("正在录音...")
        self.record_button.setText("停止录音")

    def stop_recording(self):
        self.recorder.stop_recording()
        self.shift_active = False
        self.status_label.setText("录音完成，正在处理...")
        self.record_button.setText("开始录音")

    def on_recording_started(self):
        pass

    def on_recording_stopped(self):
        logging.info("录音停止，准备处理...")
        self.status_label.setText("正在识别中...")

        if not self.recorder.audio_data:
            logging.warning("没有录音数据")
            self.status_label.setText("录音时间太短")
            return

        try:
            audio_data = np.concatenate(self.recorder.audio_data, axis=0)
            logging.info(f"音频数据形状: {audio_data.shape}")
            self.recorder.audio_data = []

            if self.sensevoice_model is None:
                self.status_label.setText("模型未加载，请先检查模型文件")
                return

            self.worker = TranscriptionWorker(
                audio_data,
                self.sensevoice_model,
                self.post_processor,
            )
            self.worker.finished.connect(self.on_transcription_finished)
            self.worker.start()

            # 超时保护
            self._transcription_start_ts = time.time()
            self._transcription_timer = QTimer(self)
            self._transcription_timer.setInterval(60 * 1000)
            self._transcription_timer.setSingleShot(True)
            self._transcription_timer.timeout.connect(self._on_transcription_timeout)
            self._transcription_timer.start()

        except Exception as e:
            logging.error(f"启动处理线程失败: {e}")
            self.status_label.setText("处理启动失败")

    def _on_transcription_timeout(self) -> None:
        elapsed = time.time() - getattr(self, '_transcription_start_ts', time.time())
        logging.error(f"转录超时（{elapsed:.0f}s），尝试终止工作线程...")
        self.status_label.setText(f"⚠ 识别超时（{elapsed:.0f}s），可能模型推理卡住。正在强制停止...")
        if getattr(self, 'worker', None) is not None:
            try:
                self.worker.terminate()
                self.worker.wait(3000)
            except Exception as e:
                logging.error(f"终止工作线程失败: {e}")
            self.worker = None
        gc.collect()
        self.status_label.setText("⚠ 识别超时，请重试。如果反复出现，请检查模型文件是否完整。")

    def on_transcription_finished(self, raw_text, corrected_text):
        if getattr(self, '_transcription_timer', None) is not None:
            self._transcription_timer.stop()
        self._handle_transcription_result(raw_text, corrected_text)
        self.worker = None
        gc.collect()

    def _handle_transcription_result(self, raw_text, corrected_text):
        self._last_raw_text = raw_text
        self._last_corrected_text = corrected_text
        self.update_text(corrected_text)
        self.accept_correction_btn.setEnabled(True)
        self.reject_correction_btn.setEnabled(True)
        self.repaste_btn.setEnabled(True)
        self.manual_edit.setPlainText(corrected_text)
        self.save_manual_btn.setEnabled(True)
        self._last_asr_activity_ts = time.time()

    def update_text(self, text):
        self.result_text.setText(text)
        self.status_label.setText("处理完成")

        if not config.is_input_injection_enabled():
            return

        # 焦点防护：前台窗口是本 App 自己时跳过粘贴（可点『重新粘贴』）
        try:
            import pygetwindow as gw
            aw = gw.getActiveWindow()
            if aw is not None and aw.title == self.windowTitle():
                self.status_label.setText(
                    "识别完成 ✓（焦点在本窗口，已跳过粘贴。请先切换到目标窗口，再点『重新粘贴』）"
                )
                return
        except Exception:
            pass

        injector = get_text_injector(config)
        if injector.inject_text_safe(text):
            self.status_label.setText("处理完成，已自动粘贴")
        else:
            err = getattr(injector, 'last_error', '') or ''
            if err:
                self.status_label.setText(f"处理完成，自动粘贴失败：{err[:60]}")
            else:
                self.status_label.setText("处理完成，自动粘贴失败")

    def on_repaste(self):
        if not self._last_corrected_text:
            self.status_label.setText("没有可粘贴的内容")
            return
        if not config.is_input_injection_enabled():
            self.status_label.setText("全局输入注入未启用（设置 → 启用输入注入）")
            return
        injector = get_text_injector(config)
        if injector.inject_text_safe(self._last_corrected_text):
            self.status_label.setText("已重新粘贴到当前窗口")
        else:
            err = getattr(injector, 'last_error', '') or ''
            self.status_label.setText(f"重新粘贴失败：{err[:60]}" if err else "重新粘贴失败")

    # ------------------------------------------------------------------
    # 手动修正 / 采纳 / 拒绝（档位4 自学习）
    # ------------------------------------------------------------------
    def on_save_manual_correction(self):
        if not self._last_raw_text:
            self.status_label.setText("无原始识别文本")
            return
        manual_text = self.manual_edit.toPlainText().strip()
        if not manual_text:
            self.status_label.setText("编辑框为空")
            return
        raw = self._last_raw_text.strip()
        corrected = manual_text
        if raw == corrected:
            self.status_label.setText("内容未变化，无需保存")
            return
        # 手动保存：用户明确纠正，直接升级为全局即时生效（不跨句子累计）
        self._learn_correction(raw, corrected, promote_now=True)
        self.save_manual_btn.setEnabled(False)

    def on_accept_correction(self):
        if not self._last_raw_text or not self._last_corrected_text:
            return
        if self._last_raw_text.strip() == self._last_corrected_text.strip():
            self.status_label.setText("无修正内容，无需保存")
            self._disable_correction_buttons()
            return
        self._learn_correction(self._last_raw_text.strip(), self._last_corrected_text.strip())
        self._disable_correction_buttons()

    def _learn_correction(self, raw: str, corrected: str, promote_now: bool = False):
        """档位4 自学习：diff 抽取 → 登记规则（同音/形近变体 + 新词入表）。

        promote_now=True 时（手动保存）：登记的精确规则直接升级为全局实时生效；
        False（采纳）时走跨句子累计（≥2 次才全局），防误学。
        """
        diffs = []
        promoted_count = 0
        variant_count = 0
        shape_count = 0
        new_words = []
        try:
            from post_processing import (
                extract_general_diff,
                expand_homophone_variants,
                expand_shape_variants,
                add_word_to_user_dict,
                _MAX_LEARNED_WORDS_PER_SAVE,
            )
            diffs = extract_general_diff(raw, corrected)
            for old, new in diffs:
                action = record_learned_rule(old, new, raw, kind='exact', promote_now=promote_now)
                if action == 'promoted':
                    promoted_count += 1
                logging.info(f'档位4 自学习[{action}]: "{old}" → "{new}"')
                for variant in expand_homophone_variants(old, new):
                    record_learned_rule(variant, new, raw, kind='variant', base=(old, new))
                    variant_count += 1
                for variant in expand_shape_variants(old, new):
                    record_learned_rule(variant, new, raw, kind='variant', base=(old, new))
                    shape_count += 1
                if (len(new_words) < _MAX_LEARNED_WORDS_PER_SAVE
                        and add_word_to_user_dict(new)):
                    new_words.append(new)
        except Exception as e:
            logging.warning(f'档位4 diff 抽取失败: {e}')

        if diffs:
            msg_parts = [f"已学习 {len(diffs)} 条修正"]
            if not promote_now:
                pending = len(diffs) - promoted_count
                if pending > 0:
                    msg_parts.append(f"{pending} 条待确认")
            if promoted_count:
                msg_parts.append(f"{promoted_count} 条升级为全局")
            total_variants = variant_count + shape_count
            if total_variants:
                msg_parts.append(f"{total_variants} 条变体已登记")
            if new_words:
                msg_parts.append(f"新词已入表:{'、'.join(new_words)}")
            self.status_label.setText("，".join(msg_parts))
        elif _de_swap_conflicts_rule_only(raw, corrected):
            self.status_label.setText("修正与规则层判定相悖，未学习（规则层判对处保持原样）")
        elif len(raw) <= 50 and len(corrected) <= 50:
            action = record_learned_rule(raw, corrected, raw, kind='exact', promote_now=promote_now)
            if action == 'promoted':
                self.status_label.setText(f"整句修正已确认，立即生效: \"{raw}\" → \"{corrected}\"")
            elif promote_now:
                self.status_label.setText(f"已记录整句修正并立即生效: \"{raw}\" → \"{corrected}\"")
            else:
                self.status_label.setText(f"已记录整句修正（待确认）: \"{raw}\" → \"{corrected}\"")
        else:
            self.status_label.setText("文本较长，仅记录日志，建议手动添加到设置")
            logging.info(f"手动保存长文本修正: {raw} -> {corrected}")

    def on_reject_correction(self):
        if self._last_raw_text:
            self.update_text(self._last_raw_text)
            self.status_label.setText("已拒绝修正，恢复原始识别结果")
        else:
            self.status_label.setText("无原始文本可恢复")
        self._disable_correction_buttons()

    def _disable_correction_buttons(self):
        self.accept_correction_btn.setEnabled(False)
        self.reject_correction_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # 全局热键：右 Shift 长按开始、松开停止
    # ------------------------------------------------------------------
    def on_key_press(self, key):
        try:
            if key == keyboard.Key.shift_r:
                if self.shift_pressed_time is None:
                    self.shift_pressed_time = time.time()
        except AttributeError:
            pass

    def on_key_release(self, key):
        try:
            if key == keyboard.Key.shift_r:
                if self.shift_pressed_time is not None:
                    duration = time.time() - self.shift_pressed_time
                    if self.recorder.is_recording:
                        self.shift_active = False
                        logging.info("触发停止录音")
                        self.stop_recording_signal.emit()
                    else:
                        if duration >= 0.3:
                            self.shift_active = True
                            logging.info("触发开始录音")
                            self.start_recording_signal.emit()
                self.shift_pressed_time = None
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # 设置
    # ------------------------------------------------------------------
    def show_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec_():
            # 注入配置可能变化，重置注入器单例缓存
            try:
                reset_text_injector()
            except Exception:
                pass
            # 模型目录变更 → 重新加载模型
            if dialog.model_dir_changed:
                self.status_label.setText("模型目录已变更，正在重新加载模型...")
                QApplication.processEvents()
                self.load_models()
            self.status_label.setText("设置已保存")


class SettingsDialog(QDialog):
    """设置对话框：输入注入、SenseVoice 参数、后处理开关、热词/替换表。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(560)
        self.model_dir_changed = False
        self.init_ui()
        self.load_values()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- 输入注入 ---
        inj_group = QGroupBox("输入注入")
        inj_layout = QFormLayout()
        self.inj_enabled = QCheckBox("识别完成后自动粘贴到当前活动窗口")
        inj_layout.addRow(self.inj_enabled)
        self.inj_method = QComboBox()
        for m, label in [("auto", "自动"), ("sendinput", "SendInput (Windows)"),
                         ("xdotool", "xdotool (Linux X11)"), ("wtype", "wtype (Linux Wayland)")]:
            self.inj_method.addItem(label, m)
        inj_layout.addRow("注入方法:", self.inj_method)
        self.inj_clipboard = QCheckBox("使用剪贴板注入")
        inj_layout.addRow(self.inj_clipboard)
        self.inj_delay = QDoubleSpinBox()
        self.inj_delay.setRange(0.0, 5.0)
        self.inj_delay.setSingleStep(0.05)
        self.inj_delay.setDecimals(2)
        inj_layout.addRow("注入延迟(秒):", self.inj_delay)
        inj_group.setLayout(inj_layout)
        layout.addWidget(inj_group)

        # --- SenseVoice ---
        sv_group = QGroupBox("SenseVoice")
        sv_layout = QFormLayout()
        self.sv_model_dir = QLineEdit()
        self.sv_model_dir.setToolTip(
            "模型目录：相对名（voice-models/ 下）或绝对路径，\n"
            "默认为官方标准版目录名，通常无需修改。"
        )
        sv_layout.addRow("模型目录:", self.sv_model_dir)
        self.sv_itn = QCheckBox("启用 ITN（数字/日期/货币反标点规范化）")
        sv_layout.addRow(self.sv_itn)
        self.sv_threads = QSpinBox()
        self.sv_threads.setRange(1, 32)
        sv_layout.addRow("推理线程数:", self.sv_threads)
        self.sv_gpu = QCheckBox("启用 GPU 加速（需 CUDA 版 sherpa-onnx）")
        sv_layout.addRow(self.sv_gpu)
        self.idle_unload = QSpinBox()
        self.idle_unload.setRange(0, 240)
        sv_layout.addRow("闲置自动卸载(分钟, 0=禁用):", self.idle_unload)
        sv_group.setLayout(sv_layout)
        layout.addWidget(sv_group)

        # --- 后处理 ---
        pp_group = QGroupBox("文本后处理（纯规则）")
        pp_layout = QFormLayout()
        self.pp_number = QCheckBox("数字转换（如 一百 → 100）")
        pp_layout.addRow(self.pp_number)
        self.pp_confusion = QCheckBox("同音字纠错（ConfusionSet + jieba）")
        pp_layout.addRow(self.pp_confusion)
        pp_group.setLayout(pp_layout)
        layout.addWidget(pp_group)

        # --- 热词 / 替换 ---
        term_group = QGroupBox("术语表（每行一条）")
        term_layout = QVBoxLayout()
        term_layout.addWidget(QLabel("热词（格式：词 分数，如：人工智能 2.0）:"))
        self.hotwords_edit = QPlainTextEdit()
        self.hotwords_edit.setMaximumHeight(90)
        term_layout.addWidget(self.hotwords_edit)
        term_layout.addWidget(QLabel("替换（格式：原词：新词 或 原词→新词，如：语系：语音）:"))
        self.replacements_edit = QPlainTextEdit()
        self.replacements_edit.setMaximumHeight(90)
        term_layout.addWidget(self.replacements_edit)
        term_group.setLayout(term_layout)
        layout.addWidget(term_group)

        # --- 按钮 ---
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save_values)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def load_values(self):
        self.sv_model_dir.setText(config.get_sensevoice_model_dir())
        self.inj_enabled.setChecked(config.is_input_injection_enabled())
        method = config.get_injection_method()
        idx = self.inj_method.findData(method)
        self.inj_method.setCurrentIndex(idx if idx >= 0 else 0)
        self.inj_clipboard.setChecked(config.is_injection_use_clipboard())
        self.inj_delay.setValue(config.get_injection_delay())
        self.sv_itn.setChecked(config.is_sensevoice_itn_enabled())
        self.sv_threads.setValue(config.get_sensevoice_num_threads())
        self.sv_gpu.setChecked(config.is_gpu_enabled())
        self.idle_unload.setValue(config.get_idle_unload_minutes())
        self.pp_number.setChecked(config.is_number_conversion_enabled())
        self.pp_confusion.setChecked(config.get('confusion_set.enabled', True))

        hot_lines = []
        for h in config.get_hotwords():
            word = h.get("word", "")
            score = h.get("score", 2.0)
            if word:
                hot_lines.append(f"{word} {score}")
        self.hotwords_edit.setPlainText("\n".join(hot_lines))

        rep_lines = []
        for r in config.get_replacements():
            old = r.get("old", "")
            new = r.get("new", "")
            if old:
                rep_lines.append(f"{old}：{new}")
        self.replacements_edit.setPlainText("\n".join(rep_lines))

    def save_values(self):
        new_model_dir = self.sv_model_dir.text().strip()
        if new_model_dir and new_model_dir != config.get_sensevoice_model_dir():
            config.set_sensevoice_model_dir(new_model_dir)
            self.model_dir_changed = True
        config.set_input_injection(self.inj_enabled.isChecked())
        method = self.inj_method.itemData(self.inj_method.currentIndex())
        config.set_injection_method(method or "auto")
        config.set_injection_use_clipboard(self.inj_clipboard.isChecked())
        config.set_injection_delay(self.inj_delay.value())
        config.set_sensevoice_itn(self.sv_itn.isChecked())
        config.set_sensevoice_num_threads(self.sv_threads.value())
        config.set_gpu_enabled(self.sv_gpu.isChecked())
        config.set_idle_unload_minutes(self.idle_unload.value())
        config.set_number_conversion_enabled(self.pp_number.isChecked())
        config.set('confusion_set.enabled', self.pp_confusion.isChecked())

        # 热词
        hotwords = []
        for line in self.hotwords_edit.toPlainText().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0]
            score = 2.0
            if len(parts) > 1:
                try:
                    score = float(parts[1])
                except ValueError:
                    pass
            hotwords.append({"word": word, "score": score})
        config.set_hotwords(hotwords)

        # 替换（支持分隔符：中文冒号 / 英文冒号 / 箭头 → ->）
        replacements = []
        for line in self.replacements_edit.toPlainText().splitlines():
            line = line.strip()
            if not line:
                continue
            # 从最靠前的分隔符切分，避免值本身含冒号
            sep = None
            idx = None
            for cand in ("→", "->", "：", ":"):
                pos = line.find(cand)
                if pos >= 0 and (idx is None or pos < idx):
                    idx = pos
                    sep = cand
            if sep is None:
                continue
            old = line[:idx].strip()
            new = line[idx + len(sep):].strip()
            if old:
                replacements.append({"old": old, "new": new})
        config.set_replacements(replacements)

        self.accept()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"项目根目录: {get_project_root()}")

    # 模型文件存在性预检（友好提示）
    model_path = os.path.join(_resolve_sensevoice_model_dir(), "model.int8.onnx")
    if not os.path.exists(model_path):
        print("=" * 60)
        print("未找到 SenseVoice 模型文件:")
        print(f"  {model_path}")
        print("请先运行:  python scripts/download_models.py")
        print("或修改 config.json 中的 asr.sensevoice_model_dir 指向已有模型目录")
        print("=" * 60)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
