"""跨平台工具模块 - 运行时检测操作系统并提供平台相关的辅助函数。

- 模型路径：统一使用项目根目录相对路径（voice-input-lite/voice-models/）。
- CUDA：Windows 预加载 onnxruntime CUDA DLL；Linux/macOS 不做额外处理。
- 权限：Windows 检查管理员；Linux/macOS 检查 root/sudo。
"""

import os
import platform
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# 打包环境识别
# ---------------------------------------------------------------------------

def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包的 exe 中（sys.frozen 由 PyInstaller 注入）。"""
    return bool(getattr(sys, "frozen", False))


def get_app_dir() -> str:
    """应用数据目录（可写）。

    - 源码模式：pc/ 目录（config.json、user_data 所在处）
    - 打包 exe：exe 所在目录（用户把 config.json / user_data / voice-models 放 exe 旁）
    """
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 平台检测
# ---------------------------------------------------------------------------

def get_platform() -> str:
    """返回当前平台标识：'windows' / 'linux' / 'macos' / 'unknown'"""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "linux":
        return "linux"
    elif system == "darwin":
        return "macos"
    return "unknown"


def is_windows() -> bool:
    return get_platform() == "windows"


def is_linux() -> bool:
    return get_platform() == "linux"


def is_macos() -> bool:
    return get_platform() == "macos"


# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Optional[str] = None


def get_project_root() -> str:
    """返回项目根目录。

    - 源码模式：voice-input-lite/（pc/ 的上一级）
    - 打包 exe：exe 所在目录（voice-models 放 exe 旁）
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        if is_frozen():
            _PROJECT_ROOT = os.path.dirname(os.path.abspath(sys.executable))
        else:
            _PROJECT_ROOT = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../")
            )
    return _PROJECT_ROOT


def get_voice_models_dir() -> str:
    """返回 voice-models 目录路径。"""
    return os.path.join(get_project_root(), "voice-models")


def get_model_path(relative_path: str) -> str:
    """返回相对于 voice-models/ 目录的模型文件/目录完整路径。

    Args:
        relative_path: 使用 POSIX 斜杠，例如：
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09/model.int8.onnx"
    """
    return os.path.join(get_voice_models_dir(), *relative_path.split("/"))


# ---------------------------------------------------------------------------
# 权限检查
# ---------------------------------------------------------------------------

def is_admin() -> bool:
    """跨平台检查是否拥有管理员/root权限。"""
    if is_windows():
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    else:
        try:
            return os.geteuid() == 0  # type: ignore[attr-defined]
        except AttributeError:
            return False


# ---------------------------------------------------------------------------
# CUDA / GPU 环境
# ---------------------------------------------------------------------------

def preload_onnx_cuda_dlls() -> None:
    """在支持的平台上预加载 ONNX Runtime CUDA 相关动态库。

    onnxruntime.preload_dlls() 主要是 Windows 行为；在 Linux 上调用会
    触发 CUDA/CuDNN 库搜索，即使配置使用 CPU 也会打印缺失警告，
    因此仅在 Windows 下执行。
    """
    if not is_windows():
        return
    try:
        import onnxruntime
        onnxruntime.preload_dlls()
    except Exception:
        pass


def setup_cuda_environment() -> None:
    """入口函数：按平台初始化 CUDA 相关环境。

    Linux 上 CUDA 库通常已在 ldconfig 路径中，无需手动注入 PATH。
    """
    preload_onnx_cuda_dlls()
