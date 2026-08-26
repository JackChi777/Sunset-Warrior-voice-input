"""跨平台文本注入抽象层 - 自动检测操作系统并选择合适的注入器"""
import platform
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional


class TextInjector(ABC):
    """文本注入器基类"""

    @abstractmethod
    def inject_text(self, text: str, delay: Optional[float] = None) -> bool:
        """将文本注入到当前活动窗口

        Args:
            text: 要注入的文本
            delay: 注入前的延迟时间（秒）

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def inject_text_safe(self, text: str) -> bool:
        """安全的文本注入（包含完整错误处理）"""
        pass


class NullInjector(TextInjector):
    """空实现 - 禁用注入时使用"""

    def inject_text(self, text: str, delay: Optional[float] = None) -> bool:
        return False

    def inject_text_safe(self, text: str) -> bool:
        return False


def _get_platform() -> str:
    """获取当前平台标识"""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "linux":
        return "linux"
    elif system == "darwin":
        return "macos"
    return "unknown"


def _get_linux_display_server() -> str:
    """检测 Linux 显示服务器 (X11 或 Wayland)"""
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


def create_text_injector(config=None) -> TextInjector:
    """工厂函数：根据平台和配置创建合适的文本注入器

    Args:
        config: 配置管理器实例（可选），用于读取启用/禁用设置

    Returns:
        TextInjector 实例
    """
    # 检查配置是否禁用注入
    if config and not config.is_input_injection_enabled():
        return NullInjector()

    system = _get_platform()

    if system == "windows":
        try:
            from windows_input import text_injector as windows_injector
            return windows_injector
        except ImportError:
            pass

    elif system == "linux":
        display_server = _get_linux_display_server()
        try:
            from linux_input import create_linux_injector
            return create_linux_injector(display_server)
        except ImportError:
            pass

    elif system == "macos":
        try:
            from macos_input import create_macos_injector
            return create_macos_injector()
        except ImportError:
            pass

    # 回退到空实现
    return NullInjector()


# 全局注入器实例（延迟初始化）
_global_injector: Optional[TextInjector] = None


def get_text_injector(config=None) -> TextInjector:
    """获取全局文本注入器实例（单例模式）"""
    global _global_injector
    if _global_injector is None:
        _global_injector = create_text_injector(config)
    return _global_injector


def reset_text_injector() -> None:
    """清空全局注入器缓存，下次 get_text_injector 时按最新配置重建。

    用途：用户在设置里打开/关闭『启用全局输入注入』后调用，
    避免单例缓存了启动时的 NullInjector（启动时注入关闭 → 之后即使
    打开开关也永远拿到 NullInjector，注入静默失效）。
    """
    global _global_injector
    _global_injector = None