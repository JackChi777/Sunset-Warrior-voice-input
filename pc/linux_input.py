"""Linux 文本注入器 - 支持 X11 (xdotool) 和 Wayland (wtype + wl-clipboard)"""
import os
import subprocess
import time
import platform
import pyperclip
from typing import Optional

from input_injector import TextInjector


class LinuxTextInjector(TextInjector):
    """Linux 基础文本注入器"""

    def __init__(self, use_clipboard: bool = True, delay: float = 0.1):
        self.use_clipboard = use_clipboard
        self.default_delay = delay
        self._last_clipboard = ""

    def _save_clipboard(self):
        """保存当前剪贴板内容"""
        try:
            self._last_clipboard = pyperclip.paste()
        except Exception:
            self._last_clipboard = ""

    def _restore_clipboard(self):
        """恢复剪贴板内容"""
        try:
            if self._last_clipboard:
                pyperclip.copy(self._last_clipboard)
        except Exception:
            pass

    def _copy_to_clipboard(self, text: str) -> bool:
        """复制文本到剪贴板"""
        try:
            pyperclip.copy(text)
            return True
        except Exception as e:
            print(f"[LinuxInjector] 剪贴板复制失败: {e}")
            return False

    def inject_text(self, text: str, delay: Optional[float] = None) -> bool:
        """注入文本 - 子类必须实现"""
        raise NotImplementedError

    def inject_text_safe(self, text: str) -> bool:
        """安全的文本注入"""
        try:
            return self.inject_text(text)
        except Exception as e:
            print(f"[LinuxInjector] 注入异常: {e}")
            return False

    def _check_command(self, cmd: str) -> bool:
        """检查命令是否可用"""
        try:
            subprocess.run(["which", cmd], capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False


class X11TextInjector(LinuxTextInjector):
    """X11 文本注入器 - 使用 xdotool"""

    def __init__(self, use_clipboard: bool = True, delay: float = 0.1):
        super().__init__(use_clipboard, delay)
        self._xdotool_available = self._check_command("xdotool")
        if not self._xdotool_available:
            print("[X11Injector] 警告: xdotool 未安装，请运行: sudo apt install xdotool")

    def _send_ctrl_v(self) -> bool:
        """发送 Ctrl+V 组合键"""
        if not self._xdotool_available:
            return False
        try:
            subprocess.run(["xdotool", "key", "ctrl+v"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[X11Injector] xdotool key 失败: {e}")
            return False

    def _type_text(self, text: str) -> bool:
        """直接输入文本（备选方案）"""
        if not self._xdotool_available:
            return False
        try:
            subprocess.run(["xdotool", "type", "--clearmodifiers", text], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[X11Injector] xdotool type 失败: {e}")
            return False

    def inject_text(self, text: str, delay: Optional[float] = None) -> bool:
        if not text:
            return True

        delay = delay or self.default_delay

        if self.use_clipboard:
            # 方案1: 剪贴板 + Ctrl+V (最可靠)
            self._save_clipboard()
            if self._copy_to_clipboard(text):
                time.sleep(delay)
                if self._send_ctrl_v():
                    time.sleep(delay * 2)
                    self._restore_clipboard()
                    return True
                self._restore_clipboard()

        # 方案2: 直接模拟键盘输入
        if self._type_text(text):
            return True

        return False


class WaylandTextInjector(LinuxTextInjector):
    """Wayland 文本注入器 - 使用 wtype + wl-clipboard"""

    def __init__(self, use_clipboard: bool = True, delay: float = 0.1):
        super().__init__(use_clipboard, delay)
        self._wtype_available = self._check_command("wtype")
        self._wl_clipboard_available = self._check_command("wl-copy")

        if not self._wtype_available:
            print("[WaylandInjector] 警告: wtype 未安装，请运行: sudo apt install wtype")
        if not self._wl_clipboard_available:
            print("[WaylandInjector] 警告: wl-clipboard 未安装，请运行: sudo apt install wl-clipboard")

    def _send_ctrl_v(self) -> bool:
        """发送 Ctrl+V (使用 wtype)"""
        if not self._wtype_available:
            return False
        try:
            subprocess.run(["wtype", "-M", "ctrl", "-k", "v"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[WaylandInjector] wtype 失败: {e}")
            return False

    def _type_text(self, text: str) -> bool:
        """直接输入文本（使用 wtype）"""
        if not self._wtype_available:
            return False
        try:
            subprocess.run(["wtype", text], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[WaylandInjector] wtype type 失败: {e}")
            return False

    def inject_text(self, text: str, delay: Optional[float] = None) -> bool:
        if not text:
            return True

        delay = delay or self.default_delay

        if self.use_clipboard and self._wl_clipboard_available:
            # 方案1: Wayland 剪贴板 + wtype Ctrl+V
            self._save_clipboard()
            try:
                # 使用 wl-copy 写入 Wayland 剪贴板
                subprocess.run(["wl-copy"], input=text.encode(), check=True)
                time.sleep(delay)
                if self._send_ctrl_v():
                    time.sleep(delay * 2)
                    self._restore_clipboard()
                    return True
            except subprocess.CalledProcessError as e:
                print(f"[WaylandInjector] wl-copy 失败: {e}")
            finally:
                self._restore_clipboard()

        # 方案2: 直接使用 wtype 输入文本
        if self._type_text(text):
            return True

        # 方案3: 回退到 X11/XWayland 注入器
        # GNOME 等 Wayland 合成器可能不支持虚拟键盘协议，导致 wtype 失效。
        # 大多数 Ubuntu 桌面都运行 XWayland，因此 xdotool 通常可用。
        print("[WaylandInjector] Wayland 注入受限，尝试回退到 X11/XWayland (xdotool)...")
        print("[WaylandInjector] 提示：GNOME 可能会弹出“远程桌面/屏幕共享”授权窗口，请点击“允许”以继续输入。")
        try:
            fallback_injector = X11TextInjector(use_clipboard=self.use_clipboard, delay=delay)
            return fallback_injector.inject_text(text, delay)
        except Exception as e:
            print(f"[WaylandInjector] X11/XWayland 回退也失败: {e}")

        return False


def get_linux_display_server() -> str:
    """检测 Linux 显示服务器类型"""
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


def create_linux_injector(display_server: Optional[str] = None, config=None) -> TextInjector:
    """工厂函数：创建适合当前环境的 Linux 注入器

    Args:
        display_server: 强制指定显示服务器 ("x11" 或 "wayland")，None 为自动检测
        config: 配置管理器实例（可选），用于读取配置

    Returns:
        TextInjector 实例
    """
    # 从配置读取设置
    use_clipboard = True
    delay = 0.1

    if config:
        use_clipboard = config.get("input_injection.use_clipboard", True)
        delay = config.get("input_injection.delay", 0.1)

    if display_server is None:
        display_server = get_linux_display_server()

    print(f"[LinuxInjector] 检测到显示服务器: {display_server}")

    if display_server == "wayland":
        return WaylandTextInjector(use_clipboard=use_clipboard, delay=delay)
    elif display_server == "x11":
        return X11TextInjector(use_clipboard=use_clipboard, delay=delay)
    else:
        print("[LinuxInjector] 无法检测显示服务器，尝试 X11 注入器")
        return X11TextInjector(use_clipboard=use_clipboard, delay=delay)


# 为了兼容性保留的导出
__all__ = [
    "LinuxTextInjector",
    "X11TextInjector",
    "WaylandTextInjector",
    "get_linux_display_server",
    "create_linux_injector",
]