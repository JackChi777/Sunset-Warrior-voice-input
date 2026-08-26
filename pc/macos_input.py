"""macOS 文本注入器 - 使用 osascript / pbpaste / pbcopy"""
import subprocess
import time
import pyperclip
from typing import Optional

from input_injector import TextInjector


class MacOSTextInjector(TextInjector):
    """macOS 文本注入器 - 使用 AppleScript (osascript) 模拟键盘输入"""

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
            print(f"[MacOSInjector] 剪贴板复制失败: {e}")
            return False

    def _send_cmd_v(self) -> bool:
        """发送 Cmd+V 组合键 (使用 osascript)"""
        script = '''
        tell application "System Events"
            keystroke "v" using {command down}
        end tell
        '''
        try:
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[MacOSInjector] osascript Cmd+V 失败: {e}")
            return False

    def inject_text(self, text: str, delay: Optional[float] = None) -> bool:
        if not text:
            return True

        delay = delay or self.default_delay

        if self.use_clipboard:
            self._save_clipboard()
            if self._copy_to_clipboard(text):
                time.sleep(delay)
                if self._send_cmd_v():
                    time.sleep(delay * 2)
                    self._restore_clipboard()
                    return True
                self._restore_clipboard()

        return False

    def inject_text_safe(self, text: str) -> bool:
        try:
            return self.inject_text(text)
        except Exception as e:
            print(f"[MacOSInjector] 注入异常: {e}")
            return False


def create_macos_injector(use_clipboard: bool = True, delay: float = 0.1) -> TextInjector:
    """创建 macOS 文本注入器"""
    return MacOSTextInjector(use_clipboard, delay)