"""跨平台全局热键监听器 - 支持 X11、Wayland (通过 libei/ydotool/portal)"""
import platform
import os
import sys
import threading
import time
import subprocess
from typing import Optional, Callable
from abc import ABC, abstractmethod


class GlobalHotkeyListener(ABC):
    """全局热键监听器抽象基类"""

    @abstractmethod
    def start(self) -> bool:
        """启动监听器，返回是否成功"""
        pass

    @abstractmethod
    def stop(self):
        """停止监听器"""
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """检查监听器是否在运行"""
        pass


class PynputHotkeyListener(GlobalHotkeyListener):
    """使用 pynput 的热键监听器 (X11/Windows/macOS)"""

    def __init__(self, on_press: Callable, on_release: Callable):
        self.on_press = on_press
        self.on_release = on_release
        self._listener = None
        self._thread = None

    def start(self) -> bool:
        try:
            from pynput import keyboard
            self._listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            )
            self._listener.start()
            return True
        except Exception as e:
            print(f"[PynputHotkeyListener] 启动失败: {e}")
            return False

    def stop(self):
        if self._listener:
            self._listener.stop()
            self._listener = None

    def is_alive(self) -> bool:
        return self._listener is not None and self._listener.running


class WaylandPortalHotkeyListener(GlobalHotkeyListener):
    """Wayland 热键监听器 - 使用 systemd 用户服务 + dbus 激活 (适用于 KDE/GNOME)"""

    def __init__(self, on_press: Callable, on_release: Callable):
        self.on_press = on_press
        self.on_release = on_release
        self._process = None

    def start(self) -> bool:
        """尝试通过各种 Wayland 方式启动全局热键"""
        # 方法 1: 尝试使用 kglobalaccel5 (KDE)
        if self._try_kde_global_accel():
            return True

        # 方法 2: 尝试使用 gnome-shell 扩展/快捷键
        if self._try_gnome_shortcuts():
            return True

        # 方法 3: 尝试 ydotool + uinput (需要 root/用户组权限)
        if self._try_ydotool_listener():
            return True

        print("[WaylandPortalHotkeyListener] 无可用 Wayland 全局热键方案")
        return False

    def _try_kde_global_accel(self) -> bool:
        """KDE: 使用 kglobalaccel5 注册全局快捷键"""
        try:
            # 检查是否在 KDE 环境
            if os.environ.get("XDG_CURRENT_DESKTOP", "").lower().find("kde") == -1:
                return False

            # 使用 dbus 激活 kglobalaccel
            # 这需要预先配置 .desktop 文件或使用 qdbus
            print("[WaylandPortalHotkeyListener] 检测到 KDE，尝试 kglobalaccel...")
            # 实际实现需要预配置，这里暂时返回 False
            return False
        except Exception:
            return False

    def _try_gnome_shortcuts(self) -> bool:
        """GNOME: 使用 gsettings 设置自定义快捷键"""
        try:
            if os.environ.get("XDG_CURRENT_DESKTOP", "").lower().find("gnome") == -1:
                return False

            print("[WaylandPortalHotkeyListener] 检测到 GNOME，可配置 gsettings 快捷键...")
            # 需要用户预先在设置中配置，或通过脚本配置
            return False
        except Exception:
            return False

    def _try_ydotool_listener(self) -> bool:
        """使用 ydotool 监听键盘事件 (需要 ydotool + ydotoold 运行)"""
        try:
            # 检查 ydotoold 是否在运行
            result = subprocess.run(
                ["pgrep", "-x", "ydotoold"],
                capture_output=True
            )
            if result.returncode != 0:
                print("[WaylandPortalHotkeyListener] ydotoold 未运行")
                return False

            print("[WaylandPortalHotkeyListener] ydotoold 正在运行，但需要额外实现键盘监听...")
            # ydotool 主要用于模拟输入，监听需要额外实现
            return False
        except Exception:
            return False

    def stop(self):
        if self._process:
            self._process.terminate()
            self._process = None

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None


class X11FallbackHotkeyListener(GlobalHotkeyListener):
    """X11 回退方案 - 强制使用 XWayland"""

    def __init__(self, on_press: Callable, on_release: Callable):
        self.on_press = on_press
        self.on_release = on_release
        self._listener = None
        self._original_gdk_backend = os.environ.get("GDK_BACKEND")
        self._original_wayland = os.environ.get("WAYLAND_DISPLAY")

    def start(self) -> bool:
        # 强制使用 X11 后端
        os.environ["GDK_BACKEND"] = "x11"
        os.environ.pop("WAYLAND_DISPLAY", None)

        try:
            from pynput import keyboard
            self._listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            )
            self._listener.start()

            # 验证是否真正工作 (等待一下检查)
            time.sleep(0.2)
            if self._listener.running:
                print("[X11FallbackHotkeyListener] 成功启动 (XWayland 模式)")
                return True
            else:
                print("[X11FallbackHotkeyListener] 启动但未运行")
                return False
        except Exception as e:
            print(f"[X11FallbackHotkeyListener] 启动失败: {e}")
            return False
        finally:
            # 恢复环境变量 (仅影响子进程)
            if self._original_gdk_backend:
                os.environ["GDK_BACKEND"] = self._original_gdk_backend
            else:
                os.environ.pop("GDK_BACKEND", None)
            if self._original_wayland:
                os.environ["WAYLAND_DISPLAY"] = self._original_wayland

    def stop(self):
        if self._listener:
            self._listener.stop()
            self._listener = None

    def is_alive(self) -> bool:
        return self._listener is not None and self._listener.running


class LibeiHotkeyListener(GlobalHotkeyListener):
    """使用 libei (Library for Emulated Input) - Wayland 原生方案
    
    需要: libei >= 1.0, python-libei (pip install libei)
    这是最现代的 Wayland 全局输入方案
    """

    def __init__(self, on_press: Callable, on_release: Callable):
        self.on_press = on_press
        self.on_release = on_release
        self._context = None
        self._seat = None
        self._running = False

    def start(self) -> bool:
        try:
            import libei
        except ImportError:
            print("[LibeiHotkeyListener] libei 未安装 (pip install libei)")
            return False

        try:
            # libei 需要与 compositor 协商
            # 这是简化示例，实际需要完整的 EI 客户端实现
            print("[LibeiHotkeyListener] libei 可用，但需要完整实现...")
            return False
        except Exception as e:
            print(f"[LibeiHotkeyListener] 启动失败: {e}")
            return False

    def stop(self):
        self._running = False
        if self._context:
            self._context = None

    def is_alive(self) -> bool:
        return self._running


def create_hotkey_listener(on_press: Callable, on_release: Callable) -> GlobalHotkeyListener:
    """工厂函数：根据平台创建合适的热键监听器"""

    system = platform.system().lower()

    if system == "windows":
        return PynputHotkeyListener(on_press, on_release)

    elif system == "darwin":
        return PynputHotkeyListener(on_press, on_release)

    elif system == "linux":
        # 检测显示服务器
        is_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        is_x11 = bool(os.environ.get("DISPLAY")) and not is_wayland

        if is_wayland:
            print("[HotkeyFactory] 检测到 Wayland 环境")

            # 优先尝试 libei (原生 Wayland)
            listener = LibeiHotkeyListener(on_press, on_release)
            if listener.start():
                return listener

            # 尝试 Wayland portal / desktop 特定方案
            listener = WaylandPortalHotkeyListener(on_press, on_release)
            if listener.start():
                return listener

            # 最后回退到 XWayland
            print("[HotkeyFactory] Wayland 方案均不可用，回退到 XWayland")
            listener = X11FallbackHotkeyListener(on_press, on_release)
            if listener.start():
                return listener

            print("[HotkeyFactory] 警告：所有 Wayland 热键方案均失败")
            return NullHotkeyListener()

        elif is_x11:
            print("[HotkeyFactory] 检测到 X11 环境")
            return PynputHotkeyListener(on_press, on_release)

        else:
            print("[HotkeyFactory] 未知显示服务器，尝试 pynput")
            return PynputHotkeyListener(on_press, on_release)

    else:
        print(f"[HotkeyFactory] 不支持的平台: {system}")
        return NullHotkeyListener()


class NullHotkeyListener(GlobalHotkeyListener):
    """空实现 - 热键不可用时使用"""

    def start(self) -> bool:
        print("[NullHotkeyListener] 热键功能不可用")
        return False

    def stop(self):
        pass

    def is_alive(self) -> bool:
        return False


# 便捷函数：创建并启动
def start_global_hotkey(on_press: Callable, on_release: Callable) -> GlobalHotkeyListener:
    """创建并启动全局热键监听器"""
    listener = create_hotkey_listener(on_press, on_release)
    listener.start()
    return listener


# 测试代码
if __name__ == "__main__":
    def test_press(key):
        print(f"Press: {key}")

    def test_release(key):
        print(f"Release: {key}")

    print(f"Platform: {platform.system()}")
    print(f"DISPLAY: {os.environ.get('DISPLAY')}")
    print(f"WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY')}")
    print(f"XDG_CURRENT_DESKTOP: {os.environ.get('XDG_CURRENT_DESKTOP')}")

    listener = start_global_hotkey(test_press, test_release)

    if listener.is_alive():
        print("热键监听器已启动，按 Ctrl+C 退出...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n停止监听器...")
            listener.stop()
    else:
        print("热键监听器启动失败")