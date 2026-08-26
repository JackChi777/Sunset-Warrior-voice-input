import pyautogui
import pyperclip
import time
import psutil
import pygetwindow as gw
import ctypes
from typing import Optional

# Windows API常量
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002

VK_CONTROL = 0x11
VK_V = 0x56
VK_SHIFT = 0x10
VK_INSERT = 0x2D

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ki", KEYBDINPUT)
    ]

class WindowsTextInjector:
    """Windows全局文本注入器"""

    def __init__(self):
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05  # 操作间的暂停时间
        # 最近一次注入失败的原因（供 UI 展示，避免只打控制台）。
        # 注意：本注入器是模块级单例，last_error 仅在 Qt 主线程被同步读写，
        # 不要从工作线程调用注入，否则会与状态栏展示产生竞态。
        self.last_error = ""

        # 加载SendInput函数
        try:
            self.SendInput = ctypes.windll.user32.SendInput
            self.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
            self.SendInput.restype = ctypes.c_uint
        except:
            self.SendInput = None

    def _check_permission_compatibility(self) -> bool:
        """检查权限兼容性 - 简化的检查"""
        try:
            # 简化的权限检查：尝试访问目标进程
            target_hwnd = ctypes.windll.user32.GetForegroundWindow()
            if not target_hwnd:
                return True

            pid = ctypes.c_ulong()
            ctypes.windll.user32.GetWindowThreadProcessId(target_hwnd, ctypes.byref(pid))

            # 尝试使用psutil访问进程 - 如果失败，通常是权限问题
            try:
                process = psutil.Process(pid.value)
                process_name = process.name()  # 尝试获取进程名
                return True  # 能访问，权限兼容
            except psutil.AccessDenied:
                print("[WARN]  检测到进程访问被拒绝 - 可能是权限级别不匹配")
                return False
            except psutil.NoSuchProcess:
                return True  # 进程不存在，假设兼容
            except Exception:
                return True  # 其他错误，假设兼容

        except Exception:
            return True  # 检查失败，假设兼容

    def _send_ctrl_v_with_sendinput(self) -> bool:
        """使用SendInput API发送Ctrl+V组合键"""
        if not self.SendInput:
            print("[WARN]  SendInput API不可用")
            return False

        try:
            # Ctrl按下
            ctrl_down = INPUT()
            ctrl_down.type = 1  # INPUT_KEYBOARD
            ctrl_down.ki.wVk = VK_CONTROL
            ctrl_down.ki.dwFlags = KEYEVENTF_KEYDOWN

            # V按下
            v_down = INPUT()
            v_down.type = 1
            v_down.ki.wVk = VK_V
            v_down.ki.dwFlags = KEYEVENTF_KEYDOWN

            # V释放
            v_up = INPUT()
            v_up.type = 1
            v_up.ki.wVk = VK_V
            v_up.ki.dwFlags = KEYEVENTF_KEYUP

            # Ctrl释放
            ctrl_up = INPUT()
            ctrl_up.type = 1
            ctrl_up.ki.wVk = VK_CONTROL
            ctrl_up.ki.dwFlags = KEYEVENTF_KEYUP

            # 发送按键序列
            inputs = (INPUT * 4)(ctrl_down, v_down, v_up, ctrl_up)
            result = self.SendInput(4, inputs, ctypes.sizeof(INPUT))

            return result == 4
        except Exception as e:
            print(f"SendInput失败: {e}")
            return False

    def get_active_application(self) -> Optional[str]:
        """获取当前活动应用程序信息"""
        try:
            # 优先尝试使用 Windows API 获取进程信息
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            if not hwnd:
                return self._get_active_app_by_title()
            
            pid = ctypes.c_ulong()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            
            process_name = "unknown"
            try:
                process = psutil.Process(pid.value)
                process_name = process.name().lower()
            except psutil.AccessDenied:
                print(f"警告: 无法访问进程 (PID: {pid.value})，这通常是因为目标窗口（如管理员权限的 PowerShell）权限高于本程序。请尝试以管理员身份运行本程序。")
                return self._get_active_app_by_title()
            except psutil.NoSuchProcess:
                return self._get_active_app_by_title()
            
            # 获取窗口标题作为补充
            title = ""
            try:
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value.lower()
            except:
                pass

            # 逻辑判断
            # 如果是 Windows Terminal
            if 'windowsterminal' in process_name:
                return 'powershell.exe'
            
            # 如果是 conhost (经典控制台)，通过标题区分 CMD 和 PowerShell
            if 'conhost' in process_name or 'cmd.exe' in process_name:
                if 'powershell' in title or 'pwsh' in title:
                    return 'powershell.exe'
                return 'cmd.exe'
            
            # 直接匹配进程名
            if 'powershell' in process_name or 'pwsh' in process_name:
                return 'powershell.exe'
            
            if 'notepad' in process_name:
                return 'notepad.exe'
            
            if 'code' in process_name:
                return 'code.exe'
            
            # 回退到标题检测
            return self._get_active_app_by_title()
                
        except Exception as e:
            print(f"获取活动应用程序失败: {e}")
            return self._get_active_app_by_title()

    def _get_active_app_by_title(self) -> Optional[str]:
        """通过窗口标题判断应用类型 (回退方法)"""
        try:
            active_window = gw.getActiveWindow()
            if active_window:
                title = active_window.title.lower()
                if 'powershell' in title or 'pwsh' in title:
                    return 'powershell.exe'
                elif 'command' in title or 'cmd' in title:
                    return 'cmd.exe'
                elif 'notepad' in title:
                    return 'notepad.exe'
                elif 'vscode' in title or 'code' in title:
                    return 'code.exe'
                return 'unknown'
            return None
        except:
            return None

    def get_app_specific_delay(self, app_name: Optional[str]) -> float:
        """根据应用程序类型获取延迟时间"""
        if not app_name:
            return 0.15

        delays = {
            'cmd.exe': 0.25,        # 终端需要更长的延迟来响应剪贴板
            'powershell.exe': 0.25,
            'notepad.exe': 0.1,
            'winword.exe': 0.2,
            'excel.exe': 0.2,
            'chrome.exe': 0.15,
            'msedge.exe': 0.15,
            'code.exe': 0.15,
            'default': 0.15
        }
        return delays.get(app_name, delays['default'])

    def inject_text(self, text: str, delay: Optional[float] = None) -> bool:
        """将文本注入到当前活动窗口"""
        try:
            # 获取活动窗口信息
            active_window = gw.getActiveWindow()
            app_name = self.get_active_application()

            if delay is None:
                delay = self.get_app_specific_delay(app_name)

            title = active_window.title if active_window else "Unknown"
            print(f"尝试注入文本: '{text[:20]}...' 到窗口: {title} (应用类型: {app_name})")

            # 检查权限兼容性
            permission_compatible = self._check_permission_compatibility()
            if not permission_compatible:
                print("[WARN]  权限兼容性警告: 检测到权限级别不匹配")
                print("   这通常发生在程序以管理员身份运行，而目标窗口以普通用户身份运行时")
                print("   建议: 确保程序和目标应用程序使用相同的权限级别")

            # 保存当前剪贴板内容
            try:
                original_clipboard = pyperclip.paste()
            except:
                original_clipboard = ""

            # 复制文本到剪贴板
            try:
                pyperclip.copy(text)
                print(f"[OK] 文本已复制到剪贴板 ({len(text)} 字符)")
            except Exception as e:
                self.last_error = f"剪贴板操作失败: {e}"
                print(f"[FAIL] 剪贴板操作失败: {e}")
                return False

            # 等待剪贴板稳定
            time.sleep(delay)

            # 终端窗口特殊处理
            if app_name in ['cmd.exe', 'powershell.exe']:
                print(f"🔧 检测到终端窗口: {app_name}")

                # 再次确认窗口焦点 - 多次激活确保焦点
                if active_window:
                    try:
                        active_window.activate()
                        time.sleep(delay * 0.5)
                        # 再次确认焦点
                        active_window.activate()
                        time.sleep(delay * 0.5)
                        print("[OK] 窗口焦点已激活")
                    except Exception as e:
                        print(f"[WARN] 窗口激活失败: {e}")

                # 终端粘贴尝试 - 多层备用策略
                paste_success = False

                # 方法1: 使用SendInput API (最底层，最可靠)
                if self._send_ctrl_v_with_sendinput():
                    print("[OK] 使用SendInput发送 Ctrl+V 到终端")
                    paste_success = True
                else:
                    # 方法2: 使用pyautogui的Ctrl+V
                    try:
                        pyautogui.hotkey('ctrl', 'v')
                        print("[OK] 使用pyautogui发送 Ctrl+V 到终端")
                        paste_success = True
                    except Exception as e:
                        print(f"[FAIL] pyautogui Ctrl+V 发送失败: {e}")
                        # 方法3: 备用Shift+Insert
                        try:
                            pyautogui.hotkey('shift', 'insert')
                            print("[OK] 备用方法：发送 Shift+Insert 到终端")
                            paste_success = True
                        except Exception as e2:
                            print(f"[FAIL] Shift+Insert 也失败: {e2}")

                if not paste_success:
                    self.last_error = "所有粘贴方法都失败了（SendInput / Ctrl+V / Shift+Insert）"
                    print("[FAIL] 所有粘贴方法都失败了")
                    return False

                # 如果是较长的文本，可能需要多一点时间
                time.sleep(delay)

                # 对于某些配置，如果 Shift+Insert 没反应，尝试 Ctrl+V 作为补充
                # 但这可能会导致在支持两者的终端里粘贴两次，所以暂时保持单一方式
                # 或者可以根据窗口类名进一步区分，但目前先增加延迟和稳定性

            elif app_name == 'notepad.exe':
                try:
                    pyautogui.hotkey('ctrl', 'v')
                    print("[OK] 发送 Ctrl+V 到记事本")
                except Exception as e:
                    self.last_error = f"Ctrl+V 发送失败: {e}"
                    print(f"[FAIL] Ctrl+V 发送失败: {e}")
                    return False
            else:
                # 默认使用 Ctrl+V
                try:
                    pyautogui.hotkey('ctrl', 'v')
                    print(f"[OK] 发送 Ctrl+V 到应用: {app_name}")
                except Exception as e:
                    self.last_error = f"Ctrl+V 发送失败: {e}"
                    print(f"[FAIL] Ctrl+V 发送失败: {e}")
                    return False

            # 等待粘贴动作完成再恢复剪贴板
            time.sleep(delay * 2)

            try:
                pyperclip.copy(original_clipboard)
                print("[OK] 剪贴板已恢复")
            except Exception as e:
                print(f"[WARN] 剪贴板恢复失败: {e}")

            print("[OK] 文本注入完成")
            return True

        except Exception as e:
            self.last_error = str(e)
            print(f"[FAIL] 注入失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def inject_text_safe(self, text: str) -> bool:
        """安全的文本注入，包含完整的错误处理"""
        self.last_error = ""
        try:
            return self.inject_text(text)
        except Exception as e:
            self.last_error = str(e)
            print(f"意外错误: {e}")
            return False

# 全局注入器实例
text_injector = WindowsTextInjector()