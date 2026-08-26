#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 PyInstaller 打包 Windows 单文件 EXE。

用法（Windows）:
    pip install pyinstaller
    python scripts/build_exe.py

产物:
    dist/Sunset-Warrior.exe        （单文件，双击运行）
    dist/voice-models/             （若本地已有模型则一并拷贝，可选）

说明:
    - 模型不打包进 exe（163MB 太大）。exe 从同目录 voice-models/ 读模型，
      没有时首次运行会提示，可运行 scripts/download_models.py 下载。
    - 打包后 config.json / user_data/ 都定位在 exe 所在目录（见 platform_utils.get_app_dir）。
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
PC_DIR = os.path.join(PROJECT_ROOT, "pc")
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")


def main() -> int:
    if not os.path.isdir(PC_DIR):
        print(f"[ERROR] 找不到 pc/ 目录: {PC_DIR}")
        return 1

    # PyInstaller 需要把数据文件打进 exe（打包后从 _MEIPASS 读取）：
    #   - user_data/ 词表（it_terms.txt 等）：随 exe 内置，运行时若无外部目录则用内置
    #   - 注意：learned_rules.json / it_terms_user.txt 是运行时生成，不内置，走 get_app_dir
    data_specs = []
    user_data = os.path.join(PC_DIR, "user_data")
    for f in ("it_terms.txt", "it_terms_subjects.txt", "yixing_words.txt"):
        p = os.path.join(user_data, f)
        if os.path.exists(p):
            data_specs.append(f"{p};user_data")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onefile",
        "--name", "Sunset-Warrior",
        "--distpath", DIST_DIR,
        "--workpath", BUILD_DIR,
        "--specpath", BUILD_DIR,
        "--hidden-import", "jieba.posseg",
        "--hidden-import", "pynput.keyboard._win32",
        "--hidden-import", "pynput.mouse._win32",
        "--collect-submodules", "sounddevice",
        # 平台相关注入模块
        "--hidden-import", "windows_input",
        "--hidden-import", "linux_input",
        "--hidden-import", "macos_input",
        # 排除非精简版依赖（防止装了原项目环境的机器把 hanlp/torch 等拖进包）
        "--exclude-module", "hanlp",
        "--exclude-module", "hanlp_common",
        "--exclude-module", "torch",
        "--exclude-module", "transformers",
        "--exclude-module", "sentencepiece",
        "--exclude-module", "tensorflow",
        "--exclude-module", "keras",
        "--exclude-module", "scipy",
        "--exclude-module", "sklearn",
        "--exclude-module", "librosa",
        "--exclude-module", "numba",
        "--exclude-module", "pytest",
        # 只保留用到的 PyQt5 子模块，大幅加快打包并减小体积
        "--exclude-module", "PyQt5.QtQml",
        "--exclude-module", "PyQt5.QtQuick",
        "--exclude-module", "PyQt5.QtQuickWidgets",
        "--exclude-module", "PyQt5.QtDesigner",
        "--exclude-module", "PyQt5.QtMultimedia",
        "--exclude-module", "PyQt5.QtMultimediaWidgets",
        "--exclude-module", "PyQt5.QtLocation",
        "--exclude-module", "PyQt5.QtPositioning",
        "--exclude-module", "PyQt5.QtNfc",
        "--exclude-module", "PyQt5.QtBluetooth",
        "--exclude-module", "PyQt5.QtDBus",
        "--exclude-module", "PyQt5.QtHelp",
        "--exclude-module", "PyQt5.QtOpenGL",
        "--exclude-module", "PyQt5.QtPrintSupport",
        "--exclude-module", "PyQt5.QtWebEngineWidgets",
        "--exclude-module", "PyQt5.QtWebChannel",
        "--exclude-module", "PyQt5.QtSensors",
        "--exclude-module", "PyQt5.QtSerialPort",
        "--exclude-module", "PyQt5.QtTest",
        "--exclude-module", "PyQt5.QtXml",
        "--exclude-module", "PyQt5.QtSql",
        "--exclude-module", "PyQt5.QtSvg",
        # 数据文件
    ]
    for spec in data_specs:
        cmd.append("--add-data")
        cmd.append(spec)

    cmd.append(os.path.join(PC_DIR, "main.py"))

    print(f"[build] 项目根: {PROJECT_ROOT}")
    print(f"[build] 执行: {' '.join(cmd[:8])} ...")
    rc = subprocess.call(cmd, cwd=PC_DIR)
    if rc != 0:
        print("[build] PyInstaller 失败")
        return rc

    exe = os.path.join(DIST_DIR, "Sunset-Warrior.exe")
    if os.path.exists(exe):
        size_mb = os.path.getsize(exe) / 1024 / 1024
        print(f"[build] OK: {exe} ({size_mb:.1f} MB)")
    else:
        print("[build] 未找到产物 exe")
        return 1

    # 模型不打包进 exe（163MB 太大）。首次运行程序会弹窗询问并从
    # HuggingFace 自动下载（免认证），下载到 exe 同目录 voice-models/。
    print("[build] 模型未打包：首次运行时会自动从 HuggingFace 下载")
    return 0


if __name__ == "__main__":
    sys.exit(main())
