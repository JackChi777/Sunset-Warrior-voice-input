#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 SenseVoice Small 模型（sherpa-onnx 官方 release）。

用法:
    python scripts/download_models.py

下载后模型解压到 voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/
（约 163MB，纯 int8，CPU 可跑，自带标点）。
"""

import os
import sys
import tarfile
import tempfile
import urllib.request

# GitHub 对无 UA / 默认 UA（Python-urllib）的下载请求可能返回 403，显式给一个浏览器 UA。
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
_TIMEOUT = 120  # 秒


def _download(url: str, dest: str) -> None:
    """带浏览器 UA + 超时 + 进度显示的下载。"""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r       {done // 1024 // 1024} / {total // 1024 // 1024} MB ({pct}%)", end="", flush=True)
    print()


# 标准 SenseVoice Small（FunAudioLLM/SenseVoice 转换，自带标点 + ITN，约 163MB）
# 注意：不要换成 sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09 ——
# 那是粤语专用变体（ASLP-lab sensevoice_small_yue），无标点且普通话识别丢字。
MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_NAME}.tar.bz2"
)


def get_project_root() -> str:
    """项目根目录 = scripts/ 的上两级。"""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))


def main():
    root = get_project_root()
    models_dir = os.path.join(root, "voice-models")
    target_dir = os.path.join(models_dir, MODEL_NAME)
    os.makedirs(models_dir, exist_ok=True)

    # 已存在且包含模型文件 → 跳过
    model_file = os.path.join(target_dir, "model.int8.onnx")
    if os.path.exists(model_file):
        print(f"[ok] 模型已存在: {model_file}")
        return

    # 支持镜像加速（国内用户可设 HF_ENDPOINT 风格的环境变量）
    mirror = os.environ.get("SHERPA_ONNX_MIRROR", "")
    url = URL
    if mirror:
        # 镜像需保持相同路径结构，如 https://hf-mirror.com/k2-fsa/sherpa-onnx/releases/download/asr-models/...
        url = mirror.rstrip("/") + url[len("https://github.com"):]
        print(f"[info] 使用镜像: {url}")

    archive = os.path.join(tempfile.gettempdir(), f"{MODEL_NAME}.tar.bz2")
    print(f"[1/3] 下载 {url}")
    print(f"      -> {archive}")
    try:
        _download(url, archive)
    except urllib.error.HTTPError as e:
        print(f"\n[error] 下载失败: HTTP {e.code} {e.reason}")
        print("提示：若为 403，说明被服务器拒绝；若为 404，说明 URL 已失效。")
        print("请检查网络，或手动下载后解压到 voice-models/ 目录。")
        sys.exit(1)
    except Exception as e:
        print(f"\n[error] 下载失败: {type(e).__name__}: {e}")
        print("请检查网络，或手动下载后解压到 voice-models/ 目录。")
        sys.exit(1)

    print(f"[2/3] 解压到 {models_dir}")
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(models_dir)
    except Exception as e:
        print(f"[error] 解压失败: {e}")
        sys.exit(1)

    try:
        os.remove(archive)
    except OSError:
        pass

    if not os.path.exists(model_file):
        print(f"[error] 解压后未找到模型文件: {model_file}")
        sys.exit(1)

    print(f"[3/3] 完成 ✓ 模型已就绪: {target_dir}")
    print("运行:  cd pc && python main.py")


if __name__ == "__main__":
    main()
