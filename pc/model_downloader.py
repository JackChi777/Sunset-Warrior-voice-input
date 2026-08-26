# -*- coding: utf-8 -*-
"""SenseVoice 模型下载器（免认证，HuggingFace 公开文件）。

构建的 EXE 不内嵌模型（163MB 太大），首次运行时若检测不到模型，
UI 会引导用户下载。本模块从 HuggingFace 直接下载 model.int8.onnx 和
tokens.txt 两个文件（无需登录/token），国内用户也可自行设置
HF_ENDPOINT 镜像环境变量。

用法：
    from model_downloader import download_model, MODEL_DIR_NAME
    ok, err = download_model(target_root, progress_cb=print)
"""

import os
import sys
import urllib.request

# HuggingFace 上 sherpa-onnx 官方转换的 SenseVoice Small（标准版 int8，带标点 + ITN）
# 注意：不要换成 2025-09-09 —— 那是粤语专用变体（ASLP-lab sensevoice_small_yue），
# 无标点且普通话识别丢字。
MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
HF_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

# 需要下载的文件（只取运行必需的两个；README/test_wavs 不需要）
REQUIRED_FILES = ["model.int8.onnx", "tokens.txt"]

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
_TIMEOUT = 120  # 秒


def _hf_base() -> str:
    """HuggingFace 下载基地址（支持 HF_ENDPOINT 镜像，如 https://hf-mirror.com）。"""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{HF_REPO}/resolve/main"


def download_file(url: str, dest: str, progress_cb=None) -> None:
    """下载单个文件到 dest，带进度回调 progress_cb(done_bytes, total_bytes)。"""
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
                if progress_cb:
                    progress_cb(done, total)


def download_model(target_root: str, progress_cb=None) -> tuple:
    """下载模型到 target_root/MODEL_DIR_NAME/。返回 (ok: bool, message: str)。

    progress_cb: callable(done_bytes, total_bytes, file_index, file_count)
    """
    model_dir = os.path.join(target_root, MODEL_DIR_NAME)
    os.makedirs(model_dir, exist_ok=True)
    base = _hf_base()

    for idx, name in enumerate(REQUIRED_FILES):
        dest = os.path.join(model_dir, name)
        # 已存在且非空 → 跳过（断点续用）
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        url = f"{base}/{name}"
        if progress_cb:
            progress_cb(0, 0, idx, len(REQUIRED_FILES))
        try:
            download_file(
                url, dest,
                progress_cb=lambda d, t, _i=idx, _n=len(REQUIRED_FILES): (
                    progress_cb(d, t, _i, _n) if progress_cb else None
                ),
            )
        except Exception as e:
            return False, f"下载 {name} 失败: {type(e).__name__}: {e}"

    # 校验
    model_file = os.path.join(model_dir, "model.int8.onnx")
    tokens_file = os.path.join(model_dir, "tokens.txt")
    if not os.path.exists(model_file) or not os.path.exists(tokens_file):
        return False, "下载完成后缺少必要文件，请重试"
    if os.path.getsize(model_file) < 100 * 1024 * 1024:  # 模型应 > 100MB
        return False, "模型文件疑似不完整（过小），请删除 voice-models 后重试"
    return True, f"模型就绪: {model_dir}"


def main() -> int:
    """命令行入口：python model_downloader.py [目标目录]"""
    from platform_utils import get_voice_models_dir

    target = sys.argv[1] if len(sys.argv) > 1 else get_voice_models_dir()

    def _cb(done, total, idx, n):
        if total:
            pct = done * 100 // total
            print(f"\r  [{idx + 1}/{n}] {done // 1024 // 1024}/{total // 1024 // 1024} MB ({pct}%)",
                  end="", flush=True)

    print(f"下载 SenseVoice 模型到 {target} ...")
    ok, msg = download_model(target, progress_cb=_cb)
    print()
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
