"""配置管理器 - 读写 config.json（与主程序同目录）。

只保留语音输入精简版实际用到的配置项；LLM 相关 getter 保留是因为
post_processing.TextPostProcessor 构造时会读取（未启用 LLM 时不影响）。
"""

import json
import os
from typing import Dict, Optional, List

from platform_utils import get_app_dir


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: Optional[str] = None):
        if config_file is None:
            # 默认与主程序同目录（源码 pc/；打包后为 exe 所在目录）
            base_dir = get_app_dir()
            self.config_file = os.path.join(base_dir, "config.json")
        else:
            self.config_file = config_file

        self.config = self.load_config()

    @staticmethod
    def _deep_merge(default: Dict, override: Dict) -> Dict:
        """递归合并：override 覆盖 default，缺的 key 用 default。

        用途：config.json 可能只写了部分配置（用户手工编辑/旧版本），
        load 后与默认配置合并，保证任何缺项都回落到代码里的默认值，
        而不是各模块自己的 get(key, True) 之类的硬编码 fallback。
        例如 ner_protection.enabled 这种默认应为 False 的开关。
        """
        out = dict(default)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = ConfigManager._deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def load_config(self) -> Dict:
        """加载配置文件并与默认配置合并（缺项回落到默认值）。"""
        defaults = self.get_default_config()
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    disk = json.load(f)
                if isinstance(disk, dict):
                    return self._deep_merge(defaults, disk)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
        return defaults

    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "audio": {
                "input_device": "",  # 空字符串 = 系统默认设备
            },
            "ollama": {
                "url": "http://localhost:11434",
                "default_model": "gemma3:270m",
                "current_model": "gemma3:270m",
                "available_models": [],  # 动态获取
            },
            # LLM 后处理配置（本精简版默认不接 LLM，字段保留兼容 post_processing）
            "llm": {
                "backend": "ollama",
                "openai_base_url": "",
                "openai_api_key": "",
                "openai_model": "",
                "openai_timeout": 60,
            },
            # 内存/显存管理：短句语音输入场景，闲置一段时间自动卸载 ASR 模型
            "asr": {
                "use_gpu": False,
                "idle_unload_minutes": 10,  # 0 = 禁用
                # SenseVoice 模型目录（默认名 = 官方 2024-07-17 int8 版）：
                #   - 相对路径：相对于 voice-models/ 下的目录名
                #   - 绝对路径：直接指向模型目录（含 model.int8.onnx + tokens.txt）
                "sensevoice_model_dir": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
            },
            "post_processing": {
                "mode": "basic",  # none, basic, custom（本版 LLM 阶段自动跳过，仅规则生效）
                "enabled": True,
                "number_conversion": True,  # 数字转换
                "mixed_lang_fix": True,
                # NER 专名保护 + 句法分析都依赖额外的 hanlp（TensorFlow），
                # 默认关闭：不在精简版依赖中，且与 onnxruntime 同进程加载有
                # OpenMP 运行时冲突风险（段错误/首次下载模型卡顿）。
                # 需要一个全局开关 scene（GET/覆盖），这里显式列默认值，
                # 保证磁盘 config.json 即使没写这两键(见 _deep_merge)也回落 False。
                "ner_protection": {"enabled": False},
                "syntax_analyzer": {"enabled": False},
                "yixing_words": True,  # 异形词规范层
                "external_punctuation": {"enabled": False},
            },
            "ui": {
                "auto_paste": True,
            },
            "input_injection": {"enabled": True, "delay": 0.1, "safe_mode": True, "use_clipboard": True, "method": "auto"},
            "terminology": {
                "hotwords": [
                    {"word": "人工智能", "score": 2.0},
                    {"word": "深度学习", "score": 2.0},
                ],
                "replacements": [{"old": "语系", "new": "语音"}],
                "sensevoice_itn": True,
                "sensevoice_num_threads": 6,
            },
            "confusion_set": {
                "enabled": True,  # 默认开启同音字纠错（ConfusionSet + jieba）
            },
            "debug": {
                "timing_enabled": False,  # 各阶段耗时打点
            },
        }

    def get(self, key: str, default=None):
        """获取配置值（点分路径，如 "audio.input_device"）"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value):
        """设置配置值（点分路径）并落盘"""
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save_config()

    # ---------------------------- LLM 后处理（兼容 post_processing） ----------------------------

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        models = self.get("ollama.available_models", [])
        return models if isinstance(models, list) else []

    def set_available_models(self, models: List[str]):
        """设置可用模型列表"""
        self.set("ollama.available_models", models)

    def get_current_model(self) -> str:
        """获取当前使用的模型"""
        model = self.get("ollama.current_model", "gemma3:270m")
        return model if isinstance(model, str) else "gemma3:270m"

    def set_current_model(self, model: str):
        """设置当前使用的模型"""
        self.set("ollama.current_model", model)

    def get_llm_backend(self) -> str:
        """LLM 后处理后端：ollama（原生 API）或 openai（OpenAI 兼容 API）。"""
        backend = self.get("llm.backend", "ollama")
        return backend if backend in ("ollama", "openai") else "ollama"

    def set_llm_backend(self, backend: str):
        """设置 LLM 后处理后端。"""
        self.set("llm.backend", backend if backend in ("ollama", "openai") else "ollama")

    def get_llm_openai_base_url(self) -> str:
        """OpenAI 兼容 API 的 Base URL。"""
        url = self.get("llm.openai_base_url", "")
        return url if isinstance(url, str) else ""

    def set_llm_openai_base_url(self, url: str):
        self.set("llm.openai_base_url", url or "")

    def get_llm_openai_api_key(self) -> str:
        """OpenAI 兼容 API 的密钥（本地服务可为空）。"""
        key = self.get("llm.openai_api_key", "")
        return key if isinstance(key, str) else ""

    def set_llm_openai_api_key(self, key: str):
        self.set("llm.openai_api_key", key or "")

    def get_llm_openai_model(self) -> str:
        """OpenAI 兼容 API 的模型名。"""
        model = self.get("llm.openai_model", "")
        return model if isinstance(model, str) else ""

    def set_llm_openai_model(self, model: str):
        self.set("llm.openai_model", model or "")

    def get_llm_openai_timeout(self) -> float:
        """OpenAI 兼容 API 请求超时（秒）。"""
        try:
            return float(self.get("llm.openai_timeout", 60))
        except (ValueError, TypeError):
            return 60.0

    def set_llm_openai_timeout(self, timeout: float):
        self.set("llm.openai_timeout", float(timeout))

    # ---------------------------- 后处理模式 ----------------------------

    def get_post_processing_mode(self) -> str:
        """获取后处理模式"""
        mode = self.get("post_processing.mode", "basic")
        return mode if isinstance(mode, str) else "basic"

    def set_post_processing_mode(self, mode: str):
        """设置后处理模式"""
        self.set("post_processing.mode", mode)

    def is_number_conversion_enabled(self) -> bool:
        """是否启用数字转换"""
        enabled = self.get("post_processing.number_conversion", True)
        return enabled if isinstance(enabled, bool) else True

    def set_number_conversion_enabled(self, enabled: bool):
        """设置数字转换"""
        self.set("post_processing.number_conversion", enabled)

    def get_custom_prompts(self) -> List[Dict[str, str]]:
        """获取自定义提示词列表（无 LLM 时仅保留兼容，不参与处理）。"""
        prompts = self.get("post_processing.custom_prompts")
        return prompts if isinstance(prompts, list) else []

    def set_custom_prompts(self, prompts: List[Dict[str, str]]):
        """设置自定义提示词列表"""
        self.set("post_processing.custom_prompts", prompts)

    def get_basic_prompt(self) -> str:
        """获取基础提示词（无 LLM 时仅保留兼容，不参与处理）。"""
        prompt = self.get("post_processing.basic_prompt", "")
        return prompt if isinstance(prompt, str) else ""

    def set_basic_prompt(self, prompt: str):
        """设置基础提示词"""
        self.set("post_processing.basic_prompt", prompt)

    def get_mixed_lang_fix(self) -> bool:
        """是否对中英混读文本启用 LLM 专修（无 LLM 时该开关无效，保留兼容）。"""
        enabled = self.get("post_processing.mixed_lang_fix", True)
        return enabled if isinstance(enabled, bool) else True

    def set_mixed_lang_fix(self, enabled: bool):
        """设置中英混读专修开关。"""
        self.set("post_processing.mixed_lang_fix", bool(enabled))

    # ---------------------------- 输入注入 ----------------------------

    def is_input_injection_enabled(self) -> bool:
        """是否启用输入注入（识别完成后自动粘贴到活动窗口）"""
        enabled = self.get("input_injection.enabled", True)
        return enabled if isinstance(enabled, bool) else True

    def set_input_injection(self, enabled: bool):
        """设置输入注入"""
        self.set("input_injection.enabled", enabled)

    def get_injection_delay(self) -> float:
        """获取注入延迟（秒）"""
        delay = self.get("input_injection.delay", 0.1)
        try:
            return float(delay)
        except (ValueError, TypeError):
            return 0.1

    def set_injection_delay(self, delay: float):
        """设置注入延迟"""
        self.set("input_injection.delay", float(delay))

    def is_injection_use_clipboard(self) -> bool:
        """是否使用剪贴板注入（Linux/Wayland）"""
        enabled = self.get("input_injection.use_clipboard", True)
        return enabled if isinstance(enabled, bool) else True

    def set_injection_use_clipboard(self, enabled: bool):
        """设置是否使用剪贴板注入"""
        self.set("input_injection.use_clipboard", bool(enabled))

    def get_injection_method(self) -> str:
        """获取注入方法 (auto, xdotool, wtype, sendinput)"""
        method = self.get("input_injection.method", "auto")
        return method if isinstance(method, str) else "auto"

    def set_injection_method(self, method: str):
        """设置注入方法"""
        self.set("input_injection.method", method)

    # ---------------------------- 术语/热词/替换 ----------------------------

    def get_hotwords(self) -> List[Dict]:
        """获取热词列表"""
        hotwords = self.get("terminology.hotwords")
        return hotwords if isinstance(hotwords, list) else []

    def set_hotwords(self, hotwords: List[Dict]):
        """设置热词列表"""
        self.set("terminology.hotwords", hotwords)

    def get_replacements(self) -> List[Dict]:
        """获取替换列表"""
        replacements = self.get("terminology.replacements")
        return replacements if isinstance(replacements, list) else []

    def add_replacement(self, old_text: str, new_text: str):
        """添加单条替换规则"""
        replacements = self.get_replacements()
        # 检查是否已存在
        for item in replacements:
            if item.get("old") == old_text:
                item["new"] = new_text
                self.set_replacements(replacements)
                return
        # 新增
        replacements.append({"old": old_text, "new": new_text})
        self.set_replacements(replacements)

    def set_replacements(self, replacements: List[Dict]):
        """设置替换列表"""
        self.set("terminology.replacements", replacements)

    # ---------------------------- SenseVoice ----------------------------

    def is_sensevoice_itn_enabled(self) -> bool:
        """SenseVoice是否启用ITN（数字/日期/货币等反标点规范化）"""
        enabled = self.get("terminology.sensevoice_itn", True)
        return enabled if isinstance(enabled, bool) else True

    def set_sensevoice_itn(self, enabled: bool):
        """设置SenseVoice ITN"""
        self.set("terminology.sensevoice_itn", enabled)

    def get_sensevoice_num_threads(self) -> int:
        """SenseVoice 推理线程数（默认 6）"""
        n = self.get("terminology.sensevoice_num_threads", 6)
        return n if isinstance(n, int) and n > 0 else 6

    def set_sensevoice_num_threads(self, n: int):
        """设置 SenseVoice 推理线程数"""
        self.set("terminology.sensevoice_num_threads", n)

    # ---------------------------- GPU ----------------------------

    def is_gpu_enabled(self) -> bool:
        """是否启用 GPU 加速（需要 CUDA 版 sherpa-onnx）"""
        enabled = self.get("asr.use_gpu", False)
        return enabled if isinstance(enabled, bool) else False

    def set_gpu_enabled(self, enabled: bool) -> None:
        """设置 GPU 加速启停。"""
        self.set("asr.use_gpu", bool(enabled))

    def get_gpu_provider(self) -> str:
        """获取 ONNX Runtime provider 字符串，供模型加载时使用。"""
        return "cuda" if self.is_gpu_enabled() else "cpu"

    # ---------------------------- SenseVoice 模型目录 ----------------------------

    _DEFAULT_SENSEVOICE_MODEL_DIR = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"

    def get_sensevoice_model_dir(self) -> str:
        """SenseVoice 模型目录：相对名（voice-models/ 下）或绝对路径。

        默认指向官方标准版目录名。其他用户已有模型时可在 config.json 中改
        为已有目录名或绝对路径，无需改代码。
        """
        val = self.get("asr.sensevoice_model_dir", self._DEFAULT_SENSEVOICE_MODEL_DIR)
        return val if isinstance(val, str) and val.strip() else self._DEFAULT_SENSEVOICE_MODEL_DIR

    def set_sensevoice_model_dir(self, model_dir: str) -> None:
        """设置 SenseVoice 模型目录（相对名或绝对路径）。"""
        self.set("asr.sensevoice_model_dir", model_dir.strip() or self._DEFAULT_SENSEVOICE_MODEL_DIR)

    def get_idle_unload_minutes(self) -> int:
        """闲置多少分钟后自动卸载 ASR 模型以释放显存/内存。0 表示禁用。"""
        val = self.get("asr.idle_unload_minutes", 10)
        try:
            v = int(val)
            return max(0, v)
        except Exception:
            return 10

    def set_idle_unload_minutes(self, val: int) -> None:
        self.set("asr.idle_unload_minutes", max(0, int(val)))

    # ---------------------------- 音频设备 ----------------------------

    def get_audio_input_device(self) -> str:
        """获取选中的麦克风设备名称。空字符串表示使用系统默认输入设备。"""
        dev = self.get("audio.input_device", "")
        return dev if isinstance(dev, str) else ""

    def set_audio_input_device(self, device_name: str) -> None:
        """设置麦克风设备名称。传空字符串恢复为系统默认。"""
        self.set("audio.input_device", device_name)

    # ---------------------------- 调试 ----------------------------

    def is_timing_enabled(self) -> bool:
        """是否输出各阶段耗时打点（ASR/各规则阶段，调试定位瓶颈用）。"""
        enabled = self.get("debug.timing_enabled", False)
        return enabled if isinstance(enabled, bool) else False

    def set_timing_enabled(self, enabled: bool) -> None:
        """设置耗时打点输出开关。"""
        self.set("debug.timing_enabled", bool(enabled))


# 全局配置实例
config = ConfigManager()
