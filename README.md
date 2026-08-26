# Sunset-Warrior（夕阳武士）· 语音输入

![License: MIT](https://img.shields.io/github/license/JackChi777/Sunset-Warrior-voice-input)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![CI](https://img.shields.io/github/actions/workflow/status/JackChi777/Sunset-Warrior-voice-input/ci.yml)

> 🌐 **Docs in your language：** [English](README.en.md) · [日本語](README.ja.md) · [한국어](README.ko.md)
>
> （以下为中文说明）

> **📌 项目简介**：**Sunset-Warrior（夕阳武士）**——按住右 Shift 说话，松开自动把识别文本粘贴到当前窗口。离线、免费、纯本地的语音输入工具（SenseVoice Small）。

离线语音输入工具：按住 **右 Shift** 说话，松开后自动把识别文本粘贴到当前活动窗口。

基于 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 的 **SenseVoice Small** 模型（标准版，约 163MB，纯 int8，CPU 即可流畅运行），识别自带标点和数字/日期反标点规范化（ITN）。全部推理在本机完成，无任何网络请求。

> **⚠️ 目前只针对中文（普通话）做了优化**
>
> 规则层后处理（`的/地/得` 纠错、同音字纠错、中文数字转换、异形词规范化等）目前**仅对中文生效**。SenseVoice 模型本身支持识别中文/英文/日文/韩文/粤语，这些语言的**语音识别可用**，但高级文本纠错目前**还没有对齐**，后续会逐步补齐。
>
> ⚠️ The rule-based post-processing is currently **optimized for Chinese only**. Speech recognition for English/Japanese/Korean works, but the advanced text corrections are not yet aligned for those languages.
> ⚠️ 現在、ルールベースの後処理は**中国語のみ**に最適化されています。英語・日本語・韓国語の音声認識は動作しますが、高度なテキスト訂正にはまだ対応していません。
> ⚠️ 규칙 기반 후처리는 현재 **중국어에만** 최적화되어 있습니다. 영어·일본어·한국어 음성 인식은 작동하지만 고급 텍스트 교정은 아직 지원되지 않습니다.

## 功能

- 🎤 **全局热键**：按住右 Shift ≥ 0.3 秒开始录音，松开停止；也可点击按钮
- 📋 **自动粘贴**：识别完成后自动把文本注入当前活动窗口（可关闭）
- ✍️ **纯规则后处理**（无需 LLM）：
  - “的/地/得”智能纠错（jieba 词性规则）
  - 同音字 / 形近字纠错（ConfusionSet）
  - 中文数字转换（一百 → 100）
  - 异形词规范化（《第一批异形词整理表》）
  - 热词表 / 替换表（可在设置中编辑）
  - **自学习纠错**：手动修正后自动学习规则并沉淀到 `user_data/learned_rules.json`
- 💾 **闲置自动卸载**：默认闲置 10 分钟自动释放模型内存，下次录音秒级重载

## 快速开始

### 1. 安装依赖（Python 3.8+）

```bash
cd pc
pip install -r requirements.txt
```

Linux 额外系统包：
```bash
# X11
sudo apt install xdotool
# Wayland
sudo apt install wtype wl-clipboard
```

### 2. 下载模型（约 163MB）

```bash
python scripts/download_models.py
```

模型解压到 `voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/`。

**已有模型？** 不用重新下载。在 `pc/config.json` 中设置 `asr.sensevoice_model_dir`：
- 相对名：`voice-models/` 下的目录名（默认就是上面的标准名）；
- 绝对路径：直接指向含 `model.int8.onnx` + `tokens.txt` 的目录。
也可以在「设置」对话框里改，保存后自动重新加载模型。

国内网络可设置镜像：
```bash
SHERPA_ONNX_MIRROR=https://hf-mirror.com python scripts/download_models.py
```

### 3. 运行

```bash
cd pc
python main.py
```

首次启动会自动加载模型（约 1-2 秒），然后即可按住右 Shift 说话。

## 目录结构

```
voice-input-lite/
├── pc/                      # 主程序
│   ├── main.py              # 入口：UI、录音、识别、注入
│   ├── config.py            # 配置管理（config.json）
│   ├── platform_utils.py    # 平台检测、模型路径
│   ├── post_processing.py   # 文本后处理流水线（规则层）
│   ├── proofreading.py      # 校对引擎（阶段 0-8 规则 + 可选 LLM 阶段）
│   ├── confusion_corrector.py  # 同音/形近字纠错
│   ├── syntax_analyzer.py   # 句法分析（HanLP 可选，未安装自动降级）
│   ├── input_injector.py    # 跨平台文本注入
│   ├── hotkey_listener.py   # 全局热键监听
│   └── user_data/           # 术语表、自学习纠错规则
├── scripts/
│   └── download_models.py   # 模型下载脚本
└── voice-models/            # 模型目录（下载后生成）
```

## 未完善之处与 Roadmap（欢迎贡献）

> **🐧 Linux 已知问题**：Linux 下 GUI 界面运行尚不稳定（Qt 窗口/显示问题），建议优先在 Windows 上使用；识别引擎本身跨平台正常，后续会修复 Linux UI。

**规则层仍不完善**，目前只是「够用」而非「完美」：

- **「的/地/得」**：大部分场景正确，但复杂句式（多层定语/状语、跨分句）仍有误判
- **「在/再」**：目前的规则处理还不够理想，容易漏改或误改
- **同音字/形近字纠错**：对口语化、方言味重的文本可能误改
- **数字转换、异形词表**：覆盖面有限，中文数字的复杂读法仍有遗漏

**LLM 校对接口已预留（默认关闭，零依赖）**

主程序刻意不内置任何 LLM 调用——保持纯离线、零成本、低延迟。但校对引擎 `proofreading.proofread()` 已经留好了 LLM 阶段（阶段 9），接法很简单：传一个 `llm_caller(text, prompt) -> 修正文本` 回调即可。支持任意 OpenAI 兼容接口：**Ollama**（本地免费）、DeepSeek、通义、vLLM 等。

示例（Ollama 的 OpenAI 兼容端点，`pip install openai`）：

```python
from proofreading import proofread
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def llm_caller(text: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model="qwen2.5",  # 换成你 ollama pull 的模型
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": text}],
        temperature=0.7,
    )
    return resp.choices[0].message.content

result = proofread("这句话里的在再帮我改对", mode='basic', llm_caller=llm_caller)
print(result.text)
```

`mode` 可选 `'basic'`（内置提示词）或 `'custom'`（自定义提示词，在 config 的 `custom_prompts` 里配置）。`prompt` 参数就是引擎准备好的润色提示词，直接喂给 LLM 即可；回调返回修正后的文本。

> `llm_caller` 是纯回调——你可以用它接**任何东西**：OpenAI、Ollama、DeepSeek、本地 vLLM，甚至一个本地规则函数。接入后「的/地/得」「在/再」等规则短板都能靠 LLM 大幅缓解。

**Roadmap（计划中）**

- [ ] 「的/地/得」「在/再」规则增强
- [ ] 英文 / 日文 / 韩文后处理对齐
- [ ] 手机版（Android / iOS）移植

## 常见问题

**提示模型加载失败？**
检查 `voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/` 是否存在 `model.int8.onnx` 和 `tokens.txt`，重新运行下载脚本。

**为什么不用 2025-09-09 版本？**
那是粤语专用变体（源自 ASLP-lab 的 sensevoice_small_yue），不支持标点，普通话识别还会丢字。本项目的标准版（2024-07-17，源自 FunAudioLLM/SenseVoice）才带标点和 ITN。

**NER 专名保护为什么默认关闭？**
该功能依赖额外的 hanlp（TensorFlow）模型，与 onnxruntime 同进程加载存在 OpenMP 运行时冲突风险。精简版不安装 hanlp，纯 jieba 规则即可正常运行；如需开启请在 `config.json` 中把 `post_processing.ner_protection.enabled` 设为 `true`（需自行安装 hanlp）。

**粘贴不生效？**
应用识别完成时若焦点在本窗口会跳过粘贴，切换到目标窗口后点「重新粘贴」即可。另外部分终端/游戏窗口需要以管理员身份运行本程序。

**想用 GPU？**
安装 CUDA 版 sherpa-onnx 后，在设置中勾选「启用 GPU 加速」。

## 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 推理引擎
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — 语音识别模型
- [jieba](https://github.com/fxsjy/jieba) — 中文分词与词性标注
