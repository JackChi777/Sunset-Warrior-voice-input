# Sunset-Warrior · Voice Input

![License: MIT](https://img.shields.io/github/license/JackChi777/Sunset-Warrior-voice-input)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![CI](https://img.shields.io/github/actions/workflow/status/JackChi777/Sunset-Warrior-voice-input/ci.yml)

> 🌐 **Docs in your language:** [中文](README.md) · [日本語](README.ja.md) · [한국어](README.ko.md)
>
> (English below)

> **📌 About**：**Sunset-Warrior** — hold right Shift, speak, and release to auto-paste the recognized text into the active window. An offline, free, fully-local voice input tool (SenseVoice Small).

Offline voice input tool, powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with the **SenseVoice Small** model (standard edition, ~163 MB, pure int8, runs smoothly on CPU). Hold **right Shift** to talk; when you release, the recognized text is automatically pasted into the currently active window. All inference runs locally — no network requests.

> **Language note:** The rule-based post-processing pipeline is currently **optimized for Chinese (中文) only**. The `的/地/得` correction, homophone correction, Chinese-numeral conversion, and variant-form normalization all target Chinese text. For English, Japanese, and Korean, the speech recognition itself works, but the advanced text corrections do not apply. Support for other languages is planned.

## Features

- 🎤 **Global hotkey**: hold right Shift for ≥ 0.3 s to start recording, release to stop; a button is also available
- 📋 **Auto-paste**: recognized text is injected into the active window automatically (can be disabled)
- ✍️ **Rule-based post-processing** (no LLM):
  - `的/地/得` smart correction (jieba part-of-speech rules)
  - Homophone / visually-similar-character correction (ConfusionSet)
  - Chinese numeral conversion (一百 → 100)
  - Variant-form normalization (First Batch of Variant Character Forms)
  - Custom term / replacement tables (editable in Settings)
  - **Self-learning corrections**: manual fixes are learned as rules persisted to `user_data/learned_rules.json`
- 💾 **Idle auto-unload**: by default unloads the model from memory after 10 minutes idle; reload is fast on the next use

## Quick Start

### 1. Install dependencies (Python 3.8+)

```bash
cd pc
pip install -r requirements.txt
```

Extra system packages on Linux:
```bash
# X11
sudo apt install xdotool
# Wayland
sudo apt install wtype wl-clipboard
```

### 2. Download the model (~163 MB)

```bash
python scripts/download_models.py
```

The model is extracted to `voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/`.

**Already have the model?** No need to re-download. Set `asr.sensevoice_model_dir` in `pc/config.json`:
- A relative name: a directory name under `voice-models/` (the default is the standard name above);
- An absolute path: point directly at a directory containing `model.int8.onnx` + `tokens.txt`.
You can also change it in the Settings dialog — the model reloads automatically after saving.

For users in China, a mirror is supported:
```bash
SHERPA_ONNX_MIRROR=https://hf-mirror.com python scripts/download_models.py
```

### 3. Run

```bash
cd pc
python main.py
```

The model loads on first launch (~1–2 s), then you can hold right Shift and speak.

## Directory Layout

```
voice-input-lite/
├── pc/                      # Main program
│   ├── main.py              # Entry point: UI, recording, recognition, injection
│   ├── config.py            # Config management (config.json)
│   ├── platform_utils.py    # Platform detection, model path
│   ├── post_processing.py   # Text post-processing pipeline (rules)
│   ├── proofreading.py      # Proofreading engine (stages 0–8 rules + optional LLM stage)
│   ├── confusion_corrector.py  # Homophone / visually-similar-character correction
│   ├── syntax_analyzer.py   # Syntax analysis (HanLP, optional — auto-degrades if not installed)
│   ├── input_injector.py    # Cross-platform text injection
│   ├── hotkey_listener.py   # Global hotkey listener
│   └── user_data/           # Term tables, self-learned correction rules
├── scripts/
│   └── download_models.py   # Model download script
└── voice-models/            # Model directory (created after download)
```

## Known Limitations & Roadmap (contributions welcome)

> **🐧 Linux known issue**: The GUI is not yet stable on Linux (Qt window/display issues). Windows is currently the recommended platform; the recognition engine itself is cross-platform and works fine. Linux UI will be fixed later.

**The rule layer is still rough** — it is "good enough", not "perfect":

- **`的/地/得` (de particles):** correct in most cases, but complex sentences (multi-level modifiers, cross-clause) are still misjudged sometimes
- **`在/再` (zai):** the current rules are not ideal yet — both missed corrections and false positives happen
- **Homophone / similar-character correction:** may over-correct colloquial or dialect-heavy text
- **Numeral conversion & variant-form tables:** limited coverage; complex Chinese numerals are still missed

**LLM proofreading hook is already in place (off by default, zero dependencies)**

The main program deliberately ships without any LLM calls — keeping it fully offline, free, and low-latency. But the proofreading engine `proofreading.proofread()` has a built-in LLM stage (stage 9): pass a simple `llm_caller(text, prompt) -> corrected_text` callback. Any OpenAI-compatible endpoint works: **Ollama** (local & free), DeepSeek, Qwen, vLLM, etc.

Example (Ollama's OpenAI-compatible endpoint, `pip install openai`):

```python
from proofreading import proofread
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def llm_caller(text: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model="qwen2.5",  # any model you `ollama pull`
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": text}],
        temperature=0.7,
    )
    return resp.choices[0].message.content

result = proofread("这句话里的在再帮我改对", mode='basic', llm_caller=llm_caller)
print(result.text)
```

`mode` can be `'basic'` (built-in prompt) or `'custom'` (custom prompts configured in `custom_prompts` in config). The `prompt` argument is the ready-to-use polishing prompt prepared by the engine — feed it straight to the LLM and return the corrected text.

> `llm_caller` is a pure callback — you can wire it to **anything**: OpenAI, Ollama, DeepSeek, a local vLLM server, or even a local rule function. Once connected, rule-layer gaps like `的/地/得` and `在/再` are largely mitigated by the LLM.

**Roadmap (planned)**

- [ ] Improve `的/地/得` and `在/再` rules
- [ ] Align post-processing for English / Japanese / Korean
- [ ] Mobile port (Android / iOS)

## FAQ

**Model fails to load?**
Check that `voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/` contains `model.int8.onnx` and `tokens.txt`, then re-run the download script.

**Why not the 2025-09-09 release?**
That is a Cantonese-specific variant (from ASLP-lab's sensevoice_small_yue) — it does not support punctuation and drops characters on Mandarin. This project's standard edition (2024-07-17, from FunAudioLLM/SenseVoice) supports punctuation and ITN.

**Why is NER proper-noun protection off by default?**
It depends on the extra hanlp (TensorFlow) model, which can conflict with onnxruntime at load time (OpenMP runtime). The lite version does not install hanlp — pure jieba rules are sufficient for normal use. To enable it, set `post_processing.ner_protection.enabled` to `true` in `config.json` (hanlp must be installed separately).

**Paste not working?**
If the app window has focus when recognition finishes, pasting is skipped — switch to the target window and click "Re-Paste". Some terminals/games need the program run as administrator.

**Want GPU?**
Install the CUDA build of sherpa-onnx, then check "Enable GPU acceleration" in Settings.

## Acknowledgments

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — inference engine
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — speech recognition model
- [jieba](https://github.com/fxsjy/jieba) — Chinese tokenization and part-of-speech tagging