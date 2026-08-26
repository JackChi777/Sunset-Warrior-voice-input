# Sunset-Warrior（석양의 무사）· 음성 입력

![License: MIT](https://img.shields.io/github/license/JackChi777/Sunset-Warrior-voice-input)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![CI](https://img.shields.io/github/actions/workflow/status/JackChi777/Sunset-Warrior-voice-input/ci.yml)

> 🌐 **언어별 문서:** [中文](README.md) · [English](README.en.md) · [日本語](README.ja.md)
>
> (아래는 한국어 설명)

> **📌 프로젝트 소개**: **Sunset-Warrior（석양의 무사）** — 오른쪽 Shift를 누른 채 말하고, 놓으면 인식된 텍스트가 활성 창에 자동으로 붙여넣기됩니다. 오프라인·무료·완전 로컬 음성 입력 도구(SenseVoice Small).

오프라인 음성 입력 도구입니다. [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 기반의 **SenseVoice Small** 모델(표준판, 약 163MB, 순수 int8, CPU에서 부드럽게 동작)을 사용합니다. **오른쪽 Shift를 누른 채로 말하면**, 놓는 순간 인식된 텍스트가 현재 활성 창에 자동으로 붙여넣어집니다. 모든 추론은 로컬에서 이루어지며 네트워크 요청이 없습니다.

> **언어 관련 안내:** 규칙 기반 후처리 파이프라인은 현재 **중국어(中文)에만 최적화**되어 있습니다. `的/地/得` 교정, 동음자 교정, 중국어 숫자 변환, 이체자 표준화는 모두 중국어 텍스트를 대상으로 합니다. 영어·일본어·한국어의 경우 음성 인식 자체는 작동하지만 고급 텍스트 교정은 적용되지 않습니다. 다른 언어 지원은 예정입니다.

## 기능

- 🎤 **전역 단축키**: 오른쪽 Shift를 0.3초 이상 길게 누르면 녹음 시작, 놓으면 정지. 버튼으로도 조작 가능
- 📋 **자동 붙여넣기**: 인식 완료 후 활성 창에 자동으로 삽입(비활성화 가능)
- ✍️ **규칙 기반 후처리**(LLM 불필요):
  - `的/地/得` 스마트 교정(jieba 품사 규칙)
  - 동음자 / 유사자 교정(ConfusionSet)
  - 중국어 숫자 변환(一百 → 100)
  - 이체자 표준화(第一批異形詞整理表)
  - 사용자 정의 용어표 / 치환표(설정에서 편집 가능)
  - **자기 학습 교정**: 수동 수정 시 자동으로 규칙화되어 `user_data/learned_rules.json`에 저장
- 💾 **유휴 시 자동 언로드**: 기본적으로 유휴 10분 후 모델을 메모리에서 해제. 다음 사용 시 빠르게 다시 로드

## 🪟 Windows 휴대용 버전(EXE)

Python을 설치하고 싶지 않다면? 패키징된 단일 EXE를 다운로드하여 더블클릭만 하면 됩니다:

1. [Releases](https://github.com/JackChi777/Sunset-Warrior-voice-input/releases) 페이지 열기
2. 최신 `Sunset-Warrior-win64.zip`(exe만, 수십 MB) 다운로드
3. 아무 폴더에 압축 해제 후 `Sunset-Warrior.exe` 더블클릭
4. **첫 실행 시** 모델을 자동 감지: 없으면 대화상자에서 확인 후 HuggingFace에서 자동 다운로드(약 163MB, 로그인 불필요·1회만, 이후 오프라인 동작)

> EXE는 [GitHub Actions](.github/workflows/build-windows.yml)이 자동 빌드합니다(`v*` 태그에서 트리거).
> 첫 실행 시 Windows SmartScreen 경고가 뜨면 "추가 정보" → "계속 실행"을 클릭하세요(서명되지 않은 프로그램의 일반적인 표시).
> 자동 다운로드를 건너뛰려면 `python scripts/download_models.py`를 실행하거나 exe 옆 `voice-models/`에 모델을 넣으세요.

## 빠른 시작

### 1. 의존성 설치(Python 3.8+)

```bash
cd pc
pip install -r requirements.txt
```

Linux의 경우 추가 시스템 패키지 필요:
```bash
# X11
sudo apt install xdotool
# Wayland
sudo apt install wtype wl-clipboard
```

### 2. 모델 다운로드(약 163MB)

```bash
python scripts/download_models.py
```

모델은 `voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/`에 압축이 풀립니다.

**이미 모델이 있나요?** 다시 다운로드할 필요 없습니다. `pc/config.json`의 `asr.sensevoice_model_dir`을 설정하세요:
- 상대 이름: `voice-models/` 아래의 디렉터리 이름(기본값은 위 표준 이름)
- 절대 경로: `model.int8.onnx` + `tokens.txt`가 포함된 디렉터리를 직접 지정
설정 대화상자에서도 변경할 수 있으며, 저장 후 모델이 자동으로 다시 로드됩니다.

중국 내 사용자는 미러를 이용할 수 있습니다:
```bash
SHERPA_ONNX_MIRROR=https://hf-mirror.com python scripts/download_models.py
```

### 3. 실행

```bash
cd pc
python main.py
```

첫 실행 시 모델을 로드합니다(약 1~2초). 이후 오른쪽 Shift를 누른 채로 말하면 됩니다.

## 디렉터리 구조

```
voice-input-lite/
├── pc/                      # 메인 프로그램
│   ├── main.py              # 진입점: UI, 녹음, 인식, 삽입
│   ├── config.py            # 설정 관리(config.json)
│   ├── platform_utils.py    # 플랫폼 감지, 모델 경로
│   ├── post_processing.py   # 텍스트 후처리 파이프라인(규칙 계층)
│   ├── proofreading.py      # 교정 엔진(스테이지 0-8 규칙 + 선택적 LLM 스테이지)
│   ├── confusion_corrector.py  # 동음/유사자 교정
│   ├── syntax_analyzer.py   # 구문 분석(HanLP, 선택 — 미설치 시 자동 다운그레이드)
│   ├── input_injector.py    # 크로스 플랫폼 텍스트 삽입
│   ├── hotkey_listener.py   # 전역 단축키 리스너
│   └── user_data/           # 용어표, 자기 학습 교정 규칙
├── scripts/
│   └── download_models.py   # 모델 다운로드 스크립트
└── voice-models/            # 모델 디렉터리(다운로드 후 생성)
```

## 미완성 부분 및 로드맵(기여 환영)

> **🐧 Linux 알려진 문제**: Linux에서 GUI가 아직 불안정합니다(Qt 창/표시 문제). 당분간은 Windows 사용을 권장합니다. 인식 엔진 자체는 크로스 플랫폼으로 정상 동작합니다. Linux UI는 추후 수정 예정입니다.

**규칙 계층은 아직 불완전합니다.** "쓸 만한" 수준이지 "완벽한" 수준은 아닙니다:

- **`的/地/得`**: 대부분의 경우 정확하지만 복잡한 문장(다중 수식어, 절을 넘는 문장)에서는 오판이 남아 있음
- **`在/再`**: 현재 규칙 처리가 아직 이상적이지 않습니다. 누락도 오수정도 발생할 수 있습니다
- **동음자/유사자 교정**: 구어체·방언 색이 강한 텍스트에서 오수정 가능성
- **숫자 변환·이체자 표**: 커버리지가 제한적이며 복잡한 중국어 숫자 읽기는 미지원

**LLM 교정 훅은 이미 구현됨(기본 꺼짐, 의존성 제로)**

메인 프로그램은 의도적으로 LLM 호출을 포함하지 않습니다 — 완전 오프라인·무료·저지연을 유지하기 위해서입니다. 하지만 교정 엔진 `proofreading.proofread()`에는 LLM 스테이지(스테이지 9)가 내장되어 있어, `llm_caller(text, prompt) -> 수정 텍스트` 콜백만 넘기면 연결할 수 있습니다. OpenAI 호환 엔드포인트라면 무엇이든 사용 가능합니다: **Ollama**(로컬·무료), DeepSeek, Qwen, vLLM 등.

예시(Ollama의 OpenAI 호환 엔드포인트, `pip install openai`):

```python
from proofreading import proofread
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def llm_caller(text: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model="qwen2.5",  # ollama pull 한 임의의 모델
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": text}],
        temperature=0.7,
    )
    return resp.choices[0].message.content

result = proofread("这句话里的在再帮我改对", mode='basic', llm_caller=llm_caller)
print(result.text)
```

`mode`는 `'basic'`(내장 프롬프트) 또는 `'custom'`(config의 `custom_prompts`에 설정한 사용자 프롬프트)을 지정할 수 있습니다. `prompt` 인자는 엔진이 준비한 다듬기용 프롬프트입니다. 그대로 LLM에 넘기고 수정된 텍스트를 반환하면 됩니다.

> `llm_caller`는 순수 콜백입니다 — OpenAI·Ollama·DeepSeek·로컬 vLLM·심지어 로컬 규칙 함수까지 **무엇에든** 연결할 수 있습니다. 연결하면 `的/地/得`, `在/再` 같은 규칙 계층의 약점도 크게 개선됩니다.

**로드맵(계획 중)**

- [ ] `的/地/得`, `在/再` 규칙 강화
- [ ] 영어·일본어·한국어 후처리 지원
- [ ] 모바일 버전(Android / iOS) 이식

## 자주 묻는 질문

**모델 로드에 실패하나요?**
`voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/`에 `model.int8.onnx`와 `tokens.txt`가 있는지 확인하고 다운로드 스크립트를 다시 실행하세요.

**2025-09-09 버전을 쓰지 않는 이유는?**
그것은 광둥어 전용 변종(ASLP-lab의 sensevoice_small_yue에서 유래)으로 문장 부호를 지원하지 않으며 표준 중국어에서 문자가 누락됩니다. 이 프로젝트의 표준판(2024-07-17, FunAudioLLM/SenseVoice 유래)은 문장 부호와 ITN을 지원합니다.

**NER 고유명사 보호가 기본적으로 꺼져 있는 이유는?**
추가 hanlp(TensorFlow) 모델에 의존하며, 로드 시 onnxruntime과 OpenMP 런타임이 충돌할 수 있습니다. 라이트 버전은 hanlp를 설치하지 않습니다 — 일반적인 사용에서는 순수 jieba 규칙으로 충분합니다. 활성화하려면 `config.json`에서 `post_processing.ner_protection.enabled`를 `true`로 설정하세요(hanlp를 별도로 설치해야 합니다).

**붙여넣기가 작동하지 않나요?**
인식 완료 시 본 앱 창에 포커스가 있으면 붙여넣기가 건너뜁니다 — 대상 창으로 전환한 후 "다시 붙여넣기"를 클릭하세요. 일부 터미널/게임은 관리자 권한으로 실행해야 합니다.

**GPU를 사용하고 싶나요?**
CUDA 버전의 sherpa-onnx를 설치하고 설정에서 "GPU 가속 활성화"를 체크하세요.

## 감사의 말

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 추론 엔진
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — 음성 인식 모델
- [jieba](https://github.com/fxsjy/jieba) — 중국어 토큰화 및 품사 태깅