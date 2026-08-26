# Sunset-Warrior（夕陽の侍）· ボイス入力

![License: MIT](https://img.shields.io/github/license/JackChi777/Sunset-Warrior-voice-input)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![CI](https://img.shields.io/github/actions/workflow/status/JackChi777/Sunset-Warrior-voice-input/ci.yml)

> 🌐 **対応言語のドキュメント：** [中文](README.md) · [English](README.en.md) · [한국어](README.ko.md)
>
> （以下は日本語の説明）

> **📌 プロジェクト紹介**：**Sunset-Warrior（夕陽の侍）**——右 Shift を押しながら話し、離すと認識テキストをアクティブウィンドウへ自動貼り付け。オフライン・無料・完全ローカルの音声入力ツール（SenseVoice Small）。

オフライン音声入力ツールです。[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) を基にした **SenseVoice Small** モデル（標準版・約 163MB・純 int8・CPU で快適に動作）を使用します。**右 Shift を押しながら話す**と、離したときに認識したテキストが現在アクティブなウィンドウへ自動的に貼り付けられます。推論はすべてローカルで完結し、ネットワーク通信は一切ありません。

> **言語についての注意:** ルールベースの後処理パイプラインは現在 **中国語（中文）専用の最適化** です。`的/地/得` の訂正、同音字訂正、中国語数字変換、異体字正規化はすべて中国語テキストを対象としています。英語・日本語・韓国語については、音声認識自体は動作しますが、高度なテキスト訂正は適用されません。他言語への対応は予定です。

## 機能

- 🎤 **グローバルショートカット**: 右 Shift を 0.3 秒以上長押しで録音開始、離すと停止。ボタンでも操作可能
- 📋 **自動貼り付け**: 認識完了後、アクティブウィンドウへ自動で挿入（無効化可能）
- ✍️ **ルールベース後処理**（LLM 不要）:
  - 「的/地/得」のスマート訂正（jieba 品詞ルール）
  - 同音字 / 近似字訂正（ConfusionSet）
  - 中国語数字変換（一百 → 100）
  - 異体字正規化（第一批異形詞整理表）
  - カスタム用語表 / 置換表（設定から編集可能）
  - **自己学習訂正**: 手動で修正すると自動的にルール化され `user_data/learned_rules.json` に保存
- 💾 **アイドル時自動アンロード**: 既定ではアイドル 10 分でモデルをメモリから解放。次回利用時は高速に再読み込み

## 🪟 Windows ポータブル版（EXE）

Python をインストールしたくない方は、パッケージ済みの単一 EXE（モデル同梱）をダウンロードしてダブルクリックするだけで使えます：

1. [Releases](https://github.com/JackChi777/Sunset-Warrior-voice-input/releases) ページを開く
2. 最新の `Sunset-Warrior-win64.zip`（exe + モデル同梱、解凍後約 400MB）をダウンロード
3. 任意のフォルダに解凍し、`Sunset-Warrior.exe` をダブルクリック

> EXE は [GitHub Actions](.github/workflows/build-windows.yml) が自動ビルドします（`v*` タグで発火）。
> 初回実行時に Windows SmartScreen の警告が出たら「詳細情報」→「実行」をクリックしてください（未署名プログラムでは通常の表示です）。

## クイックスタート

### 1. 依存関係のインストール（Python 3.8以降）

```bash
cd pc
pip install -r requirements.txt
```

Linux では追加のシステムパッケージが必要です:
```bash
# X11
sudo apt install xdotool
# Wayland
sudo apt install wtype wl-clipboard
```

### 2. モデルのダウンロード（約163MB）

```bash
python scripts/download_models.py
```

モデルは `voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/` に展開されます。

**すでにモデルをお持ちの場合:** 再ダウンロードは不要です。`pc/config.json` の `asr.sensevoice_model_dir` を設定してください:
- 相対名: `voice-models/` 配下のディレクトリ名（既定は上記の標準名）
- 絶対パス: `model.int8.onnx` + `tokens.txt` を含むディレクトリを直接指定
設定ダイアログからも変更でき、保存後にモデルが自動再読み込みされます。

中国国内の方はミラーが利用できます:
```bash
SHERPA_ONNX_MIRROR=https://hf-mirror.com python scripts/download_models.py
```

### 3. 実行

```bash
cd pc
python main.py
```

初回起動時にモデルを読み込みます（約1〜2秒）。その後、右 Shift を押しながら話してください。

## ディレクトリ構成

```
voice-input-lite/
├── pc/                      # メインプログラム
│   ├── main.py              # エントリポイント: UI・録音・認識・挿入
│   ├── config.py            # 設定管理（config.json）
│   ├── platform_utils.py    # プラットフォーム検出・モデルパス
│   ├── post_processing.py   # テキスト後処理パイプライン（ルール層）
│   ├── proofreading.py      # 校正エンジン（ステージ0〜8 ルール + 任意のLLMステージ）
│   ├── confusion_corrector.py  # 同音/近似字訂正
│   ├── syntax_analyzer.py   # 構文解析（HanLP、任意 — 未インストールなら自動ダウングレード）
│   ├── input_injector.py    # クロスプラットフォームのテキスト挿入
│   ├── hotkey_listener.py   # グローバルショートカット監視
│   └── user_data/           # 用語表・自己学習訂正ルール
├── scripts/
│   └── download_models.py   # モデルダウンロードスクリプト
└── voice-models/            # モデルディレクトリ（ダウンロード後に生成）
```

## 未完成の点とロードマップ（貢献歓迎）

> **🐧 Linux 既知の問題**：Linux では GUI がまだ不安定です（Qt ウィンドウ/表示の問題）。当面は Windows での利用を推奨します。認識エンジン自体はクロスプラットフォームで正常動作します。Linux UI は後日修正予定です。

**ルール層はまだ不完全**です。「使える」レベルであり「完璧」ではありません：

- **「的/地/得」**：多くのケースで正しいが、複雑な文（多重修飾・節をまたぐ文）では誤判定が残る
- **「在/再」**：現在のルール処理はまだ理想的ではありません。見落としも誤修正も起こり得ます
- **同音字/近似字訂正**：口語的・方言色の強いテキストで誤訂正の可能性あり
- **数字変換・異体字表**：カバー範囲が限定的で、複雑な中国語数字の読みは未対応

**LLM 校正フックは実装済み（既定オフ・依存ゼロ）**

メインプログラムは意図的に LLM 呼び出しを含んでいません — 完全オフライン・無料・低遅延を保つためです。ただし校正エンジン `proofreading.proofread()` には LLM ステージ（ステージ9）が組み込まれており、`llm_caller(text, prompt) -> 修正テキスト` というコールバックを渡すだけで接続できます。OpenAI 互換エンドポイントなら何でも使えます：**Ollama**（ローカル・無料）、DeepSeek、Qwen、vLLM など。

例（Ollama の OpenAI 互換エンドポイント、`pip install openai`）：

```python
from proofreading import proofread
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def llm_caller(text: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model="qwen2.5",  # ollama pull した任意のモデル
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": text}],
        temperature=0.7,
    )
    return resp.choices[0].message.content

result = proofread("这句话里的在再帮我改对", mode='basic', llm_caller=llm_caller)
print(result.text)
```

`mode` は `'basic'`（内蔵プロンプト）または `'custom'`（config の `custom_prompts` で設定したカスタムプロンプト）を指定できます。`prompt` 引数はエンジンが準備した推敲用プロンプトです。そのまま LLM に渡し、修正テキストを返してください。

> `llm_caller` は純粋なコールバックです — OpenAI・Ollama・DeepSeek・ローカル vLLM・あるいはローカルのルール関数まで、**何にでも**接続できます。接続すれば「的/地/得」「在/再」などのルール層の弱点も大幅に改善されます。

**ロードマップ（計画中）**

- [ ] 「的/地/得」「在/再」ルールの強化
- [ ] 英語・日本語・韓国語の後処理対応
- [ ] モバイル版（Android / iOS）移植

## よくある質問

**モデルの読み込みに失敗する場合は？**
`voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/` に `model.int8.onnx` と `tokens.txt` があるか確認し、ダウンロードスクリプトを再実行してください。

**2025-09-09 版を使わないのはなぜ？**
それは広東語専用の変種（ASLP-lab の sensevoice_small_yue 由来）で、句読点に対応しておらず、標準中国語では字が欠落します。本プロジェクトの標準版（2024-07-17、FunAudioLLM/SenseVoice 由来）は句読点と ITN に対応しています。

**NER 固有名詞保護が既定でオフなのはなぜ？**
追加の hanlp（TensorFlow）モデルに依存し、読み込み時に onnxruntime と OpenMP ランタイムが衝突する可能性があります。ライト版は hanlp をインストールしません — 通常利用では純粋な jieba ルールで十分です。有効化するには `config.json` で `post_processing.ner_protection.enabled` を `true` にしてください（hanlp を別途インストールする必要があります）。

**貼り付けが動作しない場合は？**
認識完了時に本アプリのウィンドウにフォーカスがある場合は貼り付けがスキップされます — 対象ウィンドウに切り替えて「再貼り付け」をクリックしてください。一部の端末・ゲームでは管理者権限で実行する必要があります。

**GPU を使いたい場合は？**
CUDA 版の sherpa-onnx をインストールし、設定の「GPU アクセラレーションを有効化」にチェックを入れてください。

## 謝辞

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 推論エンジン
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — 音声認識モデル
- [jieba](https://github.com/fxsjy/jieba) — 中国語の分かち書きと品詞タグ付け