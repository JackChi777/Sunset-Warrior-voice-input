# 开发与发布流程（DEVELOPING）

> 本文记录 Sunet-Warrior（夕阳武士）语音输入项目的**开发、测试、构建、发布**全流程，
> 供项目维护者和潜在贡献者参考。需要了解功能用法请看 [README.md](README.md)。

---

## 一、项目是什么 / 为什么开源

一个**离线语音输入工具**：按住右 Shift 说话，松开自动把识别文本粘贴到当前窗口。
基于 sherpa-onnx + SenseVoice Small，纯本地推理，规则层中文后处理，可选挂 LLM 校对。

这个项目用了半年多打磨。市面上功能更强的语音输入很多，开源它不是因为它最强，
而是因为它**完全离线、免费、代码透明**——如果有人愿意顺手帮忙补齐其他语言的
后处理规则、优化「的/地/得」「在/再」等细节，那更是意外之喜。

**开源动机给贡献者的话**：可能性很小，但没关系。你如果愿意看代码、提 PR、
修一行规则，都欢迎。

---

## 二、代码结构

```
voice-input-lite/
├── pc/                      # 主程序（核心代码都在这里）
│   ├── main.py              # ★ 入口：UI、录音、识别、注入、首次运行下载模型
│   ├── config.py            # 配置管理（config.json，深度合并默认值）
│   ├── platform_utils.py    # 平台检测、打包环境识别、路径定位
│   ├── model_downloader.py  # ★ 模型自动下载（HuggingFace 免认证）
│   ├── post_processing.py   # 文本后处理流水线（规则层）
│   ├── proofreading.py      # 校对引擎（阶段 0-8 规则 + 可选 LLM 阶段 9）
│   ├── confusion_corrector.py # 同音/形近字纠错
│   ├── syntax_analyzer.py   # 句法分析（HanLP，可选，默认关闭）
│   ├── input_injector.py    # 跨平台文本注入
│   ├── hotkey_listener.py   # 全局热键监听
│   └── user_data/           # 术语表、自学习纠错规则
├── scripts/
│   ├── build_exe.py         # PyInstaller 打包脚本
│   └── download_models.py   # 命令行下载模型（可选，现多在首次运行时自动下载）
├── .github/workflows/
│   ├── ci.yml               # 语法检查 + 规则冒烟测试（每次 push）
│   └── build-windows.yml    # 打 v* 标签 → 自动构建 EXE → 发布到 Release
├── README.{md,en,ja,ko}     # 四语说明
└── DEVELOPING.md            # 本文档
```

---

## 三、本地运行（源码方式）

```bash
# 1. 装依赖
cd pc
pip install -r requirements.txt

# 2. 运行（首次会弹窗询问是否从 HuggingFace 下载模型，约 163MB）
python main.py
```

模型保存在 `voice-models/`（已被 `.gitignore` 排除，不会进仓库）。

**如果不想自动下载**，自己放模型：
```bash
python scripts/download_models.py        # 或
# 手动下载到 voice-models/<模型目录名>/，含 model.int8.onnx + tokens.txt
```

---

## 四、日常开发流程

### 1. 改代码 → 自测

改完在 `pc/` 下做快速验证（无需真实麦克风）：

```bash
# 规则后处理冒烟（应该输出：新华社报道了这件事，水分和成分都要核对）
python -X utf8 -c "
from proofreading import proofread
print(proofread('新华社报导了这件事，水份和成份都要核对', mode='none').text)
"

# 语法检查
python -m py_compile pc/main.py pc/*.py scripts/*.py
```

### 2. 代码规范

- 只加 `requirements.txt` 里实际用到的依赖；HanLP/torch 等**绝不新增**（保持精简 + 避免打包问题）
- 中文注释随意，但 **print / logging 尽量避开非 ASCII 字符**（Windows 控制台 GBK/cp1252 会崩，见第五节坑 4）
- 后处理是纯规则优先，LLM 只作为可选的 `llm_caller` 回调，主程序不依赖任何网络

### 3. 提交（本地）

```bash
cd voice-input-lite
git add -A
git commit -m "描述改动的原因，不只写改了什么"
git push origin master
```

push 后 GitHub 的 `ci.yml` 会自动跑语法检查 + 规则冒烟测试，通过为绿 ✓。

---

## 五、发布 Windows EXE（打标签即可，全自动）

```bash
cd voice-input-lite
git tag v1.1.0                     # 版本号自定，v 开头
git push origin v1.1.0
```

推送标签后，GitHub Actions（`build-windows.yml`）自动执行：

1. 装依赖（Python 3.12 + requirements + PyInstaller）
2. 打包单文件 EXE（不含模型，约 88MB）
3. 组装发布目录、打成 `Sunset-Warrior-win64.zip`
4. 自动创建 GitHub Release 并附上 zip

**用户拿到 zip**：解压双击 `Sunset-Warrior.exe`，首次运行弹窗询问 → 从 HuggingFace
自动下载模型 → 直接可用，无需装 Python。

发布完可以到 Actions 页面确认绿 ✓，到 Release 页面确认 zip 已挂上。

---

## 六、踩过的坑（维护者避坑指南）

这些都是在打通自动构建时逐个排查出来的，遇到类似问题先对照看看：

| 坑 | 根因 | 修复 |
|---|---|---|
| CI 模型下载 403 | 默认 Python-urllib UA 被 GitHub 拒 | 改为**首次运行从 HuggingFace 免认证下载**（`model_downloader.py`） |
| PyInstaller 打包崩溃 | 机器装有 hanlp/torch，被递归拖入包 → sentencepiece 崩溃 | `build_exe.py` 里 `--exclude-module` 排除非精简依赖 |
| 打包步骤立刻退出 1 | Windows runner 默认 PowerShell，`exit ${PIPESTATUS[0]}` 是 bash 语法 | 步骤显式 `shell: bash` |
| `UnicodeEncodeError` (cp1252) | CI 控制台非 UTF-8，中文 print 崩溃 | 构建脚本英文输出 + `sys.stdout.reconfigure(encoding='utf-8')` |
| Release 发布 403 | workflow 默认只读权限 | 加顶层 `permissions: contents: write`；去掉 `generate_release_notes` |
| 偶发段错误（运行时） | PyQt5 自带旧版 MSVC 运行时与 onnxruntime 冲突 | `main.py` 顶部预加载系统版 `msvcp140.dll` |
| UI 白板 | `init_ui` 漏了 `central_widget.setLayout(layout)` | 补上该行 |

---

## 七、给贡献者的指引（简述）

- **提 Issue**：讲清楚「现象 / 期望 / 环境（OS、Python 版本、源码 or EXE）」。
- **提 PR**：从 master 开分支，符合第二节规范，尽量带一个改动说明。规则层改完
  请确认 `python -X utf8 -c "from proofreading import proofread; print(proofread('<例句>').text)"`
  输出符合预期。
- **想做别的语言**：重点是 `post_processing.py` 的规则层 + `confusion_corrector.py`，
  目前只针对中文优化，欢迎补齐英文 / 日文 / 韩文。

更多细节见 [README.md](README.md) 的「未完善之处与 Roadmap」。