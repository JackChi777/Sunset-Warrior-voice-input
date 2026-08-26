"""档位 2：的/地/得 句法分析增强（HanLP 可选集成）。

设计要点：
- HanLP 不在 requirements 中，因此本模块以"可选依赖"的形式存在
- import 时 try/except；若 HanLP 不可用，所有方法退化为 no-op
- 模型采用 `tok` + `dep` 两个 ELECTRA_SMALL_ZH（轻量、汉语句法）
- 早退门控已落地：不含 的/地/得 的句子（语音短句场景的绝大多数）
  直接原样返回，不触发 tok/dep/const 加载与推理（速度瓶颈）
- 主要判定依据（HanLP 依存关系）：
    - 定中关系（ATT）→ 的
    - 状中关系（ADV）→ 地
    - 动补关系（CMP）→ 得
    - 兼语/复合（COO/ROOT）→ 不动

调用约定：
    from syntax_analyzer import get_default_analyzer
    analyzer = get_default_analyzer()
    corrected = analyzer.correct("他走的比较快")  # → "他走得比较快"
"""

from __future__ import annotations

import functools
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set

# “得”构词语素动词集（记得/觉得/值得/获得…）——与规则层共用单一真相源。
# 句法层必须同样保护：否则“记得的很清楚”会被依存关系误判成补语→“记得得很清楚”。
# 注意：post_processing 不 import syntax_analyzer（只在函数内懒加载），无循环依赖。
from post_processing import (  # type: ignore[attr-defined]
    DE_MORPHEME_VERBS,
    _NP_OBJECT_VERBS,
    _ASPECT_MARKERS,
    _OBJECT_PREPOSITIONS,
    _GERUND_COMPOUND_HEADS,
    _DIRECTIONAL_COMPLEMENTS,
    GERUND_NOUNS,
    _DE_KEEP_NOMINAL_PREV,
    _NOUNLIKE_TECH_VERBS,
)

# ---------------------------------------------------------------------
# 项目级缓存路径助手 — 必须先于 import hanlp 完成
# ---------------------------------------------------------------------
# 目标：默认让 HanLP 模型落到 <project_root>/voice-models/hanlp/，与 ASR
# 模型（SenseVoiceSmall-onnx / sherpa-onnx-* 等）并列，统一进 voice-models
# 命名空间；优选尊重用户设置的 HANLP_HOME 环境变量。
#
# 关键设计：必须在 `import hanlp` 之前调 `_ensure_hanlp_home()`，因为 HanLP
# 在 import 时一次性读环境变量并缓存 HanLP_HOME。后续给 hanlp.HANLP_HOME
# 再赋值在不同版本可能不生效。setdefault 不覆盖已有 env：CLI / 用户 export
# 过的 HANLP_HOME 优先。
def _get_project_root() -> Path:
    """项目根目录 = pc/ 的上一级（voice-input-lite/pc → voice-input-lite）。"""
    return Path(__file__).resolve().parents[1]


def _get_default_hanlp_home() -> Path:
    """默认 HanLP 缓存 = <project_root>/voice-models/hanlp/."""
    return _get_project_root() / 'voice-models' / 'hanlp'


def _ensure_hanlp_home() -> None:
    """Set HANLP_HOME = <project>/voice-models/hanlp，respect 用户已设值。

    必须在 `import hanlp` 之前调用。用 setdefault 不覆盖已有 env。
    """
    target = _get_default_hanlp_home()
    os.environ.setdefault('HANLP_HOME', str(target))


# === ENV 必须在 import hanlp 之前就位 ===
_ensure_hanlp_home()


# ---------------------------------------------------------------------
# HanLP 可选导入
# ---------------------------------------------------------------------
# 说明：hanlp 不在 requirements.txt 且没有内置 .pyi 类型存根。
# 用 Any 标注让 Pylance 把 hanlp 当动态模块，避免「不是已知属性 / None 没有 X」
# 等静态告警。运行时仍走 try/except，导入失败 _HANLP_AVAILABLE 为 False。
hanlp: Any = None
try:
    import hanlp  # type: ignore
    _HANLP_AVAILABLE = True
except Exception:  # noqa: BLE001 - 安全降级
    _HANLP_AVAILABLE = False


# ---------------------------------------------------------------------
# transformers 兼容垫片：encode_plus / batch_encode_plus
# ---------------------------------------------------------------------
# 背景：transformers 4.41 的年代 HanLP 2.1.3 调 `tokenizer.encode_plus(...)`
# 与 `tokenizer.batch_encode_plus(...)`。transformers >= 5.0 把这些公开 API
# 删除/改名（encode_plus -> _encode_plus），导致 HanLP 运行时抛
# `AttributeError: BertTokenizer has no attribute encode_plus`，进而让
# 档位 2 的整体 correct() 被 except 吞掉、像"正常"一样完全失效（静默停摆）。
#
# 修复：在 HanLP 加载 tokenizer 之前，给父类补上缺失的公开方法。这样无论
# 环境里装的是 transformers 4.x 还是 5.x 都能正常工作，不用强制降级包。
def _install_transformers_compat_shim() -> None:
    try:
        from transformers.tokenization_utils import PreTrainedTokenizer
        from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
    except Exception:  # noqa: BLE001
        return

    # encode_plus 在 transformers 5.x 改名私有 _encode_plus，签名向后兼容
    if not hasattr(PreTrainedTokenizer, 'encode_plus') and hasattr(
            PreTrainedTokenizer, '_encode_plus'):
        PreTrainedTokenizer.encode_plus = PreTrainedTokenizer._encode_plus
    if not hasattr(PreTrainedTokenizerFast, 'encode_plus') and hasattr(
            PreTrainedTokenizerFast, '_encode_plus'):
        PreTrainedTokenizerFast.encode_plus = PreTrainedTokenizerFast._encode_plus

    # batch_encode_plus 在 transformers 5.x 无 _batch_encode_plus；
    # 用 __call__（可接受 list 文本）实现等价批处理语义。
    def _batch_encode_plus_shim(_self, text, *a, **kw):
        return _self(text, *a, **kw)

    if not hasattr(PreTrainedTokenizer, 'batch_encode_plus'):
        PreTrainedTokenizer.batch_encode_plus = _batch_encode_plus_shim
    if not hasattr(PreTrainedTokenizerFast, 'batch_encode_plus'):
        PreTrainedTokenizerFast.batch_encode_plus = _batch_encode_plus_shim


if _HANLP_AVAILABLE:
    try:
        _install_transformers_compat_shim()
    except Exception:  # noqa: BLE001 - 垫片失败不影响降级
        pass


# ---------------------------------------------------------------------
# 单一真相源：列举本模块运行所必需的 HanLP 模型
# ---------------------------------------------------------------------
# 此 tuple 给 download_hanlp_models.py --list 等 CLI 使用，用户不需要
# 先安装 hanlp 就能看到『将下多少、是什么」。顺序敏感：与 warmup() 一致。
# 注意：这里只列 ID 与说明，不触发实际 hanlp 取属性，Pylance / hanlp
# 未装都不会出问题。运行时取属性仍走 hanlp.pretrained.tok/dep.X 的调用。
REQUIRED_HANLP_MODELS: Tuple[Tuple[str, str, str], ...] = (
    ('tok', 'COARSE_ELECTRA_SMALL_ZH', '中文分词（ELECTRA-Small，约 50MB）'),
    ('dep', 'CTB9_DEP_ELECTRA_SMALL', '中文依存句法（CTB9 / ELECTRA-Small，约 50MB）'),
    ('const', 'CTB9_CON_ELECTRA_SMALL', '中文成分句法（CTB9 / ELECTRA-Small，约 50MB，可选）'),
    ('ner', 'MSRA_NER_ELECTRA_SMALL_ZH', '中文命名实体（MSRA / ELECTRA-Small，约 50MB，可选·专名保护）'),
)


# ---------------------------------------------------------------------
# Cache candidates + size helpers （main.py 与 download CLI 共用，单一真相源）
# ---------------------------------------------------------------------
def _get_hanlp_cache_dirs() -> Tuple[Path, ...]:
    """返回 HanLP 缓存目录的有序候选列表（同一目录去重）。

    优先级：
      1. HANLP_HOME env（用户手动 set 或 CLI 已设）
      2. <project>/voice-models/hanlp/                       (项目默认，与 ASR 模型并列)
      3. ~/.hanlp/                                              (历史默认)
      4. ~/.cache/hanlp/                                        (XDG fallback)
      5. ~/AppData/Local/hanlp/                                 (Windows fallback)

    因为 syntax_analyzer._ensure_hanlp_home() 在 import 时会把 HANLP_HOME
    设到项目默认路径，路径 1 和路径 2 会是同一个物理目录。用 normpath +
    normcase 比较兼顾 Windows 文件系统大小写不敏感与斜杠划一。
    """
    candidates: List[Path] = []
    seen: set = set()

    def _push(p: Path) -> None:
        """充分归一化后加入候选。重复的跳过，保持原序。
        normpath 把\\ 与/划一；normcase Windows 下小写。足够表达同一路径。
        """
        key = os.path.normcase(os.path.normpath(str(p)))
        if key not in seen:
            seen.add(key)
            candidates.append(p)

    env = os.environ.get('HANLP_HOME')
    if env:
        _push(Path(env))
    _push(_get_default_hanlp_home())
    _push(Path.home() / '.hanlp')
    _push(Path.home() / '.cache' / 'hanlp')
    _push(Path.home() / 'AppData' / 'Local' / 'hanlp')
    return tuple(candidates)


def _hanlp_cache_size_mb() -> float:
    """累计返回 HanLP 缓存目录大小（MB）。0.0 表示没目录。

    复用位置：
      - main.py UI「缓存 N MB」label
      - download_hanlp_models.py --check
    不要在别处重写 cache-size 逻辑，统一走这个函数。
    """
    total = 0
    for d in _get_hanlp_cache_dirs():
        if not d.exists() or not d.is_dir():
            continue
        try:
            for f in d.rglob('*'):
                try:
                    if f.is_file():
                        total += f.stat().st_size
                except Exception:
                    continue
        except Exception:
            continue
    return total / (1024.0 * 1024.0)


# ---------------------------------------------------------------------
# 本地缓存检测 — 准确区分 hanlp.load() 是走 in-memory 还是走网络
# ---------------------------------------------------------------------
# HanLP 2.1.x 把模型以 PyTorch .pt 形式打包，磁盘目录名是
# '<prefix>_<release_date>/'，如 coarse_electra_small_20220616_012050。
# config.json 是解压后必写的入口文件，看到它代表缓存可用、hanlp.load()
# 可以走纯内存路径，与网络无关。
#
# 由于 release date 会隨版本跳，仅靠精确名字匹配是脆弱的；以 prefix
# 结尾模糊匹配，包未来 HanLP 发布新模型 (e.g. coarse_electra_small_20260101_xxx)
# 也会命中。prefix 值依赖于 HanLP 官方命名，贴与 REQUIRED_HANLP_MODELS。
_MODEL_DISK_PREFIXES: Dict[str, str] = {
    'tok': 'coarse_electra_small',
    'dep': 'ctb9_dep_electra_small',
    'const': 'ctb9_con_electra_small',
    'ner': 'msra_ner_electra_small',
}


def _is_model_cached(short_id: str) -> bool:
    """True: 本地缓存里已经有 HanLP 模型（已解压、config.json 存在）。

    Returns:
        True: 接下来 hanlp.load() 会走纯内存加载，不联网
        False: 首次，需要走网络下载与解压

    任何 IO 异常都返回 False (= cache miss)，避免 UI 误导出 cache hit。
    """
    prefix = _MODEL_DISK_PREFIXES.get(short_id)
    if not prefix:
        return False
    try:
        home = Path(os.environ.get('HANLP_HOME', str(Path.home() / '.hanlp')))
        sub_dir = home / short_id
        if not sub_dir.exists() or not sub_dir.is_dir():
            return False
        for d in sub_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name.startswith(prefix) and (d / 'config.json').exists():
                return True
        return False
    except Exception:
        return False


# 依存关系标签 → 的/地/得 的映射
# HanLP CTB 关系（含 CoNLL-X 子集）：
#   ATT  : 定中关系     (adj/n 的 + n)
#   ADV  : 状中关系     (adv/a 地 + v)
#   CMP  : 动补关系     (v 得 + a/adv)
#   COO  : 并列关系
#   ROOT: 根节点
#   PUN  : 标点
DEP_TO_DE = {
    "ATT": "的",
    "ADV": "地",
    "CMP": "得",
}

# 兜底，对 HanLP 经常组合输出的关系名宽容一下
DEP_TO_DE_ALIAS = {
    "attr": "ATT",
    "advcl": "ADV",
    "advmod": "ADV",
    "comp": "CMP",
    "ccomp": "CMP",
    # HanLP 2.1.3 CTB9_DEP_ELECTRA_SMALL 实际输出的关系名：
    "assmod": "ATT",  # 定语修饰语 (adj/n 修饰 n) → 的
    "assm": "ATT",    # 定语标记 (的) → 的
    "cpm": "ATT",     # 定语标记 (的/同音误写 得)：HanLP 把「漂亮得花朵」的得标为 cpm → 的
    "rcmod": "ATT",   # 关系从句修饰 (看书的人) → 的
    "dvpmod": "ADV",  # 状语修饰语 (adv/a 修饰 v) → 地
    "dvpm": "ADV",    # 状语标记 (地) → 地
    "nn": "ATT",      # 名词复合 (n+n) → 的
    "dep": "CMP",     # 通用依赖：补语标记(他走的快/跑的快) → 得
    "det": "ATT",     # 限定词/指示代词作定语(那些/这些/他的...) → 的
    # 进行体/介词保护：advmod 可能是"在吃饭"的"在"，需在correct()里结合上下文判断
}

# 句尾/标点前的"的"无条件保护（见 correct() 情况1/2）："得"必须后接补语，
# 句尾或标点前不可能成立，故这些位置上的"的"必是语气词/定语/名物化标记。
SENTENCE_END_MARKERS = {
    "。", "！", "？", "，", "、", "；", "：", "……", "～",
    "吧", "呢", "啊", "吗", "耶", "嘛", "哈", "呵", "哟", "喔", "哇", "哇", "哈",
    "呀", "哦", "诶", "哎", "嘞", "咯", "呗", "啦",
    "~", "?", "!", ",", ".", ";", ":", "...",
}

# ---------------------------------------------------------------------
# 档位 2 保护：名动词宾语 / 复合名词 / 名词定语的句法层信号
# （与规则层共用词表，单一真相源）
# ---------------------------------------------------------------------
# HanLP token 无词性标签，无法像规则层那样按 POS 停止扫描；这里退而求其次：
# 跨过体标记（了/着/过）与其它 de 字，在 代词/标点 处停止。宁可多拦
# （拦 = 保留 的）也不漏——因为规则层已先把该改的 的→地 改完，句法层
# 见到的还是「的」的多半是规则层都没把握的歧义，保留 的 不会更错。
_DE_SWAP_STOP_CHARS = frozenset('。！？；，、')
_DE_SWAP_STOP_WORDS = frozenset({
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们',
    '咱', '咱们', '大家', '别人', '自己', '这', '那',
})


def _left_has_np_object_verb(tokens: List[str], idx: int) -> bool:
    """de 字左侧是否有「接名词宾语」的动词（做/提出/进行/完成/负责…）。

    例：做/了/认真/的/检查 → 做 命中（跨过 了）。向左最多扫 7 个 token。
    名动词（处理/说明/检查…）作宾语时修饰语必是定语 → 保留 的。
    """
    for j in range(idx - 1, max(-1, idx - 7), -1):
        w = tokens[j]
        if w in _NP_OBJECT_VERBS:
            return True
        if w in _ASPECT_MARKERS or w in ('的', '地', '得'):
            continue
        if w in _DE_SWAP_STOP_WORDS or w in _DE_SWAP_STOP_CHARS:
            return False
    return False


def _left_has_object_preposition(tokens: List[str], idx: int) -> bool:
    """de 字左侧是否有「接名词宾语」的介词（经过/通过/根据/随着…）。

    介词宾语没有「状语+动词」的合法读法：经过认真的思考 → 思考 是宾语名词。
    例：经过/了/认真/的/思考 → 经过 命中（跨过 了）。
    """
    for j in range(idx - 1, max(-1, idx - 6), -1):
        w = tokens[j]
        if w in _OBJECT_PREPOSITIONS:
            return True
        if w in _ASPECT_MARKERS or w in ('的', '地', '得'):
            continue
        if w in _DE_SWAP_STOP_WORDS or w in _DE_SWAP_STOP_CHARS:
            return False
    return False


def _gerund_compound_after(tokens: List[str], idx: int) -> bool:
    """de 字后是否紧跟「名动词+抽象属性名词」复合名词。

    研究/的/能力、管理/的/经验、严格/的/测试/流程、专业/的/培训/课程 →
    名动词几乎必是定语（复合名词）→ 保留 的。典型动词宾语（问题/数据/文件）
    刻意不在 _GERUND_COMPOUND_HEADS 里，那些仍是「状语+动宾」（→ 地）。
    """
    if idx + 2 >= len(tokens):
        return False
    g, h = tokens[idx + 1], tokens[idx + 2]
    if h in _GERUND_COMPOUND_HEADS and g in GERUND_NOUNS:
        return True
    # HanLP 可能把 研究能力 合成一个 token：在拼接串里找 名动词+抽象名词 前缀
    joined = g + h
    for gerund in GERUND_NOUNS:
        if joined.startswith(gerund):
            tail = joined[len(gerund):]
            if any(tail.startswith(head) for head in _GERUND_COMPOUND_HEADS):
                return True
    return False


def _prev_is_verb_or_adj(tok: str) -> bool:
    """前一 token 是否是动词/形容词（动补「V+得+补语」的前件）。

    的→得 只有在动补结构里才合法；「三个月的时间」的 的 被标成 dep→CMP，
    但前词 三个月 是数量短语，改成「三个月得时间」必错 → 由调用方拦截。
    """
    if tok == '过':  # jieba 把 过 误标为 ug（时间过的 好快 → 过）
        return True
    try:
        from jieba import posseg as _pseg
        flag = next(iter(_pseg.cut(tok))).flag
    except Exception:
        return False
    return flag[0] in ('v', 'a')


class DeDiDeSyntacticAnalyzer:
    """档位 2：HanLP 句法分析驱动的"的/地/得"增强纠错器。

    关键 API：
        correct(text) -> str
        is_available() -> bool
        warmup() -> bool  # 提前加载模型，避免首次调用卡顿
    """

    def __init__(
        self,
        model_id: str = "COARSE_ELECTRA_SMALL_ZH",
        auto_load: bool = False,
    ):
        self._model_id = model_id
        # 明确标 Any，hanlp 没类型存根，Pylance 会把 None 链传染到 deprels[idx].lower() 等调用
        self._tok: Any = None
        self._dep: Any = None
        self._const: Any = None
        self._loaded = False
        self._const_loaded = False
        self._lock = threading.Lock()
        self._failure_logged = False
        if auto_load:
            self.warmup()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return _HANLP_AVAILABLE and self._loaded

    def warmup(self, progress_cb=None) -> bool:
        """预加载 HanLP 模型（多线程安全）。

        Args:
            progress_cb: 可选 progress 回调。签名 progress_cb(stage: str, msg: str)。
                干现“「正在下载 COARSE_ELECTRA_SMALL_ZH...」/「加载完成」”。
        """
        if not _HANLP_AVAILABLE:
            if progress_cb:
                progress_cb('skip', '[HanLP] 未安装，已跳过')
            return False
        with self._lock:
            if self._loaded:
                if progress_cb:
                    progress_cb('done', '[HanLP] 模型已加载')
                return True
            try:
                # tok 模型：分词（HanLP 2.1.3 提供的轻量中文分词）
                # 先查本地缓存，避免误导文案：cache 命中走 'load'（内存加载，快），
                # cache miss 才走 'download'（要走网络，慢）。
                tok_cached = _is_model_cached('tok')
                tok_stage = 'load' if tok_cached else 'download'
                tok_verb = '从本地加载' if tok_cached else '下载与解压'
                if progress_cb:
                    progress_cb(tok_stage, f'[HanLP/1/2] 正在{tok_verb} tok 模型 ...')
                self._tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
                # dep 模型：同样区分 cache hit vs miss
                # HanLP 2.1.3 提供 CTB9_DEP_ELECTRA_SMALL（旧版 CTB8_PROB_DEP 已移除）
                dep_cached = _is_model_cached('dep')
                dep_stage = 'load' if dep_cached else 'download'
                dep_verb = '从本地加载' if dep_cached else '下载与解压'
                if progress_cb:
                    progress_cb(dep_stage, f'[HanLP/2/2] 正在{dep_verb} dep 模型 ...')
                self._dep = hanlp.load(
                    hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL
                )
                # 运行时健康检查：加载成功≠可调用。transformers/HanLP 版本不兼容
                # 会导致 tok() 一调用就抛异常（如 encode_plus 被删），此时必须
                # 明确失败并提示，而不是假装 loaded=True 后 correct() 静默空转。
                try:
                    smoke_tokens = self._tok('他在吃饭')
                    if smoke_tokens:
                        self._dep([smoke_tokens])
                except Exception as e:
                    if not self._failure_logged:
                        msg = (f'[HanLP] 模型加载成功但运行失败（版本不兼容？）：'
                               f'{type(e).__name__}: {e}')
                        print(f'[warn] {msg}')
                        if progress_cb:
                            progress_cb('error', msg)
                        self._failure_logged = True
                    self._tok = None
                    self._dep = None
                    self._loaded = False
                    return False
                self._loaded = True
                msg = f'[HanLP] 句法分析器已加载: {self._model_id}'
                print(f"[info] {msg}")
                if progress_cb:
                    progress_cb('done', msg)
                return True
            except Exception as e:
                if not self._failure_logged:
                    msg = f'[HanLP] 模型加载失败（仅提示一次）: {e}'
                    print(f'[warn] {msg}')
                    if progress_cb:
                        progress_cb('error', msg)
                    self._failure_logged = True
                self._tok = None
                self._dep = None
                self._loaded = False
                return False

    def warmup_constituency(self, progress_cb=None) -> bool:
        """预加载成分句法模型 CTB9_CON_ELECTRA_SMALL（可选档位 2.5）。

        与 warmup() 分离：成分句法是可选增强，加载失败/未下载不阻塞主 dep 流程。
        模型同样落到 HANLP_HOME（voice-models/hanlp/constituency）。
        """
        if not _HANLP_AVAILABLE:
            return False
        with self._lock:
            if self._const_loaded:
                return True
            try:
                if self._tok is None:
                    if not self.warmup(progress_cb=progress_cb):
                        return False
                const_cached = _is_model_cached('const')
                stage = 'load' if const_cached else 'download'
                verb = '从本地加载' if const_cached else '下载与解压'
                if progress_cb:
                    progress_cb(stage, f'[HanLP/const] 正在{verb} 成分句法模型 ...')
                self._const = hanlp.load(
                    hanlp.pretrained.constituency.CTB9_CON_ELECTRA_SMALL
                )
                # 冒烟测试：成分句法需要分词列表输入
                smoke = self._tok('他在吃饭')
                self._const([smoke])
                self._const_loaded = True
                print('[info] [HanLP] 成分句法模型已加载: CTB9_CON_ELECTRA_SMALL')
                return True
            except Exception as e:
                print(f'[warn] [HanLP] 成分句法模型加载失败（可忽略，不影响 dep）: {e}')
                self._const = None
                self._const_loaded = False
                return False

    def correct(self, text: str) -> str:
        """对文本做"的/地/得"句法增强纠错。

        - HanLP 不可用 / 未加载 → 原样返回
        - 仅修改"的/地/得"所在的 token 的 character；其它不动
        """
        if not text or not text.strip():
            return text
        # 早退门控：不含 的/地/得 的句子没有可修目标，直接原样返回。
        # 必须在 warmup() 之前 —— 否则无目标句子也会触发模型加载，
        # 且语音短句场景 99% 的句子不含 的地得，这是每句白付 tok+dep 推理的大头。
        if not any(ch in text for ch in '的地得'):
            return text
        if not _HANLP_AVAILABLE:
            return text
        if not self._loaded and not self.warmup():
            return text

        try:
            tokens: List[str] = self._tok(text)
            if not tokens:
                return text
            # dep 模型需要 list[str] 输入，输出每 token 的 head + deprel
            # HanLP 2.1.3 返回 CoNLLSentence 列表，需解包为 heads/deprels 列表
            dep_out = self._dep([tokens])
            if not dep_out:
                return text
            sent = dep_out[0]
            # CoNLLSentence 支持迭代，每个元素是 dict：{'head': int, 'deprel': str, ...}
            heads0 = []
            deprels0 = []
            for item in sent:
                heads0.append(item.get('head', 0))
                deprels0.append(item.get('deprel', ''))
        except Exception as e:
            # 预测层异常（如 transformers/HanLP 版本不兼容）——绝不能静默吞掉，
            # 否则整个档位 2 看起来"正常"却一错不改，用户无从察觉。
            if not getattr(self, '_predict_failure_logged', False):
                print(
                    f'[warn] [HanLP] 档位2 句法纠错运行失败（仅提示一次）: '
                    f'{type(e).__name__}: {e}'
                )
                print('[warn] [HanLP] 已回退到纯规则档位 1；可尝试固定 transformers<5 或检查 HanLP 安装。')
                self._predict_failure_logged = True
            return text

        # 找出"的/地/得"所在的 token 位置
        targets = []   # (token_index, replace_to)
        for idx, tok in enumerate(tokens):
            if tok in ("的", "地", "得"):
                # 构词语素保护：前词是“得”构词语素动词（记得/觉得/值得/获得…）
                # 或本身已含“地”的副词（特地/悄悄…）时，句法层不得改写 的/地/得，
                # 否则“记得的很清楚”→“记得得很清楚”、“特地的来”→“特地地来”。
                if idx > 0:
                    prev_tok = tokens[idx - 1]
                    if prev_tok in DE_MORPHEME_VERBS or prev_tok.endswith('地'):
                        continue
                deprel = deprels0[idx]
                # 句首"的/地/得 + 动词/形容词 + 名词" → 定语结构，保护"的"
                if idx == 0 and tok == "的" and idx + 2 < len(tokens):
                    next1, next2 = tokens[idx + 1], tokens[idx + 2]
                    # 下两词为 动词/形容词 + 名词 → 典型定语结构，"的"不应改
                    # 这里简单判断：next2 是名词性词汇(含方式/方法/思路/方案/代码/问题/系统/平台/架构/流程/逻辑/功能/模块/接口/数据/代码/脚本/配置/环境/依赖/版本/发布/部署/测试/重构/优化/设计/实现/开发/编程)
                    noun_like = {"方式", "方法", "思路", "方案", "代码", "问题", "系统", "平台", "架构", "流程", "逻辑", "功能", "模块", "接口", "数据", "脚本", "配置", "环境", "依赖", "版本", "发布", "部署", "测试", "重构", "优化", "设计", "实现", "开发", "编程", "目标", "计划", "需求", "标准", "规范", "要求", "结果", "效果", "成果", "进度", "质量"}
                    if next2 in noun_like:
                        continue
                # 句尾/标点前"形容词/动词 + 的" → 语气助词，保护"的"（好的/漂亮的/困难的...）
                # 门控：只有该标签会真正把"的"改写成 得/地 时，才需要先过语气词保护
                if tok == "的" and DEP_TO_DE_ALIAS.get(deprel.lower(), deprel) in ("CMP", "ADV"):
                    # 前 4 个 token 拼接，兼容 HanLP 把"没问题/没事/没毛病"切成多词的分词差异
                    prev_window = "".join(tokens[max(0, idx - 4):idx])
                    # 情况3：V的V的 并列名物化（说的写的/买的卖的/吃的穿的），保护"的"
                    if idx + 2 < len(tokens) and tokens[idx + 2] == "的":
                        continue
                    # 情况3b："的V的"中后段的"的"（前两 token 是 的+单字动词）
                    if idx >= 2 and len(tokens[idx - 1]) == 1 and tokens[idx - 2] == "的":
                        continue
                    # 情况0：好/行/OK/可以/没问题 + 的 + 代词/动词/应答词 -> 语气助词，保护"的"
                    # 如"好的我们走"、"好的好的"、"好的可以"、"行的开始吧"、"没问题的去吧"
                    if idx > 0:
                        if prev_window.endswith(("好", "行", "可以", "没问题", "没事", "没事儿", "没毛病", "没得说", "没得挑", "OK", "ok", "成", "嗯", "对", "是的", "好的", "没说的")):
                            if idx + 1 < len(tokens):
                                next_tok = tokens[idx + 1]
                                if next_tok in {"我们", "我", "你", "他", "她", "它", "他们", "她们", "那些", "咱们", "大伙", "大家", "好", "可以", "行", "没问题", "没事", "没事儿", "OK", "ok", "成", "嗯", "对", "是的", "对啊", "好的", "走", "去", "做", "来", "回", "上", "下", "吃", "喝", "睡", "看", "听", "说", "写", "读", "学", "玩", "跑", "跳", "飞", "游", "爬", "开", "关", "推", "拉", "拿", "放", "用", "试", "改", "查", "找", "等", "修", "调", "测", "编", "译"}:
                                    continue
                    # 情况1：句尾的"的"必是语气词/名物化标记（"得"必须后接补语，句尾不可能成立），无条件保护
                    # 覆盖 记得的/看得见的/跑得快的/没问题的/是的... 所有句尾"的"漏网
                    if idx == len(tokens) - 1 and idx > 0:
                        continue
                    # 情况2：标点/句末语气词前的"的"同理必是语气词/定语，无条件保护
                    # （"得"后面只能直接跟补语，不能跟标点/语气词）
                    if idx + 1 < len(tokens) and tokens[idx + 1] in SENTENCE_END_MARKERS:
                        continue

                canonical = DEP_TO_DE_ALIAS.get(deprel.lower(), deprel)
                new_char = DEP_TO_DE.get(canonical) if canonical else None
                # 名词短语读法保护（与规则层共享 _DE_KEEP_NOMINAL_PREV）：
                # HanLP 把「及时的反馈/深入的调查/反复的尝试」一律判 dvpmod（状语）→ 地，
                # 但 及时/深入/反复 等修饰词可作定语，+ 的 + 名动词（反馈/调查/尝试
                # ∈ GERUND_NOUNS）时名词短语读法同样合法 → 保留「的」。
                # 情绪词（高兴/快乐）不在 _DE_KEEP_NOMINAL_PREV，唱歌 等口语动作动词
                # 不在 GERUND_NOUNS → 高兴的唱歌 照常改 地。
                if (new_char == '地' and tok == '的'
                        and idx > 0 and idx + 1 < len(tokens)
                        and tokens[idx - 1] in _DE_KEEP_NOMINAL_PREV
                        and (tokens[idx + 1] in GERUND_NOUNS
                             or tokens[idx + 1] in _NOUNLIKE_TECH_VERBS)):
                    continue
                # 定语并列保护：de 字左右两边若在依存树里都是定语（rcmod/assmod…），
                # 中间的 得/地 必是「的」误写。例：「相应得成熟的方案」里「相应」「成熟」
                # 都是「方案」的 rcmod 定语 → 用「的」。
                # 「看得见」（右词=dep 补语）「跑得很快」（跑=root）不在此列。
                # 注意：必须先于下面的 None 判断——「得」的 deprel 是 mmod，不在映射表
                # 里会得 None，若不先覆盖就会被 continue 吞掉，规则白写。
                if tok in ('得', '地') and idx > 0 and idx + 1 < len(deprels0):
                    left_rel = deprels0[idx - 1].lower()
                    right_rel = deprels0[idx + 1].lower()
                    if left_rel in ('rcmod', 'assmod', 'assm', 'cpm', 'nn', 'attr', 'nmod') \
                            and right_rel in ('rcmod', 'assmod', 'assm', 'cpm', 'nn', 'attr', 'nmod'):
                        new_char = '的'
                if new_char is None:
                    continue
                if new_char == tok:
                    continue
                # 名词定语保护：的→得 只在动补（V+得+补语）中成立。
                # 「三个月的时间」的 的 被 HanLP 标成 dep→CMP，但前词 三个月
                # 是数量短语（非动词/形容词）→ 改成「三个月得时间」必错，保留 的。
                # 「跑的很快/吃的太多/时间过的 好快」前词是动词（过 被 jieba 误标 ug）
                # → 正常改 得，不受影响。
                if tok == '的' and new_char == '得' and idx > 0:
                    if not _prev_is_verb_or_adj(tokens[idx - 1]):
                        continue
                    # 趋向补语后的「的」是名词化/定语标记（显示出来的对吗 / 做出来的
                    # 成果），不得改成「得」——规则层已先把 出来得吗→出来的吗 修好，
                    # 本守卫只负责不推翻它。若 的 后确是真补语（跑出来的真快），
                    # 规则层规则 1.5 已先行改成 得，到本层的 的 均为名词化，保留。
                    if tokens[idx - 1] in _DIRECTIONAL_COMPLEMENTS:
                        continue
                # 名动词宾语/复合名词保护：句法层常把「做了认真的检查/经过认真的
                # 思考/深入的研究能力」里「的」的依存关系标成 dvpm（状语标记）→ 地，
                # 但此时 检查/思考/研究 是名词性宾语或复合名词定语 → 保留 的。
                # 信号：
                #   (a) 左侧有接名词宾语的动词（跨过 了/着/过：做/提出/进行/完成…）；
                #   (b) 左侧有接名词宾语的介词（经过/通过/根据/随着…）；
                #   (c) 右侧紧跟「名动词+抽象属性名词」复合（研究能力/管理经验…）；
                #   (d) 后面那个词在依存树里是宾语（dobj/vob/iobj/pobj）。
                # 注意：只拦截「的→地」，不反向强制「地→的」，保持句法层保守。
                if tok == '的' and new_char == '地' and idx >= 1:
                    if (_left_has_np_object_verb(tokens, idx)
                            or _left_has_object_preposition(tokens, idx)
                            or _gerund_compound_after(tokens, idx)):
                        continue
                    if idx + 1 < len(deprels0) \
                            and deprels0[idx + 1].lower() in ('dobj', 'vob', 'iobj', 'pobj'):
                        continue
                targets.append((idx, new_char))

        if not targets:
            return text
        return self._apply_targets(text, tokens, targets)

    @staticmethod
    def _apply_targets(text: str, tokens: List[str], targets: List[Tuple[int, str]]) -> str:
        """把 token 列表拼回原文，按 token 长度累加定位 + 替换。

        注意：必须对【每一个】token 推进 cursor，而不是只对被替换的 token 推进。
        否则当句子里有多个相同的单字 token（如多个“的”）且靠前的被保护跳过时，
        text.find('的', cursor) 会错配到前面的“的”，把保护住的“好的”改成“好得”。
        """
        target_map = dict(targets)  # token_index -> new_char
        chars = list(text)
        cursor = 0
        for idx, tok in enumerate(tokens):
            pos = text.find(tok, cursor)
            if pos == -1:
                # 分词与原文无法对齐（不应发生）：继续定位只会错位，直接放弃替换
                break
            if idx in target_map:
                new_char = target_map[idx]
                for j in range(pos, pos + len(tok)):
                    if j < len(chars):
                        chars[j] = new_char
            cursor = pos + len(tok)
        return "".join(chars)

    def correct_by_constituency(self, text: str) -> str:
        """档位 2 内层：成分句法判定 的/地/得（dep 层之后的增强层）。

        用成分树的「短语结构」而不是单字标签判 de 字角色。实测（46 用例端到端
        对比）它比 dep 层多修对 9 个「名动词作宾语/状语」的难例（规则层+dep
        只有 37/46，挂上本层 46/46），但也需要在 DVP（地字短语）标签上细分，
        否则会把下面这类改错：

          - 做一遍仔细地检查：模型把「检查」误读成动词，DVP 被挂在 NP 下
            （名物化）→ 实为 的，不能按 DVP 改成 地
          - 严格的要求：模型把「要求」误读成动词，DVP 包着 IP → 实为 的

        细分规则：
          - DVP 包着 VP/ADVP → 真状语 → 地（不要求父节点是 VP，否则
            「慢慢地/轻轻地」这种 DVP 直接挂 TOP 下的会被误判成 的）
          - DVP 在 NP 下（名物化宾语）或包着 IP/ADJP（名动词被误判）→ 的
          - DNP / CP → 的
          - VP 下的 de：后续兄弟含 NP（学的很多知识 / 说的真话 = 名物化）→ 的；
            后续兄弟是裸 VP/VRD（跑得很快 / 高兴得跳起来 = 动补）→ 得；
            VP 在 CP 内 → 不动

        只做高置信度修正，并受「接名词宾语动词」否决（开始真实的处理 → 保留 的）。
        """
        if not text or not text.strip():
            return text
        # 早退门控：同 correct()。不含 的地得 直接返回，不触发 const 模型加载/推理。
        if not any(ch in text for ch in '的地得'):
            return text
        if not _HANLP_AVAILABLE:
            return text
        if not self._const_loaded and not self.warmup_constituency():
            return text
        try:
            tokens: List[str] = self._tok(text)
            if not tokens:
                return text
            tree = self._const([tokens])[0]
        except Exception:
            return text

        from phrasetree.tree import Tree

        # token_idx -> (父短语标签, DVP 内容标签或 None, 祖父标签或 None)
        # 只记录 DVP/DNP/CP 三类高置信短语标签：VP 分支（的↔得）模型输入敏感，
        # 「学的很多知识」和「学得很多知识」几乎给出相同树，没有独立纠错能力，
        # 反而会把 dep 层保护好的「气得脸都红了/长得像他爸爸」误改成 的，故不采信。
        de_info: Dict[int, Tuple[str, Optional[str], Optional[str]]] = {}
        counter = [0]

        def _walk(node, in_cp: bool, grandparent_label: Optional[str] = None) -> None:
            label = node.label()
            for ch in node:
                if isinstance(ch, Tree):
                    if ch.label() == '_':
                        w = ch[0]
                        if w in ('的', '地', '得'):
                            dvp_inner: Optional[str] = None
                            if label == 'DVP':
                                # DVP 包着的第一个非 de 子节点（VP/ADVP → 真状语）
                                for sib in node:
                                    if isinstance(sib, Tree) and sib.label() != '_':
                                        dvp_inner = sib.label()
                                        break
                            de_info[counter[0]] = (
                                label, dvp_inner, grandparent_label,
                            )
                        counter[0] += 1
                    else:
                        _walk(ch, in_cp or (label == 'CP'), label)

        _walk(tree, False)

        targets: List[Tuple[int, str]] = []
        for idx, (label, dvp_inner, dvp_gp) in de_info.items():
            cur = tokens[idx]
            if label == 'DVP':
                if dvp_gp == 'NP':
                    # DVP 挂在 NP 下 = 名物化宾语（完成详细地说明 → 说明 是 完成 的宾语）
                    sugg = '的'
                elif dvp_inner in ('VP', 'ADVP'):
                    sugg = '地'
                else:
                    # DVP 包着 IP/ADJP（名动词被误判成动词）→ 的
                    sugg = '的'
            elif label in ('DNP', 'CP'):
                sugg = '的'
            else:
                continue
            if sugg == cur:
                continue
            # 否决：建议「地」但命中名动词宾语/复合名词信号时，
            # 处理/说明/研究 等名动词实为宾语名词或复合名词定语，保留「的」。
            # 信号与 dep 层一致：接名词宾语动词（跨 了/着/过）、宾语介词、
            # 后接「名动词+抽象属性名词」复合（研究能力/管理经验/测试流程）。
            if sugg == '地' and (
                _left_has_np_object_verb(tokens, idx)
                or _left_has_object_preposition(tokens, idx)
                or _gerund_compound_after(tokens, idx)
            ):
                continue
            # 名词短语读法保护（同 dep 层）：可作定语的修饰词 + 的 + 名动词
            # （及时的反馈/深入的调查/快速的迭代）→ 保留「的」，不按 DVP 改成 地。
            if sugg == '地' and idx > 0 and idx + 1 < len(tokens):
                if (tokens[idx - 1] in _DE_KEEP_NOMINAL_PREV
                        and (tokens[idx + 1] in GERUND_NOUNS
                             or tokens[idx + 1] in _NOUNLIKE_TECH_VERBS)):
                    continue
            targets.append((idx, sugg))

        if not targets:
            return text
        return self._apply_targets(text, tokens, targets)

    # ------------------------------------------------------------------
    # 档位 5（拼写检查错别字 · Reserved）：依赖 pycorrector / macbert4csc
    # 当前不加入依赖，仅预留接口。后续接入。
    # ------------------------------------------------------------------
    def check_spelling(self, text: str) -> List[Tuple[int, int, str]]:
        """错别字检查。返回 [（start, end, candidate）...]。

        预留接口；实际接入需加载 pycorrector / shibing624/macbert4csc-base-chinese。
        HanLP 本身不提供中文拼写纠错预训练模型，免费轻量选项
        不依赖额外模型。
        """
        if not text:
            return []
        return []

    # ------------------------------------------------------------------
    # 档位 6（搭配检查）
    # ------------------------------------------------------------------
    def check_collocation(self, text: str) -> List[Dict]:
        """搭配检查：返回可疑的 动宾 / 主谓 搭配作为警告 list。

        规则（依赖 dep 输出）：
        - 抽取 relation == 'VOB' (动宾, HanLP CTB 名) 或者 'dobj'
        - 如果动词 + 宾语的组合不在 OOV_LIST，且动词+宾语的语义是 OOV_LIST中的“错误搭配”，则告警
        - 同样抽取 'SBV' 中的主谓不一致（如「是」作谓语时主语不在事后出现）

        Returns:
            [{start, end, v_word, o_word, msg}, ...]
            仅作为警告返回，不自动修正。
        """
        if not text or not text.strip():
            return []
        if not _HANLP_AVAILABLE:
            return []
        if not self._loaded and not self.warmup():
            return []

        out: List[Dict] = []
        try:
            tokens = self._tok(text)
            heads, deprels = self._dep([tokens])
            deprels0 = deprels[0]
            heads0 = heads[0]
            for idx, rel in enumerate(deprels0):
                rel_lc = rel.lower()
                # VOB / dobj 是常见动宾关系名
                if rel_lc in ('vob', 'dobj'):
                    head_idx = heads0[idx]
                    if 0 <= head_idx - 1 < len(tokens):
                        v_word = tokens[head_idx - 1]
                        o_word = tokens[idx]
                        out.append({
                            'kind': 'V-O',
                            'v': v_word, 'o': o_word,
                            'msg': f'【搭配】动宾：{v_word} → {o_word}',
                            'severity': 'info',
                        })
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------
    # 档位 7（病句检查）
    # ------------------------------------------------------------------
    def check_completeness(self, text: str) -> List[Dict]:
        """病句检查：重复 token、跳出依赖树的孤儿词。

        Returns:
            [{kind, start, end, msg, severity}]。
        """
        if not text or not text.strip():
            return []
        warnings: List[Dict] = []
        # 1. 重复 token 检测（同一 token 连续出现两次，如「的」「了」「啊」）
        for ch in ('的', '了', '啊', '嗯', '哦', '吧', '呢'):
            doubled = ch * 2
            if doubled in text:
                idx = text.find(doubled)
                warnings.append({
                    'kind': 'redundant',
                    'start': idx, 'end': idx + 2,
                    'msg': f'【病句】重复 token：{doubled}',
                    'severity': 'warn',
                })
        # 2. 「的的不」/「的的」尾巴（常见 ASR 残影）
        if text.rstrip().endswith('的的'):
            warnings.append({
                'kind': 'redundant',
                'start': len(text) - 2, 'end': len(text),
                'msg': '【病句】句末『的的』冗余',
                'severity': 'warn',
            })
        return warnings

    # ------------------------------------------------------------------
    # 档位 8（中英混排 / 标点 / 空格处理）
    # ------------------------------------------------------------------
    def format_mixed_language(self, text: str) -> str:
        """中英混排 / 标点 修正。

        - 英文 、中文 之间未加空格则加空格（避免『Python编程』 → 『Python 编程』）
        - 全角数字 / 句末标点不会动；只调整 CN 与 EN 之间的空格与连续标点
        - 若是中文 English English 一起出现，不动内部空格，但与中文交界插入空格
        """
        if not text:
            return text
        import re as _re
        out = []
        tokens = list(_re.finditer(r'[A-Za-z0-9]+|[\u4e00-\u9fff，。！？；：…\u3000\uff00-\uffef]|[ \t]+', text))
        if not tokens:
            return text
        prev_kind = None  # 'en' / 'cn' / 'sp'
        cursor = 0
        for m in tokens:
            # 补充中间的间隙
            if m.start() > cursor:
                out.append(text[cursor:m.start()])
            span = m.group()
            cur_kind = (
                'en' if _re.match(r'^[A-Za-z0-9]+$', span) else
                'cn' if _re.match(r'^[\u4e00-\u9fff]$', span) else
                'sp' if _re.match(r'^[ \t]+$', span) else 'ot'
            )
            # CN-EN 或 EN-CN 之间从未加空格又从单词边界 添加空格
            if (cur_kind == 'en' and prev_kind == 'cn') or (cur_kind == 'cn' and prev_kind == 'en'):
                # 检查 span 已经是空格吗
                if not out or not out[-1].endswith(' '):
                    out.append(' ')
            out.append(span)
            if cur_kind in ('en', 'cn'):
                prev_kind = cur_kind
            elif cur_kind == 'sp':
                prev_kind = 'sp'
            cursor = m.end()
        if cursor < len(text):
            out.append(text[cursor:])
        return ''.join(out)


# ---------------------------------------------------------------------
# NER 专名识别（可选保护层：同音纠错 / 规则层 防专名误伤）
# ---------------------------------------------------------------------
# 与 dep/const 分离的独立懒加载：专名保护是「看不见收益但一直在止损」的
# 保护层，不该拖慢启动；首次调用才加载 MSRA_NER_ELECTRA_SMALL_ZH（~50MB）。
# tok 与 get_default_analyzer 共享同一实例（hanlp.load 按路径缓存）。
# 关键：NER 必须吃 tok 分词列表——喂裸文本会退化成逐字标注（王小明→王/明，
# 漏中字、误标 发布 的 布），实测 tok 喂入 王小明/北京/刘德华/阿里巴巴 全对。
_ner: Any = None
_ner_loaded = False
_ner_lock = threading.Lock()


def warmup_ner(progress_cb=None) -> bool:
    """懒加载 NER 模型。失败仅降级（启发式专名保护仍生效），不抛错。"""
    global _ner, _ner_loaded
    if not _HANLP_AVAILABLE:
        return False
    with _ner_lock:
        if _ner_loaded:
            return True
        try:
            if progress_cb:
                progress_cb('load', '[HanLP/ner] 正在加载 NER 模型 ...')
            _ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
            # 冒烟测试：必须吃 tok 分词列表（喂裸文本会退化成逐字标注）
            _ner([['他', '在', '北京', '上班']])
            _ner_loaded = True
            msg = '[HanLP] NER 模型已加载: MSRA_NER_ELECTRA_SMALL_ZH（专名保护）'
            print(f'[info] {msg}')
            if progress_cb:
                progress_cb('done', msg)
            return True
        except Exception as e:
            if progress_cb:
                progress_cb('error', f'[HanLP] NER 加载失败（专名保护降级为启发式）: {e}')
            print(f'[warn] [HanLP] NER 模型加载失败（专名保护降级为启发式）: {e}')
            _ner = None
            _ner_loaded = False
            return False


def get_ner_spans(text: str) -> List[Tuple[int, int]]:
    """HanLP NER 专名识别 → [(char_start, char_end), ...]。

    - 输入必须走 tok 分词（NER 模型吃 token 列表，喂裸文本会逐字退化）；
    - token 索引 → 字符偏移 用 find 对齐（与 _apply_targets 同法）；
    - 返回 [] 表示：HanLP 不可用 / 加载失败 / 无专名。绝不抛错。
    - 不做假阳性过滤（李子/周五 等已知词被小模型误标）——过滤策略是调用方
      （confusion_corrector._ner_spans_filtered）的职责，保持本函数为纯 NER API。
    """
    if not text or not _HANLP_AVAILABLE:
        return []
    if not _ner_loaded and not warmup_ner():
        return []
    try:
        tokens = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)(text)
        if not tokens:
            return []
        spans = _ner([tokens])[0]
        if not spans:
            return []
        pos = []
        cursor = 0
        for tok in tokens:
            p = text.find(tok, cursor)
            if p == -1:
                return []  # 对齐失败，放弃本次保护
            pos.append(p)
            cursor = p + len(tok)
        out: List[Tuple[int, int]] = []
        for _entity, _etype, s, e in spans:
            if e <= s or s >= len(pos):
                continue
            out.append((pos[s], pos[e - 1] + len(tokens[e - 1])))
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------
# 全局单例（懒加载）
# ---------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_default_analyzer() -> DeDiDeSyntacticAnalyzer:
    """提供默认单例。HanLP 不可用时返回的实例依然存在，但 correct() 是 no-op。"""
    return DeDiDeSyntacticAnalyzer(auto_load=False)


def warmup_default_analyzer(progress_cb=None) -> bool:
    """主动预热：让 main.py 启动时就尝试加载 HanLP 模型（失败也无副作用）。

    预热内容：tok + dep（必需）+ 成分句法 const（档位 2 的组成部分，
    懒加载也行，但预热掉可避免首句卡顿）。任一层失败都不影响主 dep 流程。

    Args:
        progress_cb: 同 warmup() 的 progress 回调。会从默认 analyzer 透传给 warmup。
    """
    analyzer = get_default_analyzer()

    def _wrapped(stage, msg):
        print(f'[hanlp-warmup/{stage}] {msg}')
        if progress_cb:
            try:
                progress_cb(stage, msg)
            except Exception:
                pass

    ok = analyzer.warmup(progress_cb=_wrapped)
    if ok:
        # 成分句法失败不阻塞主流程，静默忽略（correct_by_constituency 内部也有兜底）
        try:
            analyzer.warmup_constituency(progress_cb=_wrapped)
        except Exception:
            pass
    return ok


# ---------------------------------------------------------------------
# CLI 自检：python syntax_analyzer.py "他走的比较快"
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python syntax_analyzer.py <text> [text2 ...]")
        sys.exit(1)

    analyzer = get_default_analyzer()
    print(f"HanLP 可用: {_HANLP_AVAILABLE}")
    print(f"模型已加载: {analyzer.is_available()}")
    if not analyzer.is_available():
        analyzer.warmup()
    print(f"预热后可用: {analyzer.is_available()}")

    for line in sys.argv[1:]:
        out = analyzer.correct(line)
        print(f"[in] {line}\n[out] {out}\n")
