# -*- coding: utf-8 -*-
"""双用校对引擎（语音输入 + 专业校对共用）。

设计目标（详见 docs/proofreading_plan.md）：
- 引擎层是纯函数：文本进 → ProofResult（修正后文本 + 建议列表）出。
- 每条建议携带 位置/原文/建议/原因/来源/置信度 —— 语音输入静默应用建议，
  专业校对逐条审阅、生成报告；其他项目 import 本模块即可复用全部规则层。

与语音流水线（post_processing.process_text_mode）的关系：
- 阶段顺序与 correct_common_errors 完全一致（本模块是同一流水线的可观测版本）；
- 语音应用继续走 process_text_mode（静默），本模块供 校对/其他项目 使用；
- test_proofreading.py 的 parity 测试保证两者输出逐字节一致，防止漂移。

用法：
    from proofreading import proofread
    r = proofread('新华社报导了这件事，水份和成份都要核对')
    r.text          # 修正后文本（语音输入用它）
    r.suggestions   # 建议列表（校对用它）

CLI：
    python proofreading.py 文本 [mode]
"""

import difflib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from . import post_processing as pp
    from .config import config as config_instance
except ImportError:  # 独立使用（其他项目直接 import）
    import post_processing as pp
    import config
    config_instance = config.config

from dataclasses import dataclass, field

# 延迟实例化（normalize_chinese_numbers 是 TextPostProcessor 的方法）
_PROCESSOR = None


def _get_processor():
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = pp.TextPostProcessor()
    return _PROCESSOR


@dataclass
class Suggestion:
    """一条校对建议。start/end 相对『原始输入文本』（经各阶段位置映射回算）。"""

    start: int
    end: int
    original: str
    replacement: str
    reason: str
    source: str
    confidence: float


@dataclass
class ProofResult:
    text: str
    suggestions: list = field(default_factory=list)


def _timing_enabled() -> bool:
    """耗时打点开关（debug.timing_enabled，默认关）。关时不产生任何额外输出。"""
    try:
        return bool(config_instance.get('debug.timing_enabled', False))
    except Exception:
        return False


def _diff_stage(text, old_map, stage_fn, reason, source, confidence):
    """跑一个规则阶段，返回 (新文本, 新位置映射, 该阶段建议)。

    old_map[k] = 原始输入文本中与 text[k] 对应的索引。
    """
    _t0 = time.perf_counter()
    new_text = stage_fn(text)
    if _timing_enabled():
        _dt = (time.perf_counter() - _t0) * 1000
        print(f"[timing] 阶段·{source}/{reason}: {_dt:.1f} ms")
    if new_text == text:
        return text, old_map, []
    suggs = []
    sm = difflib.SequenceMatcher(None, text, new_text)
    new_map = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(j1, j2):
                new_map.append(old_map[i1 + (k - j1)])
        elif tag == 'replace':
            orig_idx = old_map[i1] if i1 < len(old_map) else (old_map[-1] + 1 if old_map else 0)
            suggs.append(Suggestion(orig_idx, orig_idx + (i2 - i1),
                                    text[i1:i2], new_text[j1:j2],
                                    reason, source, confidence))
            for k in range(j1, j2):
                new_map.append(orig_idx)
        elif tag == 'insert':
            orig_idx = old_map[i1] if i1 < len(old_map) else (old_map[-1] + 1 if old_map else 0)
            suggs.append(Suggestion(orig_idx, orig_idx,
                                    '', new_text[j1:j2],
                                    reason, source, confidence))
            for k in range(j1, j2):
                new_map.append(orig_idx)
        elif tag == 'delete':
            orig_idx = old_map[i1] if i1 < len(old_map) else (old_map[-1] + 1 if old_map else 0)
            suggs.append(Suggestion(orig_idx, orig_idx + (i2 - i1),
                                    text[i1:i2], '',
                                    reason, source, confidence))
    return new_text, new_map, suggs


def proofread(text: str, mode: str = 'none', llm_caller=None,
              config=None) -> ProofResult:
    """双用校对引擎入口。

    mode: 'none' 仅规则层（无 LLM）；'basic'/'custom' 追加 LLM 润色（需 llm_caller）。
    llm_caller: callable(原始文本, 提示词) -> 修正文本；缺省 None 时 LLM 阶段跳过。
    config: 配置对象；缺省用全局 config_instance（保持与语音应用一致）。
    """
    if not text:
        return ProofResult(text, [])
    conf = config or config_instance
    suggs: list = []
    old_map = list(range(len(text)))

    # --- 阶段 0：数字规范化（跟随 number_conversion 开关）---
    def _num_conv(t):
        try:
            if conf.get('post_processing.number_conversion', True):
                return _get_processor().normalize_chinese_numbers(t)
        except Exception:
            pass
        return t

    text, old_map, s = _diff_stage(text, old_map, _num_conv,
                                   '中文数字 → 阿拉伯数字', '数字规范', 0.95)
    suggs += s

    # --- 阶段 0.5：繁 → 简 ---
    text, old_map, s = _diff_stage(text, old_map, pp.traditional_to_simplified,
                                   '繁体 → 简体', '繁简转换', 1.0)
    suggs += s

    # --- 阶段 1：的/地/得（syntax 增强随配置）---
    use_syntax = False
    try:
        use_syntax = bool(conf.get('post_processing.syntax_analyzer.enabled', False))
    except Exception:
        pass

    def _dedede(t):
        if use_syntax:
            try:
                from syntax_analyzer import get_default_analyzer  # type: ignore
                return pp.correct_de_di_de_with_syntax(t, get_default_analyzer())
            except Exception:
                pass
        return pp.correct_de_di_de(t)

    text, old_map, s = _diff_stage(text, old_map, _dedede,
                                   '的/地/得 语法修正', '语法', 0.9)
    suggs += s

    # --- 阶段 2：术语 fuzzy 还原 ---
    def _fuzzy(t):
        try:
            return pp.apply_it_term_fuzzy_replace(t, pp._DEFAULT_TERMS_PATH)
        except Exception:
            return t

    text, old_map, s = _diff_stage(text, old_map, _fuzzy,
                                   '术语还原', '术语', 0.95)
    suggs += s

    # --- 阶段 3：错别字/异形词固定表（带子串守卫） ---
    text, old_map, s = _diff_stage(text, old_map, pp._apply_typo_fixes,
                                   '错别字/异形词修正', '错别字', 0.95)
    suggs += s

    # --- 阶段 4：同音字纠错（CSC）---
    text, old_map, s = _diff_stage(text, old_map, pp.correct_with_csc,
                                   '同音字纠错', '同音字', 0.85)
    suggs += s

    # --- 阶段 4.5：系统性 ASR 声母混淆修正（把X切入→接入）---
    text, old_map, s = _diff_stage(text, old_map, pp.correct_ba_qieru_confusion,
                                   'ASR 声母混淆修正（把X切入→接入）', '语境混淆', 0.95)
    suggs += s

    # --- 阶段 5：异形词规范层（跟随 yixing_words 开关）---
    def _yixing(t):
        try:
            if conf.get('post_processing.yixing_words', True):
                return pp.apply_yixing_standardization(t)
        except Exception:
            pass
        return t

    text, old_map, s = _diff_stage(text, old_map, _yixing,
                                   '异形词规范（GF 1001-2001）', '异形词', 0.9)
    suggs += s

    # --- 阶段 6：用户自定义替换（复用子串守卫，防历史规则误伤）---
    # 用户表与 _TYPO_FIXES 有重复条目（安祥/渡假/凑和/做月子…），早期是纯 replace，
    # 会把 平安祥和→平安详和 等合法短语改坏；守卫命中处保留原文。
    def _user_repl(t):
        try:
            for item in conf.get_replacements():
                o, n = item.get('old'), item.get('new')
                if o and n:
                    t = pp._guarded_replace(t, o, n)
        except Exception:
            pass
        return t

    text, old_map, s = _diff_stage(text, old_map, _user_repl,
                                   '用户自定义替换', '用户替换', 1.0)
    suggs += s

    # --- 阶段 7：自学习规则（B 档确认制）---
    def _learned(t):
        try:
            return pp.apply_learned_rules(t)
        except Exception:
            return t

    text, old_map, s = _diff_stage(text, old_map, _learned,
                                   '自学习规则', '自学习', 0.9)
    suggs += s

    # --- 阶段 8：标点归一 / 引号配对 ---
    text, old_map, s = _diff_stage(text, old_map, pp.normalize_punctuation,
                                   '标点归一 / 引号配对', '标点', 0.95)
    suggs += s

    # --- 阶段 9：LLM 润色（basic / custom，需 llm_caller）---
    if mode == 'basic':
        prompt = conf.get_basic_prompt()
        try:
            if pp._is_mixed_language(text) and conf.get_mixed_lang_fix():
                prompt = pp.MIXED_LANG_FIX_INSTRUCTION + prompt
        except Exception:
            pass

        def _llm(t):
            if llm_caller is None:
                return t
            try:
                return llm_caller(t, prompt)
            except Exception:
                return t

        text, old_map, s = _diff_stage(text, old_map, _llm,
                                       'LLM 润色', 'LLM', 0.7)
        suggs += s
        text, old_map, s = _diff_stage(text, old_map, pp.normalize_punctuation,
                                       'LLM 后标点归一', '标点', 0.95)
        suggs += s
    elif mode != 'none':
        # custom 模式：查找自定义提示词
        found = False
        try:
            for item in conf.get_custom_prompts():
                if item.get('name') == mode:
                    found = True

                    def _llm_custom(t):
                        if llm_caller is None:
                            return t
                        try:
                            return llm_caller(t, item.get('prompt', ''))
                        except Exception:
                            return t

                    text, old_map, s = _diff_stage(text, old_map, _llm_custom,
                                                   '自定义提示词润色', 'LLM', 0.7)
                    suggs += s
                    text, old_map, s = _diff_stage(text, old_map, pp.normalize_punctuation,
                                                   'LLM 后标点归一', '标点', 0.95)
                    suggs += s
                    break
        except Exception:
            pass
        if not found:
            print(f"未找到处理模式: {mode}，将不进行处理")

    return ProofResult(text, suggs)


def _cli():
    if len(sys.argv) < 2:
        print('用法: python proofreading.py 文本 [mode]')
        sys.exit(1)
    text = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'none'
    r = proofread(text, mode=mode)
    print('原文    :', text)
    print('修正后  :', r.text)
    print(f'建议({len(r.suggestions)} 条):')
    for i, s in enumerate(r.suggestions, 1):
        print(f'  {i}. [{s.source}] {s.reason}  置信度 {s.confidence}')
        print(f'     「{s.original}」→「{s.replacement}」 @ [{s.start}:{s.end}]')


if __name__ == '__main__':
    _cli()
