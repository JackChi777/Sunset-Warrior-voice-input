# -*- coding: utf-8 -*-
"""字符级 bigram 语言模型：给同音/形近字纠错器（2.5 级）提供「相邻共现」信号。

为什么需要它：
    jieba 词典本质是 unigram LM——只有词/单字的频率，没有「字与字的共现」。
    纯 unigram 打分会把 树根 里的 根 误改成高频介词 跟（跟 单字频 >> 根），
    把 再教 里的 再 毁成 在。字符级 bigram 捕捉「树+根」「再+教」「飞+掉」
    这类相邻共现，正是 unigram 缺的那层信号。

训练数据：
    - 小模型：CGED 2017+2018 CORRECTION（干净文本，仅供实验——脚本会警告）
    - 正式模型：中文维基快照 zhwiki-latest-pages-articles.xml.bz2（3.4GB，
      流式解析不落盘中间文本，繁转简后计数）。训练：cged/_wiki_train.py。

实验定论（CGED 2020/2021，全变体均已验证）：
  通用字符 n-gram（bigram/trigram）作为纠错器的全局 tie-breaker 是结构性负结果：
    - bigram 主动版：CGED 2020/2021 上 0/10 决策全错；保护率 96.5→96.4%、94.4→94.3%
    - trigram（±2 窗口，1 亿字）：命中 +2，但保护 96.5→96.3%、94.4→94.1%，仍净亏
    - 双向一致 + 高阈值：伤害减少但收益归零
    - 否决版（只拦不改）：零变化（jieba 词级评分已覆盖）
  根因：① 纠错器已充分利用 jieba 词典（本身就是 unigram LM），n-gram 只能补充
    jieba 沉默的模糊位点——那恰是连…都/自由地飞 这类 ±2 窗口区分不了的长程或
    低频结构；② 频率先验（国家联合≫国家连）会系统性带偏统计模型。
  模型工件：cged/char_bigram_wiki_400m.npz（bigram 4 亿字 7MB）、
  cged/char_tri_100m.npz（trigram 1 亿字 25MB），均未随产品发布。
  要走通 n-gram 思路需像 LanguageTool 那样逐混淆对建规则 + 3~5 字窗口 + 大语料。

打分（纠错器 _should_replace 的 2.5 级调用）：
    delta(cand) = [logP(cand|prev) + logP(next|cand)]
                - [logP(orig|prev) + logP(next|orig)]
    正 = 候选更符合语料的字序共现 → 替换；负 = 保留原字；|delta| 低于阈值
    或证据不足（训练语料没见过这种对比）时不裁决。

平滑：Jelinek-Mercer 插值  P(c|prev) = λ1·count(prev,c)/count(prev)
                                     + λ2·count(c)/N + λ3·1/V
存储：numpy 紧凑格式（bigram 键 = a + b*BASE 的 uint32 排序数组），
    几百万对时模型文件 ~10-40MB，加载后逐对查询用 searchsorted。
"""

import math
import os
import re
from typing import Dict, Iterator, List, Optional

# 仅统计 CJK + 常用中文标点 + 数字字母（标点也是有用的上下文：得，/ 的。）
_KEEP_RE = re.compile(r'[\u4e00-\u9fff0-9a-zA-Z，。？！；：、…“”‘’（）《》·—-]')

# 插值权重：bigram 为主，unigram 兜底，均匀兜底极小权重
_L1 = 0.85
_L2 = 0.12
_L3 = 0.03
_MIN_COUNT = 2  # 训练时丢弃出现 <2 次的 bigram 对

# bigram 键编码：a + b*BASE（a=前字码点, b=后字码点）。_KEEP_RE 允许的最大
# 码点 < 0x10000（CJK 统一表意 + 常用标点），BASE 取 50000 保证乘积 < 2^32。
_BASE = 50000


def _clean(text: str) -> str:
    """保留 CJK/标点/数字字母，其余（XML 标签、空白）剔除。"""
    return ''.join(_KEEP_RE.findall(text))


def _strip_tags(text: str) -> str:
    """CGED CORRECTION 里可能残留 <...> 标注（<ERROR .../> 等），剥掉。"""
    return re.sub(r'<[^>]*>', '', text)


def extract_corrections(xml_path: str) -> List[str]:
    """从 CGED XML 提取 CORRECTION（干净文本），逐句返回。"""
    raw = open(xml_path, encoding='utf-8', errors='replace').read()
    corrs = re.findall(r'<CORRECTION[^>]*>(.*?)</CORRECTION>', raw, re.S)
    out = []
    for c in corrs:
        c = _clean(_strip_tags(c))
        if len(c) >= 2:
            out.append(c)
    return out


def _sentences(text: str) -> Iterator[str]:
    """按句末标点/换行切句（只统计句内 bigram，避免跨句噪声）。"""
    for seg in re.split(r'[。！？；\n]', text):
        if len(seg) >= 2:
            yield seg


class CharBigramLM:
    """字符级 bigram/trigram LM（count>=2 截断 + Jelinek-Mercer 插值 + numpy 紧凑存储）。

    trigram 可选（trigram_keys/trigram_counts 为 None 时退化为纯 bigram）：
    P(c | a,b) = α·count(a,b,c)/count(a,b) + β·count(b,c)/count(b) + γ·count(c)/N + δ·1/V
    """

    _T1 = 0.80   # trigram 项权重（有 trigram 时）
    _T2 = 0.15   # bigram 回退
    _T3 = 0.04   # unigram
    _T4 = 0.01   # 均匀

    def __init__(self, vocab_list: List[str], unigram_counts, bigram_keys, bigram_counts,
                 trigram_keys=None, trigram_counts=None):
        self.vocab_list = vocab_list
        self.char2idx: Dict[str, int] = {c: i for i, c in enumerate(vocab_list)}
        self.vocab = len(vocab_list)
        self.unigram = unigram_counts          # np.ndarray uint64 (V,)
        self.bigram_keys = bigram_keys         # np.ndarray uint32 升序 (K,)
        self.bigram_counts = bigram_counts     # np.ndarray uint32 (K,)
        self.trigram_keys = trigram_keys       # np.ndarray uint64 升序 (K3,) 或 None
        self.trigram_counts = trigram_counts   # np.ndarray uint32 (K3,) 或 None
        self.total = int(unigram_counts.sum())
        self._log_v = math.log(self.vocab)
        # unigram 平滑对数概率：log((count+1)/(total+vocab))
        self._log_uni = {
            c: math.log(int(unigram_counts[i]) + 1) - math.log(self.total + self.vocab)
            for i, c in enumerate(vocab_list)
        }
        # 每个前字的 count(prev)（用于 bigram 项分母）
        self._log_prev = {
            c: math.log(max(int(unigram_counts[i]), 1))
            for i, c in enumerate(vocab_list)
        }
        # 常量：均匀兜底项对数概率
        self._flat = -self._log_v

    # ------------------------------------------------------------------
    def _pair_count(self, a: str, b: str) -> int:
        ia = self.char2idx.get(a)
        ib = self.char2idx.get(b)
        if ia is None or ib is None:
            return 0
        return self._lookup_bigram(ia, ib)

    # ------------------------------------------------------------------
    def _lookup_bigram(self, ia: int, ib: int) -> int:
        key = ia * self.vocab + ib
        lo, hi = 0, len(self.bigram_keys) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            k = int(self.bigram_keys[mid])
            if k == key:
                return int(self.bigram_counts[mid])
            if k < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return 0

    def _lookup_trigram(self, ia: int, ib: int, ic: int) -> int:
        if self.trigram_keys is None:
            return 0
        V = self.vocab
        key = ia * V * V + ib * V + ic
        lo, hi = 0, len(self.trigram_keys) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            k = int(self.trigram_keys[mid])
            if k == key:
                return int(self.trigram_counts[mid])
            if k < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return 0

    def logp_next2(self, prev2: Optional[str], prev1: Optional[str], ch: str) -> float:
        """log P(ch | prev2, prev1)，trigram 插值 + bigram/unigram 回退。

        prev1/prev2 缺失（句首）时逐级回退。
        """
        lu = self._log_uni.get(ch)
        if lu is None:
            return self._flat
        if self.trigram_keys is None or prev1 is None or prev2 is None:
            return self.logp_next(prev1, ch)
        ib = self.char2idx.get(prev1)
        ia = self.char2idx.get(prev2)
        ic = self.char2idx.get(ch)
        if ib is None or ia is None or ic is None:
            return self.logp_next(prev1, ch)
        # count(a,b) 与 count(a,b,c)
        cab = self._lookup_bigram(ia, ib)
        t3 = self._lookup_trigram(ia, ib, ic)
        if cab == 0 or t3 == 0:
            return self.logp_next(prev1, ch)
        tri = math.log(t3) - math.log(cab)
        bi = self.logp_next(prev1, ch)
        m = max(tri, bi, lu, self._flat)
        return m + math.log(self._T1 * math.exp(tri - m)
                            + self._T2 * math.exp(bi - m)
                            + self._T3 * math.exp(lu - m)
                            + self._T4 * math.exp(self._flat - m))

    def delta3(self, text: str, i: int, orig: str, cand: str,
               min_evidence: int = 3) -> Optional[float]:
        """trigram 版裁决：±2 上下文。

        delta = [logP(cand|p2,p1) - logP(orig|p2,p1)]
              + [logP(n1|cand,p1) - logP(n1|orig,p1)]
              + [logP(n2|n1,cand) - logP(n2|n1,orig)]
        证据：任一方向的 (a,b,c) 计数差 ≥ min_evidence 才裁决。
        """
        n = len(text)
        p2 = text[i - 2] if i > 1 else None
        p1 = text[i - 1] if i > 0 else None
        n1 = text[i + 1] if i + 1 < n else None
        n2 = text[i + 2] if i + 2 < n else None
        ev = 0
        if p1 is not None:
            ev += abs(self._pair_count(p1, cand) - self._pair_count(p1, orig))
        if n1 is not None:
            ev += abs(self._pair_count(cand, n1) - self._pair_count(orig, n1))
        if ev < min_evidence:
            return None
        d = 0.0
        d += self.logp_next2(p2, p1, cand) - self.logp_next2(p2, p1, orig)
        if n1 is not None:
            d += self.logp_next2(p1, cand, n1) - self.logp_next2(p1, orig, n1)
        if n2 is not None:
            d += self.logp_next2(cand, n1, n2) - self.logp_next2(orig, n1, n2)
        return d

    # ------------------------------------------------------------------
    def logp_next(self, prev: Optional[str], ch: str) -> float:
        """log P(ch | prev)，prev 为 None 时退化为 unigram。"""
        lu = self._log_uni.get(ch)
        if lu is None:
            return self._flat  # 生僻字：均匀兜底
        if prev is None:
            return lu
        ia = self.char2idx.get(prev)
        if ia is None:
            return lu
        ib = self.char2idx.get(ch)
        if ib is None:
            return lu
        key = ia * self.vocab + ib
        # 二分查找 count(prev,ch)
        lo, hi = 0, len(self.bigram_keys) - 1
        cnt = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            k = int(self.bigram_keys[mid])
            if k == key:
                cnt = int(self.bigram_counts[mid])
                break
            if k < key:
                lo = mid + 1
            else:
                hi = mid - 1
        if cnt > 0:
            big_term = math.log(cnt) - self._log_prev[prev]
            m = max(big_term, lu, self._flat)
            return m + math.log(_L1 * math.exp(big_term - m)
                                + _L2 * math.exp(lu - m)
                                + _L3 * math.exp(self._flat - m))
        m = max(lu, self._flat)
        return m + math.log(_L2 * math.exp(lu - m) + _L3 * math.exp(self._flat - m))

    # ------------------------------------------------------------------
    def delta(self, text: str, i: int, orig: str, cand: str,
              min_evidence: int = 3) -> Optional[float]:
        """候选相对原字的对数概率差；证据不足时返回 None（不裁决）。

        只看 ±1 邻域：logP(cand|prev)+logP(next|cand) 减 logP(orig|prev)+logP(next|orig)。
        正 = 候选更合字序，负 = 原字更合。

        证据守卫：小语料里绝大多数 (prev,cand) 组合从未见过，插值会塌缩成
        unigram——而 unigram 对单字功能词（跟/在/非/常）有系统偏差。因此只有
        在任一方向存在 ≥min_evidence 的计数差（训练语料里真见过这种对比）时
        才给出裁决；否则返回 None 交给上层保持原样。
        """
        prev = text[i - 1] if i > 0 else None
        nxt = text[i + 1] if i + 1 < len(text) else None
        ev = 0
        if prev is not None:
            ev += abs(self._pair_count(prev, cand) - self._pair_count(prev, orig))
        if nxt is not None:
            ev += abs(self._pair_count(cand, nxt) - self._pair_count(orig, nxt))
        if ev < min_evidence:
            return None
        d = 0.0
        d += self.logp_next(prev, cand) - self.logp_next(prev, orig)
        if nxt is not None:
            d += self.logp_next(cand, nxt) - self.logp_next(orig, nxt)
        return d

    # ------------------------------------------------------------------
    @staticmethod
    def _build(vocab_list: List[str], unigram: Dict[str, int],
               bigram: Dict[int, int], trigram: Optional[Dict[int, int]] = None) -> 'CharBigramLM':
        """由训练计数构建模型（bigram 键 = ia*vocab+ib，trigram 键 = ia*V²+ib*V+ic）。"""
        char2idx = {c: i for i, c in enumerate(vocab_list)}
        import numpy as np
        uni = np.zeros(len(vocab_list), dtype=np.uint64)
        for c, cnt in unigram.items():
            uni[char2idx[c]] = cnt
        keys = []
        counts = []
        for k, cnt in bigram.items():
            if cnt >= _MIN_COUNT:
                keys.append(k)
                counts.append(cnt)
        order = sorted(range(len(keys)), key=lambda j: keys[j])
        keys = np.asarray([keys[j] for j in order], dtype=np.uint32)
        counts = np.asarray([counts[j] for j in order], dtype=np.uint32)
        tk = tc = None
        if trigram:
            tkeys = []
            tcounts = []
            for k, cnt in trigram.items():
                if cnt >= _MIN_COUNT:
                    tkeys.append(k)
                    tcounts.append(cnt)
            torder = sorted(range(len(tkeys)), key=lambda j: tkeys[j])
            tk = np.asarray([tkeys[j] for j in torder], dtype=np.uint64)
            tc = np.asarray([tcounts[j] for j in torder], dtype=np.uint32)
        return CharBigramLM(vocab_list, uni, keys, counts, tk, tc)

    # ------------------------------------------------------------------
    @staticmethod
    def train_pages(page_factory, out_path: str,
                    t2s=None, progress_every: int = 20_000,
                    max_pairs: int = 8_000_000,
                    max_chars: int = 400_000_000) -> 'CharBigramLM':
        """从页面文本流训练（两遍：收集 vocab → 计数）。页面需已清洗。

        page_factory: 无参可调用，每次调用返回一个全新的页面文本迭代器
        （wiki 训练会重开 bz2 文件流两遍）。
        max_chars: 达到该字数后停止（控制模型体积/内存/训练时长）；
        两遍按同一页面数停止，保证词表与计数一致。
        """
        # ── 第一遍：收集 vocab + 总字数（只保留出现 >= 2 次的字符）──
        unigram: Dict[str, int] = {}
        n_chars = 0
        n_pages = 0
        pages_to_use = None
        for page in page_factory():
            if t2s:
                page = t2s(page)
            n_pages += 1
            for seg in _sentences(_clean(page)):
                for ch in seg:
                    unigram[ch] = unigram.get(ch, 0) + 1
                    n_chars += 1
            if n_pages % progress_every == 0:
                print(f'  [pass1] pages={n_pages} chars={n_chars} vocab={len(unigram)}')
            if n_chars >= max_chars:
                pages_to_use = n_pages
                print(f'[pass1] 达到字数上限 {max_chars}（pages={n_pages}），第二遍只用前 {pages_to_use} 页')
                break
        if pages_to_use is None:
            pages_to_use = n_pages
        vocab_list = [c for c, cnt in unigram.items() if cnt >= 2]
        vocab_list.sort()
        char2idx = {c: i for i, c in enumerate(vocab_list)}
        V = len(vocab_list)
        print(f'[pass1] done: pages={pages_to_use} chars={n_chars} vocab={V}')

        # ── 第二遍：计数 bigram + trigram（只扫前 pages_to_use 页）──
        uni2: Dict[str, int] = {}
        bigram: Dict[int, int] = {}
        trigram: Dict[int, int] = {}
        n_chars2 = 0
        n_pages2 = 0
        for page in page_factory():
            if t2s:
                page = t2s(page)
            n_pages2 += 1
            if n_pages2 > pages_to_use:
                break
            for seg in _sentences(_clean(page)):
                prev2 = None
                prev1 = None
                for ch in seg:
                    ia = char2idx.get(ch)
                    if ia is None:
                        prev2 = prev1 = None
                        continue
                    uni2[ch] = uni2.get(ch, 0) + 1
                    if prev1 is not None:
                        key = prev1 * V + ia
                        bigram[key] = bigram.get(key, 0) + 1
                        if prev2 is not None:
                            key3 = prev2 * V * V + prev1 * V + ia
                            trigram[key3] = trigram.get(key3, 0) + 1
                    prev2 = prev1
                    prev1 = ia
                    n_chars2 += 1
            if n_pages2 % progress_every == 0:
                print(f'  [pass2] pages={n_pages2} pairs={len(bigram)} '
                      f'triples={len(trigram)} ({len(bigram)*8//1048576}MB 预估)')
                if len(bigram) > max_pairs:
                    print(f'[warn] bigram 对数超过 {max_pairs}，停止（语料过大需换更大内存）。')
                    break
        lm = CharBigramLM._build(vocab_list, uni2, bigram, trigram)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        saved = lm.save(out_path)
        n_pairs = len(lm.bigram_keys)
        n_tri = len(lm.trigram_keys) if lm.trigram_keys is not None else 0
        print(f'[train] 页面 {n_pages2}, 字 {n_chars2}, 词典 {V}, '
              f'bigram 对 {n_pairs}, trigram 对 {n_tri} -> {saved} '
              f'({os.path.getsize(saved)//1048576}MB)')
        return lm

    # ------------------------------------------------------------------
    @staticmethod
    def train(corpus_paths: List[str], out_path: str) -> 'CharBigramLM':
        """从本地文本/XML 文件训练（CGED 路径；wiki 请用 train_pages）。"""
        unigram: Dict[str, int] = {}
        bigram: Dict[str, Dict[str, int]] = {}
        n_sent = 0
        for path in corpus_paths:
            if path.endswith('.xml'):
                sents = extract_corrections(path)
            else:
                sents = [_clean(_strip_tags(ln)) for ln in
                         open(path, encoding='utf-8', errors='replace')]
            for s in sents:
                if len(s) < 2:
                    continue
                n_sent += 1
                prev = None
                for ch in s:
                    unigram[ch] = unigram.get(ch, 0) + 1
                    if prev is not None:
                        row = bigram.setdefault(prev, {})
                        row[ch] = row.get(ch, 0) + 1
                    prev = ch
        # 折叠成 CharBigramLM 需要的结构
        import numpy as np
        vocab_list = [c for c, cnt in unigram.items() if cnt >= 2]
        vocab_list.sort()
        char2idx = {c: i for i, c in enumerate(vocab_list)}
        uni = {c: unigram[c] for c in vocab_list}
        flat: Dict[int, int] = {}
        for a, row in bigram.items():
            ia = char2idx.get(a)
            if ia is None:
                continue
            for b, cnt in row.items():
                ib = char2idx.get(b)
                if ib is None:
                    continue
                flat[ia * len(vocab_list) + ib] = cnt
        lm = CharBigramLM._build(vocab_list, uni, flat)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        lm.save(out_path)
        print(f'[train] 句子 {n_sent}, 字 {lm.total}, 词典 {lm.vocab}, '
              f'bigram 对 {len(lm.bigram_keys)} -> {out_path}')
        return lm

    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """保存模型，返回实际写入的文件路径（自动补 .npz 后缀）。"""
        import numpy as np
        if not path.endswith('.npz'):
            path += '.npz'
        kw = dict(
            vocab=np.asarray(self.vocab_list, dtype=object),
            unigram=self.unigram,
            bigram_keys=self.bigram_keys,
            bigram_counts=self.bigram_counts,
        )
        if self.trigram_keys is not None:
            kw['trigram_keys'] = self.trigram_keys
            kw['trigram_counts'] = self.trigram_counts
        np.savez_compressed(path, **kw)
        return path

    @staticmethod
    def load(path: str) -> 'CharBigramLM':
        import numpy as np
        if not path.endswith('.npz'):
            path += '.npz'
        data = np.load(path, allow_pickle=True)
        vocab_list = [str(c) for c in data['vocab'].tolist()]
        return CharBigramLM(
            vocab_list, data['unigram'],
            data['bigram_keys'], data['bigram_counts'],
            data.get('trigram_keys'), data.get('trigram_counts'),
        )


def _get_t2s():
    """返回繁→简转换函数；OpenCC 缺席时用内置兜底表。失败返回 None。"""
    try:
        from post_processing import traditional_to_simplified
        return traditional_to_simplified
    except Exception:
        return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cged_dir = os.path.join(here, 'cged')
    out = os.path.join(here, 'user_data', 'char_bigram.npz')
    paths = [
        os.path.join(cged_dir, 'cged2017_train.xml'),
        os.path.join(cged_dir, 'cged2018_train.xml'),
    ]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print('[err] 未找到 CGED 语料，先运行 cged 目录下的下载/转换。')
        return
    lm = CharBigramLM.train(paths, out)
    if lm.total < 10_000_000:
        print('\n[重要警告] 训练语料仅 %d 字，远低于 N-gram 模型的实际需求（约 1 亿字级）。' % lm.total)
        print('    CGED 实验证明：小语料上 bigram 计数会被语料自身的方向性偏差主导'
              '（如「，在」>「，再」、「你已经」>「你以」），产生系统性误改。')
        print('    此模型仅供实验，不要随产品发布。建议用 100MB+ 干净语料'
              '（中文维基 / Leipzig zho / 你自己的转录文本）重训，'
              '或者直接删除 user_data/char_bigram.npz 让 2.5 级评分保持关闭。')
        print('    正式训练：python cged/_wiki_train.py <zhwiki bz2 路径>')


if __name__ == '__main__':
    main()
