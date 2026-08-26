# -*- coding: utf-8 -*-
"""同音/形近字混淆集 + jieba 三级评分纠错器。

替代被 INT8 动态量化搞残了的 MacBERT4CSC ONNX 模型。

三级评分策略：
  1. 局部 bigram/trigram 频率比
  2. 全文本 jieba 分词总频分
  3. 保持原样（无法可靠决策时）

可选 2.5 级：字符级 bigram/trigram 语言模型（char_bigram.py，证据守卫 + 懒加载）。
  实验结论（CGED 2020/2021 实证，全变体已验证，定论为结构性负结果）：
  ① jieba 词典本质是 unigram LM，Viterbi 全句打分能提升修正率但会被单字功能
    词（跟/在/非）带偏、损害保护率——净负面；
  ② 字符级 bigram（52 万字→4 亿字）与 trigram（1 亿字）作为 1、2 级无结论时的
    全局 tie-breaker 均净亏：命中 +1~2，保护 -1~6。失败机理：n-gram 只补充
    jieba 沉默的模糊位点，那恰是 连…都/自由地飞 等 ±2 窗口区分不了的长程/
    低频结构；频率先验（国家联合≫国家连）系统性带偏。否决版零变化。
  因此：模型文件不随产品发布，2.5 级默认静默关闭；如需实验（如自学习语料），
  用 cged/_wiki_train.py 重训并放回 user_data/char_bigram.npz 才会启用。

注意：混淆集只包含"错误侧不在 jieba 词典中"或"频率差距极大"的高置信对。
     "好象/好像"、"一棵/一颗"这类双方都在词典中的不包含，避免误伤。

调用约定：
    from confusion_corrector import get_default_corrector
    corrector = get_default_corrector()
    corrected = corrector.correct('这个方案比教完善')
"""

import functools
import math
import os
from typing import Dict, List, Optional, Set, Tuple

import jieba
import jieba.posseg as pseg  # 供 在/再 判定验证"双字动词"词性（再缴费 vs 再银行）

# =====================================================================
# 混淆集（双向/单向均可，配合语境判定函数）
# =====================================================================
# 筛选标准：错误形式在上下文中不构成有效词，正确形式构成有效词
# 特殊处理："在/再" 需要语境判定（介词/动词 vs 副词），加入双向候选
CONFUSION_SETS: Dict[str, List[str]] = {
    # 再↔在：双向候选，实际替换由 _should_replace_zai_zai 语境判定
    '再': ['在'],
    '在': ['再'],
    # 教→较：裸映射已移除——"教"作动词太常见（老师教我们/教学/教室/请教），
    # 泛化评分会把 老师教我们 误改成 老师较我们。精确场景 比教→比较 改放
    # FORCE_REPLACE（同 道→到 先例：错误侧是 比教/教完善，正确侧 教 是高频动词）。
    # 板→版："板本" freq=3 几乎不存在
    '板': ['版'],
    # 长→常："非长" freq=0 → "非常" freq=15958
    '长': ['常'],
    # 飞→非："飞常" freq=0 → "非常" freq=15958
    '飞': ['非'],
    # 以→已："以经" freq=0 → "已经" freq=big
    '以': ['已'],
    # 队→对："队不" freq=0 → "对不起" freq=1286
    '队': ['对'],
    # 连→联："连系" freq=0 → "联系" freq=2801
    '连': ['联'],
    # 按→安：裸映射已移除——"按"作介词/动词太常见（按顺序/按照/按钮/按时/
    # 按键/按住/按计划），泛化评分会把 按顺序 误改成 安顺序（安顺=54 是地名）。
    # 精确场景 按排→安排 改放 FORCE_REPLACE（同 教→较 先例）。
    # 代→带："代上" freq=0 → "带上" freq=432
    '代': ['带'],
    # 至→致："导至" freq=0 → "导致" freq=6498
    '至': ['致'],
    # 决→觉："决得" freq=0 → "觉得" freq=11580
    '决': ['觉'],
    # 注：裸 "道→到" 已移除——"头头是道/知道/报道/味道/道谢" 等合法词会被误伤。
    # 精确场景 "做道→做到" 改放 FORCE_REPLACE（ASR 常把"做到"误听成"做道"）。
    # 密→蜜："甜密" freq=0 → "甜蜜" freq=208
    '密': ['蜜'],
    # 才→材："才料" freq=0 → "材料" freq=768
    '才': ['材'],
    # 原→源："资原" freq=0 → "资源" freq=4645
    '原': ['源'],
    # 根→跟："根我" freq=0 → "跟我" freq=570
    '根': ['跟'],
    # 己→已："自忆" freq=0 ... actual: "自己" → "自忆" but "自己" is "自"+"己"
    '己': ['已'],
    # ===== 形近字（视觉相似，多为打字/OCR 错误；双向独立评分）=====
    # 未/末：周未→周末、期未→期末；未来/末日 等成词由相邻成词保护。
    # 周未 需专名检测的「周+星期字」守卫放行（见 _detect_proper_noun_spans）。
    '未': ['末'],
    '末': ['未'],
    # 侯/候：时侯→时候、等侯→等候（侯爵 是词，保护）
    '侯': ['候'],
    # 幸/辛：幸苦→辛苦、辛福→幸福（幸福/辛苦 高频词，方向自动对齐）
    '幸': ['辛'],
    '辛': ['幸'],
    # 拨/拔：拨河→拔河、拔打→拨打（拔河/拨打 成词）
    '拨': ['拔'],
    '拔': ['拨'],
    # 枪/抢：枪救→抢救、手抢→手枪（手机/抢救 成词保护）
    '枪': ['抢'],
    '抢': ['枪'],
    # 祟→崇：祟高→崇高（崇高 成词保护）。反向 崇→祟 因无法处理
    # 鬼鬼祟祟 的 4 字成语（祟祟 不成词无评分信号），改走 FORCE_REPLACE。
    '祟': ['崇'],
    # 炙/灸：针炙→针灸（脍炙人口 成词保护）
    '炙': ['灸'],
    # 肓/盲：肓人→盲人、膏盲→膏肓（盲人/盲目 成词保护）
    '肓': ['盲'],
    '盲': ['肓'],
    # 荼/茶：荼叶→茶叶、茶毒→荼毒
    '荼': ['茶'],
    '茶': ['荼'],
    # 博/搏：脉博→脉搏、搏士→博士（博士/博客/搏斗/拼搏 成词保护）
    '博': ['搏'],
    '搏': ['博'],
    # 侍/待：等侍→等待（侍候 是词，保护）
    '侍': ['待'],
    # 帅/师：老帅→老师、帅长→师长（帅哥/元帅 成词保护）
    '帅': ['师'],
    '师': ['帅'],
    # 幕/墓：坟幕→坟墓（屏幕/幕布 成词保护）
    '幕': ['墓'],
    # 奖/浆：豆奖→豆浆（奖金/奖励 成词保护）
    '奖': ['浆'],
    # 粟/栗：板粟→板栗（粟米 是词，保护）
    '粟': ['栗'],
    # 廉/镰：廉刀→镰刀（廉洁 成词保护）
    '廉': ['镰'],
    # 燥/躁：急燥→急躁、干躁→干燥（干燥/急躁 高频词）
    '燥': ['躁'],
    '躁': ['燥'],
    # 蓝/篮：蓝球→篮球、篮天→蓝天（蓝色/篮球 成词保护）
    '蓝': ['篮'],
    '篮': ['蓝'],
    # 仅/尽：仅管→尽管（不仅 成词保护）
    '仅': ['尽'],
}

# 形近字对（视觉相似）——与 CONFUSION_SETS 中的形近条目同源，供自学习
# 变体展开等场景做双向映射复用（同音映射走 pypinyin，形近走本表）。
SHAPE_SIMILAR_PAIRS = [
    ('未', '末'), ('侯', '候'), ('幸', '辛'), ('拨', '拔'), ('洒', '酒'),
    ('枪', '抢'), ('仓', '苍'), ('仓', '沧'), ('崇', '祟'), ('炙', '灸'),
    ('肓', '盲'), ('荼', '茶'), ('梁', '粱'), ('博', '搏'), ('侍', '待'),
    ('帅', '师'), ('幕', '墓'), ('奖', '浆'), ('粟', '栗'), ('廉', '镰'),
    ('燥', '躁'), ('蓝', '篮'), ('己', '已'), ('仅', '尽'),
]

# token 级短语修正表：(错误短语, 正确短语)
# 三种匹配模式（错误短语绝不在词典、freq=0；正确短语必高频）：
#   1) 整词粘合：jieba 把错误短语 HMM 粘成一个 token（因该/跟本/知到/应响…）
#   2) 错误字独立成单字 token + 后接其余部分（常/时间、在/见、想/信…）
#   3) 错误字在 HMM 粘词末尾 + 后接其余部分（别常/时间、你常/试一下…）
# 整词挡开：原因该（原因 是整词、freq 高）→ 不匹配模式 1/2/3，安全。
_TOKEN_PHRASE_FIXES = [
    # cháng：ASR 常把 长/尝 听成 常
    ('常时间', '长时间'),
    ('常试', '尝试'),
    # yīng：因该→应该
    ('因该', '应该'),
    # jì：既使→即使 / 即然→既然
    ('既使', '即使'),
    ('即然', '既然'),
    # gēn：跟本→根本
    ('跟本', '根本'),
    # zhī dào：知到→知道
    ('知到', '知道'),
    # yóu yú：有于→由于
    ('有于', '由于'),
    # xiāng xìn：想信→相信
    ('想信', '相信'),
    # jí：级时→及时（高级 整词挡开 高级时尚）/ 及使→即使（以及 整词挡开 以及使用）
    ('级时', '及时'),
    ('及使', '即使'),
    # yǐng xiǎng：应响→影响（响应/反应 整词挡开）
    ('应响', '影响'),
    # jìng：竞然→竟然（竞争/竞赛/竞技 整词挡开）
    ('竞然', '竟然'),
    # xiǎng fǎ：相法→想法
    ('相法', '想法'),
    # zài jiàn：在见→再见（仅句尾/语气词前，避开 在见他的时候）
    ('在见', '再见'),
]

# 在见→再见 的后续白名单：要求 见 之后的 token ∈ 此集合或已是句尾。
# 在见他的时候 / 在见你之前（在+见+代词）是合法用法，不修正。
_ZAI_JIAN_AFTER = frozenset({'吧', '了', '啊', '呀', '哦', '嗯', '！', '？', '，', '。'})


def _fix_token_phrases(text: str) -> str:
    """token 级同音短语修正（ASR 把正确字听成同音别字）。

    只动独立单字 token（或 HMM 粘出的非词典词末尾），绝不碰整词词素：
    - 那么常时间→那么长时间 ✓；正常时间/经常时间 不动 ✓
    - 因该→应该 ✓；原因该由你决定（原因 整词）不动 ✓
    - 在见→再见 ✓；在见到他之前（在+见到）不动 ✓
    """
    freq = _get_freq()
    chars = list(text)
    toks = list(jieba.tokenize(text))
    n = len(toks)
    for wrong, right in _TOKEN_PHRASE_FIXES:
        L = len(wrong)
        head, tail = wrong[:1], wrong[1:]
        for idx, (tok, s, e) in enumerate(toks):
            # 模式 1：错误短语被 HMM 粘成一个 token（因该/跟本/知到/应响…）
            if tok == wrong and freq.get(tok, 0) == 0:
                for k, rc in enumerate(right):
                    chars[s + k] = rc
                continue
            # 后接 token 匹配：完全相等，或 HMM 粘出的非词典词以 tail 开头
            # （试一下 等）；试探/时光 这类真词不匹配，避免误伤。
            if idx + 1 >= n:
                continue
            nxt = toks[idx + 1][0]
            # 阈值 <50：试一下(3) 这类冷僻 HMM 粘词算错误侧；试探(468)/时光(高)
            # 这类真词算正确侧，不匹配。
            nxt_ok = nxt == tail or (nxt.startswith(tail) and freq.get(nxt, 0) < _ORIG_NEIGHBOR_MIN_FREQ)
            if not nxt_ok:
                continue
            # 模式 2：错误字独立成单字 token（常/时间、在/见、想/信…）
            if tok == head:
                if wrong == '在见' and idx + 2 < n and toks[idx + 2][0] not in _ZAI_JIAN_AFTER:
                    continue
                for k, rc in enumerate(right):
                    chars[s + k] = rc
                continue
            # 模式 3：错误字在 HMM 粘词末尾（别常/时间、你常/试一下…）
            if tok.endswith(head) and len(tok) > 1 and freq.get(tok, 0) == 0:
                for k, rc in enumerate(right):
                    chars[e - len(head) + k] = rc
    return ''.join(chars)


# =====================================================================
# 专名保护：人名/地名/机构名跨度内的字，禁止被同音评分改动
# =====================================================================
# 动机：语音/校对场景中，专名里的字可能被同音评分带偏（李原→李源、
# 王才→王材、刘长→刘常）。专名一旦改错比不改更糟——「李源」可能是别人。
# 精度优先：只保护「jieba 不认识的名字段」——
#   - 整段 freq < 50 才保护：已知词（上海市/长安/方法/马虎/张罗/李子…）
#     是合法词形，模式检测不掺和（它们需要时自有词级保护兜底）；
#   - 姓字与混淆集键（在/再/长/原/才/决/密/根/己…）零交集，纯名检测
#     不会拦截现有修正；名位字用功能字排除表挡住 王在/李是 这类
#     「姓+虚词」伪名，避免把 王在→王再 这类真实修正也保护掉。

_SURNAMES = frozenset(
    '王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁'
    '潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦'
    '邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤'
)
_SURNAMES_2 = frozenset({
    '欧阳', '司马', '上官', '诸葛', '东方', '皇甫', '尉迟', '公孙', '长孙',
    '慕容', '司徒', '司空', '夏侯', '令狐', '轩辕', '闻人', '独孤', '南宫',
    '端木', '西门', '呼延', '宇文', '申屠', '钟离', '淳于', '单于', '太叔',
    '公冶', '百里', '东郭', '南门', '左丘', '濮阳', '宗政',
})
# 名位字排除：虚词/功能字几乎不会出现在名字里（王在/李是/王再 是伪名）。
# 任一命中即拒——王在开会（在开）不能因 开 不在表里就被当成 2 字名。
_NAME_GIVEN_EXCLUDE = frozenset(
    '在再以已己是的了和或与都被把让从向其为之这那怎什哪吗呢吧啊呀哦么得地'
    '着过将只可但而则若因由并且虽按至代队教板决总自来去走说看想要会能可就也都又还'
    '一上下中前后里外间'
)
_PLACE_SUFFIXES = (
    '省', '市', '县', '区', '镇', '乡', '村', '庄', '屯',
    '山', '河', '江', '湖', '海', '岛', '湾', '港', '峰', '岭', '坡', '谷',
    '溪', '泉', '潭', '桥', '路', '街', '巷', '胡同', '大道', '大街',
    '广场', '公园', '水库', '机场', '车站', '码头', '新区', '开发区',
    '工业园', '高新区', '古镇', '古城', '半岛', '群岛', '海峡', '平原',
    '盆地', '草原', '沙漠', '油田', '矿区', '山脉', '口岸',
)
_ORG_SUFFIXES = (
    '公司', '集团', '大学', '学院', '研究院', '研究所', '医院', '银行',
    '中学', '小学', '幼儿园', '中心', '协会', '基金会', '出版社', '电视台',
    '杂志社', '事务所', '工厂', '酒店', '饭店', '餐厅', '大厦', '大楼',
    '园区', '基地', '委员会', '指挥部', '俱乐部', '商会', '联盟', '机构',
    '支行', '分行', '商行',
)
# 称谓词：按长度从长到短（总经理 先于 经理/总），人名+称谓是强信号
_PERSON_TITLES = (
    '总经理', '董事长', '研究院', '工程师', '设计师', '事务所', '幼儿园',
    '先生', '女士', '小姐', '老师', '教授', '医生', '律师', '经理',
    '总裁', '局长', '主任', '处长', '部长', '科长', '同学', '师傅',
    '博士', '硕士', '专家', '记者', '编辑', '作家', '顾问', '秘书',
    '老板', '厂长', '校长', '院长', '教练', '护士', '店长', '村长',
    '镇长', '市长', '县长', '区长', '总',
)


def _detect_proper_noun_spans_impl(text: str) -> Set[int]:
    """启发式专名检测实现（只保护 jieba 不认识的名字段）。

    供 _detect_proper_noun_spans_heuristic（纯启发式）与
    _detect_proper_noun_spans（启发式 + NER 增强）共用。
    """
    freq = _get_freq()
    n = len(text)
    protected: Set[int] = set()

    def freq_of(s: int, e: int) -> int:
        if s < 0 or e > n or e <= s:
            return 0
        return freq.get(text[s:e], 0)

    def protect(s: int, e: int):
        for k in range(s, e):
            protected.add(k)

    # 1) 称谓人名：姓(+名 0~2 字) + 称谓（王建国先生/李总/张老师）——称谓是强信号
    for title in _PERSON_TITLES:
        start = 0
        while True:
            t = text.find(title, start)
            if t < 0:
                break
            for ln in (1, 2, 3):
                s = t - ln
                if s < 0:
                    break
                if s > 0 and _is_cjk(text[s - 1]):
                    continue
                name_part = text[s:t]
                if not all(_is_cjk(c) for c in name_part):
                    continue
                sur_len = (2 if name_part[:2] in _SURNAMES_2
                           else (1 if name_part[0] in _SURNAMES else 0))
                if sur_len == 0:
                    continue
                given = name_part[sur_len:]
                # 纯姓+称谓（王先生/李总）也保护姓；有名位字时须非全虚词
                if not given or any(c not in _NAME_GIVEN_EXCLUDE for c in given):
                    protect(s, t)
                    break
            start = t + len(title)

    # 2) 地名：2~4 字 + 地名后缀（柳溪镇/朝阳公园）；名字段连后缀都不成词才保护
    for suf in _PLACE_SUFFIXES:
        start = 0
        while True:
            t = text.find(suf, start)
            if t < 0:
                break
            for ln in (4, 3, 2):
                s = t - ln
                if s < 0:
                    continue
                if s > 0 and _is_cjk(text[s - 1]):
                    continue
                name_part = text[s:t]
                if (all(_is_cjk(c) for c in name_part)
                        and freq_of(s, t) < _ORIG_NEIGHBOR_MIN_FREQ
                        and freq_of(s, t + len(suf)) < _ORIG_NEIGHBOR_MIN_FREQ):
                    protect(s, t)
                    break
            start = t + len(suf)

    # 3) 机构名：2~6 字 + 机构后缀（远山科技有限公司）
    for suf in _ORG_SUFFIXES:
        start = 0
        while True:
            t = text.find(suf, start)
            if t < 0:
                break
            for ln in (6, 5, 4, 3, 2):
                s = t - ln
                if s < 0:
                    continue
                if s > 0 and _is_cjk(text[s - 1]):
                    continue
                name_part = text[s:t]
                if (all(_is_cjk(c) for c in name_part)
                        and freq_of(s, t) < _ORIG_NEIGHBOR_MIN_FREQ
                        and freq_of(s, t + len(suf)) < _ORIG_NEIGHBOR_MIN_FREQ):
                    protect(s, t)
                    break
            start = t + len(suf)

    # 4) 纯人名（无称谓）：姓 + 1~2 字名（李莉/王建国/欧阳娜娜）
    i = 0
    while i < n:
        ch = text[i]
        if not _is_cjk(ch):
            i += 1
            continue
        sur_len = (2 if text[i:i + 2] in _SURNAMES_2
                   else (1 if ch in _SURNAMES else 0))
        if sur_len == 0:
            i += 1
            continue
        s = i
        e_sur = i + sur_len
        for glen in (2, 1):
            e2 = e_sur + glen
            if e2 > n:
                continue
            given = text[e_sur:e2]
            if not all(_is_cjk(c) for c in given):
                continue
            # 名位字含任一虚词（王在开会/李原自 里的 在/自）→ 伪名，跳过
            if any(c in _NAME_GIVEN_EXCLUDE for c in given):
                continue
            # 周+星期字（周一~周日/周末/周未）：星期语义优先于人名检测，
            # 否则 周未→周末 这类形近修正会被专名保护当成 周+未 人名挡住。
            if ch == '周' and all(c in '一二三四五六日天末未' for c in given):
                continue
            if freq_of(s, e2) >= _ORIG_NEIGHBOR_MIN_FREQ:
                continue
            # 2 字名须保证 1 字前缀也不成词（李子很/周五见 里的 李子/周五
            # 是合法词，不能因「+1 字不成词」被误当名字段整体保护）
            if glen == 2 and freq_of(s, e_sur + 1) >= _ORIG_NEIGHBOR_MIN_FREQ:
                continue
            protect(s, e2)
            break
        i = e_sur

    return protected


def _detect_proper_noun_spans_heuristic(text: str) -> Set[int]:
    """纯启发式专名检测（确定性、零模型依赖）。

    与 _detect_proper_noun_spans 的区别：不含 HanLP NER 增强。
    供 yixing（异形词归一）等纯词典层使用——它们不该背 NER 推理成本，
    也不该被 NER 对未知词串的长 span 幻觉挡住本应进行的修正
    （实测 HanLP tok 把 保安保镳队 合成单 token、NER 标 ORG 全保护，
    导致 保镳→保镖 不替换）。
    """
    return _detect_proper_noun_spans_impl(text)


def _detect_proper_noun_spans(text: str) -> Set[int]:
    """检测专名跨度，返回应保护的字符索引集合（启发式 + HanLP NER 增强）。

    NER 增强：模型识别的人名/地名/机构名（已过滤假阳性）。
    - 与启发式互补：名含 得 的人名（王得福）、非姓氏外国人名、无后缀机构名。
    - 实测 MSRA_NER_ELECTRA_SMALL_ZH 会把 李子/周五/上海市/长安街/张罗 等
      已知词误标成专名，故过滤规则与启发式保持一致（见 _ner_spans_filtered）。
    - 长 span 幻觉（>4 字）丢弃：HanLP tok 把 保安保镳队 合成单 token、
      NER 标 ORG 全保护——真专名 span 都 ≤4（王得福/柳溪镇/欧阳娜娜）。
    """
    protected = _detect_proper_noun_spans_impl(text)
    protected |= _ner_spans_filtered(text)
    return protected


def _ner_protection_enabled() -> bool:
    """NER 专名保护开关（post_processing.ner_protection.enabled，默认开）。"""
    try:
        from .config import config as _c
    except Exception:
        try:
            import config as _c2
            _c = _c2.config
        except Exception:
            return True
    try:
        return bool(_c.get('post_processing.ner_protection.enabled', True))
    except Exception:
        return True


@functools.lru_cache(maxsize=1024)
def _ner_spans_raw(text: str) -> Tuple[Tuple[int, int], ...]:
    """HanLP NER 原始 span（仅推理缓存；CSC 与 把X切入 两处调用避免重复推理）。"""
    try:
        from syntax_analyzer import get_ner_spans  # 懒加载；HanLP 不可用返回 []
        return tuple(get_ner_spans(text))
    except Exception:
        return ()


def _ner_spans_filtered(text: str) -> Set[int]:
    """HanLP NER span → 受保护字符索引（过滤假阳性后）。

    过滤规则（与启发式语义一致）：
      - 单字符 span 丢弃：王/李 单独标 PERSON 是伪名，启发式从不保护单字；
      - 已知词（jieba freq >= _ORIG_NEIGHBOR_MIN_FREQ）丢弃：李子/周五/
        上海市/长安街/张罗 全被小模型误标成专名，但它们是高频词，CSC 自身
        的频率守卫不会改它们，无需保护；
      - 长 span（>4 字）丢弃：HanLP tok 把 保安保镳队 合成单 token、NER
        标 ORG 保护全部 5 字——真专名 span 都 ≤4（王得福/柳溪镇/欧阳娜娜），
        5+ 字整串标专名是 tok 合并 + 模型幻觉的典型特征；
      - 未知/低频真专名保留：王得福(0)/柳溪镇(0)/欧阳娜娜(0)。
    """
    if not text or not _ner_protection_enabled():
        return set()
    freq = _get_freq()
    out: Set[int] = set()
    for s, e in _ner_spans_raw(text):
        if e - s <= 1 or e - s > 4:
            continue
        if freq.get(text[s:e], 0) >= _ORIG_NEIGHBOR_MIN_FREQ:
            continue
        out.update(range(s, e))
    return out


# 强制替换表：错误侧词频虽高但确为误用，绕过频率评分直接替换
# 这些在 FIXED_DE_DU_DE 也有定义，但为让 confusion_corrector 单元测试独立通过，在此也强制生效
FORCE_REPLACE: Dict[str, str] = {
    # ===== 同音短语强制修正（错误侧绝无合法用法，子串级安全）=====
    '什末': '什么',   # 什末/为什末（什么 同音 什末，莫/末 混）
    '什莫': '什么',
    '毕竞': '毕竟',   # 毕/必 与 竞/竟 同音（毕竞→毕竟）
    '付责': '负责',   # 付/负 同音
    '决对': '绝对',   # 决/绝 同音
    '绝定': '决定',
    '兰球': '篮球',   # 兰/蓝/篮 同音
    '步署': '部署',   # 步/部 同音
    '辩别': '辨别',   # 辨/辩 同音（辨别用 辨）
    '辨论': '辩论',   # 辩论用 辩
    # 坐/座
    '坐位': '座位',
    '坐落': '座落',
    '作下': '坐下',
    # 做/作
    '做为': '作为',
    # 注：'做工→作工'/'做事→作事' 曾在旧版被反向收录——但「做工/做事」本就是
    # 规范写法（作工/作事 为罕用/非规范变体），会破坏正确文本，已移除。
    # 量词（仅明确错误的具体短语）
    '一棵星星': '一颗星星',
    '两棵星星': '两颗星星',
    '一棵草': '一颗草',
    '两棵草': '两颗草',
    # 介词/动词固定搭配
    '根我': '跟我',
    # 道→到 精确场景："做道" 不是有效词 → "做到"
    '做道': '做到',
    # 教→较 精确场景：比教 不是有效词 → 比较（老师教我们 等动词用法不受影响）
    '比教': '比较',
    # 按→安 精确场景：按排 不是有效词 → 安排（按顺序/按照 等介词用法不受影响）
    '按排': '安排',
    # ===== 形近字精确场景（错误侧绝无合法用法；泛化对会误伤的降级到此）=====
    # 仓/苍/沧：泛化 仓→苍/沧 会因单字频率误伤（仓海→苍海），改精确短语
    '仓白': '苍白',
    '仓海': '沧海',
    '仓茫': '苍茫',
    '仓凉': '苍凉',
    '仓天': '苍天',
    # 崇→祟：鬼鬼祟祟 是 4 字成语，祟祟 不成词评分无信号，精确短语兜底
    '鬼鬼崇崇': '鬼鬼祟祟',
    # 洒→酒：洒水 不是 jieba 词（freq=8），泛化会把 洒水车 误改成 酒水车
    '洒店': '酒店',
    '喝洒': '喝酒',
    # 梁→粱：高梁 会被专名检测误判成 高+梁 人名，泛化评分被保护挡掉
    '高梁': '高粱',
    # 已→己：自已 是 jieba 冷僻词（freq=263），相邻成词保护拦截泛化评分
    '自已': '自己',
}

# 保护白名单：这些高频正确形式绝不被混淆集替换（子串匹配即拦截）
PROTECT_PATTERNS = frozenset({
    '做主', '作家', '做梦', '做客', '做东', '做媒', '做证', '做法', '做工', '做事', '做人',
    '作家', '作文', '作业', '作息', '作风', '作战', '作罢', '作古', '作揖', '作别',
    '坐下', '坐位', '坐落', '坐镇', '坐骑', '坐标', '坐席', '坐庄', '坐享', '坐拥',
    '座位', '座落', '座右铭', '座谈', '座机', '座驾', '座椅', '座钟', '座头', '座上宾',
})

# 保护字符：人称代词 + 的地得 + "不/一" 等高频功能字
_GENDER_PRONOUNS = frozenset({'他', '她', '它', '祂'})
_PROTECTED_CHARS = _GENDER_PRONOUNS | frozenset({'不', '一', '的', '地', '得'})

# 「相邻成词即放行」保护的最低词典频率。
# 混淆对的筛选标准是「错误侧不构成有效词」，但 jieba 词典里仍残留少量
# 冷僻异体词（板本/连系/按排/教完…，频率均为 3）——它们正是要修正的
# 错误侧，不能保护。而正确侧（可以 70958 / 以为 / 所以 / 才学 83 …）最低
# 也在 83 以上。取 50 居中：≥50 视为可靠成词 → 放行；<50 交评分逻辑。
_ORIG_NEIGHBOR_MIN_FREQ = 50

# =====================================================================
# "在/再" 语境白名单与判定逻辑
# =====================================================================

# "再"作副词的高频合法用法（保护不被误改为"在"）
# 这些模式中出现"再"时，应直接跳过替换
ZAI_ADVERB_WHITELIST: Set[str] = frozenset({
    # 再 + 单字动词（重复/继续义）
    '再说', '再来', '再看', '再走', '再做', '再试', '再想', '再等', '再找', '再问',
    '再给', '再去', '再回', '再打', '再买', '再换', '再查', '再改', '再发', '再收',
    # 单字"再"单独保护较难，靠后缀匹配
    '再也不', '再也', '再不', '再三', '再四', '再次', '再度', '再行', '再接', '再励', '再会', '再见',
    # 再 + 量词/数词短语
    '再来一次', '再看一眼', '再走一遍', '再做一次', '再试一次',
    # 再 + 双字动作动词（重复/继续义；防词频评分把正确的"再处理问题"误改回"在"）。
    # 只收录几乎总是作动词的高频词，避免"再北京/再桌子"（处所/名词）被误保护。
    '再处理', '再检查', '再考虑', '再安排', '再确认', '再调整', '再讨论',
    '再分析', '再研究', '再修改', '再测试', '再部署', '再运行', '再学习',
    '再尝试', '再等待', '再准备', '再说明', '再解释', '再商量', '再折腾',
    '再等等', '再想想', '再看看', '再试试', '再听听', '再写写', '再聊聊', '再坐坐',
})

# "在"作介词/动词的高频合法用法（保护不被误改为"再"）
# 这些模式中出现"在"时，应直接跳过替换
ZAI_PREP_VERB_WHITELIST: Set[str] = frozenset({
    # 在 + 处所/方位
    '在家', '在校', '在公司', '在单位', '在医院', '在学校', '在路上', '在楼上', '在楼下',
    '在里面', '在外面', '在前面', '在后面', '在左边', '在右边', '在旁边', '在对面',
    '在附近', '在附近', '在身边', '在手边', '在眼前', '在心里', '在梦里', '在天上',
    '在地上', '在水里', '在火里', '在风里', '在雨里', '在雪里', '在雾里',
    # 在 + 时间/抽象方位
    '在过去', '在现在', '在将来', '在未来', '在当时', '在那时', '在这些',
    '在最近', '在最早', '在最后', '在开始', '在结束', '在中间', '在期间',
    # 正在/正好 + 动词（进行体标记）
    '正在', '正在做', '正在写', '正在想', '正在吃', '正在睡', '正在玩',
    '正在干', '正在跑', '正在走', '正在说', '正在听', '正在等', '正在找',
    # 在 + 处所/方位/时间（不含动词，避免与动量豁免冲突）
    # 原有的 '在看/在想/在吃/在睡/在玩/在干/在跑/在走/在说/在听/在等/在找/在学/在工作/在学习/在编程' 已移除
    # 由 _should_replace_zai_zai 的动量豁免逻辑直接处理
    '在这里', '在那里', '在这里面', '在那里面', '在前面', '在后面',
    # "在" 作构词语素的高频合法双字词（避免误改为"再"）
    '在于', '在乎', '在意', '在场', '在逃', '在押', '在岗', '在编', '在职',
    '在世', '在坐', '在案', '现在', '现存',
})

# "在"后接瞬时动词/单字动词+了/过/着 → 可能是"再"（"在吃了"误写应为"再吃了"）
# 仅用于启发式判断，不直接决定
ZAI_VERB_ASPECT_SUFFIXES = frozenset({'了', '过', '着', '一下', '一遍', '一次'})

# =====================================================================
# 数量追加结构：「再/在 + 追加类动词(+了) + 名量短语」→ 再
# =====================================================================
# 语法依据：《现代汉语八百词》"再"义项①"表示又一次（多指将发生的）"（例：再唱一个）、
# 义项⑤"表示另外有所补充"——「再增加一个功能 / 再添加一些内容 / 再吃一碗面」中，
# 动词后的 数量短语（数词+量词+名词）表示"追加一个/再来一个"，是口语中 在/再
# 混淆的高发结构。而"在"作进行体时与"一次性追加"语义冲突——"数量在增加 / 成本
# 在增加"的 增加 不带宾语，是持续义，不在此列（由名量短语条件天然排除）。
#
# 只收「再+V 几乎总是追加义」的双字动词。进行体高频的动词（编辑/补充/处理/制作/
# 修改/整理/设计…）刻意不收，否则"他在编辑一个视频"这类合法进行体会被误改成"再"。
ZAI_ZAI_ADDITIVE_VERBS: frozenset = frozenset({
    '增加', '添加', '增设', '增补', '加装', '补加', '添补',
    '加设', '扩增', '增建', '增添', '追加',
})

# 名量短语首字（数词/不定量）：一/两/几/三…十/多/好/数。
# 刻意不含 这/那："再增加那个功能"与"在增加那个功能"两种读法都成立，不强行翻转。
ZAI_ZAI_NUM_PREFIX: frozenset = frozenset('一两几三四五六七八九十多好数')

# 常用名量词（配合 NUM_PREFIX 组成 一个/两份/几条/一笔…）。
# 刻意不含时点/时段量词（点/时/分/秒/天/年/月/周/日/号/刻/钟/旬/载/季）：
# "在两点见/在三个小时里" 是合法的时间介词用法，不能卷入 在/再 翻转。
ZAI_ZAI_MEASURE: frozenset = frozenset(
    '个些份碗篇项款条只台部套张堆批种类位名本首件段册支根块片层级别门节题行页组回次'
    '笔元杯瓶盒包袋双对面幅封束把顶辆架艘栋座间行列排串群队班叠捆朵颗粒滴股剂')

# 不定量/复合名量短语（两字起，不落入 数词+量词 模式）：很多/不少/许多…
ZAI_ZAI_QUANT_PHRASES: frozenset = frozenset({
    '很多', '不少', '许多', '好多', '更多', '多个', '数个', '若干', '有些',
    '好几个', '好多个', '好几种', '好几分', '好几项', '好几款',
})

# 「再」后直接接双字动词时，首字为处所起始字的排除（再前面一点/再上面一些
# 由既有的 "再+处所词 → 在" 规则处理，本规则不覆盖）。
ZAI_ZAI_LOCATIVE_FIRST: frozenset = frozenset(
    '这那哪前后左右上下里外内旁边面中间家校公司院所厂站')

# G2 守卫：再+动词+时间框架（在...之前/之后/以前/以来/期间）是 在 的合法
# 介词用法（在开会之前/在出发之后），不保护——交给通用评分修成"在"。
ZAI_ZAI_TIME_FRAME: frozenset = frozenset({'之前', '之后', '以前', '以来', '期间'})


def _zai_zai_has_measure(text: str, vi: int) -> bool:
    """判断 text[vi:] 是否紧接 名量短语（数词+量词 / 不定量短语）。

    用于「在→再」方向的门槛："数量在增加/成本在增加"（增加 不带宾语、
    持续义）必须保留"在"，只有后接名量宾语时才翻转成"再"。
    """
    if vi >= len(text):
        return False
    if text[vi:vi + 3] in ZAI_ZAI_QUANT_PHRASES:
        return True
    if text[vi:vi + 2] in ZAI_ZAI_QUANT_PHRASES:
        return True
    return (text[vi] in ZAI_ZAI_NUM_PREFIX and vi + 1 < len(text)
            and text[vi + 1] in ZAI_ZAI_MEASURE)


def _is_cjk(ch: str) -> bool:
    return len(ch) == 1 and '\u4e00' <= ch <= '\u9fff'


def _has_whitelist_suffix(text: str, pos: int, whitelist: Set[str], max_len: int = 6) -> bool:
    """检查 text[pos:] 是否以 whitelist 中任一词为前缀（向后最长 max_len）。"""
    end = min(len(text), pos + max_len)
    for l in range(2, end - pos + 1):
        if text[pos:pos + l] in whitelist:
            return True
    return False


def _has_whitelist_prefix(text: str, pos: int, whitelist: Set[str], max_len: int = 6) -> bool:
    """检查 text[:pos+1] 是否以 whitelist 中任一词为后缀（向前最长 max_len）。"""
    start = max(0, pos - max_len + 1)
    for l in range(2, pos - start + 2):
        if text[pos - l + 1:pos + 1] in whitelist:
            return True
    return False


def _should_replace_zai_zai(chars: List[str], i: int, cand: str, freq: Dict[str, int],
                            token_info_at: Optional[Dict[int, tuple]] = None,
                            pos_at: Optional[Dict[int, str]] = None) -> Optional[bool]:
    """
    "在/再" 专用语境判定。
    返回 True/False/None（None 表示交给通用三级评分）。
    """
    orig = chars[i]
    text = ''.join(chars)
    n = len(text)

    # ── 0* 数量追加结构「再/在 + 追加类动词(+了) + 名量短语」→ 再 ──
    # 优先级最高，须在场景 D（在+双字动词→进行体保护）之前拦截。
    # 再→在（保护，防毁）：
    #   a) 再+追加类动词（无条件）——再增加/再添加/再增设 等几乎总是"追加义"，
    #      而通用三级评分会被 在 的单字高频带偏，把正确的"再"毁成"在"（连
    #      "数量再增加/成本再增加"这类无宾语持续义都被毁过）；
    #   b) 再+任意双字动词+名量短语——"再编辑一个视频/再制作两个页面" 同理被毁，
    #      白名单救不了无穷的动词，必须结构化保护（处所起始字除外，不覆盖既有
    #      的 "再+处所词→在" 规则）；
    #   c) 再+名量短语（无动词）——"再两个功能/再几个选项" 是"再来几个"义。
    # 在→再（修正）：仅 在+追加类动词+名量短语——"数量在增加/成本在增加"
    #   （增加 不带宾语）是合法持续义必须保留"在"；非追加类动词（编辑/处理…）
    #   的进行体与追加义同等常见，尊重输入不翻转。
    if orig == '再' and cand == '在':
        if text.startswith(tuple(ZAI_ZAI_ADDITIVE_VERBS), i + 1):
            return False
        # b) 再 + 双字动词（首字非处所）+ 名量短语
        if (i + 3 < n and _is_cjk(text[i + 1]) and _is_cjk(text[i + 2])
                and text[i + 1] not in ZAI_ZAI_LOCATIVE_FIRST):
            vi = i + 3
            if vi < n and text[vi] == '了':
                vi += 1
            if _zai_zai_has_measure(text, vi):
                return False
        # c) 再 + 名量短语（无动词，时间量词已被 MEASURE 排除）
        if i + 2 < n and _zai_zai_has_measure(text, i + 1):
            return False
        # d) 再 + 单字动词 + 名量短语——"再唱一个/再画一幅/再扔一块" 等
        #    common_v 表外的单字动词被通用评分毁成"在"（再唱一个→在唱一个）。
        #    用 jieba 词性验证动词身份，避免 再+非动词+名量 误保护。
        if (pos_at and pos_at.get(i + 1, '') == 'v' and i + 2 < n
                and text[i + 1] not in ZAI_ZAI_LOCATIVE_FIRST
                and _zai_zai_has_measure(text, i + 2)):
            return False
        # e) 再 + 双字动词（词性验证）——"先报名再缴费/做完再缴费/然后再缴费"
        #    等承接/继续义被通用评分毁成"在"（在+双字动词读作进行体，带偏评分）。
        #    处所名词（再银行/再医院/再公司）词性为 n/ns，不会被保护，保持既有修正。
        _tok_info = token_info_at.get(i + 1) if token_info_at else None
        if (_tok_info and pos_at and pos_at.get(i + 1, '') in ('v', 'vn')):
            _tok, _tok_start = _tok_info
            if _tok_start == i + 1:
                if len(_tok) == 2:
                    _tail = text[i + 3:i + 6]
                    # 时间框架（在...之前/之后）不保护，交给通用评分修成"在"
                    if not (_tail[:2] in ZAI_ZAI_TIME_FRAME
                            or _tail[:3] == '的时候'):
                        return False
                else:
                    return False
        # g) 再 + 单字动词 + 句尾/语气词——"等找到垃圾桶后再丢/然后再走" 等
        #    承接义被通用评分毁成"在"（CGED 2021 实证：再丢→在丢）。带宾语
        #    （再丢垃圾）可能是"在丢"进行体，不保护；句尾/语气词前几乎必是
        #    承接义"然后再做"。
        if (pos_at and pos_at.get(i + 1, '') == 'v'
                and text[i + 1] not in ZAI_ZAI_LOCATIVE_FIRST
                and (i + 2 >= n
                     or text[i + 2] in ('，', '。', '、', '；', '！', '？',
                                        '了', '吧', '呢', '吗', '啊', '呀'))):
            return False
        # f) 再 + 形容词 + 也/都/不过——让步/程度结构（再难也要做/再合适不过），
        #    是 再 的标志性用法，被通用评分系统性毁成"在"。
        if (i + 2 < n and _is_cjk(text[i + 1])
                and text[i + 1] not in ZAI_ZAI_LOCATIVE_FIRST):
            if text[i + 2] in ('也', '都'):
                return False
            if (i + 3 < n and _is_cjk(text[i + 2]) and text[i + 3] in ('也', '都')):
                return False
            if text[i + 2:i + 4] == '不过' or text[i + 3:i + 5] == '不过':
                return False
    elif orig == '在' and cand == '再':
        if text.startswith(tuple(ZAI_ZAI_ADDITIVE_VERBS), i + 1):
            vi = i + 3  # 追加类动词均为双字
            if vi < n and text[vi] == '了':
                vi += 1
            if _zai_zai_has_measure(text, vi):
                return True

    # ── 0. 动量/程度短语豁免：优于白名单的最高优先级判定 ──
    # 语法依据：汉典"再"副词义项②"动作重复/继续，多指未然：再试一次、再看一下、再高一点"
    # "在"作进行体时不带动量短语；"在+动词+一下/一遍/一次/一回/几下/几遍/多下/多遍/多次/再下/再遍/再次"
    # 强烈倾向于"再"（重复/继续义）。此处在白名单短路前拦截，修复"在看一下/在折腾一下"类漏改。
    # 处所词排除集：若后字是处所起始字，不触发豁免（如"在家一下"不改）
    LOCATIVE_CHARS = {'这', '那', '哪', '前', '后', '左', '右', '上', '下',
                      '里', '外', '内', '旁', '边', '面', '中', '间', '家', '校', '公', '司', '院', '所', '厂', '站', '口', '头', '尾', '旁', '畔'}
    # 核心动量短语集（覆盖 95%+ ASR 误识场景）
    MOTION_QUANTIFIERS = {
        '一下', '一遍', '一次', '一回',   # 核心四款
        '一眼', '一会', '一会儿', '一点', '一看', '一试', '一查', '一改', '一想', '一等',  # 常用口语量词
        '两下', '两遍', '两次', '两回',   # 二量
        '几下', '几遍', '几次', '几回', '几眼', '几会',   # 不定量
        '多下', '多遍', '多次', '多回',   # 程度/数量
        '再下', '再遍', '再次', '再回',   # 复合
        '些下', '些遍', '些次',           # 少见但可能
    }

    def _find_quantifier(window: str):
        """返回 (动量词在窗口中的起始位置, 动量词)；未命中返回 (-1, None)。"""
        for q in sorted(MOTION_QUANTIFIERS, key=len, reverse=True):
            qi = window.find(q)
            if qi != -1:
                return qi, q
        return -1, None

    def _between_ok(between: str, allow_short: bool = True) -> bool:
        """在/再 与动量词之间的中间内容是否"干净"。

        只允许两类：
        1) 状语结构（含"地"，如 认真地/仔细地）——在认真地看一下 / 再仔细地检查一遍；
        2) 极短动词/语气词（≤2 字且无处所/体标记）——在看一下 / 再折腾一下。

        出现处所字/体标记（了/的/在/呢…）说明"在"挂在处所或持续体上
        （他在群里看了一下消息、他在讲了一下方案），不应触发 在→再。
        """
        if '地' in between:
            return True
        if not allow_short:
            return False
        if len(between) <= 2 and not any(c in between for c in LOCATIVE_CHARS):
            return True
        return False

    if orig == '在' and cand == '再':
        # 场景 A：单字动词 —— "在" + X(单字动词) + 动量（i+1=动词，i+2 起是动量）
        if (i + 3 <= n and chars[i + 1] not in LOCATIVE_CHARS and chars[i + 1] != '了'):
            tail2 = text[i + 2:i + 4]  # 取"一下"、"一遍"等双字
            if tail2 in MOTION_QUANTIFIERS:
                return True
        # 场景 B：双字动词 —— "在" + XY(双字动词) + 动量（i+1~i+2=动词，i+3 起是动量）
        if (i + 4 <= n and chars[i + 1] not in LOCATIVE_CHARS):
            tail3 = text[i + 3:i + 5]
            if tail3 in MOTION_QUANTIFIERS and text[i + 2] != '了':
                return True
        # 场景 C：通用兜底 —— 动词后窗口内出现动量短语（宽容 ASR 插词/语气词）。
        # 收紧：中间内容必须含"地"（状语）或极短（≤2 字），否则视为
        # "在"挂在处所/持续体上（他在群里看了一下消息 → 他再群里… 的错误原罪）。
        if i + 1 < n and chars[i + 1] not in LOCATIVE_CHARS:
            window = text[i + 1:i + 9]
            qi, _q = _find_quantifier(window)
            if qi != -1 and _between_ok(window[:qi]):
                return True
        # 场景 D：在 + 单字/双字动词（无处所、无动量、无体标记）→ 进行体，保持"在"。
        # 关键回归修复："他在慢慢地走" 原被词频评分误改为"他再慢慢地走"。
        # 语法上"在+动词"默认读作进行体；只有动量短语（场景 A/B/C）才倾向"再"。
        # 例外：后接体标记（了/过/着）时交给下方启发式（在吃了→再吃了），此处不拦。
        if (i + 1 < n and _is_cjk(chars[i + 1])
                and chars[i + 1] not in LOCATIVE_CHARS and chars[i + 1] != '了'):
            if not (i + 2 < n and chars[i + 2] in ('了', '过', '着')):
                return False

    # ── 0.5 再 + 动词/状语 + 动量 → 保持"再"（再折腾一下/再处理一下/再仔细地检查一遍）──
    # 与 0 对称：避免"再折腾一下"在 再→在→再 之间震荡（原实现 5 轮后停在错误侧，
    # 把正确的"再折腾一下"改成"在折腾一下"）。
    if orig == '再' and cand == '在':
        if i + 1 < n and chars[i + 1] not in LOCATIVE_CHARS:
            window = text[i + 1:i + 9]
            qi, _q = _find_quantifier(window)
            if qi != -1 and _between_ok(window[:qi]):
                return False

    # 1. 白名单保护：若周围命中明确合法模式，直接放行
    if orig == '再' and cand == '在':
        # 原是"再"，候选"在" -> 检查是否为副词"再"合法用法
        if _has_whitelist_suffix(text, i, ZAI_ADVERB_WHITELIST):
            return False  # "再说/再来/再看..." 保护
        # 向前看：若前字是"不/也/都/还/就/才/又/再"等副词修饰语，常为副词"再"
        if i > 0 and chars[i - 1] in {'不', '也', '都', '还', '就', '才', '又', '总', '老', '偏'}:
            return False
        # "再"后接处所词（这里/那里/前面/上面/下面...） -> 应改为"在"
        if i + 1 < n and chars[i + 1] in {'这', '那', '哪', '前', '后', '左', '右', '上', '下',
                                           '里', '外', '内', '旁', '边', '面', '中', '间'}:
            return True
    elif orig == '在' and cand == '再':
        # 原是"在"，候选"再" -> 检查是否为介词/动词"在"合法用法
        if _has_whitelist_suffix(text, i, ZAI_PREP_VERB_WHITELIST):
            return False  # "在家/在公司/正在做..." 保护
        if _has_whitelist_prefix(text, i, ZAI_PREP_VERB_WHITELIST):
            return False
        # "在"后接处所词（这里/那里/前面/里面...） -> 保护"在"
        if i + 1 < n and chars[i + 1] in {'这', '那', '哪', '前', '后', '左', '右', '上', '下',
                                           '里', '外', '内', '旁', '边', '面', '中', '间'}:
            return False

    # 2. 启发式：单字"再"后跟动词、且无处所标记 -> 倾向保留"再"（副词义）
    #    单字"在"后跟单字动词+体标记(了/过/着) -> 倾向改为"再"（如"在吃了"->"再吃了"）
    if orig == '再' and cand == '在':
        # "再" + 单字/双字动词（常见动词集合），且后无处所词
        if i + 1 < n:
            nxt = chars[i + 1]
            common_v = {'说', '看', '做', '来', '去', '走', '跑', '吃', '睡', '想', '试',
                        '等', '找', '问', '给', '买', '换', '查', '改', '发', '收', '写',
                        '读', '听', '学', '玩', '干', '忙', '帮', '借', '还', '送', '拿'}
            if nxt in common_v:
                # 进一步看后面是不是处所词
                if not (i + 2 < n and chars[i + 2] in {'在', '这', '那', '里', '上', '下', '前', '后'}):
                    return False  # 倾向保留"再"

    if orig == '在' and cand == '再':
        # "在" + 单字动词 + 体标记(了/过/着/一下/一遍) -> 可能是误写的"再"
        if i + 1 < n:
            nxt = chars[i + 1]
            common_v = {'吃', '睡', '看', '做', '玩', '跑', '走', '说', '听', '想', '写',
                        '读', '学', '买', '卖', '洗', '刷', '扫', '拖', '倒', '切', '剁',
                        # ── 新增高频通用动词（覆盖用户反馈的试/改/用/搞/弄/聊/查/找/等/修/调/测/编/译/推/拉/开/关/换/拿/放）──
                        '试', '改', '用', '搞', '弄', '聊', '查', '找', '等', '修',
                        '调', '测', '编', '译', '推', '拉', '开', '关', '换', '拿',
                        '放', '跑', '跳', '飞', '游', '爬', '骑', '推', '拉', '买'}
            if nxt in common_v:
                # 向后再看一两字是否有体标记
                if i + 2 < n and chars[i + 2] in ZAI_VERB_ASPECT_SUFFIXES:
                    return True
                if i + 3 < n and text[i + 2:i + 4] in ZAI_VERB_ASPECT_SUFFIXES:
                    return True

    # 3. 无法可靠判断，交给通用三级评分
    return None


# =====================================================================
# jieba 词典初始化
# =====================================================================
_FREQ_REF: Optional[Dict[str, int]] = None


def _get_freq() -> Dict[str, int]:
    global _FREQ_REF
    if _FREQ_REF is None:
        list(jieba.cut('初始化'))
        _FREQ_REF = jieba.dt.FREQ  # type: ignore[attr-defined]
    return _FREQ_REF


# =====================================================================
# 三级评分纠错器
# =====================================================================
TEN_FOLD = 10
TEXT_SCORE_RATIO = 1.15

# =====================================================================
# 可选字符级 bigram 语言模型（2.5 级评分）
# =====================================================================
# 为什么需要它：jieba 词典本质是 unigram LM——只有「单字/成词的频率」，
# 没有「字与字的共现」。用整句 unigram 打分会把 树根 里的 根 误改成
# 高频介词 跟（跟 单字频 >> 根），把 再教 里的 再 毁成 在。而字符级
# bigram 捕捉的是「树+根」「再+教」「飞+掉」这类相邻共现，正是 unigram
# 缺的那层信号。模型训练自干净语料（CGED 2017+2018 CORRECTION），
# 训练脚本 cged/_train_bigram.py，产物 char_bigram.pkl 可选加载：
# 未训练/未加载时 2.5 级自动跳过，行为与旧版完全一致。
BIGRAM_LOG_RATIO = math.log(1.3)  # 2.5 级阈值：候选对数概率比原字高 30%

# 延迟加载的单例（_BIGRAM_LM 为 None = 未启用）
_BIGRAM_LM: Optional['CharBigramLM'] = None
_BIGRAM_LM_TRIED = False


def _get_bigram_lm():
    """懒加载字符级 bigram LM；文件缺失/加载失败返回 None（静默降级）。"""
    global _BIGRAM_LM, _BIGRAM_LM_TRIED
    if _BIGRAM_LM_TRIED:
        return _BIGRAM_LM
    _BIGRAM_LM_TRIED = True
    try:
        from char_bigram import CharBigramLM
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'user_data', 'char_bigram.npz')
        if os.path.exists(path):
            _BIGRAM_LM = CharBigramLM.load(path)
            print(f'[info] [BigramLM] 字符级 bigram LM 已加载: {os.path.getsize(path)//1024} KB')
    except Exception as e:
        print(f'[warn] [BigramLM] 加载失败，2.5 级评分跳过: {e}')
    return _BIGRAM_LM


class ConfusionSetCorrector:
    """同音/形近字混淆集纠错器。API 兼容 CscCorrector。"""

    def __init__(self, threshold: float = 0.0):
        self._loaded = False
        self._warned = False

    def is_available(self) -> bool:
        return self._loaded

    def warmup(self, progress_cb=None) -> bool:
        if self._loaded:
            return True
        try:
            if progress_cb:
                progress_cb('load', '[ConfusionSet/1/1] 加载 jieba 词典...')
            _get_freq()
            self._loaded = True
            msg = '[ConfusionSet] 同音字混淆集纠错器已加载'
            print(f'[info] {msg}')
            if progress_cb:
                progress_cb('done', msg)
            return True
        except Exception as e:
            msg = f'[ConfusionSet] 初始化失败: {e}'
            print(f'[warn] {msg}')
            if progress_cb:
                progress_cb('error', msg)
            return False

    def correct(self, text: str) -> str:
        if not text or not text.strip():
            return text
        if not any(_is_cjk(c) for c in text):
            return text
        # 单字/极短文本无上下文，无法可靠判断在/再等歧义，直接返回
        if len(text) <= 1:
            return text
        if not self._loaded:
            if not self.warmup():
                return text
        try:
            return self._run_correction(text)
        except Exception as e:
            if not self._warned:
                print(f'[warn] [ConfusionSet] 纠错失败（仅提示一次）: {e}')
                self._warned = True
            return text

    # ------------------------------------------------------------------
    # 三级评分
    # ------------------------------------------------------------------
    @staticmethod
    def _best_local_score(chars, i, freq):
        n = len(chars)
        best = 0
        if i > 0:
            best = max(best, freq.get(chars[i - 1] + chars[i], 0))
        if i < n - 1:
            best = max(best, freq.get(chars[i] + chars[i + 1], 0))
        if i > 0 and i < n - 1:
            best = max(best, freq.get(chars[i - 1] + chars[i] + chars[i + 1], 0))
        return best

    @staticmethod
    def _full_text_score(text: str, freq: Dict[str, int], center: int) -> int:
        """对文本中目标字符附近的上下文区域评分，避免高频词淹没信号。"""
        start = max(0, center - 5)
        end = min(len(text), center + 6)
        region = text[start:end]
        return sum(freq.get(w, 0) for w in jieba.lcut(region))

    def _should_replace(self, chars, i, cand, freq, token_freq_at,
                        token_info_at=None, pos_at=None) -> Optional[bool]:
        orig = chars[i]

        # 特殊处理：在/再 语境判定
        if (orig == '在' and cand == '再') or (orig == '再' and cand == '在'):
            decision = _should_replace_zai_zai(chars, i, cand, freq,
                                               token_info_at, pos_at)
            if decision is not None:
                return decision
            # 语境判定无果，继续走通用三级评分

        # 通用「相邻成词即放行」保护：被纠错的原字与左/右相邻字构成词典词且
        # 频率足够高时（可以/以为/所以/比较/版本/安排/觉得…），说明它是合法
        # 词形的一部分，直接放行。依据：混淆集的筛选标准就是「错误侧在上下
        # 文中不构成有效词」，故原字能与邻居成高频词基本必是正确写法。防止
        # 「不过也可以不用 Docker」的 可以 被二级整段评分误改成 可已——
        # 「已」作单字频率极高，会把整段分词总分拉高，掩盖 可以(70958) ≫
        # 可已(0) 的局部事实。
        # 阈值过滤：jieba 词典残留的冷僻异体词（板本/连系/按排/教完…，频率
        # 均为 3）正是要修正的错误侧，不能保护；正确侧最低（才学 83）也在
        # 阈值之上。
        # 在/再 双向均高频成词，但 _should_replace_zai_zai 已先行判定（处所/
        # 进行体/副词/动量），此保护只兜底其未决场景，且仅拦截「原字成词」
        # （错误侧 再家/再这 不成词，不受影响）。
        if (i > 0 and freq.get(chars[i - 1] + orig, 0) >= _ORIG_NEIGHBOR_MIN_FREQ) or (
            i < len(chars) - 1 and freq.get(orig + chars[i + 1], 0) >= _ORIG_NEIGHBOR_MIN_FREQ
        ):
            return False

        # 词级保护：目标字符所在的整词若成词且频率足够（长时间/比较/版本…）
        # 直接放行——比单字 bigram 检查更稳：「长时间」freq=119 是整词，而
        # 「长时」=0，旧 bigram 检查看不到整词，导致 长→常 把 长时间 误改成
        # 常时间（“很多地方都是”的元凶）。只对 ≥2 字词生效：单字（常/长/教…）
        # 频率高但必须留给上下文判定（非长→非常 里 长 是单字 token，不能保护）。
        if token_freq_at.get(i, 0) >= _ORIG_NEIGHBOR_MIN_FREQ:
            return False

        # 成语/低频成词保护：目标字符所在整词是 jieba 认识的词（freq>0，即使
        # 低于 _ORIG_NEIGHBOR_MIN_FREQ——荣幸之至=3/拔苗助长=13/沧海一粟=8
        # 这类成语），且替换为候选后整词不成词（freq=0）→ 保留原文。
        # 这是「词级保护」的低频版：高频词（长时间/比较）走上方直接放行，
        # 低频成语在这里按「候选不成词」判死。对照错误侧：板本→版本、
        # 连系→联系、按排→安排 的候选均成词（版本 3743/联系 9767/安排 7836），
        # 不受影响；非长/飞常/比教/导至 等原文整词 freq=0，同样不触发。
        tok_info = token_info_at.get(i) if token_info_at else None
        if tok_info:
            tok, tok_start = tok_info
            if freq.get(tok, 0) > 0:
                off = i - tok_start
                cand_tok = tok[:off] + cand + tok[off + 1:]
                if freq.get(cand_tok, 0) == 0:
                    return False

        # 一级：局部 bigram/trigram 频率比
        orig_local = self._best_local_score(chars, i, freq)
        chars[i] = cand
        cand_local = self._best_local_score(chars, i, freq)
        chars[i] = orig

        if orig_local > 0 or cand_local > 0:
            orig_eff = orig_local if orig_local > 0 else 1
            cand_eff = cand_local if cand_local > 0 else 1
            ratio = cand_eff / orig_eff
            if ratio >= TEN_FOLD and cand_local > 0:
                return True
            if cand_local > 0 and orig_local == 0:
                return True
            if cand_local == 0 and orig_local > 0:
                return False

        # 二级：分词总频分（只评目标字符附近 11 字符区域）
        text = ''.join(chars)
        orig_total = self._full_text_score(text, freq, center=i)
        chars[i] = cand
        cand_total = self._full_text_score(''.join(chars), freq, center=i)
        chars[i] = orig

        if cand_total > orig_total * TEXT_SCORE_RATIO and cand_total > orig_total:
            return True
        if orig_total > cand_total * TEXT_SCORE_RATIO and orig_total > cand_total:
            return False

        # 2.5 级：字符级 bigram LM（可选加载；1、2 级无结论时的裁决）
        # 用「候选 与 前后字的共现对数概率和」对比原字——树+根/再+教/飞+掉
        # 这类相邻共现是 jieba unigram 词典给不了的信号。
        lm = _get_bigram_lm()
        if lm is not None:
            delta = lm.delta(text, i, orig, cand)
            if delta is not None:
                if delta > BIGRAM_LOG_RATIO:
                    return True
                if delta < -BIGRAM_LOG_RATIO:
                    return False

        return None

# ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _run_correction(self, text: str) -> str:
        # token 级修正：ASR 常把正确字听成同音别字（常时间/因该/在见…）
        text = _fix_token_phrases(text)

        chars = list(text)
        n = len(chars)
        freq = _get_freq()

        # 专名保护：人名/地名/机构名跨度内的字符不允许被同音评分改动
        # （李原 里的 原 不能因 原来/源 的高频语境而改成 李源）
        proper_noun_spans = _detect_proper_noun_spans(text)

        # 词级保护映射：字符索引 → 所在整词的词频（仅 ≥2 字词，单字留给上下文）
        token_freq_at: Dict[int, int] = {}
        # 字符索引 → (所在整词, 整词起始位置)：供成语/低频成词保护用
        # （荣幸之至/拔苗助长 等 freq 3~13 的成语，候选不成词时保留原文）
        token_info_at: Dict[int, tuple] = {}
        for tok, s, e in jieba.tokenize(text):
            if len(tok) >= 2:
                tf = freq.get(tok, 0)
                for idx in range(s, e):
                    token_freq_at[idx] = tf
                    token_info_at[idx] = (tok, s)

        # 词性映射：字符索引 → 所在整词的 jieba 词性（供 在/再 判定区分
        # "双字动词"与"处所名词"——再缴费/再决定 是动词应保护，
        # 再银行/再医院/再公司 是处所名词不能保护，必须由词性区分）。
        pos_at: Dict[int, str] = {}
        _tok_off = 0
        for _tok, _flag in pseg.cut(text):
            for _k in range(len(_tok)):
                pos_at[_tok_off + _k] = _flag
            _tok_off += len(_tok)

        # 保护白名单模式（用于跳过位置）
        PROTECT_PATTERNS = (
            ZAI_ADVERB_WHITELIST | ZAI_PREP_VERB_WHITELIST |
            frozenset({'做主', '作家', '坐下', '座位', '作主', '作文', '作业', '作息', '作风', '作用', '作画', '作曲', '作文'})
        )

        # FORCE_REPLACE 锁定的位置：确定性替换后，概率评分不得再改（防止
        # 二级区域分被 鬼鬼祟(1083) 这类前缀词带偏，把修好的 祟 又翻回 崇）
        force_locked: Set[int] = set()

        for _round in range(5):
            changed = False
            for i in range(n):
                ch = chars[i]

                # 0. 强制替换表：检查以 i 为起点的前向窗口是否匹配错误模式
                # 优先于保护字符检查，允许"一棵"等以受保护字符开头的模式
                force_replaced = False
                force_span = 0
                for err_pattern, correct_pattern in FORCE_REPLACE.items():
                    L = len(err_pattern)
                    if i + L <= n and ''.join(chars[i:i + L]) == err_pattern:
                        for k, cc in enumerate(correct_pattern):
                            if i + k < n:
                                chars[i + k] = cc
                        changed = True
                        force_replaced = True
                        force_span = L
                        break
                if force_replaced:
                    # 锁定本窗口，评分器后续（含下一轮）不得翻改
                    force_locked.update(range(i, i + force_span))
                    continue

                # 0b. FORCE 锁定位置：确定性修正结果不再交给概率评分
                if i in force_locked:
                    continue

                # 1. 专名保护：人名/地名/机构名内部字不改动（优先于普通保护）
                if i in proper_noun_spans:
                    continue

                # 1b. 保护白名单/字符：受保护字符直接跳过
                if not _is_cjk(ch) or ch in _PROTECTED_CHARS:
                    continue

                # 2. 保护白名单：以 i 为中心的窗口若命中保护模式，直接跳过该位置
                protect_window = ''.join(chars[max(0, i - 2):min(n, i + 3)])
                if any(p in protect_window for p in PROTECT_PATTERNS):
                    continue

                candidates = CONFUSION_SETS.get(ch)
                if not candidates:
                    continue
                for cand in candidates:
                    if not _is_cjk(cand):
                        continue
                    # 2. 语境/频率评分判定
                    decision = self._should_replace(chars, i, cand, freq,
                                                   token_freq_at, token_info_at, pos_at)
                    if decision is True:
                        chars[i] = cand
                        changed = True
                        break
                    elif decision is False:
                        continue
            if not changed:
                break
        return ''.join(chars)


# =====================================================================
# 单例
# =====================================================================
_default_corrector: Optional[ConfusionSetCorrector] = None


def get_default_corrector() -> ConfusionSetCorrector:
    global _default_corrector
    if _default_corrector is None:
        _default_corrector = ConfusionSetCorrector()
    return _default_corrector


def warmup_default_corrector(progress_cb=None) -> bool:
    return get_default_corrector().warmup(progress_cb=progress_cb)


# =====================================================================
# CLI
# =====================================================================
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('用法: python confusion_corrector.py <text> [text2 ...]')
        print('或:   python confusion_corrector.py --test')
        sys.exit(1)

    if sys.argv[1] == '--test':
        tests = [
            ('比教→比较', '这个方案比教完善', '这个方案比较完善'),
            ('再→在', '他再这里等着', '他在这里等着'),
            ('板→版', '这个板本有问题', '这个版本有问题'),
            ('非长→非常', '她非长漂亮', '她非常漂亮'),
            ('飞常→非常', '今天飞常热', '今天非常热'),
            ('以→已', '以经来了', '已经来了'),
            ('队→对', '队不起', '对不起'),
            ('连→联', '请连系我', '请联系我'),
            ('按→安', '按排一下', '安排一下'),
            ('棵→颗', '一棵星星', '一颗星星'),
            ('代→带', '代上钱包', '带上钱包'),
            ('至→致', '导至问题', '导致问题'),
            ('决→觉', '我决得可以', '我觉得可以'),
            ('道→到', '做道最好', '做到最好'),
            # 在/再 双向测试
            ('再→在_处所', '他再这里等着', '他在这里等着'),
            ('再→在_处所2', '放再桌子上', '放在桌子上'),
            ('在→再_副词_再说', '我再想想再说', '我再想想再说'),
            ('在→再_副词_再来', '再来一次', '再来一次'),
            ('在→再_副词_再看', '我再看一眼', '我再看一眼'),
            ('在→再_副词_再走', '再走快点', '再走快点'),
            ('在→再_进行体', '他在吃饭', '他在吃饭'),  # 保护"在"作进行体
            ('在→再_处所', '我在家', '我在家'),
            ('在→再_进行体2', '正在写代码', '正在写代码'),
            # 不应误改
            ('正确_在_处所', '他在这里工作', '他在这里工作'),
            ('正确_在_进行体', '我还在想问题', '我还在想问题'),
            ('正确_再_副词', '我再想想', '我再想想'),
            ('正确_再_副词2', '再说一遍', '再说一遍'),
            ('正确_比较', '比较两款产品', '比较两款产品'),
            ('正确_版本', '版本更新了', '版本更新了'),
            ('正确_是非常', '她非常漂亮', '她非常漂亮'),
            ('正确_做好', '我要做好这件事', '我要做好这件事'),
            ('正确_到达', '安全到达', '安全到达'),
            ('正确_联系', '保持联系', '保持联系'),
            ('正确_安排', '我会安排', '我会安排'),
            ('正确_一颗', '一颗珍珠', '一颗珍珠'),
            ('正确_一棵', '一棵大树', '一棵大树'),
            ('正确_已经', '他已经走了', '他已经走了'),
            ('正确_好像', '他好像很忙', '他好像很忙'),
            ('正确_可以', '我觉得可以', '我觉得可以'),
            ('正确_可以2', '不过也可以不用Docker', '不过也可以不用Docker'),  # 二级评分曾把 可以 误改 可已
            ('正确_可以3', '也可以', '也可以'),
            ('正确_可以4', '可以的', '可以的'),
            ('正确_以为', '我以为是这样', '我以为是这样'),
            ('正确_所以', '所以呢', '所以呢'),
            ('正确_以上', '以上内容', '以上内容'),
            ('正确_难以', '难以想象', '难以想象'),
            ('正确_自己', '我自己来', '我自己来'),
            # ===== 通用「相邻成词即放行」：原字与邻居成词的正确词形保持 =====
            ('正确_决策', '我做了个决策', '我做了个决策'),
            ('正确_材料', '这些材料很齐全', '这些材料很齐全'),
            ('正确_资源', '合理利用资源', '合理利用资源'),
            ('正确_根源', '找到问题的根源', '找到问题的根源'),
            ('正确_成长', '他在成长', '他在成长'),
            ('正确_飞翔', '鸟儿在飞翔', '鸟儿在飞翔'),
            ('正确_队长', '他是队长', '他是队长'),
            ('正确_严密', '逻辑很严密', '逻辑很严密'),
            # ===== 错误侧仍按评分修正（原字与邻居不成词）=====
            ('密→蜜_甜密', '甜密的感觉', '甜蜜的感觉'),
            ('才→材_才料', '这批才料', '这批材料'),
            ('原→源_资原', '资原很丰富', '资源很丰富'),
            ('正确_还是', '他还是来了', '他还是来了'),
            ('正确_道', '知道就好', '知道就好'),
            ('正确_事', '没事的', '没事的'),
            # ===== 长/常：长时间 保护 + 常时间→长时间 =====
            # 旧 bug：长→常 词级保护缺失，每个「长时间」都被改成「常时间」
            ('正确_长时间', '长时间没有见面', '长时间没有见面'),
            ('正确_长时间2', '别长时间看手机', '别长时间看手机'),
            ('正确_长时间3', '那么长时间没见了', '那么长时间没见了'),
            ('正确_延长时间', '延长时间', '延长时间'),
            # ASR 把「长时间」听成「常时间」→ 修回（常 为独立 token 或 HMM 粘词）
            ('常时间→长时间', '那么常时间', '那么长时间'),
            ('常时间→长时间2', '我们常时间没见了', '我们长时间没见了'),
            ('常时间→长时间3', '别常时间看手机', '别长时间看手机'),
            ('常时间→长时间4', '常时间没见了', '长时间没见了'),
            # 边界：常 是词素的合法词不受影响
            ('正确_正常时间', '在正常时间内完成', '在正常时间内完成'),
            ('正确_经常时间', '他经常时间观念不强', '他经常时间观念不强'),
            ('正确_异常时间', '异常时间数据', '异常时间数据'),
            ('正确_日常时间', '日常时间安排', '日常时间安排'),
            ('正确_非常时间', '非常时间观念', '非常时间观念'),
            # 非长→非常 仍生效（长 是单字 token，不能被词级保护挡住）
            ('非长→非常', '她非长漂亮', '她非常漂亮'),
            # ===== 同音短语扫描：FORCE_REPLACE（子串级，错误侧绝无合法用法）=====
            ('什末→什么', '你什末时候来', '你什么时候来'),
            ('为什末→为什么', '为什末要这样', '为什么要这样'),
            ('什莫→什么', '这是什莫', '这是什么'),
            ('毕竞→毕竟', '毕竞是第一次', '毕竟是第一次'),
            ('付责→负责', '由你付责', '由你负责'),
            ('决对→绝对', '这决对不行', '这绝对不行'),
            ('绝定→决定', '我绝定去', '我决定去'),
            ('兰球→篮球', '打兰球', '打篮球'),
            ('步署→部署', '完成步署', '完成部署'),
            ('辩别→辨别', '很难辩别', '很难辨别'),
            ('辨论→辩论', '参加辨论', '参加辩论'),
            # ===== 同音短语扫描：token 级（整词粘合/单字+后接/HMM 粘词末尾）=====
            ('因该→应该', '你因该早点来', '你应该早点来'),
            ('既使→即使', '既使下雨也要去', '即使下雨也要去'),
            ('即然→既然', '即然你来了', '既然你来了'),
            ('竞然→竟然', '他竞然走了', '他竟然走了'),
            ('常试→尝试', '你常试一下', '你尝试一下'),
            ('跟本→根本', '跟本没必要', '根本没必要'),
            ('知到→知道', '我知到这个', '我知道这个'),
            ('有于→由于', '有于时间不够', '由于时间不够'),
            ('想信→相信', '我想信你', '我相信你'),
            ('相法→想法', '我的相法', '我的想法'),
            ('级时→及时', '级时赶到', '及时赶到'),
            ('及使→即使', '及使有困难', '即使有困难'),
            ('应响→影响', '不受应响', '不受影响'),
            ('在见→再见', '明天在见', '明天再见'),
            # 同音扫描边界：正确写法绝不能改坏（整词挡开）
            ('保护_原因', '原因很简单', '原因很简单'),
            ('保护_响应', '响应速度很快', '响应速度很快'),
            ('保护_即使', '即使如此', '即使如此'),
            ('保护_既然', '既然这样', '既然这样'),
            ('保护_篮球', '篮球比赛', '篮球比赛'),
            ('保护_绝对', '绝对正确', '绝对正确'),
            ('保护_决定', '决定权在你', '决定权在你'),
            ('保护_相信', '相信你', '相信你'),
            ('保护_想法', '有想法', '有想法'),
            ('保护_感觉', '感觉不错', '感觉不错'),
            ('保护_知道', '他知道这件事', '他知道这件事'),
            ('保护_方向', '方向不对', '方向不对'),
            ('保护_周末', '周末休息', '周末休息'),
            ('保护_试探', '他常试探我', '他常试探我'),
            ('保护_高级时尚', '高级时尚', '高级时尚'),
            ('保护_以及使用', '以及使用方法', '以及使用方法'),
            ('保护_在见到', '在见到他之前', '在见到他之前'),
            ('保护_原因该', '原因该由你决定', '原因该由你决定'),
            # 边界
            ('纯英文', 'Hello World', 'Hello World'),
            ('纯数字', '12345', '12345'),
            ('短文本', '他', '他'),
            ('空文本', '', ''),
            # ===== Phase 2 新增：在/再 动量短语豁免（原漏改 8/8）=====
            ('在看一下→再看一下', '在看一下', '再看一下'),
            ('在想一下→再想一下', '在想一下', '再想一下'),
            ('在试一下→再试一下', '在试一下', '再试一下'),
            ('在改一下→再改一下', '在改一下', '再改一下'),
            ('在用一下→再用一下', '在用一下', '再用一下'),
            ('在搞一下→再搞一下', '在搞一下', '再搞一下'),
            ('在玩一下→再玩一下', '在玩一下', '再玩一下'),
            ('在折腾一下→再折腾一下', '在折腾一下', '再折腾一下'),
            # 保护：进行体/处所不应误改
            ('保护_正在看书', '正在看书', '正在看书'),
            ('保护_在家里', '在家里', '在家里'),
            ('保护_他在写代码', '他在写代码', '他在写代码'),
            # ===== Phase 3 新增：数量追加结构「再/在 + 追加类动词 + 名量短语」→ 再 =====
            # 语法依据：《现代汉语八百词》"再"义项①"表示又一次（多指将发生的）"（例：再唱一个）、
            # 义项⑤"表示另外有所补充"。ASR 常把 再 误听成 在（在 单字频率更高），
            # 且通用三级评分会反向把正确的"再"毁成"在"——本组双向锁定。
            # 用户场景：
            ('在→再_增加_东西', '其实这个在增加一个视频的编辑和动画的东西',
             '其实这个再增加一个视频的编辑和动画的东西'),
            ('再保护_增加_东西', '其实这个再增加一个视频的编辑和动画的东西',
             '其实这个再增加一个视频的编辑和动画的东西'),
            # 再→在：追加类动词无条件保护（曾被通用评分毁成"在"）
            ('再保护_增加一个', '再增加一个功能', '再增加一个功能'),
            ('再保护_数量再增加', '数量再增加', '数量再增加'),
            ('再保护_成本再增加', '成本再增加', '成本再增加'),
            ('再保护_增加无宾语', '再增加投入', '再增加投入'),
            ('再保护_增加带了', '再增加了一个功能', '再增加了一个功能'),
            ('再保护_添加', '再添加一个文件', '再添加一个文件'),
            ('再保护_增设', '再增设一个部门', '再增设一个部门'),
            ('再保护_追加', '再追加一笔投资', '再追加一笔投资'),
            # 在→再：追加类动词 + 名量短语 修正
            ('在→再_增加一个', '在增加一个功能', '再增加一个功能'),
            ('在→再_主语我', '我在增加一个功能', '我再增加一个功能'),
            ('在→再_带体标记', '在增加了一个功能', '再增加了一个功能'),
            ('在→再_添加', '在添加一些内容', '再添加一些内容'),
            ('在→再_增设', '在增设一个部门', '再增设一个部门'),
            ('在→再_追加', '在追加一笔投资', '再追加一笔投资'),
            # 保护：增加 不带宾语是合法持续义，不翻转
            ('保护_数量在增加', '数量在增加', '数量在增加'),
            ('保护_成本在增加', '成本在增加', '成本在增加'),
            ('保护_他在增加投入', '他在增加投入', '他在增加投入'),
            # 再→在：通用双字动词 + 名量短语 保护（白名单救不了的动词族）
            ('再保护_编辑', '再编辑一个视频', '再编辑一个视频'),
            ('再保护_制作', '再制作两个页面', '再制作两个页面'),
            ('再保护_编写', '再编写一个模块', '再编写一个模块'),
            ('再保护_优化', '再优化一个功能', '再优化一个功能'),
            ('再保护_开发', '再开发一个模块', '再开发一个模块'),
            ('再保护_设计', '再设计一个界面', '再设计一个界面'),
            # 再→在：无动词名量短语保护（再来几个义）
            ('再保护_两个', '再两个功能', '再两个功能'),
            ('再保护_几个', '再几个选项', '再几个选项'),
            ('再保护_三个', '再三个方案', '再三个方案'),
            # 保护：时点/时段是 在 的合法介词用法（量词表排除时间量词）
            ('保护_在两点见', '在两点见', '在两点见'),
            ('保护_在三个小时里', '在三个小时里', '在三个小时里'),
            ('在→再_两点不动', '再两点见', '在两点见'),
            # 保护：非追加类双字动词 双向尊重输入（进行体与追加义同等常见）
            ('保护_他在编辑一个视频', '他在编辑一个视频', '他在编辑一个视频'),
            ('保护_他在补充一个方案', '他在补充一个方案', '他在补充一个方案'),
            ('保护_他在吃一碗面', '他在吃一碗面', '他在吃一碗面'),
            # ===== Phase 4 新增：按《现代汉语八百词》义项全量审计后补漏 =====
            # 再+单字动词+名量（common_v 表外动词曾被通用评分毁成"在"）
            ('再保护_唱一个', '再唱一个', '再唱一个'),
            ('再保护_画一幅', '再画一幅', '再画一幅'),
            ('再保护_扔一块', '再扔一块', '再扔一块'),
            ('再保护_拍一张', '再拍一张', '再拍一张'),
            ('再保护_煮一碗', '再煮一碗', '再煮一碗'),
            # 再+双字动词（承接/继续义，jieba 词性验证；处所名词不误保护）
            ('再保护_先报名再缴费', '先报名再缴费', '先报名再缴费'),
            ('再保护_做完再缴费', '做完再缴费', '做完再缴费'),
            ('再保护_然后再缴费', '然后再缴费', '然后再缴费'),
            ('再保护_下班再缴费', '下班再缴费', '下班再缴费'),
            ('再保护_再缴费', '再缴费', '再缴费'),
            ('再保护_再决定', '再决定', '再决定'),
            # 再+形容词+也/都/不过（让步/程度结构）
            ('再保护_再难也要做', '再难也要做', '再难也要做'),
            ('再保护_再贵也要买', '再贵也要买', '再贵也要买'),
            ('再保护_再麻烦也要做', '再麻烦也要做', '再麻烦也要做'),
            ('再保护_再合适不过', '再合适不过', '再合适不过'),
            ('再保护_再清楚不过', '再清楚不过', '再清楚不过'),
            # 处所名词/时间框架的「再」仍应修成「在」（不被新保护误拦）
            ('在→再_银行', '再银行取钱', '在银行取钱'),
            ('在→再_医院', '再医院看病', '在医院看病'),
            ('在→再_公司', '再公司开会', '在公司开会'),
            ('在→再_北京', '再北京工作', '在北京工作'),
            ('在→再_外面', '再外面等着', '在外面等着'),
            ('在→再_房间', '再房间里面', '在房间里面'),
            ('在→再_开会之前', '再开会之前', '在开会之前'),
            ('在→再_出发之前', '再出发之前', '在出发之前'),
            ('在→再_下班之后', '再下班之后', '在下班之后'),
            # ===== Phase 2 新增：做/作 单向（仅为 做为→作为；做工/做事 是规范写法，不再反向纠正）=====
            ('做为→作为', '做为', '作为'),
            # 保护：双向高频词不应误改
            ('保护_做主', '做主', '做主'),
            ('保护_作家', '作家', '作家'),
            # ===== Phase 2 新增：坐/座 单向（坐位→座位、坐落→座落、作下→坐下）=====
            ('坐位→座位', '坐位', '座位'),
            ('坐落→座落', '坐落', '座落'),
            ('作下→坐下', '作下', '坐下'),
            # 保护：双向高频/正向正确
            ('保护_坐下', '坐下', '坐下'),
            ('保护_座位', '座位', '座位'),
            # ===== 专名保护：人名/地名/机构名不被同音评分改坏 =====
            ('专名_李原', '李原自远方来', '李原自远方来'),
            ('专名_陈原来', '陈原来找我', '陈原来找我'),
            ('专名_称谓', '王建国先生来了', '王建国先生来了'),
            ('专名_李总', '李总同意这个方案', '李总同意这个方案'),
            ('专名_地名', '柳溪镇很安静', '柳溪镇很安静'),
            ('专名_机构', '远山科技有限公司发布了产品', '远山科技有限公司发布了产品'),
            ('专名_双姓', '欧阳娜娜唱歌很好听', '欧阳娜娜唱歌很好听'),
            # 伪名/已知词：既不该被误保护，也不该被误改
            ('伪名_王在开会', '王在开会', '王在开会'),
            ('已知词_李子', '李子很好吃', '李子很好吃'),
            ('已知词_上海市', '上海市欢迎你', '上海市欢迎你'),
            # ===== 形近字（未/末、侯/候、幸/辛、拨/拔 等）=====
            ('形近_周未→周末', '周未', '周末'),
            ('形近_期未→期末', '期未', '期末'),
            ('形近_时侯→时候', '时侯', '时候'),
            ('形近_等侯→等候', '等侯', '等候'),
            ('形近_幸苦→辛苦', '幸苦', '辛苦'),
            ('形近_辛福→幸福', '辛福', '幸福'),
            ('形近_拨河→拔河', '拨河', '拔河'),
            ('形近_拔打→拨打', '拔打', '拨打'),
            ('形近_洒店→酒店', '洒店', '酒店'),
            ('形近_喝洒→喝酒', '喝洒', '喝酒'),
            ('形近_枪救→抢救', '枪救', '抢救'),
            ('形近_手抢→手枪', '手抢', '手枪'),
            ('形近_仓白→苍白', '仓白', '苍白'),
            ('形近_仓海→沧海', '仓海', '沧海'),
            ('形近_鬼鬼崇崇→祟祟', '鬼鬼崇崇', '鬼鬼祟祟'),
            ('形近_祟高→崇高', '祟高', '崇高'),
            ('形近_针炙→针灸', '针炙', '针灸'),
            ('形近_肓人→盲人', '肓人', '盲人'),
            ('形近_膏盲→膏肓', '膏盲', '膏肓'),
            ('形近_荼叶→茶叶', '荼叶', '茶叶'),
            ('形近_茶毒→荼毒', '茶毒', '荼毒'),
            ('形近_高梁→高粱', '高梁', '高粱'),
            ('形近_脉博→脉搏', '脉博', '脉搏'),
            ('形近_搏士→博士', '搏士', '博士'),
            ('形近_等侍→等待', '等侍', '等待'),
            ('形近_老帅→老师', '老帅', '老师'),
            ('形近_帅长→师长', '帅长', '师长'),
            ('形近_坟幕→坟墓', '坟幕', '坟墓'),
            ('形近_豆奖→豆浆', '豆奖', '豆浆'),
            ('形近_板粟→板栗', '板粟', '板栗'),
            ('形近_廉刀→镰刀', '廉刀', '镰刀'),
            ('形近_急燥→急躁', '急燥', '急躁'),
            ('形近_干躁→干燥', '干躁', '干燥'),
            ('形近_蓝球→篮球', '蓝球', '篮球'),
            ('形近_篮天→蓝天', '篮天', '蓝天'),
            ('形近_自已→自己', '自已', '自己'),
            ('形近_仅管→尽管', '仅管', '尽管'),
            # 形近保护：正确词形绝不被形近对误改
            ('保形_未来', '未来已来', '未来已来'),
            ('保形_周末', '周末愉快', '周末愉快'),
            ('保形_幸福', '很幸福', '很幸福'),
            ('保形_辛苦', '辛苦了', '辛苦了'),
            ('保形_拔河', '拔河比赛', '拔河比赛'),
            ('保形_酒店', '住酒店', '住酒店'),
            ('保形_抢救', '抢救成功', '抢救成功'),
            ('保形_崇高', '崇高理想', '崇高理想'),
            ('保形_针灸', '去针灸', '去针灸'),
            ('保形_盲人', '盲人学校', '盲人学校'),
            ('保形_茶叶', '喝茶叶', '喝茶叶'),
            ('保形_高粱', '种高粱', '种高粱'),
            ('保形_脉搏', '测脉搏', '测脉搏'),
            ('保形_博士', '博士生', '博士生'),
            ('保形_等待', '耐心等待', '耐心等待'),
            ('保形_老师', '李老师', '李老师'),
            ('保形_坟墓', '扫墓', '扫墓'),
            ('保形_豆浆', '喝豆浆', '喝豆浆'),
            ('保形_板栗', '吃板栗', '吃板栗'),
            ('保形_镰刀', '拿镰刀', '拿镰刀'),
            ('保形_急躁', '性格急躁', '性格急躁'),
            ('保形_篮球', '打篮球', '打篮球'),
            ('保形_蓝天', '蓝天白云', '蓝天白云'),
            ('保形_自己', '靠自己', '靠自己'),
            ('保形_尽管', '尽管试试', '尽管试试'),
            ('保形_洒水车', '洒水车', '洒水车'),
            ('保形_元帅', '元帅', '元帅'),
            ('保形_帅哥', '帅哥', '帅哥'),
            ('保形_栋梁', '栋梁', '栋梁'),
            ('保形_脍炙人口', '脍炙人口', '脍炙人口'),
            ('保形_不仅', '不仅', '不仅'),
            ('保形_已经', '已经', '已经'),
            ('保形_粟米', '粟米', '粟米'),
            ('保形_侍候', '侍候', '侍候'),
            ('保形_侯爵', '侯爵', '侯爵'),
            # fuzz 修复：成语保护（原文整词成词但候选整词不成词 → 不动）
            ('保形_荣幸之至', '荣幸之至', '荣幸之至'),
            ('保形_拔苗助长', '拔苗助长', '拔苗助长'),
            ('保形_沧海一粟', '沧海一粟', '沧海一粟'),
            # fuzz 修复：按 作介词保留（按顺序/按照/按钮），按排→安排 走 FORCE_REPLACE
            ('保形_按顺序', '按顺序排列', '按顺序排列'),
            ('保形_按照', '按照要求执行', '按照要求执行'),
            ('保形_按下按钮', '按下按钮', '按下按钮'),
            # fuzz 修复：教 作动词保留（老师教/教学/请教），比教→比较 走 FORCE_REPLACE
            ('保形_老师教', '老师教我们', '老师教我们'),
            ('保形_教我们', '老师教我们学拼音', '老师教我们学拼音'),
        ]
        print('=== ConfusionSet 纠错测试 ===')
        passed = 0
        failed = 0
        corrector = get_default_corrector()
        corrector.warmup()
        for label, inp, expected in tests:
            result = corrector.correct(inp)
            status = 'PASS' if result == expected else 'FAIL'
            if status == 'PASS':
                passed += 1
            else:
                failed += 1
            print(f'  [{status}] {label}')
            if status == 'FAIL':
                print(f'    input:    {repr(inp)}')
                print(f'    expected: {repr(expected)}')
                print(f'    got:      {repr(result)}')
        print(f'结果: {passed} passed, {failed} failed / {len(tests)} total')
        sys.exit(0 if failed == 0 else 1)

    corrector = get_default_corrector()
    corrector.warmup()
    for line in sys.argv[1:]:
        out = corrector.correct(line)
        print(f'[in]  {line}')
        print(f'[out] {out}')