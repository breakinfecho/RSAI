import jieba
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict
import sys
sys.setrecursionlimit(2000)  # 增加递归深度限制
# 原始文本
candidate = "餐馆的地点完全决定了档次，东西还是比较辣这家店开在国金所以环境还算不错可能算是新开，服务人员有点跟不上，上菜速度慢。"
reference = " 这家餐馆位于国金中心，地理位置优越，环境还算不错。菜品口味偏辣，符合一部分人的喜好。虽然服务人员有点跟不上，上菜速度较慢，但整体来看，这些问题可能是新开业时的磨合期问题，有待改进。综合考虑，餐馆的整体表现还是可以的。"


# 中文分词
def tokenize_chinese(text):
    return list(jieba.cut(text))  # 精确模式分词


def generate_su4_ngrams(tokens):
    """生成SU4特征：uni-gram + 跳词bi-gram（间隔≤4）"""
    ngrams = defaultdict(int)
    n = len(tokens)

    # 1. 添加uni-gram
    for token in tokens:
        ngrams[(token,)] += 1

    # 2. 添加跳词bi-gram（最大间隔4）
    for i in range(n):
        max_j = min(i + 5, n)  # j ∈ [i+1, i+4]
        for j in range(i + 1, max_j):
            bigram = (tokens[i], tokens[j])
            ngrams[bigram] += 1

    return ngrams


def calculate_rouge_su4(candidate, reference):
    """计算ROUGE-SU4的F1值"""
    # 生成n-gram统计
    cand_ngrams = generate_su4_ngrams(candidate)
    ref_ngrams = generate_su4_ngrams(reference)

    # 统计匹配的n-gram数量
    overlap = 0
    for gram, count in cand_ngrams.items():
        overlap += min(count, ref_ngrams.get(gram, 0))

    # 计算分母
    cand_total = sum(cand_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    # 避免除以零
    precision = overlap / cand_total if cand_total > 0 else 0
    recall = overlap / ref_total if ref_total > 0 else 0

    # 计算F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return round(f1, 3)


def calc(candidate, reference):
    candidate_tokens = tokenize_chinese(candidate)
    reference_tokens = tokenize_chinese(reference)

    # 计算BLEU（使用1-gram）
    smooth = SmoothingFunction().method1
    bleu_score = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        weights=(1, 0, 0, 0),  # 仅用1-gram
        smoothing_function=smooth
    )

    # 计算ROUGE（需转为空格分隔的字符串）
    rouge = Rouge()
    rouge_scores = rouge.get_scores(
        " ".join(candidate_tokens),
        " ".join(reference_tokens),
        avg=True
    )
    rouge_su4_score = calculate_rouge_su4(candidate_tokens, reference_tokens)
    # 输出结果
    results = {
        "BLEU": round(bleu_score, 3),
        "ROUGE-1": round(rouge_scores["rouge-1"]["f"], 3),
        "ROUGE-2": round(rouge_scores["rouge-2"]["f"], 3),
        "ROUGE-L": round(rouge_scores["rouge-l"]["f"], 3),
        "ROUGE-SU4": rouge_su4_score  # 新增SU4计算结果
    }

    return results

