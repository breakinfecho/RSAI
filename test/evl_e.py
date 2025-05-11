from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
from collections import defaultdict

# 原始文本
candidate = """
There are so many reasons to love Lee's Sandwiches. Among many baked goods and pastries they have baguettes that are freshly baked every 30 minutes, ~1.60 a loaf (or you can get 2 x 1 day old baguettes for ~$2), among other beverages they have fresh sugarcane juice that is absolutely delicious, and also sandwiches made with their fresh bread. Overall everything is super affordable and the staff are really nice, I love having this place so close by and being able to support a local business.
"""

reference = """   Lee's Sandwiches offers a variety of baked goods, including freshly bak
ed baguettes every 30 minutes at a great price. They also have delightful fresh sugarcane juice and sandwiches made with their own fresh bread. The affordability and friendly staff make it a pleasure to support this local business.
 """

# 预处理：移除标点，统一小写，按空格分词
def preprocess(text):
    text = text.lower()
    text = "".join([c if c.isalnum() or c.isspace() else " " for c in text])
    return text.split()


def preprocess(text):
    """文本预处理：小写化、去除非字母数字字符、分词"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # 去除非字母数字字符
    tokens = text.split()  # 简单空格分词
    return tokens


def generate_su4_ngrams(tokens):
    """生成SU4特征：uni-gram + 跳词bi-gram（间隔≤4）"""
    ngrams = defaultdict(int)
    n = len(tokens)

    # 1. 添加uni-gram
    for token in tokens:
        ngrams[(token,)] += 1  # 使用元组作为键

    # 2. 添加跳词bi-gram（最大间隔4）
    for i in range(n):
        for j in range(i + 1, min(i + 5 + 1, n)):  # j ∈ [i+1, i+5]
            bigram = (tokens[i], tokens[j])
            ngrams[bigram] += 1

    return ngrams


def rouge_su4(candidate, reference):
    """计算ROUGE-SU4的Precision, Recall, F1"""
    # 生成n-gram统计
    cand_ngrams = generate_su4_ngrams(preprocess(candidate))
    ref_ngrams = generate_su4_ngrams(preprocess(reference))

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

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }
def cal(candidate, reference):
    candidate_tokens = preprocess(candidate)
    reference_tokens = preprocess(reference)

    # 计算BLEU（仅使用1-gram）
    smooth = SmoothingFunction().method1
    bleu_score = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        weights=(1, 0, 0, 0),  # 仅用1-gram
        smoothing_function=smooth
    )

    # 计算ROUGE（需将列表转为空格分隔的字符串）
    rouge = Rouge()
    rouge_scores = rouge.get_scores(
        " ".join(candidate_tokens),
        " ".join(reference_tokens),
        avg=True
    )
    # ROUGE-SU4
    su4 = rouge_su4(candidate, reference)

    # 结果输出
    results = {
        "BLEU": round(bleu_score, 3),
        "ROUGE-1": round(rouge_scores["rouge-1"]["f"], 3),
        "ROUGE-2": round(rouge_scores["rouge-2"]["f"], 3),
        "ROUGE-L": round(rouge_scores["rouge-l"]["f"], 3),
        "ROUGE-SU4": su4['f1'],
    }

    # print("Metrics:")
    # for metric, value in results.items():
    #     print(f"{metric}: {value:.3f}")

    return results


'''
              V3         GLM         QWen        R1
BLEU:        0.303      0.337       0.262       0.180
ROUGE-1:     0.414      0.384       0.312       0.224
ROUGE-2:     0.088      0.193       0.048       0.029
ROUGE-L:     0.288      0.320       0.220       0.138
'''