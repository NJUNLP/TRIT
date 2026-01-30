"""
使用 math_verify 库判断数学答案是否一致的工具函数
与 MMATH 评估方式完全一致，并支持语言奖励计算
"""

import re
import os

try:
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify
except ImportError:
    mv_parse = None
    mv_verify = None

try:
    from langdetect import detect_langs
except ImportError:
    detect_langs = None

# 支持直接运行和作为模块导入两种方式
try:
    from .mmath_utils import math_postprocess_v2
except ImportError:
    from mmath_utils import math_postprocess_v2
from verl.utils.reward_score.repeat import check_repetition_valid

# 语言代码映射
LANGUAGE_MAP = {
    "ZH": ["zh-cn", "zh"],
    "FR": ["fr"],
    "BN": ["bn"],
    "EN": ["en"],
    "DE": ["de"],
    "JA": ["ja"],
    "ES": ["es"],
    "AR": ["ar"],
    "TH": ["th"],
    "KO": ["ko"],
    "PT": ["pt"],
    "VI": ["vi"]
}



def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(0)
    return None


def detect_language_probability(text, language_code):
    """
    使用langdetect检测指定语言的概率
    """
    language_probabilities = detect_langs(text)
    target_language_code = LANGUAGE_MAP[language_code]
    total_probability = 0.0
    for lang_detection in language_probabilities:
        lang_code = lang_detection.lang
        probability = lang_detection.prob
        if lang_code in target_language_code:
            total_probability += probability
    return total_probability


def get_language_probability_strict(text, language_code):
    """
    严格的语言概率检测，处理边界情况
    
    Args:
        text: 要检测的文本
        language_code: 目标语言代码
    
    Returns:
        float: 语言概率（0.0 - 1.0）
    """
    cleaned_text = text
    
    if not cleaned_text.strip():
        # 空文本给予中性奖励，不鼓励也不惩罚
        return 0.0
    elif len(cleaned_text.strip()) < 5:
        # 对于极短文本，langdetect 可能不可靠
        # 给予较低的置信度，避免过度奖励或惩罚短文本
        return 0.1
    else:
        # 使用 langdetect 进行语言检测
        try:
            return detect_language_probability(cleaned_text, language_code)
        except Exception:
            # 如果检测失败，返回中性奖励
            # 这确保训练过程不会因为异常而中断
            return 0.0


def is_answer_correct(prediction: str, ground_truth: str) -> bool:
    """
    判断预测答案是否与标准答案一致（使用 MMATH 的方式）
    
    Args:
        prediction: 预测的答案文本（可以是完整的回答，会自动提取答案）
        ground_truth: 标准答案（格式如 "$42$" 或 "42"）
    
    Returns:
        bool: 答案是否正确
        
    Example:
        >>> is_answer_correct("The answer is \\boxed{42}", "$42$")
        True
        >>> is_answer_correct("<think>...</think>\\boxed{6\\pi}", "$6\\pi$")
        True
    """
    if mv_parse is None or mv_verify is None:
        raise ImportError(
            "`math_verify` is not installed. Please install it with `pip install math_verify`."
        )
    
    try:
        # 使用 MMATH 的方式提取预测答案
        pred_answer = math_postprocess_v2(prediction)
        if pred_answer is None:
            return False
        
        # 确保标准答案格式正确（带美元符号）
        if not ground_truth.startswith('$'):
            ground_truth = f'${ground_truth}$'
        
        # 解析并验证
        gold = mv_parse(ground_truth)
        pred = mv_parse(f'${pred_answer}$')
        
        return mv_verify(gold, pred)
    
    except Exception as e:
        # 验证失败视为答案错误
        return False

def compute_score_strict_language_reward_language_first(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # 这个奖励函数只关注了语言一致性和答案正确性，只有语言一致的情况下才会获得答案正确性奖励
    # 特殊的一些环境变量
    # ONLY_ACCURACY_REWARD: 只关注答案正确性，不关注语言一致性
    # USE_ALL_EN_THINK: 思考和总结部分都使用英文
    # USE_EN_THINK: 思考使用英文，总结使用原始语言
    if "</think>" not in solution_str or solution_str.count("</think>") > 1:
        # 如果不存在总结部分直接返回0分或者存在多个总结部分直接返回0分
        return {
            'score': 0,
            'acc': 0,
            'format_correctness': 0,
            'pred': None,
            'accuracy_reward': 0,
            'format_reward': 0,
            'language_reward': 0,
            'language_probability': 0,
            'language_weight': 0,
            'language_code': 'EN'  # 默认语言代码应该是字符串
        }
    stage_one_en = os.getenv("STAGE_ONE_EN")
    stage_one_use_en = False
    if stage_one_en and stage_one_en.lower() in ["true", "1"]:
        first_sentence = solution_str.split("\n\n")[0]
        if detect_language_probability(first_sentence, "EN") > 0.8:
            # language_code = "EN" # 我们认为现在处于阶段一，使用英文回答
            stage_one_use_en = True
    # 判断第一句话是不是英文
    think_str = solution_str.split('</think>')[0].strip().strip("<think>").strip()
    summary_str = solution_str.split('</think>')[1].strip()
    extract_boxed_answer = extract_last_boxed(summary_str)

    
    # 安全地解析 ground_truth
    ground_truth_parts = ground_truth.split("---")
    language_code = ground_truth_parts[0]
    ground_truth = ground_truth_parts[2]
    ground_truth = f"${ground_truth}$"
    use_all_en_think = os.getenv("USE_ALL_EN_THINK")
    if use_all_en_think and use_all_en_think.lower() in ["true", "1"]:
        # print("--------------使用英文think---------------------------")
        language_code = "EN"
    if stage_one_use_en:
        language_code = "EN"
    # 安全地计算数学验证分数
    is_correct = False
    try:
        is_correct = is_answer_correct(summary_str, ground_truth)
    except Exception as e:
        print(f"Math verify error: {e}")
    
    use_en_think = os.getenv("USE_EN_THINK")
    think_language_code = language_code
    if use_en_think and use_en_think.lower() in ["true", "1"]:
        # print("--------------使用英文think---------------------------")
        think_language_code = "EN"
    think_language_probability = get_language_probability_strict(think_str, think_language_code)
    summary_language_probability = get_language_probability_strict(summary_str, language_code)
    min_language_probability = min(think_language_probability, summary_language_probability)
    language_probability = 0
    if min_language_probability < 0.99:
        language_probability = 0
    else:
        language_probability = 1

    final_score = 0
    if language_probability > 0:
        final_score = 0.1
        if is_correct:
            final_score = 1.0
    
    only_accuracy_reward = os.getenv("ONLY_ACCURACY_REWARD")
    if only_accuracy_reward and only_accuracy_reward.lower() in ["true", "1"]:
        if is_correct:
            final_score = 1.0
        else:
            final_score = 0.0
    
    is_check_repetition = os.getenv("CHECK_REPETITION")
    if is_check_repetition and is_check_repetition.lower() in ["true", "1"]:
        has_repetition = check_repetition_valid(solution_str)
        if not has_repetition:
            if language_probability > 0:
                final_score = 0.1
            else:
                final_score = 0.0
    
    is_slc_reward = os.getenv("USE_SLC_REWARD")
    if is_slc_reward and is_slc_reward.lower() in ["true", "1"]:
        final_score = 0
        if is_correct:
            final_score += 0.9
        if language_probability > 0:
            final_score += 0.1

    
    return {
        'score': final_score,
        'acc': 1 if is_correct else 0,
        'format_correctness': 1 if extract_boxed_answer is not None else 0,
        'pred': extract_boxed_answer,
        'accuracy_reward': 1 if is_correct else 0,
        'format_reward': 1 if extract_boxed_answer is not None else 0,
        'language_reward': 1 if language_probability > 0 else 0,
        'language_probability': min_language_probability,
        'language_weight': 0.1,
        'language_code': language_code
    }


def compute_score_strict_language_reward_language_first_only_summary(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # 这个奖励函数只关注了语言一致性和答案正确性，只有语言一致的情况下才会获得答案正确性奖励
    # 特殊的一些环境变量
    # CHECK_REPETITION: 检查重复内容
    # USE_SLC_REWARD: 使用分离的奖励机制（答案0.9 + 语言0.1）
    if "</think>" not in solution_str or solution_str.count("</think>") > 1:
        # 如果不存在总结部分直接返回0分或者存在多个总结部分直接返回0分
        return {
            'score': 0,
            'acc': 0,
            'format_correctness': 0,
            'pred': None,
            'accuracy_reward': 0,
            'format_reward': 0,
            'language_reward': 0,
            'language_probability': 0,
            'language_weight': 0,
            'language_code': 'EN'  # 默认语言代码应该是字符串
        }
    
    summary_str = solution_str.split('</think>')[1].strip()
    extract_boxed_answer = extract_last_boxed(summary_str)

    # 安全地解析 ground_truth
    ground_truth_parts = ground_truth.split("---")
    language_code = ground_truth_parts[0]
    ground_truth = ground_truth_parts[2]
    ground_truth = f"${ground_truth}$"

    # 安全地计算数学验证分数
    is_correct = False
    try:
        is_correct = is_answer_correct(summary_str, ground_truth)
    except Exception as e:
        print(f"Math verify error: {e}")
    
    summary_language_probability = get_language_probability_strict(summary_str, language_code)
    language_probability = 1 if summary_language_probability >= 0.99 else 0

    final_score = 0
    if language_probability > 0:
        final_score = 0.1
        if is_correct:
            final_score = 1.0
    
    is_check_repetition = os.getenv("CHECK_REPETITION")
    if is_check_repetition and is_check_repetition.lower() in ["true", "1"]:
        has_repetition = check_repetition_valid(solution_str)
        if not has_repetition:
            if language_probability > 0:
                final_score = 0.1
            else:
                final_score = 0.0
    
    is_slc_reward = os.getenv("USE_SLC_REWARD")
    if is_slc_reward and is_slc_reward.lower() in ["true", "1"]:
        final_score = 0
        if is_correct:
            final_score += 0.9
        if language_probability > 0:
            final_score += 0.1

    return {
        'score': final_score,
        'acc': 1 if is_correct else 0,
        'format_correctness': 1 if extract_boxed_answer is not None else 0,
        'pred': extract_boxed_answer,
        'accuracy_reward': 1 if is_correct else 0,
        'format_reward': 1 if extract_boxed_answer is not None else 0,
        'language_reward': 1 if language_probability > 0 else 0,
        'language_probability': summary_language_probability,
        'language_weight': 0.1,
        'language_code': language_code
    }



if __name__ == "__main__":
    # 测试示例
    print("测试1:", is_answer_correct("The answer is \\boxed{42}", "$42$"))
    print("测试2:", is_answer_correct("最終的に簡略化された形式は：$$\\boxed{6\\pi}$$", "$6\\pi$"))
    print("测试3:", is_answer_correct("Wrong answer \\boxed{100}", "$42$"))
