from typing import Any, Dict, List

from verl.utils.reward_score.math_parsing_util import extract_answer, math_equal
import os
from verl.utils.reward_score.base import Scorer
import re
from langdetect import detect_langs, DetectorFactory
import random
from verl.utils.reward_score.repeat import check_repetition_valid

LANGUAGE_MAP = {
    "ZH": ["zh-cn", "zh"],
    "FR": ["fr"],
    "BN": ["bn"],
    "EN": ["en"],
    "DE": ["de"],
    "JA": ["ja"]
}
try:
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify
except ImportError:
    # print("math_verify is not installed")
    mv_parse = None
    mv_verify = None


class MathEqualScorer(Scorer):
    """Scorer for math based on the `math_equal` function from Qwen Math

    Args:
        response_column: The column name for the model generated response. (str)
        answer_column: The column name for the ground truth answer. (str)
    """

    SCORE_COLUMN = "math_equal_score"

    def __init__(self, response_column: str, answer_column: str):
        self.response_column = response_column
        self.answer_column = answer_column

    def score(self, row: dict) -> Dict[str, Any]:
        try:
            # print(row[self.response_column])
            pred = extract_answer(row[self.response_column])
            # print(pred)
            ref = extract_answer(row[self.answer_column])
            # print(pred, ref)
        except Exception:
            # print("-----------")
            return False
        return {self.SCORE_COLUMN: math_equal(pred, ref)}

    @property
    def expected_keys(self) -> List[str]:
        return [self.response_column, self.answer_column]


class MathVerifyScorer(Scorer):
    """Scorer for math based on the `math_verify` function from HuggingFace

    Args:
        response_column: The column name for the model generated response. (str)
        answer_column: The column name for the ground truth answer. (str)
    """

    SCORE_COLUMN = "math_verify_score"

    def __init__(self, response_column: str, answer_column: str):
        self.response_column = response_column
        self.answer_column = answer_column
        if mv_parse is None or mv_verify is None:
            raise ImportError(
                "`math_verify` is not installed. Please install it with `pip install math_verify`."
            )

    def score(self, row: dict) -> Dict[str, Any]:
        try:
            # print(self.response_column)
            # print(row[self.response_column])
            pred = mv_parse(row[self.response_column])
            # print(pred)
            ref = mv_parse(row[self.answer_column])
            # print("pred", pred)
            # print("ref", ref)
            # print(pred, ref)
        except Exception as e:
            # print("-----------")
            # print(e)
            return False
        return {self.SCORE_COLUMN: mv_verify(pred, ref)}

    @property
    def expected_keys(self) -> List[str]:
        return [self.response_column, self.answer_column]

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
    cleaned_text = text
    if not cleaned_text.strip():
        # 空文本给予中性奖励，不鼓励也不惩罚
        result = 0.0
    elif len(cleaned_text.strip()) < 5:
        # 对于极短文本，langdetect可能不可靠
        # 我们给予较低的置信度，避免过度奖励或惩罚短文本
        result = 0.1
    else:
        # 第三步：使用langdetect进行语言检测
        try:
            result = detect_language_probability(cleaned_text, language_code)
        except Exception as e:
            # 如果检测失败，返回中性奖励
            # 这确保训练过程不会因为异常而中断
            result = 0.0
    return result

def get_language_probability_strict_qwen3_origin(text, language_code):
    cleaned_text = text
    if not cleaned_text.strip():
        # 空文本给予中性奖励，不鼓励也不惩罚
        result = 0.0
    elif len(cleaned_text.strip()) < 5:
        # 对于极短文本，langdetect可能不可靠
        # 我们给予较低的置信度，避免过度奖励或惩罚短文本
        result = 0.1
    else:
        # 第三步：使用langdetect进行语言检测
        try:
            result = detect_language_probability(cleaned_text, language_code)
        except Exception as e:
            # 如果检测失败，返回中性奖励
            # 这确保训练过程不会因为异常而中断
            result = 0.0
    return result

def compute_score_strict_language_reward_qwen3(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
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
    if "</think>" not in solution_str:
        # 如果不存在总结部分直接返回0分
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
    think_str = solution_str.split('</think>')[0].strip().strip("<think>").strip()
    summary_str = solution_str.split('</think>')[1].strip()
    extract_boxed_answer = extract_last_boxed(summary_str)
    # print(f"ground_truth: {ground_truth}")
    # print(f"extract_boxed_answer: {extract_boxed_answer}")
    
    # 安全地解析 ground_truth
    ground_truth_parts = ground_truth.split("---")
    if len(ground_truth_parts) < 3:
        # 如果格式不正确，返回错误
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
            'language_code': 'EN'
        }
    language_code = ground_truth_parts[0]
    ground_truth = ground_truth_parts[2]
    ground_truth = f"${ground_truth}$"
    use_all_en_think = os.getenv("USE_ALL_EN_THINK")
    if use_all_en_think and use_all_en_think.lower() in ["true", "1"]:
        # print("--------------使用英文think---------------------------")
        language_code = "EN"

    format_reward = 0.1
    accuracy_reward = 0.0
    language_weight = 0.1
    if extract_boxed_answer is None:
        format_reward = 0.0
    
    # 安全地计算数学验证分数
    try:
        math_verify_score = MathVerifyScorer(response_column="response", answer_column="answer")
        is_correct = math_verify_score.score({"response": solution_str, "answer": ground_truth})["math_verify_score"]
        if is_correct:
            accuracy_reward = 0.8
    except Exception as e:
        print(f"Math verify error: {e}")
        accuracy_reward = 0.0
    
    use_en_think = os.getenv("USE_EN_THINK")
    think_language_code = language_code
    if use_en_think and use_en_think.lower() in ["true", "1"]:
        # print("--------------使用英文think---------------------------")
        think_language_code = "EN"
    think_language_probability = get_language_probability_strict_qwen3_origin(think_str, think_language_code)
    summary_language_probability = get_language_probability_strict_qwen3_origin(summary_str, language_code)
    min_language_probability = min(think_language_probability, summary_language_probability)
    if min_language_probability < 0.99:
        language_probability = 0.0
    else:
        language_probability = 1.0
    # if random.random() < 0.05:
    #     print("=========================================================")
    #     print(f"solution_str: {solution_str[-50:]}")
    #     print(f"ground_truth: {ground_truth}")
    #     print(f"当前的language weight: {language_weight}, language_code: {language_code}")
    #     print(f"accuracy_reward: {accuracy_reward}, format_reward: {format_reward}, language_probability: {min_language_probability}")
    #     print("=========================================================")
    
    language_reward = language_probability * language_weight
    total_score = accuracy_reward + format_reward + language_reward

    return {
        'score': total_score,
        'acc': 1 if accuracy_reward > 0 else 0,
        'format_correctness': 1 if format_reward > 0 else 0,
        'pred': extract_boxed_answer,
        'accuracy_reward': accuracy_reward,
        'format_reward': format_reward,
        'language_reward': language_reward,
        'language_probability': min_language_probability,
        'language_weight': language_weight,
        'language_code': language_code
    }

def compute_score_strict_language_reward_qwen3_minus(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
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
    if "</think>" not in solution_str:
        # 如果不存在总结部分直接返回0分
        return {
            'score': -1.0,
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
    
    think_str = solution_str.split('</think>')[0].strip().strip("<think>").strip()
    summary_str = solution_str.split('</think>')[1].strip()
    extract_boxed_answer = extract_last_boxed(summary_str)
    # print(f"ground_truth: {ground_truth}")
    # print(f"extract_boxed_answer: {extract_boxed_answer}")
    
    # 安全地解析 ground_truth
    ground_truth_parts = ground_truth.split("---")
    if len(ground_truth_parts) < 3:
        # 如果格式不正确，返回错误
        return {
            'score': -1.0,
            'acc': 0,
            'format_correctness': 0,
            'pred': None,
            'accuracy_reward': 0,
            'format_reward': 0,
            'language_reward': 0,
            'language_probability': 0,
            'language_weight': 0,
            'language_code': 'EN'
        }
    language_code = ground_truth_parts[0]
    ground_truth = ground_truth_parts[2]
    ground_truth = f"${ground_truth}$"
    use_all_en_think = os.getenv("USE_ALL_EN_THINK")
    if use_all_en_think and use_all_en_think.lower() in ["true", "1"]:
        # print("--------------使用英文think---------------------------")
        language_code = "EN"
    format_reward = 0.1
    accuracy_reward = 0.0
    language_weight = 0.1
    if extract_boxed_answer is None:
        format_reward = 0.0
    
    # 安全地计算数学验证分数
    try:
        math_verify_score = MathVerifyScorer(response_column="response", answer_column="answer")
        is_correct = math_verify_score.score({"response": solution_str, "answer": ground_truth})["math_verify_score"]
        if is_correct:
            accuracy_reward = 0.8
    except Exception as e:
        print(f"Math verify error: {e}")
        accuracy_reward = 0.0
    
    use_en_think = os.getenv("USE_EN_THINK")
    think_language_code = language_code
    if use_en_think and use_en_think.lower() in ["true", "1"]:
        # print("--------------使用英文think---------------------------")
        think_language_code = "EN"
    think_language_probability = get_language_probability_strict_qwen3_origin(think_str, think_language_code)
    summary_language_probability = get_language_probability_strict_qwen3_origin(summary_str, language_code)
    min_language_probability = min(think_language_probability, summary_language_probability)
    if min_language_probability < 0.99:
        language_probability = 0.0
    else:
        language_probability = 1.0
    # if random.random() < 0.05:
    #     print("=========================================================")
    #     print(f"solution_str: {solution_str[-50:]}")
    #     print(f"ground_truth: {ground_truth}")
    #     print(f"当前的language weight: {language_weight}, language_code: {language_code}")
    #     print(f"accuracy_reward: {accuracy_reward}, format_reward: {format_reward}, language_probability: {min_language_probability}")
    #     print("=========================================================")
    
    language_reward = language_probability * language_weight
    acc_score = 1 if accuracy_reward > 0 else 0
    language_score = 0 if language_reward > 0 else -1
    total_score = accuracy_reward + language_reward

    return {
        'score': total_score,
        'acc': 1 if accuracy_reward > 0 else 0,
        'format_correctness': 1 if format_reward > 0 else 0,
        'pred': extract_boxed_answer,
        'accuracy_reward': accuracy_reward,
        'format_reward': format_reward,
        'language_reward': language_reward,
        'language_probability': min_language_probability,
        'language_weight': language_weight,
        'language_code': language_code
    }

# 写一个函数提取solution_str中有多少个\boxed{}，如果summary部分超过1个则奖励直接为0
import re

def count_boxed(solution_str: str) -> int:
    """
    统计字符串中出现了多少个 \boxed{...} 结构
    """
    pattern = r'\\boxed\s*\{[^{}]*\}'
    matches = re.findall(pattern, solution_str)
    return len(matches)

def compute_score_strict_language_reward_qwen3_language_important(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
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
    if "</think>" not in solution_str:
        # 如果不存在总结部分直接返回0分
        return {
            'score': -1.0,
            'acc': -1.0,
            'format_correctness': 0,
            'pred': None,
            'accuracy_reward': 0,
            'format_reward': 0,
            'language_reward': 0,
            'language_probability': 0,
            'language_weight': 0,
            'language_code': 'EN'  # 默认语言代码应该是字符串
        }
    think_str = solution_str.split('</think>')[0].strip().strip("<think>").strip()
    summary_str = solution_str.split('</think>')[1].strip()
    extract_boxed_answer = extract_last_boxed(summary_str)
    # print(f"ground_truth: {ground_truth}")
    # print(f"extract_boxed_answer: {extract_boxed_answer}")
    
    # 安全地解析 ground_truth
    ground_truth_parts = ground_truth.split("---")
    language_code = ground_truth_parts[0]
    ground_truth = ground_truth_parts[2]
    ground_truth = f"${ground_truth}$"

    format_reward = 0.1
    accuracy_reward = 0.0
    language_weight = 0.1
    if extract_boxed_answer is None:
        format_reward = 0.0
    
    # 安全地计算数学验证分数
    try:
        math_verify_score = MathVerifyScorer(response_column="response", answer_column="answer")
        is_correct = math_verify_score.score({"response": solution_str, "answer": ground_truth})["math_verify_score"]
        if is_correct:
            accuracy_reward = 0.8
    except Exception as e:
        print(f"Math verify error: {e}")
        accuracy_reward = 0.0
    
    use_en_think = os.getenv("USE_EN_THINK")
    think_language_code = language_code
    if use_en_think and use_en_think.lower() in ["true", "1"]:
        # print("--------------使用英文think---------------------------")
        think_language_code = "EN"
    think_language_probability = get_language_probability_strict_qwen3_origin(think_str, think_language_code)
    summary_language_probability = get_language_probability_strict_qwen3_origin(summary_str, language_code)
    min_language_probability = min(think_language_probability, summary_language_probability)
    if min_language_probability < 0.95:
        language_probability = 0
    else:
        language_probability = 1
    # if random.random() < 0.05:
    #     print("=========================================================")
    #     print(f"solution_str: {solution_str[-50:]}")
    #     print(f"ground_truth: {ground_truth}")
    #     print(f"当前的language weight: {language_weight}, language_code: {language_code}")
    #     print(f"accuracy_reward: {accuracy_reward}, format_reward: {format_reward}, language_probability: {min_language_probability}")
    #     print("=========================================================")
    
    language_reward = language_probability * language_weight
    if language_probability == 0:
        accuracy_reward_final = 0
        language_score = 0
    else:
        accuracy_reward_final = 0.9 if accuracy_reward > 0 else 0
        language_score = 0.1
    # 检查重复：check_repetition_valid 返回 True 表示无重复（有效），False 表示有重复（无效）
    has_repetition = check_repetition_valid(solution_str)
    if not has_repetition:  # 如果有重复（返回False），则将奖励设为0
        accuracy_reward_final = 0
    # boxed_count = max(count_boxed(summary_str), 0)
    # if boxed_count > 1:
    #     accuracy_reward_final = 0
    # acc_score = 1 if accuracy_reward > 0 else 0
    total_score = accuracy_reward_final + language_score

    return {
        'score': total_score,
        'acc': 1 if total_score > 0.1 else 0,
        'format_correctness': 1,
        'pred': extract_boxed_answer,
        'accuracy_reward': accuracy_reward,
        'format_reward': format_reward,
        'language_reward': language_reward,
        'language_probability': min_language_probability,
        'language_weight': language_weight,
        'language_code': language_code
    }

if __name__ == "__main__":
    math_verify_score = MathVerifyScorer(response_column="response", answer_column="answer")
    solution_str = """最終的に簡略化された形式は：$$\\boxed{6\pi}$$"""
    ground_truth = "$6\pi$"
    is_correct = math_verify_score.score({"response": solution_str, "answer": ground_truth})["math_verify_score"]
    print("Is correct: ", is_correct)