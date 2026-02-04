# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated
import os


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    use_en_reasoning_reward = os.environ.get("USE_EN_REASONING_REWARD", "0") == "1"
    if use_en_reasoning_reward:
        from . import math_ljx_final
        res = math_ljx_final.compute_score_strict_language_reward_language_first_only_summary(solution_str, ground_truth)
        return res
    # 优先级最高：显式使用代码奖励（通过 extra_info 传入）
    use_code_reward = os.environ.get("USE_CODE_REWARD", "0") == "1"
    # if isinstance(extra_info, dict):
    #     use_code_reward = bool(extra_info.get("use_code_reward", False))

    if use_code_reward:
        from . import code_verification

        # code_verification 接口是纯 code + tests，不依赖 data_source
        return code_verification.compute_code_score(solution_str, ground_truth)

    use_math_ljx_final = os.environ.get("USE_MATH_LJX_FINAL", "0") == "1"
    if use_math_ljx_final:
        use_mt_ljx_final = os.environ.get("USE_MT_LJX_FINAL", "0") == "1"
        if use_mt_ljx_final:
            from . import mt_ljx_final
            res = mt_ljx_final.compute_score(solution_str, ground_truth)
        else:
            from . import math_ljx_final
            res = math_ljx_final.compute_score_strict_language_reward_language_first(solution_str, ground_truth)
    else:
        # print("shibaile")
        if data_source == "openai/gsm8k":
            from . import gsm8k

            res = gsm8k.compute_score(solution_str, ground_truth)
        elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
            from . import math

            res = math.compute_score(solution_str, ground_truth)
            # [Optional] Math-Verify Integration
            # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
            # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
            # To use it, override the `compute_score` function with the following implementation:

            # from . import math_verify
            # res = math_verify.compute_score(solution_str, ground_truth)
        elif data_source == "math_dapo" or data_source.startswith("aime"):
            from . import math_dapo

            res = math_dapo.compute_score(solution_str, ground_truth)
        elif data_source == "math_ljx_strict_language_reward":
            from . import math_ljx
            use_minus = os.environ.get("MATH_LJX_STRICT_LANGUAGE_REWARD_USE_MINUS", "0") == "1"
            use_language_important = os.environ.get("MATH_LJX_STRICT_LANGUAGE_REWARD_USE_LANGUAGE_IMPORTANT", "0") == "1"
            if use_language_important:
                res = math_ljx.compute_score_strict_language_reward_qwen3_language_important(solution_str, ground_truth)
            elif use_minus:
                res = math_ljx.compute_score_strict_language_reward_qwen3_minus(solution_str, ground_truth)
            else:
                res = math_ljx.compute_score_strict_language_reward_qwen3(solution_str, ground_truth)
        elif data_source in [
            "numina_aops_forum",
            "numina_synthetic_math",
            "numina_amc_aime",
            "numina_synthetic_amc",
            "numina_cn_k12",
            "numina_olympiads",
        ]:
            from . import prime_math

            res = prime_math.compute_score(solution_str, ground_truth)
        elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
            # Use the passed sandbox_fusion_url if available
            if sandbox_fusion_url:
                from . import sandbox_fusion

                # Pass the URL directly, ground_truth likely contains test cases here
                res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True)
            else:
                # If no sandbox URL is provided, fall back to prime_code or raise error
                from . import prime_code

                # Assuming prime_code doesn't need the URL
                res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
        elif data_source in ["hiyouga/geometry3k", "geoqa", "geoinstruction_200k"]:
            from . import geo3k
            res = geo3k.compute_score(solution_str, ground_truth)
        elif data_source in ["searchR1_nq", "searchR1_triviaqa", "searchR1_popqa", "searchR1_hotpotqa", "searchR1_2wikimultihopqa", "searchR1_musique", "searchR1_bamboogle"]:
            from . import search_r1_like_qa_em

            res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

        else:
            # raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
            from . import prime_math
            res = prime_math.compute_score(solution_str, ground_truth)

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, memory_limit_mb=None):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb)


__all__ = ["default_compute_score"]
