import logging

import datasets

from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo

logger = logging.getLogger(__name__)

answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


class CustomRLHFVLDataset(RLHFDataset):
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            data_source = parquet_file
            if any([src in data_source for src in ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]]):
                dataframe = dataframe.map(
                    self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            elif any([src in data_source for src in ["hiyouga/geometry3k", "DATA/ToolVerify"]]):
                dataframe = dataframe.map(
                    self.map_fn_vl,
                    fn_kwargs={"data_source": data_source},
                    remove_columns=dataframe.column_names,
                    num_proc=16,
                )
            else:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "Maxwell-Jia/AIME_2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source == "yentinglin/aime_2025":
            problem, answer = row["problem"], row["answer"]

        prompt = problem + answer_format
        data = {
            "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        return data

    def map_fn2(self, row: dict):
        content = row["prompt"][0]["content"]
        row["prompt"][0]["content"] = content + answer_format
        row["agent_name"] = "tool_agent"
        return row

    def map_fn_vl(self, row: dict, *, data_source: str = None):
        problem, answer = row["problem"], row["answer"]
        problem = problem + answer_format
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": problem}],
            "images": row["images"],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        return data


def compute_score(data_source, solution_str, ground_truth, extra_info):
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    num_turns = extra_info["num_turns"]
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
