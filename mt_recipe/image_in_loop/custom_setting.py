import logging
import os
import re
from io import BytesIO
from uuid import uuid4

import datasets
from PIL import Image
from qwen_vl_utils import fetch_image

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
            dataframe = dataframe.map(
                self.map_fn_vl,
                fn_kwargs={"data_source": parquet_file},
                remove_columns=dataframe.column_names,
                num_proc=16,
            )
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        print(f"dataset len: {len(self.dataframe)}")

    def map_fn_vl(self, row: dict, *, data_source: str = None):
        problem, answer = row["problem"], row["answer"]
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": problem}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        if "images" in row and row["images"]:
            data["images"] = row["images"]
        return data

    def _build_messages(self, example: dict):
        """Build messages with multimodal alias mapping for input images.

        This method processes messages and creates a mapping from standardized paths
        (/workdir/image/input_image_{idx:02}.{ext}) to actual image file paths or PIL images.
        """
        messages = example.pop(self.prompt_key)

        multimodal_alias_mapping = {}
        current_image_idx = 0

        # Create a directory to store raw images
        raw_image_dir = self.config.get("raw_image_dir", "/workdir/raw_image")
        os.makedirs(raw_image_dir, exist_ok=True)

        if self.image_key not in example and self.video_key not in example:
            example["multimodal_alias_mapping"] = {}
            return messages

        for message in messages:
            content = message.get("content")
            if not isinstance(content, str) or ("<image>" not in content and "<video>" not in content):
                continue

            content_list = []
            segments = re.split(r"(<image>|<video>)", content)
            segments = [item for item in segments if item]

            for segment in segments:
                if segment == "<image>":
                    content_list.append({"type": "image"})
                    if (
                        self.image_key in example
                        and example[self.image_key]
                        and current_image_idx < len(example[self.image_key])
                    ):
                        actual_image = example[self.image_key][current_image_idx]
                        if "bytes" in actual_image:
                            pil_image = Image.open(BytesIO(actual_image["bytes"]))
                        else:
                            pil_image = fetch_image(actual_image)
                        ext = ".png"
                        if pil_image.format:
                            ext = f".{pil_image.format.lower()}"
                        temp_image_path = os.path.join(raw_image_dir, f"{uuid4()}{ext}")
                        pil_image.save(temp_image_path)

                        alias = f"/workdir/image/input_image_{current_image_idx:02}{ext}"
                        multimodal_alias_mapping[alias] = temp_image_path
                        content_list.append({"type":"text", "text": f"<image_path>{alias}</image_path>"})

                        current_image_idx += 1
                elif segment == "<video>":
                    # Not support video currently
                    content_list.append({"type": "video"})
                else:
                    content_list.append({"type": "text", "text": segment})

            message["content"] = content_list

        # Store the mapping back into example so it can be returned by __getitem__
        example["multimodal_alias_mapping"] = multimodal_alias_mapping

        return messages


def compute_score(data_source, solution_str, ground_truth, extra_info):
    # 这里的 reward 没有实现
    return 1
