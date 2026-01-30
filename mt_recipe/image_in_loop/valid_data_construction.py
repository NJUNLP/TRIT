import io

import pandas as pd
from PIL import Image


def convert_image_to_bytes(image_path):
    img = Image.open(image_path)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        output.seek(0)
        return output.read()


ig_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/DATA/ToolVerify/images/ig.png"
calculate_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/DATA/ToolVerify/images/calculate.png"
SAVE_PARQUET_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/verl/mt_recipe/image_in_loop/image_out.parquet"

data = [
    {
        "images": [
            {"bytes": convert_image_to_bytes(ig_path), "path": ig_path},
            {"bytes": convert_image_to_bytes(calculate_path), "path": calculate_path},
        ],
        "problem": "给你两张图片 <image> 和 <image>，用代码工具把里面 ig 的那张图上下翻转。",
        "answer": -1,
    },
    {
        "images": [
            {"bytes": convert_image_to_bytes(ig_path), "path": ig_path},
        ],
        "problem": "给你这张图片 <image>，用缩放工具把这张图片的白边和图像部分都裁剪掉，只保留 invictus gaming 艺术字本身",
        "answer": -1,
    },
    {
        "images": [],
        "problem": "请你使用网页访问工具查看 https://verl.readthedocs.io/en/latest/advance/agent_loop.html，然后选择里面最复杂的一张图片上下翻转展示出来",
        "answer": -1,
    },
]

df = pd.DataFrame(data)
df.to_parquet(SAVE_PARQUET_PATH)
