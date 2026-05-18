# PrimeCodeRewardManager 使用说明

## 概述

`PrimeCodeRewardManager` 是专门为代码验证任务设计的奖励管理器，解决了 `PrimeRewardManager` 在使用 Ray 时可能出现的超时问题。

## 主要特点

1. **不使用 Ray**：在主进程中直接处理，避免 Ray 相关的超时和资源问题
2. **直接调用代码验证函数**：不经过 `default_compute_score`，直接使用 `code_verification` 模块
3. **批量处理优化**：使用 `compute_code_score_batch` 接口控制并发，提高效率
4. **支持完整格式**：返回包含 `score`、`acc`、`format_score` 的字典格式

## 与 PrimeRewardManager 的区别

| 特性 | PrimeRewardManager | PrimeCodeRewardManager |
|------|-------------------|----------------------|
| 并发方式 | Ray (分布式) | ThreadPoolExecutor (本地) |
| 奖励函数 | 通过 `default_compute_score` | 直接调用 `code_verification` |
| 超时问题 | 可能出现 Ray 超时 | 避免 Ray 超时 |
| 适用场景 | 通用任务 | 代码验证任务 |

## 使用方法

### 1. 基本使用

```python
from verl.workers.reward_manager import PrimeCodeRewardManager
from transformers import AutoTokenizer

# 创建 tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-model")

# 创建 RewardManager
reward_manager = PrimeCodeRewardManager(
    tokenizer=tokenizer,
    num_examine=10,  # 每个数据源打印的样本数量
    max_workers=20,  # 并发worker数（None表示自动选择）
    timeout=10,      # 单个代码执行的超时时间（秒）
    use_batch_processing=True,  # 使用批量处理
)

# 调用
result = reward_manager(data_proto, return_dict=True)
```

### 2. 配置参数

#### 必需参数
- `tokenizer`: PreTrainedTokenizer，用于解码 token ids
- `num_examine`: int，每个数据源打印日志的样本数量

#### 可选参数
- `compute_score`: 不使用，保留接口兼容性
- `reward_fn_key`: str，默认为 `"data_source"`，用于标识数据源的 key
- `max_workers`: Optional[int]，ThreadPoolExecutor 的最大 worker 数
  - `None`: 自动选择（推荐）
  - 数字: 指定 worker 数（如 `20`）
- `timeout`: int，默认 `10`，单个代码执行的超时时间（秒）
- `use_batch_processing`: bool，默认 `True`，是否使用批量处理接口
- `batch_size`: Optional[int]，默认 `None`，批量处理的大小
- `**kwargs`: 传递给 `code_verification` 函数的额外参数

### 3. 返回值

#### `return_dict=False`（默认）
返回 `reward_tensor`，形状为 `(batch_size, seq_len)`，在响应结束位置填充分数。

#### `return_dict=True`
返回字典：
```python
{
    "reward_tensor": torch.Tensor,  # 形状 (batch_size, seq_len)
    "reward_extra_info": {
        "acc": List[float],          # 每个样本的准确性（0或1）
        "format_score": List[float], # 每个样本的格式分数（0或1）
        # ... 其他额外信息
    }
}
```

### 4. 在训练配置中使用

在训练配置文件中，将 `reward_manager` 设置为 `"prime_code"`：

```yaml
reward_model:
  reward_manager: "prime_code"
  reward_kwargs:
    num_examine: 10
    max_workers: 20
    timeout: 10
    use_batch_processing: true
```

或者在 Python 配置中：

```python
config = {
    "reward_model": {
        "reward_manager": "prime_code",
        "reward_kwargs": {
            "num_examine": 10,
            "max_workers": 20,
            "timeout": 10,
            "use_batch_processing": True,
        }
    }
}
```

## 代码验证格式要求

`PrimeCodeRewardManager` 使用 `code_verification.compute_score` 函数，该函数期望以下格式：

### 输入格式

1. **Solution（响应）**：
   - 可以包含 `</think>` 标记（用于 format_score）
   - 包含 Python 代码（通常在 markdown 代码块中）

2. **Ground Truth（标准答案）**：
   - 包含测试函数和 `<entry_point>` 标记
   - 格式：`check_function\n\n<entry_point>\nfunction_name`

示例：
```python
ground_truth = """def check(func):
    assert func(1, 2) == 3
    assert func(0, 0) == 0
<entry_point>
add"""
```

### 输出格式

返回字典包含：
- `score`: 0 或 1，只有当 `format_score` 和 `acc` 都为 1 时才为 1
- `acc`: 0 或 1，代码执行并通过测试为 1
- `format_score`: 0 或 1，存在 `</think>` 标记为 1

## 性能优化建议

1. **并发数设置**：
   - 对于 CPU 密集型任务，`max_workers` 可以设置为 CPU 核心数
   - 对于 I/O 密集型任务（如代码执行），可以设置更大的值（如 20-50）

2. **批量处理**：
   - 默认启用 `use_batch_processing=True`，利用批量处理接口提高效率
   - 对于大批量数据，批量处理可以显著提高吞吐量

3. **超时设置**：
   - 根据代码复杂度调整 `timeout`
   - 简单代码：5-10秒
   - 复杂代码：10-30秒

## 故障排查

### 问题：超时错误
- 增加 `timeout` 参数
- 减少 `max_workers` 数量
- 检查代码执行环境是否正常

### 问题：内存不足
- 减少 `max_workers` 数量
- 使用 `batch_size` 限制批量大小

### 问题：结果不正确
- 检查 `ground_truth` 格式是否正确（包含 `<entry_point>`）
- 检查响应中是否包含有效的 Python 代码
- 查看日志输出确认代码提取和执行情况

## 示例代码

完整示例请参考 `test_prime_code_reward_manager.py`。

## 注意事项

1. `PrimeCodeRewardManager` 专门用于代码验证任务，不适用于其他类型的奖励计算
2. 确保代码执行环境（Jupyter kernel）已正确配置
3. 批量处理时，所有样本共享相同的超时和配置参数
4. 如果需要不同的配置，可以创建多个 `PrimeCodeRewardManager` 实例

