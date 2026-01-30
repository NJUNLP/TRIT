#!/bin/bash

# ========================================
# LlamaFactory 训练配置脚本
# ========================================

# 模型配置
export MODEL_NAME_OR_PATH="/path/to/your/model"  # 修改为你的模型路径

# 数据集配置
export DATASET="your_dataset_name"  # 修改为你的数据集名称
export TEMPLATE="default"  # 修改为你的模板类型，如: default, alpaca, vicuna等
export CUTOFF_LEN=2048  # 最大序列长度

# 训练配置
export GRADIENT_ACCUMULATION_STEPS=8  # 梯度累积步数
export NUM_EPOCHS=3  # 训练轮数

# 输出配置
export OUTPUT_DIR="./output/$(date +%Y%m%d_%H%M%S)"  # 输出目录，自动添加时间戳

# 原始配置文件路径
CONFIG_TEMPLATE="/path/to/your/config.yaml"  # 修改为你的yaml模板路径

# ========================================
# 创建输出目录
# ========================================
mkdir -p "$OUTPUT_DIR"

# ========================================
# 生成临时配置文件
# ========================================
TEMP_CONFIG="$OUTPUT_DIR/training_config.yaml"

echo "Generating config file: $TEMP_CONFIG"

# 使用envsubst替换环境变量生成配置文件
envsubst < "$CONFIG_TEMPLATE" > "$TEMP_CONFIG"

# ========================================
# 保存训练参数记录
# ========================================
TRAINING_INFO="$OUTPUT_DIR/training_info.txt"

cat > "$TRAINING_INFO" << EOF
========================================
Training Information
========================================
Training Start Time: $(date '+%Y-%m-%d %H:%M:%S')
Hostname: $(hostname)
User: $(whoami)

========================================
Model Configuration
========================================
MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH

========================================
Dataset Configuration
========================================
DATASET: $DATASET
TEMPLATE: $TEMPLATE
CUTOFF_LEN: $CUTOFF_LEN

========================================
Training Configuration
========================================
GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS
NUM_EPOCHS: $NUM_EPOCHS

========================================
Output Configuration
========================================
OUTPUT_DIR: $OUTPUT_DIR

========================================
GPU Information
========================================
$(nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || echo "GPU info not available")

========================================
Environment Variables
========================================
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"Not set"}

========================================
Full Configuration File
========================================
$(cat "$TEMP_CONFIG")
========================================
EOF

# ========================================
# 打印配置信息
# ========================================
echo "========================================="
echo "Training Configuration"
echo "========================================="
echo "Model: $MODEL_NAME_OR_PATH"
echo "Dataset: $DATASET"
echo "Template: $TEMPLATE"
echo "Cutoff Length: $CUTOFF_LEN"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================="
echo "Config saved to: $TEMP_CONFIG"
echo "Training info saved to: $TRAINING_INFO"
echo "========================================="

# ========================================
# 启动训练
# ========================================

# 使用生成的临时配置文件启动训练
llamafactory-cli train "$TEMP_CONFIG"

# 或使用Python模块方式
# python -m llamafactory.train "$TEMP_CONFIG"

# 如果需要指定GPU
# CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train "$TEMP_CONFIG"

# ========================================
# 训练结束后记录
# ========================================
echo "" >> "$TRAINING_INFO"
echo "Training End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$TRAINING_INFO"