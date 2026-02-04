#!/bin/bash
# ============================================================================
# 分布式训练启动脚本
# 
# 使用方法：
#   1. 修改下方"用户配置区域"中的参数，或通过环境变量设置
#   2. 在集群环境中，设置 AFO_ENV_CLUSTER_SPEC 环境变量
#   3. 在本地调试时，脚本会自动使用本地配置
# 
# 环境变量配置示例：
#   export WANDB_API_KEY="your-api-key"
#   export TRAIN_DATA_PATH="./data/train.parquet"
#   export VAL_DATA_PATH="./data/val.parquet"
#   export MODEL_CHECKPOINT_PATH="./checkpoints/model"
#   export PROJECT_NAME="my-project"
#   export EXPERIMENT_NAME="my-experiment"
#   bash example.sh
# ============================================================================

set -x

# ==================== 用户配置区域 ====================
# 请根据您的环境修改以下配置参数，或通过环境变量设置

# Wandb API Key (可选，如果不需要wandb可以留空)
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# 代理设置 (可选，如果不需要代理可以注释掉)
# export http_proxy="${HTTP_PROXY:-}"
# export https_proxy="${HTTPS_PROXY:-}"

# 路径配置
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workdir/tmp}"
RAY_TMP_DIR="${RAY_TMP_DIR:-/workdir/tmp/ray}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-./tensorboard_logs}"

# 数据路径配置
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./dataset/train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-./dataset/val.parquet}"

# 模型checkpoint路径
MODEL_CHECKPOINT_PATH="${MODEL_CHECKPOINT_PATH:-./checkpoints/model}"

# 训练输出路径
CHECKPOINT_SAVE_DIR="${CHECKPOINT_SAVE_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./logs}"
LOG_FILE="${LOG_FILE:-training.log}"

# 主节点完成标志文件路径 (用于worker节点等待)
MAIN_DONE_FILE="${MAIN_DONE_FILE:-./main_done.txt}"

# 本地调试配置 (当 AFO_ENV_CLUSTER_SPEC 未设置时使用)
LOCAL_NPROC_PER_NODE="${LOCAL_NPROC_PER_NODE:-8}"
LOCAL_OBJECT_STORE_MEMORY="${LOCAL_OBJECT_STORE_MEMORY:-19999999999}"

# 其他环境配置 (可选)
export HADOOP_HOME="${HADOOP_HOME:-/opt/meituan/hadoop}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

# 训练超参数配置 (可根据需要修改)
PROJECT_NAME="${PROJECT_NAME:-my-project}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-my-experiment}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
TARGET_LANGUAGE="${TARGET_LANGUAGE:-JA}"

# ==================== 环境变量设置 ====================

export CUDA_VISIBLE_DEVICES=$(seq -s "," 0 $(nvidia-smi --list-gpus | wc -l | awk '{print $1-1}'))
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
# export TORCH_NCCL_BLOCKING_WAIT=1

if [ -n "$AFO_ENV_CLUSTER_SPEC" ]; then
    echo "AFO_ENV_CLUSTER_SPEC: $AFO_ENV_CLUSTER_SPEC"

    master_addr_script="from mt_recipe.ray_hope import env_parse;print(env_parse.get_master_addr())"
    master_port_script="from mt_recipe.ray_hope import env_parse;print(env_parse.get_master_port())"
    node_rank_script="from mt_recipe.ray_hope import env_parse;print(env_parse.get_node_rank())"
    nproc_per_node_script="from mt_recipe.ray_hope import env_parse;print(env_parse.get_nproc_per_node())"
    nnodes_script="from mt_recipe.ray_hope import env_parse;print(env_parse.get_nnodes())"

    master_addr=$(python3 -c "$master_addr_script")
    master_port=$(python3 -c "$master_port_script")
    node_rank=$(python3 -c "$node_rank_script")
    nproc_per_node=$(python3 -c "$nproc_per_node_script")
    nnodes=$(python3 -c "$nnodes_script")

    echo "MASTER_ADDR: $master_addr"
    echo "MASTER_PORT: $master_port"
    echo "NODE_RANK: $node_rank"
    echo "NPROC_PER_NODE: $nproc_per_node"
    echo "NNODES: $nnodes"

    OBJECT_STORE_MEMORY=80530636800
    RAY_WAIT_TIME=300

    # pip3 install -U "qwen-agent[gui,rag,code_interpreter,mcp]" -i https://pypi.org/simple/
else
    # 本地调试用代码
    ray stop --force
    master_addr=$(hostname -I | awk '{print $1}')
    nnodes=1
    nproc_per_node=${LOCAL_NPROC_PER_NODE}
    node_rank=0
    OBJECT_STORE_MEMORY=${LOCAL_OBJECT_STORE_MEMORY}
    RAY_WAIT_TIME=1
fi

export MASTER_ADDR="$master_addr"
export MASTER_PORT="$master_port"
mkdir -p $RAY_TMP_DIR
rm -rf $RAY_TMP_DIR/*
# ================== ABOVE IS SCRIPTS FOR RAY CLUSTER =====================

export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,NET,GRAPH,ENV,DYNDBG
export PYTHONFAULTHANDLER=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=120
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_RETRY_CNT=15
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HADOOP_HOME=/opt/meituan/hadoop
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export RAY_COLOR_PREFIX=0
export RAY_LOGGING_LEVEL=DEBUG
export OPENBLAS_NUM_THREADS=1
export HYDRA_FULL_ERROR=1


export CHECK_REPETITION=1
export USE_MATH_LJX_FINAL=1
# ================= tool setting =================
mkdir -p "${TENSORBOARD_DIR}"


if [[ "$node_rank" -ne "0" ]]; then
    connect_flag=1
    retry_times=0
    MAX_RETRY_TIMES=3
    echo "Starting running ray on WORKER node, connecting to $master_addr:6379"
    while [[ $connect_flag -ne 0 ]] && [[ $retry_times -lt $MAX_RETRY_TIMES ]]
    do
        sleep 5
        ray start --address="$master_addr:6379" --metrics-export-port 65184 --dashboard-agent-grpc-port 65185 --dashboard-agent-listen-port 65186 --temp-dir $RAY_TMP_DIR
        connect_flag=$?
        retry_times=$((retry_times + 1))
        echo "Connect status: $connect_flag, Retry times: $retry_times"
    done

    if [[ $retry_times -ge $MAX_RETRY_TIMES ]]; then
        echo "Tried for $MAX_RETRY_TIMES, Exceeds max retry times!"
        exit 1
    else
        echo "Connected to MASTER"
        sleep 20
        ray status
        while  [ ! -f "${MAIN_DONE_FILE}" ]; do
            echo "Waiting for main node to finish..."
            sleep 3600
        done
    fi
else
    echo "Starting running ray on MASTER node"
    
    ray start --head \
        --object-store-memory=$OBJECT_STORE_MEMORY \
        --dashboard-port=8414 \
        --dashboard-host='0.0.0.0' \
        --metrics-export-port 65184 \
        --dashboard-agent-grpc-port 65185 \
        --dashboard-agent-listen-port 65186 \
        --temp-dir $RAY_TMP_DIR

    echo " 等待所有节点加入 ray 集群..."
    start_time=$(date +%s)
    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))

        # 获取当前活跃节点数量
        active_nodes=$(ray status | sed -n '/Active:/,/Pending:/p' | grep "1 node_" | wc -l)

        echo "当前活跃节点数: $active_nodes / $nnodes"

        if [ "$active_nodes" -ge "$nnodes" ]; then
            echo "所有节点已加入集群!"
            break
        fi

        if [ "$elapsed" -ge "$RAY_WAIT_TIME" ]; then
            echo "错误: 等待节点加入集群超时($RAY_WAIT_TIME)，当前节点数: $active_nodes / $nnodes"
            echo "期望节点数: $nnodes，实际节点数: $active_nodes"
            echo "集群节点数量不足，训练任务无法继续执行!"
            exit 1
        fi

        echo "等待更多节点加入... (已等待 ${elapsed}s / ${RAY_WAIT_TIME}s)"
        sleep 10
    done
    
    ray status

    # 创建必要的目录
    mkdir -p "${CHECKPOINT_SAVE_DIR}"
    mkdir -p "${LOG_DIR}"
    
    ray job submit --address="http://$master_addr:8414" \
    -- python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files="${TRAIN_DATA_PATH}" \
 data.val_files="${VAL_DATA_PATH}" \
 data.train_batch_size=${TRAIN_BATCH_SIZE} \
 data.prompt_key=query \
 data.max_prompt_length=4096 \
 data.max_response_length=8192 \
 data.filter_overlong_prompts=True \
 data.truncation='error' \
 data.return_raw_input_ids=True \
 data.return_raw_chat=True \
 data.return_full_prompt=True \
 data.target_language="${TARGET_LANGUAGE}" \
 data.translation_acc_lower=0.2 \
 data.translation_acc_upper=1.0 \
 data.qt_training_ratio=1.0 \
 data.shuffle=False \
 actor_rollout_ref.model.path="${MODEL_CHECKPOINT_PATH}" \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.actor.strategy=fsdp2 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.rollout.translation_sample_n=4 \
 actor_rollout_ref.rollout.translation_temperature=0.7 \
 actor_rollout_ref.rollout.translation_top_p=0.95 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
 actor_rollout_ref.rollout.n=6 \
 actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
 actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
 actor_rollout_ref.rollout.val_kwargs.n=4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.ref.strategy=fsdp2 \
 algorithm.use_kl_in_reward=False \
 trainer.task='trit' \
 trainer.critic_warmup=0 \
 trainer.logger=['tensorboard','console'] \
 trainer.n_gpus_per_node=${nproc_per_node} \
 trainer.nnodes=${nnodes} \
 trainer.save_freq=5 \
 trainer.test_freq=5 \
 trainer.project_name="${PROJECT_NAME}" \
 trainer.experiment_name="${EXPERIMENT_NAME}" \
 trainer.default_local_dir="${CHECKPOINT_SAVE_DIR}" \
 trainer.total_epochs=${TOTAL_EPOCHS} 2>&1 | tee "${LOG_DIR}/${LOG_FILE}" 

fi