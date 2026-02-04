#!/bin/bash
set -x

export WANDB_API_KEY="bbeac77b9033d5f1954965b0eb8268566d920558"
export http_proxy=http://10.217.142.137:8080
export https_proxy=http://10.217.142.137:8080

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

    pip3 install -U "qwen-agent[gui,rag,code_interpreter,mcp]" -i https://pypi.org/simple/
else
    # 本地调试用代码
    ray stop --force
    master_addr=$(hostname -I | awk '{print $1}')
    nnodes=1
    nproc_per_node=4
    node_rank=0
    OBJECT_STORE_MEMORY=16106127360
    RAY_WAIT_TIME=1
    rm -rf /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/verl/mt_recipe/image_in_loop/tool_workdir/*
fi

export MASTER_ADDR="$master_addr"
export MASTER_PORT="$master_port"
export XDG_CACHE_HOME="/workdir/tmp"
RAY_TMP_DIR=/workdir/tmp/ray
rm -rf $RAY_TMP_DIR/*
# ================== ABOVE IS SCRIPTS FOR RAY CLUSTER =====================

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,P2P,NET,GRAPH,ENV,DYNDBG
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=20
export NCCL_IB_QPS_PER_CONNECTION=8
export DFS_CLIENT_WRITE_ZONE=hldy
export HYDRA_FULL_ERROR=1
export RAY_LOGGING_LEVEL=DEBUG


# ================= tool setting =================
export M6_CODE_INTERPRETER_WORK_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/verl/mt_recipe/image_in_loop/tool_workdir"
export INPUT_IMAGE_TMP_DIR=$M6_CODE_INTERPRETER_WORK_DIR

# ================= data/model/tool =================

MODEL_PATH=/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/hldy/model/Qwen/Qwen3-VL-4B-Thinking/main

train_files="[\
    /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/verl/mt_recipe/image_in_loop/image_out.parquet \
]"

test_files="[\
    /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/verl/mt_recipe/image_in_loop/image_out.parquet \
]"

# tool
tool_config_path=mt_recipe/image_in_loop/tool_config.yaml

# logging
project_name=retool
exp_name=qwen3_vl_4b_math_gspo_ci
CKPTS_DIR=/mnt/hdfs/zw04mlnn01/checkpoint/vlp_ckpt/zhuangziyuan/2510/checkpoint/$project_name/$exp_name
TBDIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/DATA/tblog
export TENSORBOARD_DIR=$TBDIR/$project_name/$exp_name

# ================= algorithm =================
adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
offload=true # it's a small model, offloading will just slow-down training
rollout_engine=sglang
rollout_mode=async # can be async to speedup large scale xps
gpu_memory_utilization=0.75
reward_manager=dapo
adv_estimator=grpo
shuffle_dataset=true
first_time_dataset_prep=true # prepare dataset

test_freq=5
save_freq=50
total_epochs=1
val_before_train=true

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1
train_batch_size=1
ppo_mini_batch_size=1
ppo_micro_batch_size_per_gpu=1
n_resp_per_prompt=4

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 16))
# dapo reward manager params
enable_overlong_buffer=false # true
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Sampling params at rollouts
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# agents
max_turns=8

# Performance Related Parameter
# ================= performance ==================
sp_size=4
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=true
gen_tp=4
entropy_checkpointing=true # This enables entropy recomputation specifically for the entropy calculation, lowering memory usage during training.

logdir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/verl/mt_recipe/image_in_loop/tool_workdir/logdir
rollout_data_dir=$logdir/rollout_log
validation_data_dir=$logdir/validation_log
mkdir -p $rollout_data_dir
mkdir -p $validation_data_dir


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
        while  [ ! -f ${TENSORBOARD_DIR}/connection/log/main_done_${MASTER_ADDR}.txt ]; do
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

    ray job submit --address="http://$master_addr:8414" \
    -- python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=${adv_estimator} \
        actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
        data.train_files="${train_files}" \
        data.val_files="${test_files}" \
        data.shuffle=$shuffle_dataset \
        data.prompt_key=prompt \
        data.return_raw_chat=True \
        data.image_key=images \
        data.truncation='error' \
        data.filter_overlong_prompts=true \
        data.train_batch_size=${train_batch_size} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.custom_cls.path=mt_recipe/image_in_loop/custom_setting.py \
        data.custom_cls.name=CustomRLHFVLDataset \
        +data.raw_image_dir=$INPUT_IMAGE_TMP_DIR \
        custom_reward_function.path=mt_recipe/image_in_loop/custom_setting.py \
        custom_reward_function.name=compute_score \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        algorithm.use_kl_in_reward=${use_kl_in_reward} \
        algorithm.kl_ctrl.kl_coef=${kl_coef} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.model.use_remove_padding=true \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.name=${rollout_engine} \
        actor_rollout_ref.rollout.mode=${rollout_mode} \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
        actor_rollout_ref.rollout.multi_turn.format=hermes \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.rollout.enable_chunked_prefill=true \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=true \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
        actor_rollout_ref.rollout.agent.num_workers=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
        critic.enable=false \
        trainer.rollout_data_dir=$rollout_data_dir \
        trainer.validation_data_dir=$validation_data_dir \
        trainer.logger='["console","wandb","hope"]' \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.n_gpus_per_node=${nproc_per_node} \
        trainer.nnodes=${nnodes} \
        trainer.val_before_train=${val_before_train} \
        trainer.test_freq=${test_freq} \
        trainer.save_freq=${save_freq} \
        trainer.total_epochs=${total_epochs} \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.resume_mode=auto \
        trainer.log_val_generations=2

        # reward_model.reward_manager=${reward_manager} \
        # +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
        # +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
        # +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
        # +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
        # +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
        mkdir -p ${TENSORBOARD_DIR}/connection/log
        touch ${TENSORBOARD_DIR}/connection/log/main_done_${MASTER_ADDR}.txt
        sleep 15
fi