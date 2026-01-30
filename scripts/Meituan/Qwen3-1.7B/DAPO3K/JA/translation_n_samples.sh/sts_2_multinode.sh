#!/bin/bash
set -x

export WANDB_API_KEY="9a459e2566a6644042b1b91f63e0bfacc119d240"
export http_proxy=http://10.217.142.137:8080
export https_proxy=http://10.217.142.137:8080
export CUDA_VISIBLE_DEVICES=$(seq -s "," 0 $(nvidia-smi --list-gpus | wc -l | awk '{print $1-1}'))
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
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
    nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)
    node_rank=0
    OBJECT_STORE_MEMORY=19999999999
    RAY_WAIT_TIME=1
    # rm -rf /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/verl/mt_recipe/image_in_loop/tool_workdir/*
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
export HYDRA_FULL_ERROR=1
export RAY_LOGGING_LEVEL=DEBUG
export OPENBLAS_NUM_THREADS=1

# ================= tool setting =================
# export M6_CODE_INTERPRETER_WORK_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/verl/mt_recipe/image_in_loop/tool_workdir"
# export INPUT_IMAGE_TMP_DIR=$M6_CODE_INTERPRETER_WORK_DIR

# ================= data/model/tool =================

if [[ "$node_rank" -ne "0" ]]; then
    echo "Starting Ray worker..."
    ray start --address="$MASTER_ADDR:6379" --temp-dir $RAY_TMP_DIR
    sleep 5
    ray status
    echo "Worker ready. Waiting for master to finish..."
    while [ ! -f /workdir/main_done_${MASTER_ADDR}.txt ]; do
        sleep 30
    done
    exit 0
else
    echo "Starting Ray master..."
    ray start --head \
        --object-store-memory=80530636800 \
        --dashboard-port=8414 \
        --dashboard-host=0.0.0.0 \
        --temp-dir $RAY_TMP_DIR
fi

########################################
#     等待所有 worker 加入 Ray 集群
########################################
echo "Waiting for all nodes to join..."
start_time=$(date +%s)
while true; do
    active_nodes=$(ray status | sed -n '/Active:/,/Pending:/p' | grep "1 node_" | wc -l)
    echo "Active nodes: $active_nodes / $nnodes"

    if [ "$active_nodes" -ge "$nnodes" ]; then
        echo "All nodes ready!"
        break
    fi
    elapsed=$(( $(date +%s) - $start_time ))
    if [ "$elapsed" -gt 300 ]; then
        echo "Timeout waiting for nodes!"
        exit 1
    fi
    sleep 5
done

########################################
#       MASTER 节点启动训练任务
########################################

echo "Launching distributed training..."

ray job submit --address="http://$MASTER_ADDR:8414" \
    -- python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918/data/DAPO/dapo_3k_ja_sts.parquet \
        data.val_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918/data/math500_ja_question_ja_prompt_ja_begin.parquet \
        data.train_batch_size=256 \
        data.prompt_key=query \
        data.max_prompt_length=4096 \
        data.max_response_length=8192 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_input_ids=True \
        data.return_raw_chat=True \
        data.return_full_prompt=True \
        data.target_language='JA' \
        data.translation_acc_lower=0.2 \
        data.translation_acc_upper=1.0 \
        data.qt_training_ratio=1.0 \
        actor_rollout_ref.model.path=/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen3-1.7B/main \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.translation_sample_n=2 \
        actor_rollout_ref.rollout.translation_temperature=0.6 \
        actor_rollout_ref.rollout.translation_top_p=0.95 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=6 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.strategy=fsdp2 \
        algorithm.use_kl_in_reward=False \
        trainer.task='sts' \
        trainer.critic_warmup=0 \
        trainer.logger='["tensorboard","console"]' \
        trainer.n_gpus_per_node=${nproc_per_node} \
        trainer.nnodes=${nnodes} \
        trainer.save_freq=5 \
        trainer.test_freq=5 \
        trainer.project_name=DAPO3K-Qwen3-1.7B-JA \
        trainer.experiment_name=ja_sts_8192_2_translation_samples \
        trainer.default_local_dir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918/checkpoints/DAPO3K-Qwen3-1.7B-JA/ja_sts_8192_2_translation_samples \
        trainer.total_epochs=5

touch /workdir/main_done_${MASTER_ADDR}.txt
sleep 10