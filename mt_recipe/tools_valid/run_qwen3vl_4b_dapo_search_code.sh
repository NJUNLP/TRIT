set -x

export WANDB_API_KEY="bbeac77b9033d5f1954965b0eb8268566d920558"
export http_proxy=http://10.217.142.137:8080
export https_proxy=http://10.217.142.137:8080
# ================= data/model/tool =================

model_path=/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/hldy/model/Qwen/Qwen3-VL-4B-Thinking/main

TRAIN_FILE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/DATA/ToolVerify/verify.parquet
TEST_FILE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/DATA/ToolVerify/verify.parquet
train_files="['$TRAIN_FILE']"
test_files="['$TEST_FILE']"

# tool
tool_config_path=mt_recipe/tools_valid/sandbox_search_tool_config.yaml

# wandb
project_name=retool
experiment_name=qwen3_vl_4b_dapo_tool_use
default_local_dir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/DATA/ReTool-SFT/checkpoint/$experiment_name
rollout_data_dir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/DATA/ReTool-SFT/$experiment_name/rollout_log
validation_data_dir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/zhuangziyuan/workspace/ponder/DATA/ReTool-SFT/$experiment_name/validation_log

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=8
max_prompt_length=1024
max_response_length=16384
actor_lr=1e-6

train_batch_size=2
ppo_mini_batch_size=2
n_resp_per_prompt=4
n_resp_per_prompt_val=1

# ================= perfomance =================
infer_tp=4
train_sp=4
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

python3 -m verl.trainer.main_ppo \
    ray_kwargs.ray_init.num_cpus=32 \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=mt_recipe/tools_valid/tools_valid.py \
    data.custom_cls.name=CustomRLHFVLDataset \
    data.image_key=images \
    custom_reward_function.path=mt_recipe/tools_valid/tools_valid.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    critic.enable=False \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=4 \
    trainer.total_epochs=5 $@
