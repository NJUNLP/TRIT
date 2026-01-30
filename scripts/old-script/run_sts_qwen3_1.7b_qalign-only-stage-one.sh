source "/home/data_91_d/anaconda3/etc/profile.d/conda.sh"
# 设置成对照组，看看效果
conda activate /home/nfs05/anaconda3/envs/ljx-verl
export USE_MATH_LJX_FINAL=1
# export DEBUGGING_STS=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export USE_CONSISTENT_LANGUAGE_WEIGHT=True
export WANDB_MODE=offline
export WANDB_DIR="/home/nfs05/liujx/GithubRepos/verl/wandb/MATH8K-Qwen3-1.7B-JA/StS-0.2-1.0-1.0-repair-qalign-only-stage-one-4096"
# export USE_EN_THINK=False
# export SWANLAB_API_KEY="hTFCM59njhgvcx9emSxBk"          # 设置在线跟踪模式API
# export SWANLAB_LOG_DIR="/home/nfs05/liujx/GithubRepos/verl/swanlab/grpo_qwen3-1.7b_0.1_reward_ja_ours"    # 设置本地日志存储路径
# export SWANLAB_MODE="local"     # 包含四种模式：cloud云端跟踪模式（默认）、cloud-only仅云端跟踪本地不保存文件、local本地跟踪模式、disabled完全不记录用于debug
# # export TENSORBOARD_LOG_DIR="/home/nfs05/liujx/GithubRepos/verl/tensorboard/test_grpo/grpo_test_raw_prompt_chinese_language_reward_0.1"
# export TENSORBOARD_LOG_DIR="/home/nfs05/liujx/GithubRepos/verl/tensorboard/GSM8K/grpo_qwen3-1.7b_0.1_reward_ja_ours"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=/home/nfs05/liujx/GithubRepos/SvS-0918/data/math8k-ljx-qalign.parquet \
 data.val_files=/home/nfs05/liujx/GithubRepos/verl/data/MATH/JA_Test/math500_ja_question_ja_prompt_ja_begin.parquet \
 data.train_batch_size=256 \
 data.prompt_key=query \
 data.max_prompt_length=2048 \
 data.max_response_length=4096 \
 data.filter_overlong_prompts=True \
 data.truncation='error' \
 data.return_raw_input_ids=True \
 data.return_raw_chat=True \
 data.return_full_prompt=True \
 data.target_language='JA' \
 data.translation_acc_lower=0.2 \
 data.translation_acc_upper=1.0 \
 data.qt_training_ratio=1.0 \
 actor_rollout_ref.model.path=/home/nfs05/model/Qwen3-1.7B \
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
 actor_rollout_ref.rollout.translation_sample_n=4 \
 actor_rollout_ref.rollout.translation_temperature=0.6 \
 actor_rollout_ref.rollout.translation_top_p=0.95 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.ref.strategy=fsdp2 \
 algorithm.use_kl_in_reward=False \
 trainer.task='sts-qalign-only-stage-one' \
 trainer.critic_warmup=0 \
 trainer.logger=['console','tensorboard','wandb'] \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=15 \
 trainer.test_freq=5 \
 trainer.project_name=MATH8K-Qwen3-1.7B-JA \
 trainer.experiment_name=StS-0.2-1.0-1.0-repair-qalign-only-stage-one-4096 \
 trainer.default_local_dir=/home/nfs05/liujx/GithubRepos/verl/checkpoints/MATH8K-Qwen3-1.7B-JA/StS-0.2-1.0-1.0-repair-qalign-only-stage-one-4096 \
 trainer.total_epochs=5 2>&1 | tee /home/nfs05/liujx/GithubRepos/verl/logs/MATH8K-JA-StS-0.2-1.0-1.0-repair-qalign-only-stage-one-4096.log 

# bash /home/nfs05/liujx/Inference/scripts/temp.sh

# bash /home/nfs05/liujx/GithubRepos/SvS-0918/scripts/run_sts_qwen3_1.7b_qalign-plus-remove-translation.sh
bash /home/nfs05/liujx/GithubRepos/SvS-0918/scripts/DeepSeek-Distill/StS-language-first.sh