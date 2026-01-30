# source "/home/data_91_d/anaconda3/etc/profile.d/conda.sh"
# # 设置成对照组，看看效果
# conda activate /home/nfs05/anaconda3/envs/ljx-verl
# export DEBUGGING_STS=True
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
export NCCL_TIMEOUT=3600
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CHECK_REPETITION=1
export USE_MATH_LJX_FINAL=1
# export USE_CONSISTENT_LANGUAGE_WEIGHT=True
# export WANDB_MODE=offline
# export WANDB_DIR="/home/nfs05/liujx/GithubRepos/verl/wandb/MATH8K-Qwen3-1.7B-JA/StS-0.2-1.0-1.0-repair-4096"
# export USE_EN_THINK=False
# export SWANLAB_API_KEY="hTFCM59njhgvcx9emSxBk"          # 设置在线跟踪模式API
# export SWANLAB_LOG_DIR="/home/nfs05/liujx/GithubRepos/verl/swanlab/grpo_qwen3-1.7b_0.1_reward_ja_ours"    # 设置本地日志存储路径
# export SWANLAB_MODE="local"     # 包含四种模式：cloud云端跟踪模式（默认）、cloud-only仅云端跟踪本地不保存文件、local本地跟踪模式、disabled完全不记录用于debug
# # export TENSORBOARD_LOG_DIR="/home/nfs05/liujx/GithubRepos/verl/tensorboard/test_grpo/grpo_test_raw_prompt_chinese_language_reward_0.1"
# export TENSORBOARD_LOG_DIR="/home/nfs05/liujx/GithubRepos/verl/tensorboard/GSM8K/grpo_qwen3-1.7b_0.1_reward_ja_ours"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918/dataset/Train/Qwen3/DAPO3K/dapo3k_no_shuffle_fr_ja_ko_pt_th.parquet \
 data.val_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918/dataset/Test/MMATH/Qwen3/mmath_fr_ja_ko_pt_th.parquet \
 data.train_batch_size=512 \
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
 data.shuffle=False \
 actor_rollout_ref.model.path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/SFT/SFT-Checkpoint/Qwen3-1.7B/DAPO3K-5-Language-Qwen3-1.7B-EN-Question \
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
 actor_rollout_ref.rollout.translation_temperature=0.9 \
 actor_rollout_ref.rollout.translation_top_p=0.95 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.rollout.n=6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.ref.strategy=fsdp2 \
 actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
 actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
 actor_rollout_ref.rollout.val_kwargs.n=4 \
 algorithm.use_kl_in_reward=False \
 trainer.task='normal' \
 trainer.critic_warmup=0 \
 trainer.logger=['tensorboard','console'] \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=5 \
 trainer.test_freq=5 \
 trainer.project_name=DAPO3K-5-Languages-Qwen3-1D7B \
 trainer.experiment_name=baseline_8k_no_shuffle \
 trainer.default_local_dir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liujunxiao03/checkpoints/DAPO3K-5-Languages-Qwen3-1D7B/baseline_8k_no_shuffle \
 trainer.total_epochs=15 2>&1 | tee /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liujunxiao03/logs/DAPO3K-5-Languages-Qwen3-1D7B-baseline_8k_no_shuffle.log 

# bash /home/nfs05/liujx/Inference/scripts/temp.sh

# bash /home/nfs05/liujx/GithubRepos/SvS-0918/scripts/run_sts_qwen3_1.7b-remove-translation.sh
# bash /home/nfs05/liujx/Inference/scripts/temp.sh