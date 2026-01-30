# source "/home/data_91_d/anaconda3/etc/profile.d/conda.sh"
# # 设置成对照组，看看效果
# conda activate /home/nfs05/anaconda3/envs/ljx-verl
# export DEBUGGING_STS=True
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CHECK_REPETITION=1
export USE_MATH_LJX_FINAL=1
export USE_SLC_REWARD=1


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918/dataset/Train/Qwen3-4B-DAPO3K-5-Language-Qwen3-1.7B/Without-Prefix/dapo3k_fr_ja_ko_pt_th.parquet \
 data.val_files=["/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918/dataset/Test/PolyMATH/Qwen3/Medium/fr_ja_ko_pt_th.parquet","/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918/dataset/Test/PolyMATH/Qwen3/High/fr_ja_ko_pt_th.parquet"] \
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
 actor_rollout_ref.model.path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/SFT/SFT-Checkpoint/Qwen3-4B/DAPO3K-5-Language-Qwen3-1.7B-EN-Question \
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
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
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
 trainer.project_name=DAPO3K-5-Languages-Qwen3-4B \
 trainer.experiment_name=baseline_8k \
 trainer.default_local_dir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918/checkpoints/DAPO3K-5-Languages-Qwen3-4B/baseline_8k \
 trainer.total_epochs=5 2>&1 | tee /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/SvS-0918/logs/DAPO3K-5-Languages-Qwen3-4B-baseline_8k.log 

# bash /home/nfs05/liujx/Inference/scripts/temp.sh

# bash /home/nfs05/liujx/GithubRepos/SvS-0918/scripts/run_sts_qwen3_1.7b-remove-translation.sh
# bash /home/nfs05/liujx/Inference/scripts/temp.sh