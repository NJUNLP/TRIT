source "/home/data_91_d/anaconda3/etc/profile.d/conda.sh"

conda activate /home/nfs05/anaconda3/envs/ljx-LF-newest-310/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /home/nfs05/liujx/GithubRepos/LLaMA-Factory

llamafactory-cli train /home/nfs05/liujx/GithubRepos/SvS-0918/sft_scripts/sft_deepseek-distill-en-ja.yaml

# bash /home/nfs05/liujx/Inference/scripts/inference-direct.sh

# llamafactory-cli train /home/nfs05/liujx/GithubRepos/SvS-0918/sft_scripts/sft_qwen3-ja-ja.yaml

# bash /home/nfs05/liujx/Inference/scripts/inference-direct.sh