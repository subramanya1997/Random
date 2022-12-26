#!/bin/sh
#SBATCH --job-name=ddp_pytorch
#SBATCH -o /work/pi_hzhang2_umass_edu/snagabhushan_umass_edu/ddp_pytorch/logs/sbatch_log_%j.txt
#SBATCH --time=10:00:00
#SBATCH -c 1 # Cores
#SBATCH --mem=128GB  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH --gres gpu:2
#SBATCH -G 2  # Number of GPUs
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

conda activate py39

cd /work/pi_hzhang2_umass_edu/snagabhushan_umass_edu/ddp_pytorch

# log based on the job id
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train_ddp.py > logs/run_log_${SLURM_JOB_ID}.txt