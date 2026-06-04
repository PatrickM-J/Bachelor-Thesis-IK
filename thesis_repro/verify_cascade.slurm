#!/bin/bash
#SBATCH --job-name=clt_verify
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/s4525205/clt_thesis/logs/verify_%j.out

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
source ~/clt_venv/bin/activate

cd /scratch/s4525205/clt_thesis/Bachelor-Thesis-IK/peer_handoff

python inference.py \
    --input example_input.json \
    --stage1_model models/stage1_model \
    --stage2_model models/stage2_best_model.pt \
    --output example_output.json \
    --batch_size 32
