#!/bin/bash
#SBATCH --job-name=clt_prep
#SBATCH --partition=regular
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/s4525205/clt_thesis/logs/prep_%j.out

module purge
module load Python/3.11.3-GCCcore-12.3.0
source ~/clt_venv/bin/activate

cd /scratch/s4525205/clt_thesis/Bachelor-Thesis-IK/final_dataset_creation/dataset_construction

THREADS_PATH=/scratch/s4525205/clt_thesis/data/3778298/threads.jsonl.bz2 \
PAIRS_PATH=/scratch/s4525205/clt_thesis/data/3778298/pairs.jsonl.bz2 \
THREAD_IDS=/scratch/s4525205/clt_thesis/all_thread_ids.txt \
bash run_pipeline_pre_classifier.sh
