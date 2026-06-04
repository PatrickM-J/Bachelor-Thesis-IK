#!/bin/bash
#SBATCH --job-name=clt_cascade
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/s4525205/clt_thesis/logs/cascade_%j.out

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
source ~/clt_venv/bin/activate

cd /scratch/s4525205/clt_thesis/Bachelor-Thesis-IK/peer_handoff

INDIR=/scratch/s4525205/clt_thesis/Bachelor-Thesis-IK/final_dataset_creation/dataset_construction/outputs/cascade_input
OUTDIR=/scratch/s4525205/clt_thesis/Bachelor-Thesis-IK/final_dataset_creation/dataset_construction/outputs/cascade_output
mkdir -p "$OUTDIR"

for shard in "$INDIR"/sentences_shard_*.json; do
    base=$(basename "$shard" .json)
    out="$OUTDIR/pred_${base}.json"
    if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
    fi
    echo "[$(date)] classifying $base ..."
    python inference.py \
        --input "$shard" \
        --stage1_model models/stage1_model \
        --stage2_model models/stage2_best_model.pt \
        --output "$out" \
        --batch_size 256
done

echo "[$(date)] All shards done."
