#!/bin/bash
#SBATCH --job-name="vanilla vivid H100 test"
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
eval "$(conda shell.bash hook)"
mamba activate vivid
module load cuda/12.1.1

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


echo "========================================="
echo "SLURM JOB: Running on host $(hostname)"
echo "SLURM JOB: Python path: $(which python)"
echo "========================================="

/home/stud/lavingal/miniforge3/envs/vivid/bin/python train_nvs.py --batch 256 --batch-gpu 256 --channels 64
