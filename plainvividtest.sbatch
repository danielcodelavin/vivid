#!/bin/bash
#SBATCH --job-name="a40 plain vanilla vivid test 7"
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
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

/home/stud/lavingal/miniforge3/envs/vivid/bin/python train_nvs.py --batch 64 --outdir 'plainoutput' --channels 64 --duration 19200000 --checkpoint 640000 --snapshot 12800 --metrics 64000 --samples 12800 --status 1280 --lr 0.0024