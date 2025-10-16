#!/bin/bash
#SBATCH --job-name="a40 nonvannilla dualsource singletarget final model 7"
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:a40:1
#SBATCH --time=9-00:00:00
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

/home/stud/lavingal/miniforge3/envs/vivid/bin/python train_nvs.py --channels 64 --batch 64 --duration 115200000 --checkpoint 128000 --snapshot 128000 --metrics 128000 --samples 64000 --status 1280 --lr 0.001 --outdir 'lowLRexperiment' --fp16 False