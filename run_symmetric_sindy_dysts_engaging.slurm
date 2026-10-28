#!/bin/bash
#SBATCH --job-name=symm_sindy_dysts
#SBATCH --partition=mit_preemptable,mit_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/home/songk/sindy-symmetry
#SBATCH --output=/home/songk/sindy-symmetry/logs/%x_%j.log
#SBATCH --time=4:00:00

cd /home/songk/sindy-symmetry
source .venv/bin/activate
export DYSTS_DATA_DIR=/home/songk/data-songk/dysts_hf/fine
export LOG_DIR=/home/songk/sindy-symmetry/logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python notebooks/symmetric_sindy_dysts.py "$@"
