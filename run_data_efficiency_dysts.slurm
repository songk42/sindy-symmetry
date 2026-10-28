#!/bin/bash
#SBATCH --job-name=data_eff_dysts
#SBATCH --partition=mit_preemptable,mit_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --array=0-41
#SBATCH --chdir=/home/songk/sindy-symmetry
#SBATCH --output=/home/songk/sindy-symmetry/logs/data_efficiency_%A_%a.log
#SBATCH --time=2:00:00

cd /home/songk/sindy-symmetry
source .venv/bin/activate
export DYSTS_DATA_DIR=/home/songk/data-songk/dysts_hf/fine
export LOG_DIR=/home/songk/sindy-symmetry/logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python notebooks/data_efficiency_dysts.py "$@"
