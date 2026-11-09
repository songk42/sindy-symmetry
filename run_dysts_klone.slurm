#!/bin/bash
# Generic klone runner for the dysts harnesses: sbatch [--export=ALL,SINDY_...] run_dysts_klone.slurm <script.py> [systems...]
#SBATCH --job-name=dysts
#SBATCH --partition=ckpt-all
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH --output=/mmfs1/home/songk42/ehd-sindy/logs/dysts_traj/slurm/%x_%A_%a.log

cd /mmfs1/home/songk42/ehd-sindy
source .venv/bin/activate
export PYTHONPATH=$HOME/dysts
export DYSTS_DATA_DIR=/gscratch/stf/songk42/dysts_hf/fine
export LOG_DIR=${LOG_DIR:-/mmfs1/home/songk42/ehd-sindy/logs/dysts_traj}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
python notebooks/"$@"
