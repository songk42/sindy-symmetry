#!/bin/bash
#SBATCH -J wo_tune_m
#SBATCH -p ckpt-all
#SBATCH -A stf
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 4:00:00
#SBATCH --array=0-3
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH -o /mmfs1/home/songk42/ehd-sindy/logs/dysts_traj/slurm/%x_%A_%a.log
source .venv/bin/activate
export PYTHONPATH=$HOME/dysts DYSTS_DATA_DIR=/gscratch/stf/songk42/dysts_hf/fine LOG_DIR=logs/dysts_traj
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
M=(8 16 25 50)
export WEAK_M=${M[$SLURM_ARRAY_TASK_ID]} WO_FORMS=weak WO_TAG=tune_m${M[$SLURM_ARRAY_TASK_ID]}
python notebooks/weak_orbit_dysts.py $(python -c "
import pandas as pd; s=sorted(pd.read_csv('logs/dysts_traj/sindy_si_dysts/side_info.csv').system); print(' '.join(s[::10]))")
