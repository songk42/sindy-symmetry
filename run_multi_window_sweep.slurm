#!/bin/bash
#SBATCH --job-name=multi_window_sweep
#SBATCH --partition=ckpt-all
#SBATCH --account=stf
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40|l40s   # full dense cupy SVD of a 12221x40000 matrix needs
                                          # >11GB VRAM (OOM'd on a 2080 Ti); z3006 also failed
                                          # separately with an NVRTC "invalid -arch" error
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH --output=/mmfs1/home/songk42/ehd-sindy/logs/%x_%j.log
#SBATCH --time=06:00:00

module load cuda/12.9.1
module load gcc/11.2.0

cd /mmfs1/home/songk42/ehd-sindy
source .venv/bin/activate
# unbuffered: without this, a job killed by the time limit (or preempted on ckpt-all)
# loses every print() that hadn't filled Python's stdout buffer yet -- the previous
# run had real progress (multiple fits completed, per the warnings that did leak
# through) but zero of the actual print_row() output survived.
python -u notebooks/run_multi_window_sweep.py
