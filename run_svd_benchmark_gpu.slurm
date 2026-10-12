#!/bin/bash
#SBATCH --job-name=svd_bench_gpu
#SBATCH --partition=ckpt-all
#SBATCH --account=stf
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40|l40s
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH --output=/mmfs1/home/songk42/ehd-sindy/logs/%x_%j.log
#SBATCH --time=01:00:00

module load cuda/12.9.1
module load gcc/11.2.0

set -eo pipefail
cd /mmfs1/home/songk42/ehd-sindy
source .venv/bin/activate
# dense cupy SVD only on the smaller (n_t~25000) file -- the 80000-column file OOMs
# the dense method per compute_pod's docstring, so it only gets the randomized SVD.
python -u notebooks/benchmark_svd.py /gscratch/stf/songk42/ehd/sindy_data_extended.npz cupy 6
python -u notebooks/benchmark_svd.py /gscratch/stf/songk42/ehd/sindy_data_extended.npz cupy_rsvd 6
python -u notebooks/benchmark_svd.py /gscratch/stf/songk42/ehd/sindy_data_fine.npz cupy_rsvd 6
