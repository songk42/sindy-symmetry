#!/bin/bash
#SBATCH --job-name=svd_bench_cpu
#SBATCH --partition=ckpt-all
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH --output=/mmfs1/home/songk42/ehd-sindy/logs/%x_%j.log
#SBATCH --time=01:00:00

set -eo pipefail
cd /mmfs1/home/songk42/ehd-sindy
source .venv/bin/activate
python -u notebooks/benchmark_svd.py /gscratch/stf/songk42/ehd/sindy_data_extended.npz svds 6
python -u notebooks/benchmark_svd.py /gscratch/stf/songk42/ehd/sindy_data_fine.npz svds 6
