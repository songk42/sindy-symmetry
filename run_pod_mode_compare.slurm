#!/bin/bash
#SBATCH --job-name=pod_mode_compare
#SBATCH --partition=ckpt-all
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH --output=/mmfs1/home/songk42/ehd-sindy/logs/%x_%j.log
#SBATCH --time=01:00:00

# CPU-only (svds), 5 fresh SVDs (global + 4 windows) each on a duration<=3200 slice of
# sindy_data_fine.npz -- benchmark_svd.py showed the FULL file (n_t=80000) takes ~48s
# via svds with 2 cpus/16G, so this should finish in well under the 1h budget.

set -eo pipefail
cd /mmfs1/home/songk42/ehd-sindy
source .venv/bin/activate
python -u notebooks/compare_pod_modes_across_windows.py
