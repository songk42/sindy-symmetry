#!/bin/bash
#SBATCH --job-name=r_fidelity_sweep
#SBATCH --partition=ckpt-all
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH --output=/mmfs1/home/songk42/ehd-sindy/logs/%x_%j.log
#SBATCH --time=04:00:00

# CPU-only: POD is cached (pod_cache_trimmed_t800.npz), so compute_pod hits the cache
# and never touches cupy/GPU. Runs r=4 and r=5 (r=3 is already running interactively
# and survived one session disconnect already; this exists so the REMAINING r values
# don't depend on an interactive shell/SSH session staying alive too).

set -eo pipefail
cd /mmfs1/home/songk42/ehd-sindy
source .venv/bin/activate
python -u notebooks/run_r_fidelity_sweep.py 4 5
