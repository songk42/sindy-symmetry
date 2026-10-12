#!/bin/bash
#SBATCH --job-name=r_fidelity
#SBATCH --partition=ckpt-all
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH --output=/mmfs1/home/songk42/ehd-sindy/logs/%x_r%a_%A.log
#SBATCH --time=03:00:00
#SBATCH --array=3-5

# CPU-only fidelity sweep (bounded + simulated switching rate vs real rate), one r
# value per array task. Moved off the login node after a run there was OOM-killed by
# the memcg (node was under heavy load from other users' processes at the time, not
# our own usage -- but the login node isn't a reliable place for a multi-r sweep
# regardless).

set -eo pipefail
cd /mmfs1/home/songk42/ehd-sindy
source .venv/bin/activate
python -u notebooks/run_r_fidelity.py "$SLURM_ARRAY_TASK_ID"
