#!/bin/bash
#SBATCH --job-name=symm_sindy_dysts
#SBATCH --partition=ckpt-all
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH --output=/mmfs1/home/songk42/ehd-sindy/logs/%x_%j.log
#SBATCH --time=4:00:00

cd /mmfs1/home/songk42/ehd-sindy
source .venv/bin/activate
export PYTHONPATH=$HOME/dysts
python notebooks/symmetric_sindy_dysts.py "$@"
