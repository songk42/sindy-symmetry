#!/bin/bash
#SBATCH --job-name=ehd_convert
#SBATCH --partition=ckpt-all
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH --output=/mmfs1/home/songk42/ehd-sindy/logs/%x_%j.log
#SBATCH --time=02:00:00

# Was OOM-killed (exit 137) running directly on the login node -- pandas.read_csv on
# the two ~11.8GB raw ASCII dumps (charge_data, vorticity) needs real memory headroom
# during parsing (well beyond the ~7.9GB the two final float32 arrays take), and the
# login node's per-user limit isn't enough. Run as a proper slurm allocation instead.

set -eo pipefail
cd /mmfs1/home/songk42/ehd-sindy
source .venv/bin/activate
python -u sindy_data/convert_to_sindy.py --stride 1 --out /gscratch/stf/songk42/ehd/sindy_data_fine.npz
